#!/usr/bin/env python3
"""
ACE-Step 1.5 自动化 Music Caption 脚本

使用 acestep-5Hz-lm-4B 模型对音频文件进行自动标注，生成：
  - caption（音乐风格描述）
  - bpm / keyscale / timesignature / language
  - lyrics（歌词转录）

依赖：ACE-Step-1.5 仓库（会自动克隆）
硬件：需要 NVIDIA GPU，建议 20GB+ 显存

用法:
  # 处理单个文件
  python music_caption.py --audio /path/to/song.mp3

  # 批量处理整个目录
  python music_caption.py --audio_dir /path/to/music_folder

  # 指定输出目录
  python music_caption.py --audio_dir /path/to/music_folder --output_dir /path/to/output

  # 使用更小的模型（显存不足时）
  python music_caption.py --audio_dir /path/to/music_folder --lm_model acestep-5Hz-lm-0.6B

  # CPU offload 模式（低显存）
  python music_caption.py --audio_dir /path/to/music_folder --offload_to_cpu
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".opus"}

ACEST_REPO_URL = "https://github.com/ace-step/ACE-Step-1.5.git"
ACEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACE-Step-1.5")


def patch_no_meta_tensor_loading():
    """
    Replace accelerate's init_empty_weights with a no-op in transformers.

    For sharded models, transformers ALWAYS uses init_empty_weights() (meta device)
    regardless of low_cpu_mem_usage. This breaks vector_quantize_pytorch.FSQ which
    calls .item() on buffers during __init__. Disabling init_empty_weights makes the
    model instantiate with real (random) tensors on CPU, then weights are loaded
    normally. Uses more RAM during init but avoids meta tensor issues entirely.
    """
    from contextlib import contextmanager

    @contextmanager
    def _noop_init_empty_weights(*args, **kwargs):
        yield

    patched = []

    try:
        import transformers.modeling_utils as mu
        if hasattr(mu, "init_empty_weights"):
            mu.init_empty_weights = _noop_init_empty_weights
            patched.append("transformers.modeling_utils")
    except (ImportError, AttributeError):
        pass

    try:
        import accelerate
        accelerate.init_empty_weights = _noop_init_empty_weights
        patched.append("accelerate")
    except (ImportError, AttributeError):
        pass

    try:
        import accelerate.big_modeling
        accelerate.big_modeling.init_empty_weights = _noop_init_empty_weights
        patched.append("accelerate.big_modeling")
    except (ImportError, AttributeError):
        pass

    if patched:
        print(f"[PATCH] Disabled init_empty_weights in: {', '.join(patched)}")
    else:
        print("[WARN] Could not patch init_empty_weights - meta tensor issues may occur")


def ensure_acestep_installed():
    """确保 ACE-Step-1.5 仓库已克隆并在 Python 路径中。"""
    if not os.path.isdir(ACEST_DIR):
        print(f"[INFO] 正在克隆 ACE-Step-1.5 仓库到 {ACEST_DIR} ...")
        ret = os.system(f"git clone --depth 1 {ACEST_REPO_URL} {ACEST_DIR}")
        if ret != 0:
            print("[ERROR] 克隆 ACE-Step-1.5 失败，请手动克隆:")
            print(f"  git clone {ACEST_REPO_URL} {ACEST_DIR}")
            sys.exit(1)

    if ACEST_DIR not in sys.path:
        sys.path.insert(0, ACEST_DIR)

    nano_vllm_path = os.path.join(ACEST_DIR, "acestep", "third_parts", "nano-vllm")
    if os.path.isdir(nano_vllm_path) and nano_vllm_path not in sys.path:
        sys.path.insert(0, nano_vllm_path)


def scan_audio_files(path: str) -> List[str]:
    """扫描目录中的所有音频文件，或返回单个音频文件路径。"""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in SUPPORTED_AUDIO_EXTENSIONS:
            return [path]
        print(f"[WARN] 不支持的音频格式: {ext}")
        return []

    if os.path.isdir(path):
        files = []
        for root, _, filenames in os.walk(path):
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_AUDIO_EXTENSIONS:
                    files.append(os.path.join(root, fname))
        return files

    print(f"[ERROR] 路径不存在: {path}")
    return []


def initialize_dit_handler(
    project_root: str,
    device: str = "auto",
    offload_to_cpu: bool = False,
    quantization: Optional[str] = None,
):
    """初始化 DiT handler（用于音频编码为 audio codes）。"""
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    print("[INFO] 正在初始化 DiT 模型（用于音频编码）...")
    status_msg, success = handler.initialize_service(
        project_root=project_root,
        config_path="acestep-v15-turbo",
        device=device,
        offload_to_cpu=offload_to_cpu,
        quantization=quantization,
    )
    if not success:
        print(f"[ERROR] DiT 模型初始化失败:\n{status_msg}")
        sys.exit(1)
    print(f"[INFO] DiT 模型初始化完成")
    return handler


def initialize_llm_handler(
    lm_model: str = "acestep-5Hz-lm-4B",
    backend: str = "auto",
    device: str = "auto",
):
    """初始化 5Hz LM handler（用于音频理解/caption 生成）。"""
    from acestep.llm_inference import LLMHandler

    handler = LLMHandler()
    print(f"[INFO] 正在初始化 LLM 模型: {lm_model} ...")

    model_path = os.path.join(ACEST_DIR, "checkpoints", lm_model)
    if not os.path.isdir(model_path):
        print(f"[INFO] 模型 {lm_model} 不在本地，将自动从 HuggingFace 下载...")
        model_path = f"ACE-Step/{lm_model}"

    status = handler.initialize(
        model_path=model_path,
        backend=backend,
        device=device,
    )
    if not handler.llm_initialized:
        print(f"[ERROR] LLM 模型初始化失败: {status}")
        sys.exit(1)
    print(f"[INFO] LLM 模型初始化完成 (backend={handler.llm_backend})")
    return handler


def caption_single_audio(
    audio_path: str,
    dit_handler,
    llm_handler,
    temperature: float = 0.7,
) -> Tuple[Dict[str, Any], str]:
    """
    对单个音频文件进行 caption 标注。

    返回: (metadata_dict, status_message)
    metadata_dict 包含: caption, bpm, keyscale, timesignature, language, lyrics, genres
    """
    import torch

    filename = os.path.basename(audio_path)
    print(f"  [1/2] 编码音频: {filename} ...")

    with torch.inference_mode():
        audio_codes = dit_handler.convert_src_audio_to_codes(audio_path)

    if not audio_codes or audio_codes.startswith("❌"):
        return {}, f"编码失败: {audio_codes}"

    num_codes = audio_codes.count("<|audio_code_")
    print(f"  [1/2] 编码完成，生成 {num_codes} 个 audio codes")
    print(f"  [2/2] LLM 理解音频并生成 caption ...")

    metadata, status = llm_handler.understand_audio_from_codes(
        audio_codes=audio_codes,
        temperature=temperature,
        use_constrained_decoding=True,
    )

    return metadata, status


def save_caption_result(
    audio_path: str,
    metadata: Dict[str, Any],
    output_dir: Optional[str] = None,
):
    """保存 caption 结果到 JSON 和 caption.txt 文件。"""
    stem = Path(audio_path).stem
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_path = os.path.join(output_dir, stem)
    else:
        base_path = os.path.join(os.path.dirname(audio_path), stem)

    json_data = {
        "caption": metadata.get("caption", ""),
        "bpm": metadata.get("bpm", ""),
        "keyscale": metadata.get("keyscale", ""),
        "timesignature": metadata.get("timesignature", ""),
        "language": metadata.get("vocal_language", metadata.get("language", "")),
    }
    json_path = f"{base_path}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    caption = metadata.get("caption", "")
    if caption:
        caption_path = f"{base_path}.caption.txt"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

    lyrics = metadata.get("lyrics", "")
    if lyrics:
        lyrics_path = f"{base_path}.lyrics.txt"
        with open(lyrics_path, "w", encoding="utf-8") as f:
            f.write(lyrics)

    return json_path


def process_batch(
    audio_files: List[str],
    dit_handler,
    llm_handler,
    output_dir: Optional[str] = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """批量处理音频文件并生成 caption。"""
    results = {"success": 0, "failed": 0, "total": len(audio_files), "details": []}

    for i, audio_path in enumerate(audio_files):
        filename = os.path.basename(audio_path)
        print(f"\n[{i+1}/{len(audio_files)}] 处理: {filename}")
        t0 = time.time()

        try:
            metadata, status = caption_single_audio(
                audio_path, dit_handler, llm_handler, temperature
            )

            if metadata:
                json_path = save_caption_result(audio_path, metadata, output_dir)
                elapsed = time.time() - t0
                print(f"  ✅ 完成 ({elapsed:.1f}s) -> {json_path}")
                print(f"     caption: {metadata.get('caption', '')[:100]}...")
                results["success"] += 1
                results["details"].append({
                    "file": filename,
                    "status": "success",
                    "time": round(elapsed, 1),
                    "caption": metadata.get("caption", ""),
                })
            else:
                elapsed = time.time() - t0
                print(f"  ❌ 失败 ({elapsed:.1f}s): {status}")
                results["failed"] += 1
                results["details"].append({
                    "file": filename,
                    "status": "failed",
                    "error": status,
                })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ 异常 ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            results["failed"] += 1
            results["details"].append({
                "file": filename,
                "status": "error",
                "error": str(e),
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step 1.5 自动化 Music Caption 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python music_caption.py --audio song.mp3

  # 批量处理目录
  python music_caption.py --audio_dir ./music_folder

  # 指定输出目录和模型
  python music_caption.py --audio_dir ./music_folder --output_dir ./captions --lm_model acestep-5Hz-lm-4B

  # 低显存模式
  python music_caption.py --audio_dir ./music_folder --offload_to_cpu --lm_model acestep-5Hz-lm-0.6B
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str, help="单个音频文件路径")
    input_group.add_argument("--audio_dir", type=str, help="音频文件目录路径")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认与音频同目录）")
    parser.add_argument("--lm_model", type=str, default="acestep-5Hz-lm-4B",
                        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"],
                        help="5Hz LM 模型大小（默认: 4B）")
    parser.add_argument("--lm_backend", type=str, default="auto",
                        choices=["auto", "vllm", "nano-vllm", "pt"],
                        help="LLM 推理后端（默认: auto）")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备（auto/cuda/cpu，默认: auto）")
    parser.add_argument("--offload_to_cpu", action="store_true",
                        help="启用 CPU offload（低显存时使用）")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "int8_weight_only", "fp8_weight_only"],
                        help="DiT 模型量化（降低显存占用）")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM 生成温度（默认: 0.7）")
    parser.add_argument("--summary_output", type=str, default=None,
                        help="保存处理摘要的 JSON 文件路径")

    args = parser.parse_args()

    ensure_acestep_installed()
    patch_no_meta_tensor_loading()

    audio_path = args.audio or args.audio_dir
    audio_files = scan_audio_files(audio_path)
    if not audio_files:
        print("[ERROR] 未找到支持的音频文件")
        sys.exit(1)

    print(f"[INFO] 找到 {len(audio_files)} 个音频文件")

    project_root = ACEST_DIR
    dit_handler = initialize_dit_handler(
        project_root=project_root,
        device=args.device,
        offload_to_cpu=args.offload_to_cpu,
        quantization=args.quantization,
    )

    llm_handler = initialize_llm_handler(
        lm_model=args.lm_model,
        backend=args.lm_backend,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print(f"开始批量处理 {len(audio_files)} 个音频文件")
    print("=" * 60)

    t_start = time.time()
    results = process_batch(
        audio_files=audio_files,
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )
    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print(f"处理完成！")
    print(f"  成功: {results['success']}/{results['total']}")
    print(f"  失败: {results['failed']}/{results['total']}")
    print(f"  总耗时: {total_time:.1f}s")
    if results["success"] > 0:
        avg_time = total_time / results["success"]
        print(f"  平均每首: {avg_time:.1f}s")
    print("=" * 60)

    if args.summary_output:
        results["total_time"] = round(total_time, 1)
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 处理摘要已保存到: {args.summary_output}")


if __name__ == "__main__":
    main()
