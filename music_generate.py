#!/usr/bin/env python3
"""
ACE-Step 1.5 Music Generation 脚本

支持两种生成模式:
  1. LM+DiT 模式 (--mode llm_dit): LM 先生成 audio codes，再由 DiT 合成音频
  2. 纯 DiT 模式 (--mode dit): 仅用 DiT 从文本直接生成（不加载 LM，节省显存）

用法:
  # LM+DiT 模式（最佳质量，需要更多显存）
  python music_generate.py --caption "A dreamy synthwave track" --lyrics "[Verse]\\nHello world"

  # 纯 DiT 模式（无需 LM，节省显存）
  python music_generate.py --mode dit --caption "A jazz piano piece" --duration 30

  # 批量生成（同一 prompt 生成多个变体）
  python music_generate.py --caption "Epic orchestral" --batch_size 4

  # 从 JSON 配置文件批量生成
  python music_generate.py --from_json tasks.json
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

ACEST_REPO_URL = "https://github.com/ace-step/ACE-Step-1.5.git"
ACEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACE-Step-1.5")


def patch_no_meta_tensor_loading():
    from contextlib import contextmanager as _cm

    @_cm
    def _noop(*a, **kw):
        yield

    for mod_path in [
        "transformers.modeling_utils",
        "accelerate",
        "accelerate.big_modeling",
    ]:
        try:
            mod = __import__(mod_path, fromlist=["init_empty_weights"])
            if hasattr(mod, "init_empty_weights"):
                mod.init_empty_weights = _noop
        except (ImportError, AttributeError):
            pass
    print("[PATCH] Disabled init_empty_weights for FSQ compatibility")


def ensure_acestep_installed():
    if not os.path.isdir(ACEST_DIR):
        print(f"[INFO] 正在克隆 ACE-Step-1.5 仓库到 {ACEST_DIR} ...")
        ret = os.system(f"git clone --depth 1 {ACEST_REPO_URL} {ACEST_DIR}")
        if ret != 0:
            print("[ERROR] 克隆失败，请手动克隆:")
            print(f"  git clone {ACEST_REPO_URL} {ACEST_DIR}")
            sys.exit(1)
    if ACEST_DIR not in sys.path:
        sys.path.insert(0, ACEST_DIR)
    nano_vllm = os.path.join(ACEST_DIR, "acestep", "third_parts", "nano-vllm")
    if os.path.isdir(nano_vllm) and nano_vllm not in sys.path:
        sys.path.insert(0, nano_vllm)


def init_dit_handler(device="auto", offload_to_cpu=False, quantization=None):
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    print("[INFO] 正在初始化 DiT 模型...")
    status, success = handler.initialize_service(
        project_root=ACEST_DIR,
        config_path="acestep-v15-turbo",
        device=device,
        offload_to_cpu=offload_to_cpu,
        quantization=quantization,
    )
    if not success:
        print(f"[ERROR] DiT 模型初始化失败:\n{status}")
        sys.exit(1)
    print("[INFO] DiT 模型就绪")
    return handler


def init_llm_handler(lm_model="acestep-5Hz-lm-4B", backend="auto", device="auto"):
    from acestep.llm_inference import LLMHandler

    handler = LLMHandler()
    print(f"[INFO] 正在初始化 LLM 模型: {lm_model} ...")
    checkpoint_dir = os.path.join(ACEST_DIR, "checkpoints")
    status, success = handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model,
        backend=backend,
        device=device,
    )
    if not success:
        print(f"[ERROR] LLM 初始化失败: {status}")
        sys.exit(1)
    print(f"[INFO] LLM 模型就绪 (backend={handler.llm_backend})")
    return handler


def generate_single(
    dit_handler,
    llm_handler,
    caption: str,
    lyrics: str = "",
    mode: str = "llm_dit",
    duration: float = 60.0,
    bpm: Optional[int] = None,
    key_scale: str = "",
    time_signature: str = "",
    language: str = "en",
    batch_size: int = 1,
    seed: int = -1,
    inference_steps: int = 8,
    guidance_scale: float = 7.0,
    output_dir: str = "./output",
    tag: str = "",
) -> Dict[str, Any]:
    """生成一首或多首音乐。"""
    import torch
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    audio_code_string = ""
    metadata = {}

    if mode == "llm_dit" and llm_handler is not None:
        print(f"  [Phase 0] LM 规划: caption → CoT → audio codes ...")
        lm_result = llm_handler.generate_with_stop_condition(
            caption=caption,
            lyrics=lyrics,
            infer_type="llm_dit",
            temperature=0.85,
            target_duration=duration,
            user_metadata={
                "bpm": str(bpm) if bpm else None,
                "keyscale": key_scale or None,
                "timesignature": time_signature or None,
                "duration": str(int(duration)),
            },
            batch_size=batch_size,
            use_constrained_decoding=True,
        )

        if not lm_result.get("success"):
            return {"success": False, "error": lm_result.get("error", "LM generation failed")}

        if batch_size > 1:
            audio_code_string = lm_result["audio_codes"]
            metadata = lm_result["metadata"][0] if lm_result["metadata"] else {}
        else:
            audio_code_string = lm_result["audio_codes"]
            metadata = lm_result.get("metadata", {})

        lm_time = time.time() - t0
        codes_count = (
            sum(c.count("<|audio_code_") for c in audio_code_string)
            if isinstance(audio_code_string, list)
            else audio_code_string.count("<|audio_code_")
        )
        print(f"  [Phase 0] LM 完成 ({lm_time:.1f}s), 生成 {codes_count} audio codes")
        caption = metadata.get("caption", caption)

    print(f"  [Phase 1-5] DiT 扩散生成 + VAE 解码 ...")
    t_dit = time.time()

    result = dit_handler.generate_music(
        captions=caption,
        lyrics=lyrics,
        bpm=bpm,
        key_scale=key_scale,
        time_signature=time_signature,
        vocal_language=language,
        audio_duration=duration,
        batch_size=batch_size,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        use_random_seed=(seed < 0),
        audio_code_string=audio_code_string,
        task_type="text2music",
    )

    dit_time = time.time() - t_dit
    total_time = time.time() - t0

    if not result.get("success", False) and not result.get("audios"):
        return {
            "success": False,
            "error": result.get("error", result.get("status_message", "DiT generation failed")),
        }

    audios = result.get("audios", [])
    saved_files = []
    sample_rate = getattr(dit_handler, "sample_rate", 48000)

    for i, audio_tensor in enumerate(audios):
        if hasattr(audio_tensor, "cpu"):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = audio_tensor

        if audio_np.ndim == 2:
            audio_np = audio_np.T

        suffix = f"_{tag}" if tag else ""
        filename = f"gen{suffix}_{i}" if batch_size > 1 else f"gen{suffix}"
        filepath = os.path.join(output_dir, f"{filename}.wav")
        sf.write(filepath, audio_np, sample_rate)
        saved_files.append(filepath)
        print(f"  ✅ 已保存: {filepath}")

    return {
        "success": True,
        "files": saved_files,
        "metadata": metadata,
        "time": {
            "total": round(total_time, 1),
            "dit": round(dit_time, 1),
        },
    }


def run_from_json(json_path: str, dit_handler, llm_handler, args):
    """从 JSON 配置文件批量生成。"""
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if isinstance(tasks, dict):
        tasks = [tasks]

    print(f"[INFO] 从 {json_path} 加载了 {len(tasks)} 个生成任务")
    all_results = []

    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(tasks)}] {task.get('caption', '')[:60]}...")
        print(f"{'='*60}")

        result = generate_single(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            caption=task.get("caption", ""),
            lyrics=task.get("lyrics", ""),
            mode=args.mode,
            duration=task.get("duration", args.duration),
            bpm=task.get("bpm", args.bpm),
            key_scale=task.get("key_scale", args.key_scale),
            time_signature=task.get("time_signature", args.time_signature),
            language=task.get("language", args.language),
            batch_size=task.get("batch_size", args.batch_size),
            seed=task.get("seed", args.seed),
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            output_dir=args.output_dir,
            tag=task.get("tag", f"task{i}"),
        )
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step 1.5 Music Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # LM+DiT 模式（默认，最佳质量）
  python music_generate.py \\
      --caption "A dreamy synthwave track with lush pads" \\
      --lyrics "[Verse 1]\\nWalking down the street" \\
      --duration 60

  # 纯 DiT 模式（不加载 LM，节省显存）
  python music_generate.py --mode dit \\
      --caption "Calm piano jazz" --duration 30

  # 批量生成多个变体
  python music_generate.py \\
      --caption "Epic orchestral battle theme" \\
      --batch_size 4

  # 从 JSON 文件批量生成
  python music_generate.py --from_json tasks.json

tasks.json 格式:
  [
    {"caption": "A pop song", "lyrics": "[Chorus]\\nLa la la", "duration": 60},
    {"caption": "Jazz piano", "duration": 30, "bpm": 120}
  ]
        """,
    )

    parser.add_argument("--caption", type=str, default="",
                        help="音乐描述 (caption)")
    parser.add_argument("--lyrics", type=str, default="",
                        help="歌词文本（支持 [Verse]/[Chorus] 结构标签）")
    parser.add_argument("--lyrics_file", type=str, default=None,
                        help="从文件读取歌词")
    parser.add_argument("--from_json", type=str, default=None,
                        help="从 JSON 文件加载批量生成任务")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录（默认: ./output）")

    parser.add_argument("--mode", type=str, default="llm_dit",
                        choices=["dit", "llm_dit"],
                        help="生成模式: dit=纯DiT, llm_dit=LM+DiT（默认）")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="音频时长/秒（默认: 60）")
    parser.add_argument("--bpm", type=int, default=None,
                        help="BPM（留空则自动）")
    parser.add_argument("--key_scale", type=str, default="",
                        help="调式，如 'C major', 'A minor'")
    parser.add_argument("--time_signature", type=str, default="",
                        help="拍号，如 '4'（=4/4）")
    parser.add_argument("--language", type=str, default="en",
                        help="歌词语言（默认: en）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="同时生成的变体数（默认: 1）")
    parser.add_argument("--seed", type=int, default=-1,
                        help="随机种子（-1=随机）")
    parser.add_argument("--inference_steps", type=int, default=8,
                        help="扩散步数（turbo=8, base=50，默认: 8）")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                        help="CFG 引导强度（默认: 7.0）")

    parser.add_argument("--lm_model", type=str, default="acestep-5Hz-lm-4B",
                        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"],
                        help="LM 模型（默认: 4B）")
    parser.add_argument("--lm_backend", type=str, default="auto",
                        choices=["auto", "vllm", "nano-vllm", "pt"],
                        help="LLM 推理后端")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备（auto/cuda/cpu）")
    parser.add_argument("--offload_to_cpu", action="store_true",
                        help="DiT CPU offload（低显存）")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "int8_weight_only", "fp8_weight_only"],
                        help="DiT 量化")

    args = parser.parse_args()

    if args.lyrics_file:
        with open(args.lyrics_file, "r", encoding="utf-8") as f:
            args.lyrics = f.read()

    if not args.caption and not args.from_json:
        parser.error("请提供 --caption 或 --from_json")

    ensure_acestep_installed()
    patch_no_meta_tensor_loading()

    dit_handler = init_dit_handler(
        device=args.device,
        offload_to_cpu=args.offload_to_cpu,
        quantization=args.quantization,
    )

    llm_handler = None
    if args.mode == "llm_dit":
        llm_handler = init_llm_handler(
            lm_model=args.lm_model,
            backend=args.lm_backend,
            device=args.device,
        )

    if args.from_json:
        results = run_from_json(args.from_json, dit_handler, llm_handler, args)
        success_count = sum(1 for r in results if r.get("success"))
        print(f"\n{'='*60}")
        print(f"批量生成完成: {success_count}/{len(results)} 成功")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"开始生成音乐")
        print(f"  模式: {args.mode}")
        print(f"  caption: {args.caption[:80]}...")
        print(f"  时长: {args.duration}s, batch: {args.batch_size}")
        print(f"{'='*60}")

        result = generate_single(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            caption=args.caption,
            lyrics=args.lyrics,
            mode=args.mode,
            duration=args.duration,
            bpm=args.bpm,
            key_scale=args.key_scale,
            time_signature=args.time_signature,
            language=args.language,
            batch_size=args.batch_size,
            seed=args.seed,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            output_dir=args.output_dir,
        )

        print(f"\n{'='*60}")
        if result["success"]:
            print(f"✅ 生成成功！")
            print(f"  文件: {result['files']}")
            print(f"  耗时: {result['time']['total']}s (DiT: {result['time']['dit']}s)")
        else:
            print(f"❌ 生成失败: {result['error']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
