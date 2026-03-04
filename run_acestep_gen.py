#!/usr/bin/env python3
"""
ACE-Step 1.5 Music Generation 脚本（使用与 HeartMuLa 相同的 prompt/lyrics）

使用 acestep-5Hz-lm-4B + DiT turbo 模式生成音乐。

用法:
  python run_acestep_gen.py
  python run_acestep_gen.py --mode dit          # 纯 DiT 模式（不加载 LM）
  python run_acestep_gen.py --batch_size 4      # 生成 4 个变体
"""

import os
import sys
import time
from contextlib import contextmanager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACEST_DIR = os.path.join(SCRIPT_DIR, "ACE-Step-1.5")

CAPTION = (
    "[R&B] [dreamy] [moderate tempo] [featuring guitar, piano] [Thunder] "
    "[Tube Distortion] [Hall Reverb] [Guitar delay fx] [vocal delay fx] "
    "Kicking off with syncopated drums, tight claps, and a buoyant synth bass, "
    "this punchy jpop anthem bursts with energy."
)

LYRICS = """\
[Intro]

[Verse 1]
灯影摇 晚风轻绕
舞厅里 旋律漫飘
你眼眸 星子落巢
漫过我 心头浪潮
琴音柔 萨克斯轻挑
心事随 节奏慢摇
这夜色 温柔难描
唯有你 是我解药

[Pre-Chorus]

[Chorus]
老上海的夜 藏着深情未了
与你并肩 时光都变轻巧
爵士声里 诉说岁月静好
这份情 熬成温柔的牢
老上海的风 吹着爱意昭昭
牵你的手 走过暮暮朝朝
烟火人间 有你就足够好
余生路 与你慢慢变老

[Verse 2]
杯影晃 酒香萦绕
指尖触 温柔刚好
你笑容 月色漫照
驱散我 所有寂寥
旧唱片 转着时光遥
爱意在 旋律里飘
这夜晚 漫漫长绕
唯有你 解我烦恼

[Pre-Chorus]

[Chorus]
老上海的夜 藏着深情未了
与你并肩 时光都变轻巧
爵士声里 诉说岁月静好
这份情 熬成温柔的牢
老上海的风 吹着爱意昭昭
牵你的手 走过暮暮朝朝
烟火人间 有你就足够好
余生路 与你慢慢变老

[Interlude]

[Chorus]
老上海的夜 藏着深情未了
与你并肩 时光都变轻巧
爵士声里 诉说岁月静好
余生路 与你慢慢变老

[Outro]
(ooh ooh)
"""

DURATION = 220.0


def patch_no_meta_tensor_loading():
    @contextmanager
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


def ensure_acestep():
    if not os.path.isdir(ACEST_DIR):
        print(f"[INFO] 克隆 ACE-Step-1.5 ...")
        os.system(f"git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git {ACEST_DIR}")
    if ACEST_DIR not in sys.path:
        sys.path.insert(0, ACEST_DIR)
    nano_vllm = os.path.join(ACEST_DIR, "acestep", "third_parts", "nano-vllm")
    if os.path.isdir(nano_vllm) and nano_vllm not in sys.path:
        sys.path.insert(0, nano_vllm)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ACE-Step 1.5 Music Generation (demo)")
    parser.add_argument("--mode", type=str, default="llm_dit", choices=["dit", "llm_dit"],
                        help="生成模式")
    parser.add_argument("--save_path", type=str, default="./output/acestep_output.wav",
                        help="输出文件路径")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--lm_model", type=str, default="acestep-5Hz-lm-4B",
                        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"])
    parser.add_argument("--lm_backend", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offload_to_cpu", action="store_true")
    parser.add_argument("--inference_steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    args = parser.parse_args()

    ensure_acestep()
    patch_no_meta_tensor_loading()

    import torch
    import soundfile as sf
    from acestep.handler import AceStepHandler

    print("[INFO] 初始化 DiT 模型...")
    dit_handler = AceStepHandler()
    status, success = dit_handler.initialize_service(
        project_root=ACEST_DIR,
        config_path="acestep-v15-turbo",
        device=args.device,
        offload_to_cpu=args.offload_to_cpu,
    )
    if not success:
        print(f"[ERROR] DiT 初始化失败: {status}")
        sys.exit(1)
    print("[INFO] DiT 模型就绪")

    llm_handler = None
    audio_code_string = ""
    metadata = {}

    if args.mode == "llm_dit":
        from acestep.llm_inference import LLMHandler

        print(f"[INFO] 初始化 LLM: {args.lm_model} ...")
        llm_handler = LLMHandler()
        checkpoint_dir = os.path.join(ACEST_DIR, "checkpoints")
        status, success = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=args.lm_model,
            backend=args.lm_backend,
            device=args.device,
        )
        if not success:
            print(f"[ERROR] LLM 初始化失败: {status}")
            sys.exit(1)
        print(f"[INFO] LLM 就绪 (backend={llm_handler.llm_backend})")

        print("[Phase 0] LM 规划: CoT → audio codes ...")
        t_lm = time.time()
        lm_result = llm_handler.generate_with_stop_condition(
            caption=CAPTION,
            lyrics=LYRICS,
            infer_type="llm_dit",
            temperature=0.85,
            target_duration=DURATION,
            user_metadata={"duration": str(int(DURATION))},
            batch_size=args.batch_size,
            use_constrained_decoding=True,
        )
        lm_time = time.time() - t_lm

        if not lm_result.get("success"):
            print(f"[ERROR] LM 生成失败: {lm_result.get('error')}")
            sys.exit(1)

        audio_code_string = lm_result["audio_codes"]
        metadata = (
            lm_result["metadata"][0]
            if isinstance(lm_result["metadata"], list) and lm_result["metadata"]
            else lm_result.get("metadata", {})
        )
        codes_count = (
            sum(c.count("<|audio_code_") for c in audio_code_string)
            if isinstance(audio_code_string, list)
            else audio_code_string.count("<|audio_code_")
        )
        print(f"[Phase 0] LM 完成 ({lm_time:.1f}s), {codes_count} audio codes")

    caption = metadata.get("caption", CAPTION) if metadata else CAPTION

    print("[Phase 1-5] DiT 扩散 + VAE 解码 ...")
    t_dit = time.time()

    result = dit_handler.generate_music(
        captions=caption,
        lyrics=LYRICS,
        vocal_language="zh",
        audio_duration=DURATION,
        batch_size=args.batch_size,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_random_seed=(args.seed < 0),
        audio_code_string=audio_code_string,
        task_type="text2music",
    )
    dit_time = time.time() - t_dit

    if not result.get("success", False) and not result.get("audios"):
        print(f"[ERROR] DiT 生成失败: {result.get('error', result.get('status_message'))}")
        sys.exit(1)

    audios = result.get("audios", [])
    sample_rate = getattr(dit_handler, "sample_rate", 48000)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    saved = []
    for i, audio_tensor in enumerate(audios):
        audio_np = audio_tensor.cpu().numpy() if hasattr(audio_tensor, "cpu") else audio_tensor
        if audio_np.ndim == 2:
            audio_np = audio_np.T

        if args.batch_size > 1:
            base, ext = os.path.splitext(args.save_path)
            filepath = f"{base}_{i}{ext}"
        else:
            filepath = args.save_path

        sf.write(filepath, audio_np, sample_rate)
        saved.append(filepath)
        print(f"  ✅ {filepath}")

    total_time = (lm_time if args.mode == "llm_dit" else 0) + dit_time
    print(f"\n{'='*60}")
    print(f"✅ ACE-Step 生成完成！")
    print(f"  文件: {saved}")
    print(f"  耗时: {total_time:.1f}s (DiT: {dit_time:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
