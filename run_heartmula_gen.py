#!/usr/bin/env python3
"""
HeartMuLa Music Generation 脚本

使用 HeartMuLa-oss-3B 模型生成音乐。

前置条件:
  pip install heartlib
  # 下载模型
  huggingface-cli download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
  huggingface-cli download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year'
  huggingface-cli download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss-20260123'

用法:
  python run_heartmula_gen.py --model_path ./ckpt
"""

import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEARTLIB_DIR = os.path.join(SCRIPT_DIR, "heartlib")

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

TAGS = "R&B,dreamy,moderate tempo,guitar,piano,Thunder,Tube Distortion,Hall Reverb,Guitar delay fx,vocal delay fx,jpop,syncopated drums,synth bass,claps"

DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "ckpt")


def ensure_heartlib():
    try:
        import heartlib
        return
    except ImportError:
        pass

    if os.path.isdir(HEARTLIB_DIR):
        sys.path.insert(0, HEARTLIB_DIR)
        try:
            import heartlib
            return
        except ImportError:
            pass

    print("[INFO] heartlib 未安装，正在克隆并安装...")
    os.system(f"git clone --depth 1 https://github.com/HeartMuLa/heartlib.git {HEARTLIB_DIR}")
    os.system(f"pip install -e {HEARTLIB_DIR}")


def download_models(model_path: str):
    """检查并下载模型权重。"""
    needed = []
    if not os.path.isfile(os.path.join(model_path, "gen_config.json")):
        needed.append(("HeartMuLa/HeartMuLaGen", model_path))
    if not os.path.isdir(os.path.join(model_path, "HeartMuLa-oss-3B")):
        needed.append(("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", os.path.join(model_path, "HeartMuLa-oss-3B")))
    if not os.path.isdir(os.path.join(model_path, "HeartCodec-oss")):
        needed.append(("HeartMuLa/HeartCodec-oss-20260123", os.path.join(model_path, "HeartCodec-oss")))

    if not needed:
        print("[INFO] 模型权重已就绪")
        return

    for repo_id, local_dir in needed:
        print(f"[INFO] 下载 {repo_id} → {local_dir} ...")
        os.makedirs(local_dir, exist_ok=True)
        ret = os.system(f'huggingface-cli download --local-dir "{local_dir}" "{repo_id}"')
        if ret != 0:
            ret = os.system(f'modelscope download --model "{repo_id}" --local_dir "{local_dir}"')
        if ret != 0:
            print(f"[ERROR] 下载 {repo_id} 失败，请手动下载")
            sys.exit(1)


def write_temp_files(lyrics_path, tags_path):
    """将歌词和标签写入临时文件。"""
    with open(lyrics_path, "w", encoding="utf-8") as f:
        f.write(LYRICS)
    with open(tags_path, "w", encoding="utf-8") as f:
        f.write(TAGS)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HeartMuLa Music Generation")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"模型路径（默认: {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--save_path", type=str, default="./output/heartmula_output.mp3",
                        help="输出文件路径")
    parser.add_argument("--max_audio_length_ms", type=int, default=240_000,
                        help="最大音频长度/毫秒")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--mula_device", type=str, default="cuda")
    parser.add_argument("--codec_device", type=str, default="cuda")
    parser.add_argument("--lazy_load", action="store_true",
                        help="按需加载模型（低显存）")
    parser.add_argument("--auto_download", action="store_true", default=True,
                        help="自动下载模型")
    args = parser.parse_args()

    ensure_heartlib()

    if args.auto_download:
        download_models(args.model_path)

    lyrics_path = os.path.join(SCRIPT_DIR, ".tmp_lyrics.txt")
    tags_path = os.path.join(SCRIPT_DIR, ".tmp_tags.txt")
    write_temp_files(lyrics_path, tags_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    from heartlib import HeartMuLaGenPipeline

    print("[INFO] 初始化 HeartMuLa Pipeline...")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device={
            "mula": torch.device(args.mula_device),
            "codec": torch.device(args.codec_device),
        },
        dtype={
            "mula": torch.bfloat16,
            "codec": torch.float32,
        },
        version="3B",
        lazy_load=args.lazy_load,
    )
    print("[INFO] Pipeline 就绪，开始生成...")

    t0 = time.time()
    with torch.no_grad():
        pipe(
            {
                "lyrics": lyrics_path,
                "tags": tags_path,
            },
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    elapsed = time.time() - t0

    os.remove(lyrics_path)
    os.remove(tags_path)

    print(f"\n{'='*60}")
    print(f"✅ HeartMuLa 生成完成！")
    print(f"  输出: {args.save_path}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
