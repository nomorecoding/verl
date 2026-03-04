#!/usr/bin/env python3
"""
ACE-Step 1.5 Music Generation HTTP API 服务

启动方式:
  python music_generate_server.py --port 8080

API 端点:
  POST /generate   - 提交音乐生成任务，返回生成的音频文件
  GET  /health     - 健康检查
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ACEST_REPO_URL = "https://github.com/ace-step/ACE-Step-1.5.git"
ACEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACE-Step-1.5")

dit_handler = None
llm_handler = None
_lock = asyncio.Lock()

app = FastAPI(
    title="ACE-Step Music Generation API",
    description="使用 ACE-Step 1.5 生成音乐",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    caption: str = Field(..., description="音乐描述")
    lyrics: str = Field("", description="歌词文本")
    mode: str = Field("llm_dit", description="生成模式: dit / llm_dit")
    duration: float = Field(60.0, ge=5.0, le=240.0, description="时长/秒")
    bpm: Optional[int] = Field(None, ge=40, le=300, description="BPM")
    key_scale: str = Field("", description="调式")
    time_signature: str = Field("", description="拍号")
    language: str = Field("en", description="歌词语言")
    batch_size: int = Field(1, ge=1, le=8, description="生成数量")
    seed: int = Field(-1, description="随机种子 (-1=随机)")
    inference_steps: int = Field(8, ge=1, le=100, description="扩散步数")
    guidance_scale: float = Field(7.0, ge=1.0, le=20.0, description="CFG 强度")


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


def ensure_acestep_installed():
    if not os.path.isdir(ACEST_DIR):
        os.system(f"git clone --depth 1 {ACEST_REPO_URL} {ACEST_DIR}")
    if ACEST_DIR not in sys.path:
        sys.path.insert(0, ACEST_DIR)
    nano_vllm = os.path.join(ACEST_DIR, "acestep", "third_parts", "nano-vllm")
    if os.path.isdir(nano_vllm) and nano_vllm not in sys.path:
        sys.path.insert(0, nano_vllm)


def init_models(
    lm_model="acestep-5Hz-lm-4B",
    lm_backend="auto",
    device="auto",
    offload_to_cpu=False,
    quantization=None,
    load_lm=True,
):
    global dit_handler, llm_handler
    import torch
    from acestep.handler import AceStepHandler

    print("[INFO] 初始化 DiT 模型...")
    dit_handler = AceStepHandler()
    status, success = dit_handler.initialize_service(
        project_root=ACEST_DIR,
        config_path="acestep-v15-turbo",
        device=device,
        offload_to_cpu=offload_to_cpu,
        quantization=quantization,
    )
    if not success:
        raise RuntimeError(f"DiT 初始化失败: {status}")
    print("[INFO] DiT 模型就绪")

    if load_lm:
        from acestep.llm_inference import LLMHandler

        print(f"[INFO] 初始化 LLM 模型: {lm_model} ...")
        llm_handler = LLMHandler()
        checkpoint_dir = os.path.join(ACEST_DIR, "checkpoints")
        status, success = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model,
            backend=lm_backend,
            device=device,
        )
        if not success:
            raise RuntimeError(f"LLM 初始化失败: {status}")
        print(f"[INFO] LLM 模型就绪 (backend={llm_handler.llm_backend})")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dit_ready": dit_handler is not None and dit_handler.model is not None,
        "llm_ready": llm_handler is not None and llm_handler.llm_initialized,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    """生成音乐，返回 WAV 文件（batch_size>1 时返回 ZIP）。"""
    if dit_handler is None or dit_handler.model is None:
        raise HTTPException(503, "DiT 模型未就绪")

    if req.mode == "llm_dit" and (llm_handler is None or not llm_handler.llm_initialized):
        raise HTTPException(503, "LLM 模型未就绪，请使用 mode=dit 或启动时加载 LLM")

    async with _lock:
        import torch
        import soundfile as sf
        import numpy as np

        t0 = time.time()
        audio_code_string = ""
        metadata = {}

        if req.mode == "llm_dit" and llm_handler is not None:
            lm_result = llm_handler.generate_with_stop_condition(
                caption=req.caption,
                lyrics=req.lyrics,
                infer_type="llm_dit",
                temperature=0.85,
                target_duration=req.duration,
                user_metadata={
                    "bpm": str(req.bpm) if req.bpm else None,
                    "keyscale": req.key_scale or None,
                    "timesignature": req.time_signature or None,
                    "duration": str(int(req.duration)),
                },
                batch_size=req.batch_size,
                use_constrained_decoding=True,
            )

            if not lm_result.get("success"):
                raise HTTPException(500, f"LM 生成失败: {lm_result.get('error')}")

            audio_code_string = lm_result["audio_codes"]
            metadata = (
                lm_result["metadata"][0]
                if isinstance(lm_result["metadata"], list) and lm_result["metadata"]
                else lm_result.get("metadata", {})
            )

        caption = metadata.get("caption", req.caption) if metadata else req.caption

        result = dit_handler.generate_music(
            captions=caption,
            lyrics=req.lyrics,
            bpm=req.bpm,
            key_scale=req.key_scale,
            time_signature=req.time_signature,
            vocal_language=req.language,
            audio_duration=req.duration,
            batch_size=req.batch_size,
            inference_steps=req.inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            use_random_seed=(req.seed < 0),
            audio_code_string=audio_code_string,
            task_type="text2music",
        )

        total_time = time.time() - t0

        if not result.get("success", False) and not result.get("audios"):
            raise HTTPException(500, f"生成失败: {result.get('error', result.get('status_message'))}")

        audios = result.get("audios", [])
        sample_rate = getattr(dit_handler, "sample_rate", 48000)

        if len(audios) == 1:
            audio_np = audios[0].cpu().numpy() if hasattr(audios[0], "cpu") else audios[0]
            if audio_np.ndim == 2:
                audio_np = audio_np.T
            buf = io.BytesIO()
            sf.write(buf, audio_np, sample_rate, format="WAV")
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=generated.wav",
                    "X-Processing-Time": str(round(total_time, 2)),
                    "X-Caption": (metadata.get("caption", ""))[:200],
                },
            )
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, audio_tensor in enumerate(audios):
                    audio_np = audio_tensor.cpu().numpy() if hasattr(audio_tensor, "cpu") else audio_tensor
                    if audio_np.ndim == 2:
                        audio_np = audio_np.T
                    wav_buf = io.BytesIO()
                    sf.write(wav_buf, audio_np, sample_rate, format="WAV")
                    zf.writestr(f"generated_{i}.wav", wav_buf.getvalue())

                zf.writestr("metadata.json", json.dumps({
                    "caption": req.caption,
                    "metadata": metadata,
                    "processing_time": round(total_time, 2),
                    "count": len(audios),
                }, ensure_ascii=False, indent=2))

            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=generated.zip",
                    "X-Processing-Time": str(round(total_time, 2)),
                },
            )


def main():
    parser = argparse.ArgumentParser(description="ACE-Step Music Generation API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--lm_model", type=str, default="acestep-5Hz-lm-4B",
                        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"])
    parser.add_argument("--lm_backend", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offload_to_cpu", action="store_true")
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--no_lm", action="store_true",
                        help="不加载 LM 模型（仅支持 dit 模式）")

    args = parser.parse_args()

    ensure_acestep_installed()
    patch_no_meta_tensor_loading()
    init_models(
        lm_model=args.lm_model,
        lm_backend=args.lm_backend,
        device=args.device,
        offload_to_cpu=args.offload_to_cpu,
        quantization=args.quantization,
        load_lm=not args.no_lm,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
