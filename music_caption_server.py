#!/usr/bin/env python3
"""
ACE-Step 1.5 Music Caption HTTP API 服务

提供 REST API 接口对音频文件进行自动 caption 标注。

启动方式:
  python music_caption_server.py --port 8080

API 端点:
  POST /caption        - 上传音频文件，返回 caption 结果
  POST /caption_batch  - 上传多个音频文件，返回批量 caption 结果
  GET  /health         - 健康检查
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

ACEST_REPO_URL = "https://github.com/ace-step/ACE-Step-1.5.git"
ACEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACE-Step-1.5")
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".opus"}

dit_handler = None
llm_handler = None
_lock = asyncio.Lock()

app = FastAPI(
    title="ACE-Step Music Caption API",
    description="使用 acestep-5Hz-lm 模型对音频进行自动化 caption 标注",
    version="1.0.0",
)


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
    if not os.path.isdir(ACEST_DIR):
        print(f"[INFO] 正在克隆 ACE-Step-1.5 仓库到 {ACEST_DIR} ...")
        ret = os.system(f"git clone --depth 1 {ACEST_REPO_URL} {ACEST_DIR}")
        if ret != 0:
            raise RuntimeError("克隆 ACE-Step-1.5 失败")
    if ACEST_DIR not in sys.path:
        sys.path.insert(0, ACEST_DIR)
    nano_vllm_path = os.path.join(ACEST_DIR, "acestep", "third_parts", "nano-vllm")
    if os.path.isdir(nano_vllm_path) and nano_vllm_path not in sys.path:
        sys.path.insert(0, nano_vllm_path)


def init_models(
    lm_model: str = "acestep-5Hz-lm-4B",
    lm_backend: str = "auto",
    device: str = "auto",
    offload_to_cpu: bool = False,
    quantization: Optional[str] = None,
):
    global dit_handler, llm_handler
    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    print("[INFO] 初始化 DiT 模型...")
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=ACEST_DIR,
        config_path="acestep-v15-turbo",
        device=device,
        offload_to_cpu=offload_to_cpu,
        quantization=quantization,
    )
    if not success:
        raise RuntimeError(f"DiT 初始化失败: {status_msg}")
    print("[INFO] DiT 模型就绪")

    print(f"[INFO] 初始化 LLM 模型: {lm_model} ...")
    llm_handler = LLMHandler()
    model_path = os.path.join(ACEST_DIR, "checkpoints", lm_model)
    if not os.path.isdir(model_path):
        model_path = f"ACE-Step/{lm_model}"
    llm_handler.initialize(model_path=model_path, backend=lm_backend, device=device)
    if not llm_handler.llm_initialized:
        raise RuntimeError("LLM 初始化失败")
    print(f"[INFO] LLM 模型就绪 (backend={llm_handler.llm_backend})")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dit_ready": dit_handler is not None and dit_handler.model is not None,
        "llm_ready": llm_handler is not None and llm_handler.llm_initialized,
    }


@app.post("/caption")
async def caption_audio(
    file: UploadFile = File(...),
    temperature: float = Query(0.7, ge=0.0, le=2.0),
):
    """上传单个音频文件，返回 caption、lyrics 和元数据。"""
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(400, f"不支持的音频格式: {ext}")

    async with _lock:
        import torch

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            t0 = time.time()

            with torch.inference_mode():
                audio_codes = dit_handler.convert_src_audio_to_codes(tmp_path)
            if not audio_codes or audio_codes.startswith("❌"):
                raise HTTPException(500, f"音频编码失败: {audio_codes}")

            metadata, status = llm_handler.understand_audio_from_codes(
                audio_codes=audio_codes,
                temperature=temperature,
                use_constrained_decoding=True,
            )
            elapsed = time.time() - t0

            if not metadata:
                raise HTTPException(500, f"Caption 生成失败: {status}")

            return JSONResponse({
                "filename": file.filename,
                "caption": metadata.get("caption", ""),
                "bpm": metadata.get("bpm", ""),
                "keyscale": metadata.get("keyscale", ""),
                "timesignature": metadata.get("timesignature", ""),
                "language": metadata.get("vocal_language", metadata.get("language", "")),
                "genres": metadata.get("genres", ""),
                "lyrics": metadata.get("lyrics", ""),
                "processing_time_seconds": round(elapsed, 2),
            })
        finally:
            os.unlink(tmp_path)


@app.post("/caption_batch")
async def caption_batch(
    files: list[UploadFile] = File(...),
    temperature: float = Query(0.7, ge=0.0, le=2.0),
):
    """上传多个音频文件，批量返回 caption 结果。"""
    results = []
    for file in files:
        ext = Path(file.filename or "audio.wav").suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            results.append({"filename": file.filename, "error": f"不支持的格式: {ext}"})
            continue

        async with _lock:
            import torch

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                t0 = time.time()
                with torch.inference_mode():
                    audio_codes = dit_handler.convert_src_audio_to_codes(tmp_path)
                if not audio_codes or audio_codes.startswith("❌"):
                    results.append({"filename": file.filename, "error": f"编码失败: {audio_codes}"})
                    continue

                metadata, status = llm_handler.understand_audio_from_codes(
                    audio_codes=audio_codes,
                    temperature=temperature,
                    use_constrained_decoding=True,
                )
                elapsed = time.time() - t0

                if metadata:
                    results.append({
                        "filename": file.filename,
                        "caption": metadata.get("caption", ""),
                        "bpm": metadata.get("bpm", ""),
                        "keyscale": metadata.get("keyscale", ""),
                        "timesignature": metadata.get("timesignature", ""),
                        "language": metadata.get("vocal_language", metadata.get("language", "")),
                        "genres": metadata.get("genres", ""),
                        "lyrics": metadata.get("lyrics", ""),
                        "processing_time_seconds": round(elapsed, 2),
                    })
                else:
                    results.append({"filename": file.filename, "error": f"Caption 失败: {status}"})
            finally:
                os.unlink(tmp_path)

    return JSONResponse({"results": results, "total": len(results)})


def main():
    parser = argparse.ArgumentParser(description="ACE-Step Music Caption API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--lm_model", type=str, default="acestep-5Hz-lm-4B",
                        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"])
    parser.add_argument("--lm_backend", type=str, default="auto",
                        choices=["auto", "vllm", "nano-vllm", "pt"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offload_to_cpu", action="store_true")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "int8_weight_only", "fp8_weight_only"])

    args = parser.parse_args()

    ensure_acestep_installed()
    patch_no_meta_tensor_loading()
    init_models(
        lm_model=args.lm_model,
        lm_backend=args.lm_backend,
        device=args.device,
        offload_to_cpu=args.offload_to_cpu,
        quantization=args.quantization,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
