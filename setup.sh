#!/bin/bash
# ACE-Step 1.5 Music Caption 部署脚本
# 用法: bash setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACEST_DIR="$SCRIPT_DIR/ACE-Step-1.5"

echo "============================================"
echo "ACE-Step 1.5 Music Caption 自动化部署"
echo "============================================"

# 1. 检查 GPU
echo ""
echo "[1/5] 检查 GPU 环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  ✅ NVIDIA GPU 已检测到"
else
    echo "  ⚠️  未检测到 NVIDIA GPU，将使用 CPU 模式（速度较慢）"
fi

# 2. 克隆仓库
echo ""
echo "[2/5] 准备 ACE-Step-1.5 仓库..."
if [ -d "$ACEST_DIR" ]; then
    echo "  ✅ ACE-Step-1.5 已存在，跳过克隆"
    cd "$ACEST_DIR" && git pull --ff-only 2>/dev/null || true
    cd "$SCRIPT_DIR"
else
    echo "  正在克隆 ACE-Step-1.5..."
    git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "$ACEST_DIR"
fi

# 3. 安装 Python 依赖
echo ""
echo "[3/5] 安装 Python 依赖..."
cd "$ACEST_DIR"

pip install --upgrade pip

# 安装 PyTorch（根据系统自动选择）
if command -v nvidia-smi &> /dev/null; then
    echo "  安装 PyTorch (CUDA)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  安装 PyTorch (CPU)..."
    pip install torch torchvision torchaudio
fi

echo "  安装核心依赖..."
pip install transformers>=4.51.0 diffusers safetensors accelerate
pip install scipy soundfile loguru einops
pip install vector-quantize-pytorch>=1.27.15
pip install peft>=0.18.0
pip install fastapi uvicorn[standard]
pip install numba torchao toml modelscope xxhash pyyaml

# 尝试安装 flash-attn 和 triton（非必需）
if command -v nvidia-smi &> /dev/null; then
    echo "  尝试安装 flash-attn（可选）..."
    pip install flash-attn 2>/dev/null || echo "  ⚠️  flash-attn 安装失败（不影响使用）"
    pip install triton 2>/dev/null || echo "  ⚠️  triton 安装失败（不影响使用）"
fi

# 安装 nano-vllm（如存在）
NANO_VLLM="$ACEST_DIR/acestep/third_parts/nano-vllm"
if [ -d "$NANO_VLLM" ]; then
    echo "  安装 nano-vllm..."
    pip install -e "$NANO_VLLM" 2>/dev/null || echo "  ⚠️  nano-vllm 安装失败（不影响使用，将使用 pt 后端）"
fi

cd "$SCRIPT_DIR"

# 4. 下载模型
echo ""
echo "[4/5] 检查/下载模型权重..."

CHECKPOINT_DIR="$ACEST_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# 下载 DiT 模型
if [ ! -d "$CHECKPOINT_DIR/acestep-v15-turbo" ]; then
    echo "  下载 DiT 模型 (acestep-v15-turbo)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/ACE-Step-v1-5-turbo', local_dir='$CHECKPOINT_DIR/acestep-v15-turbo')
print('  ✅ DiT 模型下载完成')
" || echo "  ⚠️  DiT 模型下载失败，运行时将自动重试"
else
    echo "  ✅ DiT 模型已存在"
fi

# 下载 VAE
if [ ! -d "$CHECKPOINT_DIR/music-dcae-v15" ]; then
    echo "  下载 VAE 模型 (music-dcae-v15)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/music-dcae-v15', local_dir='$CHECKPOINT_DIR/music-dcae-v15')
print('  ✅ VAE 模型下载完成')
" || echo "  ⚠️  VAE 模型下载失败，运行时将自动重试"
else
    echo "  ✅ VAE 模型已存在"
fi

# 下载 Text Encoder
if [ ! -d "$CHECKPOINT_DIR/umt5-xl" ]; then
    echo "  下载 Text Encoder (umt5-xl)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/umt5-xl', local_dir='$CHECKPOINT_DIR/umt5-xl')
print('  ✅ Text Encoder 下载完成')
" || echo "  ⚠️  Text Encoder 下载失败，运行时将自动重试"
else
    echo "  ✅ Text Encoder 已存在"
fi

# 下载 5Hz LM 4B
if [ ! -d "$CHECKPOINT_DIR/acestep-5Hz-lm-4B" ]; then
    echo "  下载 5Hz LM 模型 (acestep-5Hz-lm-4B)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ACE-Step/acestep-5Hz-lm-4B', local_dir='$CHECKPOINT_DIR/acestep-5Hz-lm-4B')
print('  ✅ 5Hz LM 4B 模型下载完成')
" || echo "  ⚠️  LM 模型下载失败，运行时将自动重试"
else
    echo "  ✅ 5Hz LM 4B 模型已存在"
fi

# 5. 完成
echo ""
echo "[5/5] 验证安装..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f'  显存: {total_mem / 1024**3:.1f} GB')
import transformers
print(f'  Transformers: {transformers.__version__}')
"

echo ""
echo "============================================"
echo "✅ 部署完成！"
echo ""
echo "使用方式："
echo ""
echo "  # CLI 批量处理"
echo "  python music_caption.py --audio_dir /path/to/music"
echo ""
echo "  # 启动 HTTP API 服务"
echo "  python music_caption_server.py --port 8080"
echo ""
echo "  # 使用 curl 调用 API"
echo "  curl -X POST http://localhost:8080/caption \\"
echo "       -F 'file=@song.mp3'"
echo "============================================"
