# ACE-Step 1.5 自动化 Music Caption 部署方案

使用 [acestep-5Hz-lm-4B](https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B) 模型对音频文件进行自动化标注，生成 caption（音乐风格描述）、BPM、调式、拍号、语言、歌词等元数据。适用于 LoRA 训练数据准备等场景。

## 原理

ACE-Step 1.5 的 music caption 流程分为两步：

1. **音频编码**：使用 DiT 模型 + VAE 将音频文件编码为 audio codes（`<|audio_code_XXXXX|>` 格式的语义 token 序列）
2. **音频理解**：使用 5Hz LM（语言模型）对 audio codes 进行反向推理，通过 Chain-of-Thought 生成 caption、歌词和元数据

```
音频文件 → [VAE 编码] → [DiT tokenize] → audio codes → [5Hz LM 理解] → caption + lyrics + metadata
```

## 硬件要求

| 配置 | 说明 |
|------|------|
| GPU 显存 ≥ 20GB | 同时加载 DiT + LM-4B 模型 |
| GPU 显存 ≥ 12GB | 使用 `--offload_to_cpu` 或使用较小的 LM-0.6B |
| GPU 显存 ≥ 8GB | 使用 `--offload_to_cpu --quantization int8_weight_only --lm_model acestep-5Hz-lm-0.6B` |

推荐：NVIDIA A100 / RTX 4090 / RTX 3090

## 快速开始

### 一键部署

```bash
git clone <this-repo> && cd <this-repo>
bash setup.sh
```

`setup.sh` 会自动完成：
- 克隆 ACE-Step-1.5 仓库
- 安装所有 Python 依赖
- 下载模型权重（DiT、VAE、Text Encoder、5Hz LM 4B）

### 手动部署

```bash
# 1. 克隆 ACE-Step-1.5
git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git

# 2. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers>=4.51.0 diffusers safetensors accelerate
pip install scipy soundfile loguru einops peft>=0.18.0
pip install vector-quantize-pytorch>=1.27.15 numba torchao toml modelscope xxhash pyyaml
pip install fastapi uvicorn[standard]

# 3. 模型权重会在首次运行时自动从 HuggingFace 下载
#    也可以提前手动下载到 ACE-Step-1.5/checkpoints/ 目录
```

## 使用方式

### 方式一：CLI 命令行（适合批量处理）

```bash
# 处理单个音频文件
python music_caption.py --audio song.mp3

# 批量处理整个目录
python music_caption.py --audio_dir /path/to/music_folder

# 指定输出目录
python music_caption.py --audio_dir /path/to/music_folder --output_dir /path/to/output

# 低显存模式
python music_caption.py --audio_dir /path/to/music_folder \
    --offload_to_cpu \
    --lm_model acestep-5Hz-lm-0.6B
```

**输出文件**（每个音频文件对应）：
- `song.json` — 元数据（caption、bpm、keyscale、timesignature、language）
- `song.caption.txt` — 纯文本 caption
- `song.lyrics.txt` — 歌词

### 方式二：HTTP API 服务（适合集成到系统中）

```bash
# 启动服务
python music_caption_server.py --port 8080

# 调用 API
curl -X POST http://localhost:8080/caption \
     -F "file=@song.mp3"

# 批量调用
curl -X POST http://localhost:8080/caption_batch \
     -F "files=@song1.mp3" \
     -F "files=@song2.mp3"

# 健康检查
curl http://localhost:8080/health
```

**API 响应示例：**

```json
{
    "filename": "song.mp3",
    "caption": "A high-energy J-pop track with synthesizer leads, fast tempo, and catchy vocal hooks",
    "bpm": 190,
    "keyscale": "D major",
    "timesignature": "4",
    "language": "ja",
    "genres": "j-pop, electronic, synth-pop",
    "lyrics": "[Verse 1]\nWalking down the street...\n[Chorus]\nWe are the stars...",
    "processing_time_seconds": 3.42
}
```

### 方式三：在 ACE-Step Gradio UI 中使用

```bash
cd ACE-Step-1.5
python acestep/acestep_v15_pipeline.py
```

在 Gradio UI 中：
1. 初始化时勾选 `acestep-5Hz-lm-4B` 模型
2. 切换到 **LoRA Training** 选项卡
3. 扫描数据集目录
4. 点击 **Auto Label Data** 按钮

## CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--audio` | — | 单个音频文件路径 |
| `--audio_dir` | — | 音频文件目录 |
| `--output_dir` | 与音频同目录 | 输出文件目录 |
| `--lm_model` | `acestep-5Hz-lm-4B` | LM 模型（0.6B / 1.7B / 4B） |
| `--lm_backend` | `auto` | 推理后端（auto / vllm / nano-vllm / pt） |
| `--device` | `auto` | 设备（auto / cuda / cpu） |
| `--offload_to_cpu` | False | DiT 模型 CPU offload（节省显存） |
| `--quantization` | None | DiT 量化（int8_weight_only / fp8_weight_only） |
| `--temperature` | 0.7 | LLM 生成温度 |

## Music Generation（音乐生成）

### CLI 命令行

```bash
# LM+DiT 模式（最佳质量，LM 规划 + DiT 合成）
python music_generate.py \
    --caption "A dreamy synthwave track with lush pads" \
    --lyrics "[Verse 1]\nWalking down the street" \
    --duration 60

# 纯 DiT 模式（不加载 LM，节省显存）
python music_generate.py --mode dit \
    --caption "Calm piano jazz" --duration 30

# 批量生成多个变体
python music_generate.py --caption "Epic orchestral" --batch_size 4

# 从 JSON 文件批量生成
python music_generate.py --from_json tasks.json
```

**tasks.json 格式：**

```json
[
  {"caption": "A pop song with catchy hooks", "lyrics": "[Chorus]\nLa la la", "duration": 60},
  {"caption": "Ambient piano", "duration": 30, "bpm": 80, "tag": "ambient"}
]
```

### HTTP API 服务

```bash
# 启动服务
python music_generate_server.py --port 8080

# 生成音乐（返回 WAV 文件）
curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{"caption": "A dreamy synthwave track", "duration": 30}' \
     -o output.wav

# 纯 DiT 模式启动（不加载 LM）
python music_generate_server.py --port 8080 --no_lm
```

## 文件结构

```
.
├── setup.sh                  # 一键部署脚本
├── music_caption.py          # Music Caption CLI 工具
├── music_caption_server.py   # Music Caption HTTP API
├── music_generate.py         # Music Generation CLI 工具
├── music_generate_server.py  # Music Generation HTTP API
├── architecture.md           # 模型架构与 forward 流向图
├── README.md                 # 本文档
└── ACE-Step-1.5/             # (自动克隆) ACE-Step 1.5 仓库
    ├── acestep/
    │   ├── handler.py            # DiT handler (音频编码)
    │   ├── llm_inference.py      # LLM handler (音频理解/caption)
    │   └── ...
    └── checkpoints/              # (自动下载) 模型权重
        ├── acestep-v15-turbo/    # DiT 模型
        ├── music-dcae-v15/       # VAE 模型
        ├── umt5-xl/              # Text Encoder
        └── acestep-5Hz-lm-4B/   # 5Hz LM 4B 模型
```

## 注意事项

- 首次运行会自动下载约 15GB 的模型权重，请确保网络通畅
- 如果 HuggingFace 下载速度慢，可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- caption 结果的质量与 LM 模型大小正相关，4B > 1.7B > 0.6B
- BPM 和 Key 由 LM 推理生成，可能存在误差，建议使用 [Key-BPM-Finder](https://vocalremover.org/key-bpm-finder) 获取更准确的值
- 歌词转录可能存在错别字，建议人工检查
