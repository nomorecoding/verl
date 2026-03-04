# ACE-Step 1.5 模型架构与 Forward 流向图

## 一、整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         ACE-Step 1.5 Music Generation System                        │
│                                                                                     │
│  ┌───────────────────────────────────────┐    ┌──────────────────────────────────┐  │
│  │         5Hz LM (Language Model)       │    │      DiT (Diffusion Transformer) │  │
│  │         Qwen3-4B / 1.7B / 0.6B       │    │      + Condition Encoder         │  │
│  │                                       │    │      + Audio Tokenizer           │  │
│  │  用户 prompt ──→ Chain-of-Thought     │    │                                  │  │
│  │                  ├─ caption            │    │   条件信号 ──→ N步去噪 ──→ 潜表示│  │
│  │                  ├─ bpm/key/duration   │    │                                  │  │
│  │                  └─ audio codes (5Hz)  │    │                                  │  │
│  └──────────────┬────────────────────────┘    └───────────┬──────────────────────┘  │
│                 │                                         │                         │
│                 │ caption + audio_codes                   │ latents                 │
│                 ▼                                         ▼                         │
│  ┌──────────────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │   UMT5-XL Text Encoder      │    │           Music-DCAE VAE                 │   │
│  │   (Qwen3-Embedding-0.6B)    │    │   Encoder: audio → latents (25Hz)       │   │
│  │                              │    │   Decoder: latents → audio (48kHz)      │   │
│  │   text → hidden_states       │    │                                          │   │
│  │   lyrics → embeddings        │    │   基于 AutoencoderOobleck               │   │
│  └──────────────────────────────┘    └──────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 二、模型组件详细结构

```
AceStepConditionGenerationModel (主模型)
├── decoder: AceStepDiTModel               ← 核心扩散 Transformer
│   ├── rotary_emb: Qwen3RotaryEmbedding   ← 旋转位置编码
│   ├── proj_in: Conv1d(in_ch → hidden)    ← 将 [context_latents ∥ x_t] patch 化
│   ├── time_embed: TimestepEmbedding      ← 时间步 t 的嵌入
│   ├── time_embed_r: TimestepEmbedding    ← 时间步差 (t-r) 的嵌入
│   ├── condition_embedder: Linear         ← 投影条件 encoder_hidden_states
│   ├── layers: [AceStepDiTLayer × N]      ← N 层 DiT Transformer
│   │   └── 每层包含:
│   │       ├── Self-Attention (带 RoPE)
│   │       ├── Cross-Attention (对 encoder_hidden_states)
│   │       ├── FFN (SwiGLU)
│   │       └── AdaLN (用 timestep 调制)
│   ├── norm_out: RMSNorm + scale_shift_table  ← 自适应输出归一化
│   └── proj_out: ConvTranspose1d          ← 反 patch 化，恢复序列长度
│
├── encoder: AceStepConditionEncoder       ← 条件编码器
│   ├── text_projector: Linear             ← 投影 text_hidden_states → hidden_size
│   ├── lyric_encoder: AceStepLyricEncoder ← 歌词编码器
│   │   ├── embed_tokens: Linear           ← 歌词嵌入投影
│   │   ├── rotary_emb: RoPE
│   │   └── layers: [EncoderLayer × M]    ← M 层双向 Transformer
│   └── timbre_encoder: AceStepTimbreEncoder ← 音色编码器
│       ├── embed_tokens: Linear           ← 参考音频特征投影
│       ├── special_token: [CLS] 参数      ← 聚合音色信息
│       └── layers: [EncoderLayer × K]    ← K 层双向 Transformer
│
├── tokenizer: AceStepAudioTokenizer       ← 音频 tokenizer (25Hz→5Hz)
│   ├── audio_acoustic_proj: Linear        ← 投影声学特征
│   ├── attention_pooler: AttentionPooler  ← 注意力池化 (pool_window=5)
│   └── quantizer: ResidualFSQ             ← 有限标量量化 → 离散 codes
│
├── detokenizer: AudioTokenDetokenizer     ← 反 tokenizer (5Hz→25Hz)
│
└── null_condition_emb: Parameter          ← CFG 空条件嵌入
```

## 三、Music Generation Forward 流向图

### 完整推理流程（text2music 模式）

```
用户输入
  │
  │  caption: "A dreamy synthwave track..."
  │  lyrics:  "[Verse 1]\nWalking down..."
  │  bpm: 120, key: "C minor", duration: 60s
  │
  ▼
╔══════════════════════════════════════════════════════════╗
║  Phase 0: LM Planner (可选，用 5Hz LM 时)               ║
║                                                          ║
║  caption + lyrics ──→ LLMHandler.generate_with_stop()    ║
║                       │                                  ║
║                       ├── Phase 1: CoT 推理              ║
║                       │   <think>                        ║
║                       │   bpm: 120                       ║
║                       │   caption: "dreamy synthwave..." ║
║                       │   duration: 60                   ║
║                       │   keyscale: C minor               ║
║                       │   </think>                       ║
║                       │                                  ║
║                       └── Phase 2: 生成 audio codes      ║
║                           <|audio_code_18953|>           ║
║                           <|audio_code_13833|>...        ║
║                           (5Hz × 60s = 300 tokens)       ║
╚══════════════════════════════╦═══════════════════════════╝
                               │
                               │ caption + audio_code_hints
                               ▼
╔══════════════════════════════════════════════════════════╗
║  Phase 1: 准备条件信号  (_prepare_batch)                  ║
║                                                          ║
║  ┌─────────────────────────────────────────────────────┐ ║
║  │ 1a. 文本 tokenize                                   │ ║
║  │                                                     │ ║
║  │ SFT_GEN_PROMPT.format(instruction, caption, metas)  │ ║
║  │         │                                           │ ║
║  │         ▼                                           │ ║
║  │ text_tokenizer(prompt, max_len=256)                 │ ║
║  │     → text_token_ids [B, L_text]                    │ ║
║  │                                                     │ ║
║  │ text_tokenizer(lyrics, max_len=2048)                │ ║
║  │     → lyric_token_ids [B, L_lyric]                  │ ║
║  └─────────────────────────────────────────────────────┘ ║
║                                                          ║
║  ┌─────────────────────────────────────────────────────┐ ║
║  │ 1b. 目标音频 latent 初始化                           │ ║
║  │                                                     │ ║
║  │ duration=60s → latent_length = 60 × 25 = 1500       │ ║
║  │ target_latents = silence_latent 扩展到 [B, 1500, D] │ ║
║  │ src_latents = zeros [B, 1500, D]                    │ ║
║  └─────────────────────────────────────────────────────┘ ║
║                                                          ║
║  ┌─────────────────────────────────────────────────────┐ ║
║  │ 1c. LM hints 解码 (如有 audio_codes)                │ ║
║  │                                                     │ ║
║  │ audio_code_string                                   │ ║
║  │     │                                               │ ║
║  │     ▼                                               │ ║
║  │ _decode_audio_codes_to_latents()                    │ ║
║  │     │                                               │ ║
║  │     ├── model.tokenizer.quantizer                   │ ║
║  │     │     .get_output_from_indices(codes)            │ ║
║  │     │                                               │ ║
║  │     └── model.detokenizer(quantized)                │ ║
║  │         → lm_hints_25Hz [B, 1500, D]                │ ║
║  └─────────────────────────────────────────────────────┘ ║
╚══════════════════════════════╦═══════════════════════════╝
                               │
                               ▼
╔══════════════════════════════════════════════════════════╗
║  Phase 2: 推理嵌入  (preprocess_batch)                   ║
║                                                          ║
║  ┌─────────────────────────────────────────┐             ║
║  │ 2a. 参考音频编码 (音色)                  │             ║
║  │                                         │             ║
║  │ refer_audio                             │             ║
║  │     │                                   │             ║
║  │     ▼  [VAE context]                    │             ║
║  │ vae.encode(refer_audio)                 │             ║
║  │     .latent_dist.sample()               │             ║
║  │     → refer_audio_latents [N, T, D]     │             ║
║  └─────────────────────────────────────────┘             ║
║                                                          ║
║  ┌─────────────────────────────────────────┐             ║
║  │ 2b. 文本/歌词嵌入                       │             ║
║  │                                         │             ║
║  │ [Text Encoder context]                  │             ║
║  │                                         │             ║
║  │ text_token_ids                          │             ║
║  │     │                                   │             ║
║  │     ▼                                   │             ║
║  │ text_encoder(input_ids)                 │             ║
║  │     .last_hidden_state                  │             ║
║  │     → text_hidden_states [B, L, 1024]   │             ║
║  │                                         │             ║
║  │ lyric_token_ids                         │             ║
║  │     │                                   │             ║
║  │     ▼                                   │             ║
║  │ text_encoder.embed_tokens(ids)          │             ║
║  │     → lyric_hidden_states [B, L, 1024]  │             ║
║  └─────────────────────────────────────────┘             ║
╚══════════════════════════════╦═══════════════════════════╝
                               │
                               │ text_hidden_states, lyric_hidden_states,
                               │ refer_audio_latents, src_latents,
                               │ chunk_masks, lm_hints_25Hz
                               ▼
╔══════════════════════════════════════════════════════════╗
║  Phase 3: 条件融合  (model.prepare_condition)            ║
║                                                          ║
║  ┌─────────────────────────────────────────────────────┐ ║
║  │ 3a. Condition Encoder                               │ ║
║  │                                                     │ ║
║  │ text_hidden_states ──→ text_projector ──→ text_emb  │ ║
║  │                                                     │ ║
║  │ lyric_hidden_states ──→ lyric_encoder ──→ lyric_emb │ ║
║  │   (RoPE + N层 Bidirectional Transformer)            │ ║
║  │                                                     │ ║
║  │ refer_audio_latents ──→ timbre_encoder ──→ timb_emb │ ║
║  │   (CLS token + K层 Bidirectional Transformer)       │ ║
║  │                                                     │ ║
║  │ pack_sequences:                                     │ ║
║  │   [lyric_emb ∥ timbre_emb] ──→ [.. ∥ text_emb]     │ ║
║  │                                                     │ ║
║  │ → encoder_hidden_states [B, L_cond, hidden_size]    │ ║
║  │ → encoder_attention_mask [B, L_cond]                │ ║
║  └─────────────────────────────────────────────────────┘ ║
║                                                          ║
║  ┌─────────────────────────────────────────────────────┐ ║
║  │ 3b. LM Hints → Context Latents                     │ ║
║  │                                                     │ ║
║  │ if is_cover:                                        │ ║
║  │   src_latents = lm_hints_25Hz  (用 LM 生成的 hints)│ ║
║  │ else:                                               │ ║
║  │   src_latents = zeros  (纯文本生成)                 │ ║
║  │                                                     │ ║
║  │ context_latents = [src_latents ∥ chunk_masks]       │ ║
║  │                    [B, T, D + mask_channels]         │ ║
║  └─────────────────────────────────────────────────────┘ ║
╚══════════════════════════════╦═══════════════════════════╝
                               │
                               │ encoder_hidden_states [B, L_cond, H]
                               │ context_latents [B, T, D+mask]
                               ▼
╔══════════════════════════════════════════════════════════╗
║  Phase 4: Flow Matching 扩散  (model.generate_audio)     ║
║                                                          ║
║  初始噪声: x_T ~ N(0, I), shape = [B, T, D]             ║
║                                                          ║
║  时间步: t = linspace(1.0 → 0.0, steps=8) (turbo)       ║
║  如 shift≠1: t = shift·t / (1 + (shift-1)·t)            ║
║                                                          ║
║  for (t_cur, t_next) in zip(t[:-1], t[1:]):              ║
║  ┌─────────────────────────────────────────────────────┐ ║
║  │                                                     │ ║
║  │             ┌──────────────────────┐                │ ║
║  │  x_t ──────→│  DiT Decoder (forward)│               │ ║
║  │             │                      │                │ ║
║  │  inputs:    │  1. proj_in:         │                │ ║
║  │   ├ x_t     │    [ctx ∥ x_t] →    │                │ ║
║  │   ├ t       │    Conv1d → patches  │                │ ║
║  │   ├ ctx     │                      │                │ ║
║  │   └ cond    │  2. time_embed(t)    │                │ ║
║  │             │     → timestep emb   │                │ ║
║  │             │                      │                │ ║
║  │             │  3. N × DiT Layer:   │                │ ║
║  │             │    ├ Self-Attn+RoPE  │                │ ║
║  │             │    ├ Cross-Attn      │── cond         │ ║
║  │             │    ├ FFN (SwiGLU)    │                │ ║
║  │             │    └ AdaLN (t调制)   │                │ ║
║  │             │                      │                │ ║
║  │             │  4. norm + proj_out: │                │ ║
║  │             │    ConvTranspose1d   │                │ ║
║  │             │    → v_t (velocity)  │                │ ║
║  │             └──────────┬───────────┘                │ ║
║  │                        │                            │ ║
║  │                        ▼                            │ ║
║  │  CFG: v = v_cond + scale × (v_cond - v_uncond)     │ ║
║  │                        │                            │ ║
║  │                        ▼                            │ ║
║  │  ODE step: x_{t-1} = x_t + (t_next - t_cur) × v_t │ ║
║  │                                                     │ ║
║  └─────────────────────────────────────────────────────┘ ║
║                                                          ║
║  → target_latents = x_0  [B, T=1500, D=64]              ║
╚══════════════════════════════╦═══════════════════════════╝
                               │
                               │ target_latents [B, 1500, 64]
                               ▼
╔══════════════════════════════════════════════════════════╗
║  Phase 5: VAE 解码  (vae.decode)                         ║
║                                                          ║
║  target_latents [B, 64, 1500]  (转置)                    ║
║       │                                                  ║
║       ▼                                                  ║
║  AutoencoderOobleck.decode()                             ║
║  (分块 tiled decode 以节省显存)                            ║
║       │                                                  ║
║       ▼                                                  ║
║  waveform [B, 2, 2880000]                                ║
║  (48kHz × 60s = 2,880,000 samples, 双声道)               ║
║       │                                                  ║
║       ▼                                                  ║
║  后处理: 裁剪到目标时长 → 保存为 .wav                     ║
╚══════════════════════════════════════════════════════════╝
```

### DiT Decoder 单步 Forward 详细流向

```
输入:
  x_t:            [B, T, D]           噪声化的音频 latent
  context_latents:[B, T, D+mask_ch]   源 latent + chunk mask
  timestep t:     [B]                 当前扩散时间步
  encoder_hidden_states: [B, L_cond, H]   融合后的条件序列

                      ┌─────────────────────────────────┐
                      │        Input Processing          │
                      │                                  │
  [context_latents ∥ x_t]  →  [B, T, D+mask+D]          │
                      │           │                      │
                      │     pad to patch_size             │
                      │           │                      │
                      │     Conv1d (patch_size=5)         │
                      │     stride=patch_size             │
                      │           │                      │
                      │     → patches [B, T/5, H]        │
                      └───────────┬─────────────────────┘
                                  │
                      ┌───────────▼─────────────────────┐
                      │     Timestep Embedding           │
                      │                                  │
  t ──→ time_embed(t) + time_embed_r(t-r) → temb [B, H] │
                      │                                  │
                      └───────────┬─────────────────────┘
                                  │
  encoder_hidden_states ──→ condition_embedder(Linear)
                      │       → cond [B, L_cond, H]
                      │
            ┌─────────▼───────────────────────────────────┐
            │                                              │
            │   ╔═══ DiT Layer ×N (e.g. 24 layers) ═══╗  │
            │   ║                                       ║  │
            │   ║  patches                              ║  │
            │   ║    │                                  ║  │
            │   ║    ▼                                  ║  │
            │   ║  AdaLN (modulated by temb)            ║  │
            │   ║    │                                  ║  │
            │   ║    ▼                                  ║  │
            │   ║  Self-Attention (+ RoPE)              ║  │
            │   ║  Q,K,V from patches                   ║  │
            │   ║  (滑动窗口 or 全局)                    ║  │
            │   ║    │                                  ║  │
            │   ║    ▼                                  ║  │
            │   ║  Cross-Attention                      ║  │
            │   ║  Q: patches, K/V: cond                ║  │
            │   ║    │                                  ║  │
            │   ║    ▼                                  ║  │
            │   ║  FFN (SwiGLU)                         ║  │
            │   ║    │                                  ║  │
            │   ║    ▼                                  ║  │
            │   ║  → updated patches                    ║  │
            │   ║                                       ║  │
            │   ╚═══════════════════════════════════════╝  │
            │                                              │
            └─────────┬───────────────────────────────────┘
                      │
                      ▼
            ┌─────────────────────────────────────────────┐
            │  Output Processing                           │
            │                                              │
            │  scale, shift = scale_shift_table (from temb)│
            │  patches = AdaLN(patches, scale, shift)      │
            │  patches = RMSNorm(patches)                  │
            │         │                                    │
            │         ▼                                    │
            │  ConvTranspose1d (patch_size=5)               │
            │  (反 patch: T/5 → T)                         │
            │         │                                    │
            │         ▼                                    │
            │  v_t: [B, T, D]  (predicted velocity field)  │
            └─────────────────────────────────────────────┘
```

### Music Caption (反向理解) Forward 流向

```
音频文件
  │
  ▼
╔════════════════════════════════════════════╗
║  Step 1: 音频编码为 Audio Codes            ║
║                                            ║
║  audio.wav                                 ║
║    │                                       ║
║    ▼                                       ║
║  vae.encode(audio).latent_dist.sample()    ║
║    → latents [1, T, 64]  (25Hz)            ║
║    │                                       ║
║    ▼                                       ║
║  model.tokenize(latents, silence, mask)    ║
║    ├── reshape: [1, T/5, 5, 64]            ║
║    ├── AudioTokenizer.forward():           ║
║    │   ├── audio_acoustic_proj(Linear)     ║
║    │   ├── attention_pooler (5→1)          ║
║    │   └── ResidualFSQ.quantize()          ║
║    │       → indices [1, T/5]  (5Hz)       ║
║    │                                       ║
║    └── 序列化:                              ║
║        "<|audio_code_18953|>               ║
║         <|audio_code_13833|>..."           ║
╚════════════════════╦═══════════════════════╝
                     │
                     │  audio_codes string
                     ▼
╔════════════════════════════════════════════╗
║  Step 2: LLM 反向理解                     ║
║                                            ║
║  System: "Understand the given musical     ║
║   conditions and describe the audio        ║
║   semantics accordingly:"                  ║
║                                            ║
║  User: "<|audio_code_18953|>               ║
║         <|audio_code_13833|>..."           ║
║                                            ║
║  LLM (Qwen3-4B) generates:                ║
║                                            ║
║  <think>                                   ║
║  bpm: 120                                  ║
║  caption: A dreamy synthwave track with    ║
║    lush pads and arpeggiated synths...     ║
║  duration: 60                              ║
║  keyscale: C minor                         ║
║  language: en                              ║
║  timesignature: 4                          ║
║  </think>                                  ║
║  [Verse 1]                                 ║
║  Walking down the empty street...          ║
║                                            ║
║  → metadata dict + lyrics                  ║
╚════════════════════════════════════════════╝
```

## 四、关键维度参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 48,000 Hz | 输出音频采样率 |
| VAE 时间压缩率 | ×1920 | 48000/25 = 1920 samples → 1 latent frame |
| VAE latent rate | 25 Hz | 每秒 25 个 latent frames |
| VAE latent dim | 64 | 每个 latent frame 的维度 |
| Tokenizer pool_window | 5 | 25Hz → 5Hz (注意力池化) |
| Audio code rate | 5 Hz | LM 生成的 audio code 频率 |
| Codebook size | 64,000 | ResidualFSQ 离散码本大小 |
| DiT hidden_size | 2048 | Transformer 隐藏层维度 |
| DiT patch_size | 5 | 1D Conv patch 大小 |
| Text max_length | 256 tokens | caption prompt 最大长度 |
| Lyric max_length | 2048 tokens | 歌词最大长度 |
| Text encoder dim | 1024 | Qwen3-Embedding 输出维度 |
| Turbo infer_steps | 8 | turbo 模式扩散步数 |
| Base infer_steps | 50 | base 模式扩散步数 |
| 60s 歌曲 latent 长度 | 1500 | 60 × 25Hz = 1500 frames |
| 60s 歌曲 LM tokens | 300 | 60 × 5Hz = 300 audio codes |

## 五、数据流维度变换总结

```
用户输入 caption + lyrics
        │
        ▼
[Text Tokenizer]
  caption → token_ids [B, ≤256]
  lyrics  → token_ids [B, ≤2048]
        │
        ▼
[Text Encoder (Qwen3-Embedding)]
  caption → text_hidden_states   [B, L_text, 1024]
  lyrics  → lyric_hidden_states  [B, L_lyric, 1024]  (仅 embed_tokens)
        │
        ▼
[Condition Encoder]
  text_projector:  [B, L_text, 1024]  → [B, L_text, 2048]
  lyric_encoder:   [B, L_lyric, 1024] → [B, L_lyric, 2048]
  timbre_encoder:  [N_ref, T_ref, D]  → [B, N_timbre, 2048]
  pack_sequences:  → encoder_hidden_states [B, L_total, 2048]
        │
        ▼
[prepare_condition]
  context_latents = [src_latents ∥ chunk_masks]  [B, 1500, 64+mask]
        │
        ▼
[DiT Diffusion Loop × 8 steps]
  input:   [context_latents ∥ x_t]  →  Conv1d  →  [B, 300, 2048]
  process: 24 layers DiT (self-attn + cross-attn + FFN)
  output:  ConvTranspose1d  →  v_t [B, 1500, 64]
  update:  x_{t-1} = x_t + dt × v_t
        │
        ▼
  x_0 = target_latents [B, 1500, 64]
        │
        ▼
[VAE Decoder]
  latents [B, 64, 1500]  →  AutoencoderOobleck  →  waveform [B, 2, 2880000]
        │
        ▼
  输出: 48kHz 立体声 WAV 文件
```
