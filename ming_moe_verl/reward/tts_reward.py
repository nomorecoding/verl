"""
TTS reward functions for Flow-GRPO training of Ming-omni-tts.

We define multiple reward signals that together drive the MoE+Flow model
to generate higher-quality speech:

1. **Speaker Similarity**: cosine similarity between generated and
   reference speaker embeddings (if a reference exists).
2. **Intelligibility (ASR)**: WER/CER between the ASR transcript of the
   generated audio and the original text prompt.
3. **Audio Quality (MOS predictor)**: UTMOS or similar non-intrusive MOS
   prediction.
4. **Duration Penalty**: penalises outputs that are too long or too short
   relative to the expected duration.

Each sub-reward returns a scalar per sample; we combine them with
configurable weights to produce the final reward used by GRPO.
"""

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


def compute_tts_reward(
    audio_latents: torch.Tensor,
    prompt_text: str,
    reference_audio: Optional[torch.Tensor] = None,
    audio_decoder=None,
    asr_model=None,
    mos_model=None,
    spk_model=None,
    weights: Optional[Dict[str, float]] = None,
    max_expected_steps: int = 200,
) -> Dict[str, float]:
    """
    Compute a composite reward for a single TTS generation.

    Parameters
    ----------
    audio_latents : [T, latent_dim * patch_size]
        The generated audio latent sequence.
    prompt_text : str
        The input text that was synthesised.
    reference_audio : optional Tensor
        Reference waveform for speaker similarity.
    audio_decoder : optional nn.Module
        AudioVAE decoder to convert latents → waveform.
    asr_model : optional
        ASR pipeline / model for intelligibility scoring.
    mos_model : optional
        MOS prediction model.
    spk_model : optional
        Speaker embedding model.
    weights : dict
        Reward component weights, e.g. {"intelligibility": 0.4, "mos": 0.3,
        "duration": 0.1, "speaker_sim": 0.2}.
    max_expected_steps : int
        Maximum number of steps considered normal for duration normalisation.

    Returns
    -------
    dict with "score" (float) and individual component scores.
    """
    if weights is None:
        weights = {
            "intelligibility": 0.4,
            "mos": 0.3,
            "duration": 0.1,
            "speaker_sim": 0.2,
        }

    scores = {}
    T = audio_latents.shape[0]

    # ---- Duration reward (always available) ----
    # Mild Gaussian-shaped reward centred at a "reasonable" length
    expected = max(5, min(T, max_expected_steps))
    duration_ratio = T / max(expected, 1)
    scores["duration"] = math.exp(-0.5 * ((duration_ratio - 1.0) / 0.3) ** 2)

    # ---- Decode to waveform if decoder is available ----
    waveform = None
    if audio_decoder is not None:
        try:
            with torch.no_grad():
                latent_for_decode = audio_latents.unsqueeze(0)
                waveform = audio_decoder.decode(latent_for_decode)[0]
        except Exception:
            waveform = None

    # ---- ASR Intelligibility ----
    if waveform is not None and asr_model is not None:
        try:
            hypothesis = asr_model(waveform)
            if isinstance(hypothesis, list):
                hypothesis = hypothesis[0]
            if isinstance(hypothesis, dict):
                hypothesis = hypothesis.get("text", "")
            ref_lower = prompt_text.lower().strip()
            hyp_lower = hypothesis.lower().strip()
            if ref_lower:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, ref_lower, hyp_lower).ratio()
                scores["intelligibility"] = ratio
            else:
                scores["intelligibility"] = 0.5
        except Exception:
            scores["intelligibility"] = 0.0
    else:
        scores["intelligibility"] = 0.5  # neutral fallback

    # ---- MOS prediction ----
    if waveform is not None and mos_model is not None:
        try:
            with torch.no_grad():
                mos_score = mos_model(waveform)
                if isinstance(mos_score, torch.Tensor):
                    mos_score = mos_score.item()
                scores["mos"] = mos_score / 5.0  # normalise to [0,1]
        except Exception:
            scores["mos"] = 0.5
    else:
        scores["mos"] = 0.5

    # ---- Speaker similarity ----
    if waveform is not None and spk_model is not None and reference_audio is not None:
        try:
            with torch.no_grad():
                gen_emb = spk_model(waveform)
                ref_emb = spk_model(reference_audio)
                cos_sim = F.cosine_similarity(
                    gen_emb.flatten().unsqueeze(0),
                    ref_emb.flatten().unsqueeze(0),
                ).item()
                scores["speaker_sim"] = max(0.0, cos_sim)
        except Exception:
            scores["speaker_sim"] = 0.0
    else:
        scores["speaker_sim"] = 0.0

    # ---- Weighted combination ----
    total = sum(weights.get(k, 0.0) * scores.get(k, 0.0) for k in weights)
    scores["score"] = total
    return scores


class TTSRewardManager:
    """
    Reward manager for TTS Flow-GRPO.

    Compatible with verl's reward manager interface.  In practice you can
    register this via the verl registry or pass it directly to
    RayPPOTrainer.

    Usage in standalone training loop::

        rm = TTSRewardManager(tokenizer, num_examine=2, audio_decoder=vae)
        rewards = rm(data_proto)
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        audio_decoder=None,
        asr_model=None,
        mos_model=None,
        spk_model=None,
        reward_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or compute_tts_reward
        self.reward_fn_key = reward_fn_key
        self.audio_decoder = audio_decoder
        self.asr_model = asr_model
        self.mos_model = mos_model
        self.spk_model = spk_model
        self.reward_weights = reward_weights

    def __call__(self, data, return_dict: bool = False):
        """
        Compute rewards for a batch of TTS generations.

        ``data`` should carry:
          - data.batch["audio_latents"]  : [B, T, D]
          - data.batch["response_mask"]  : [B, T]
          - data.non_tensor_batch["prompt_text"] : list[str]

        Returns:
          - reward_tensor [B, T] with the scalar reward placed at the last
            valid position of each sequence (GRPO outcome-level reward).
        """
        batch_size = data.batch["response_mask"].shape[0]
        max_resp_len = data.batch["response_mask"].shape[1]
        reward_tensor = torch.zeros(batch_size, max_resp_len, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed = 0

        for i in range(batch_size):
            resp_mask = data.batch["response_mask"][i]
            valid_len = int(resp_mask.sum().item())
            audio_latents = data.batch["audio_latents"][i, :valid_len]
            prompt_text = data.non_tensor_batch.get("prompt_text", [""] * batch_size)[i]

            ref_audio = None
            if "reference_audio" in data.non_tensor_batch:
                ref_audio = data.non_tensor_batch["reference_audio"][i]

            score_dict = self.compute_score(
                audio_latents=audio_latents,
                prompt_text=prompt_text,
                reference_audio=ref_audio,
                audio_decoder=self.audio_decoder,
                asr_model=self.asr_model,
                mos_model=self.mos_model,
                spk_model=self.spk_model,
                weights=self.reward_weights,
            )

            reward = score_dict["score"]
            reward_tensor[i, valid_len - 1] = reward

            for key, value in score_dict.items():
                reward_extra_info[key].append(value)

            if already_printed < self.num_examine:
                already_printed += 1
                print(f"[prompt] {prompt_text[:80]}...")
                print(f"[latent_steps] {valid_len}")
                for key, value in score_dict.items():
                    print(f"[{key}] {value:.4f}" if isinstance(value, float) else f"[{key}] {value}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        return reward_tensor
