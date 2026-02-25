"""
Data preprocessing for TTS Flow-GRPO training.

verl expects training data in parquet format with specific columns:
  - prompt        : list[dict] (chat format) **or** str
  - data_source   : str
  - reward_model  : dict  (ground-truth info for reward computation)
  - extra_info    : dict  (any auxiliary metadata)

For TTS training we construct:
  - prompt:  the text to synthesise
  - reward_model.ground_truth:  ground-truth info (reference audio path, text)
  - extra_info:  speaker id, emotion, etc.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def create_tts_parquet(
    manifest_path: str,
    output_dir: str,
    data_source: str = "ming_tts",
    split: str = "train",
    max_samples: int = -1,
):
    """
    Convert a TTS manifest (JSONL or CSV) into verl-compatible parquet.

    Expected manifest format (JSONL):
    ```json
    {
        "text": "Hello world",
        "audio_path": "/data/audio/sample_001.wav",
        "speaker_id": "speaker_01",
        "emotion": "neutral",
        "duration": 3.2
    }
    ```

    Or CSV with columns: text, audio_path, speaker_id, emotion, duration
    """
    if manifest_path.endswith(".jsonl"):
        records = []
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    elif manifest_path.endswith(".csv"):
        df = pd.read_csv(manifest_path)
    elif manifest_path.endswith(".tsv"):
        df = pd.read_csv(manifest_path, sep="\t")
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    if max_samples > 0:
        df = df.head(max_samples)

    verl_records = []
    for idx, row in df.iterrows():
        text = row.get("text", row.get("sentence", ""))
        audio_path = row.get("audio_path", row.get("path", ""))
        speaker_id = row.get("speaker_id", row.get("speaker", "default"))
        emotion = row.get("emotion", "neutral")
        duration = row.get("duration", 0.0)

        prompt_content = f"Please generate speech for the following text: Text input:\n{text}"

        record = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "tts",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    "text": text,
                    "audio_path": audio_path,
                    "speaker_id": speaker_id,
                }),
            },
            "extra_info": {
                "split": split,
                "index": int(idx),
                "text": text,
                "audio_path": audio_path,
                "speaker_id": speaker_id,
                "emotion": emotion,
                "duration": float(duration) if duration else 0.0,
            },
        }
        verl_records.append(record)

    out_df = pd.DataFrame(verl_records)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split}.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} records to {out_path}")
    return out_path


def create_demo_data(output_dir: str, num_train: int = 100, num_test: int = 20):
    """
    Create a small demo dataset for testing the pipeline.
    """
    import random
    os.makedirs(output_dir, exist_ok=True)

    demo_texts = [
        "Hello, how are you today?",
        "The weather is beautiful outside.",
        "Welcome to the demonstration of our text to speech system.",
        "Artificial intelligence is changing the world rapidly.",
        "Please remember to take your medication on time.",
        "The stock market showed significant gains this quarter.",
        "I love listening to music in the evening.",
        "Can you help me find the nearest restaurant?",
        "Technology continues to evolve at an unprecedented pace.",
        "Good morning, it's a pleasure to meet you.",
    ]

    speakers = ["speaker_01", "speaker_02", "speaker_03"]
    emotions = ["neutral", "happy", "sad", "excited"]

    for split, num in [("train", num_train), ("test", num_test)]:
        records = []
        for i in range(num):
            records.append({
                "data_source": "ming_tts_demo",
                "prompt": [{
                    "role": "user",
                    "content": f"Please generate speech for the following text: Text input:\n{demo_texts[i % len(demo_texts)]}",
                }],
                "ability": "tts",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({
                        "text": demo_texts[i % len(demo_texts)],
                        "audio_path": "",
                        "speaker_id": random.choice(speakers),
                    }),
                },
                "extra_info": {
                    "split": split,
                    "index": i,
                    "text": demo_texts[i % len(demo_texts)],
                    "audio_path": "",
                    "speaker_id": random.choice(speakers),
                    "emotion": random.choice(emotions),
                    "duration": round(random.uniform(1.0, 5.0), 2),
                },
            })

        df = pd.DataFrame(records)
        path = os.path.join(output_dir, f"{split}.parquet")
        df.to_parquet(path, index=False)
        print(f"Created demo {split} dataset: {path} ({len(df)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TTS data for Flow-GRPO training")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest file (JSONL/CSV)")
    parser.add_argument("--output_dir", type=str, default="./data/tts_grpo", help="Output directory")
    parser.add_argument("--data_source", type=str, default="ming_tts")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--create_demo", action="store_true", help="Create demo dataset")

    args = parser.parse_args()

    if args.create_demo:
        create_demo_data(args.output_dir)
    elif args.manifest:
        create_tts_parquet(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            data_source=args.data_source,
            split=args.split,
            max_samples=args.max_samples,
        )
    else:
        print("Please specify --manifest or --create_demo")
