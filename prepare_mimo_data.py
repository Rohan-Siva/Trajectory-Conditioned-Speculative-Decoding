"""
Convert downloaded parquet datasets to jsonl format expected by the miles training script.
- DAPO-Math-17k: prompt (list of chat dicts) -> prompt, reward_model.ground_truth -> label
- AIME-2024: Problem -> prompt (formatted), Answer -> label
"""
import json
import pandas as pd
from pathlib import Path

DAPO_PARQUET = "/home/rs63759/code/spec/data/dapo-math-17k/data/dapo-math-17k.parquet"
DAPO_JSONL   = "/home/rs63759/code/spec/data/dapo-math-17k/dapo-math-17k.jsonl"

AIME_PARQUET = "/home/rs63759/code/spec/data/aime-2024/aime_2024_problems.parquet"
AIME_JSONL   = "/home/rs63759/code/spec/data/aime-2024/aime-2024.jsonl"

# ── DAPO ──────────────────────────────────────────────────────────────────────
print("Converting DAPO-Math-17k...")
df = pd.read_parquet(DAPO_PARQUET)
print(f"  Rows: {len(df)}")

with open(DAPO_JSONL, "w") as f:
    for _, row in df.iterrows():
        prompt = row["prompt"]
        # prompt is a numpy array of dicts (chat messages)
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        label = row["reward_model"]["ground_truth"]
        f.write(json.dumps({"prompt": prompt, "label": label}) + "\n")

print(f"  Written to {DAPO_JSONL}")

# ── AIME ──────────────────────────────────────────────────────────────────────
print("Converting AIME-2024...")
df = pd.read_parquet(AIME_PARQUET)
print(f"  Rows: {len(df)}")

AIME_SYSTEM = (
    "Solve the following math competition problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is the answer to the problem.\n\n"
)

with open(AIME_JSONL, "w") as f:
    for _, row in df.iterrows():
        prompt = [
            {"role": "user", "content": AIME_SYSTEM + row["Problem"]}
        ]
        label = str(row["Answer"])
        f.write(json.dumps({"prompt": prompt, "label": label}) + "\n")

print(f"  Written to {AIME_JSONL}")
print("Done!")
