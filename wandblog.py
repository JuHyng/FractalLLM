import wandb
import pandas as pd

PROJECT = "FractalLLM"     # "entity/project" 형식 (entity 생략 시 자신의 계정)

RUN_NAMES = [
    "25_05-07_15:46-meta-llama/Llama-3.2-1B-draft-gsm8k-max_samples:100-draft_len:8",
    "25_05-07_18:01-meta-llama/Llama-3.2-1B-fractal-gsm8k-max_samples:100-quant_8bit-draft_len:8-draft_token:[DRAFT{i}]",
    "25_05-07_17:45-meta-llama/Llama-3.2-1B-fractal-gsm8k-max_samples:100-quant_8bit-draft_len:8-draft_token:[DRAFT{i}]"
]

KEY = "elapsed"            # WandB history column

api   = wandb.Api(timeout=30)
means = {}                 # run_name → elapsed 평균

for name in RUN_NAMES:
    runs = api.runs(PROJECT, filters={"display_name": name})
    if not runs:
        print(f"⚠️  '{name}' not found — skipped")
        continue

    run = runs[0]
    df  = run.history(keys=[KEY], pandas=True)
    if df.empty:
        print(f"⚠️  '{KEY}' column missing in run {run.id} — skipped")
        continue

    means[name] = df[KEY].mean()

# 보기 좋게 정렬·출력
results = pd.Series(means).sort_values(ascending=False)
print("\nAverage elapsed per run:")
print(results)
