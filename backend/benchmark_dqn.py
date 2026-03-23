"""
benchmark_dqn.py  —  Fixed DQN benchmark
BEFORE: evaluated the DQN using the exact same reward function it was trained on.
        The +437% result was the model memorising its own training signal.
AFTER:  evaluates against:
        1. Random policy (fair baseline)
        2. Always-intervene policy (naive upper bound)
        3. Always-maintain policy (conservative baseline)
        Reward computed from actual delay outcomes in real data,
        NOT from the training reward function.
"""

import os
import sys
import random
import numpy as np
import torch
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
MODEL_PATH  = os.path.join(MODELS_DIR, "dqn_railway_model.pth")
BENCH_OUT   = os.path.join(MODELS_DIR, "dqn_benchmark.json")

print("=" * 60)
print("DQN Benchmark — Production Model vs Baselines")
print("=" * 60)

# ── 1. Load model ────────────────────────────────────────────────────────────
try:
    from dqn_env import QNetwork, build_dqn_replay_dataset, _synthetic_replay_fallback
    model = QNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print(f"✅ Loaded DQN model from {MODEL_PATH}")
except FileNotFoundError:
    print("❌ No DQN model found. Run: python backend/dqn_env.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Model load failed: {e}")
    sys.exit(1)

# ── 2. Load evaluation data ──────────────────────────────────────────────────
df = build_dqn_replay_dataset()
if df.empty:
    print("⚠️  Real data unavailable — using structured synthetic evaluation set")
    df = _synthetic_replay_fallback()
    data_source = "structured synthetic"
else:
    data_source = "real operational CSV data"

N_EVAL = min(2000, len(df))
eval_df = df.sample(n=N_EVAL, random_state=99).reset_index(drop=True)
print(f"\n📊 Evaluation: {N_EVAL} scenarios | Source: {data_source}")

# ── 3. Outcome-based reward (independent of training reward) ─────────────────
# We use a DIFFERENT reward signal for evaluation:
# Based on whether the actual delay OUTCOME justified intervention.
# This tests whether the model learned the RIGHT behaviour, not just
# to mimic its training signal.

def outcome_reward(action: int, actual_delay: float, congestion: float) -> float:
    """
    Evaluation reward based on real delay outcomes, not training heuristics.
    action=1 (intervene) is justified when delay > 2 min AND congestion_norm > 0.3
    action=0 (maintain)  is correct when network is stable
    Dataset delay range: 0–6.54 min (NOT 0–60 min — thresholds adjusted accordingly)
    """
    delay_high  = actual_delay > 2.0    # top ~8.6% of dataset
    delay_mod   = actual_delay > 1.0    # top ~25% of dataset
    cong_high   = congestion > 0.5      # congestion_norm > 50%
    cong_mod    = congestion > 0.3      # congestion_norm > 30%

    if action == 1:
        if delay_high and cong_high:          return +20   # correct critical intervention
        elif delay_mod and cong_mod:          return +8    # correct moderate intervention
        elif not delay_high and not cong_mod: return -10   # unnecessary intervention
        else:                                 return +2    # marginal
    else:
        if delay_high and cong_high:          return -20   # missed critical event
        elif delay_mod and cong_high:         return -8    # missed moderate event
        elif not delay_high:                  return +5    # correct: nothing needed
        else:                                 return 0     # acceptable non-action

# ── 4. Run evaluation against all policies ────────────────────────────────────
dqn_score    = 0
random_score = 0
always_score = 0    # always intervene
never_score  = 0    # never intervene

dqn_correct   = 0   # matched ground truth label
rand_correct  = 0

for i in range(N_EVAL):
    row        = eval_df.iloc[i]
    # Real column names from build_dqn_replay_dataset():
    #   congestion_norm, active_trains_norm, priority_norm,
    #   time_of_day_norm, delay_minutes, action_label
    congestion = float(row["congestion_norm"])
    delay      = float(row["delay_minutes"])
    label      = int(row["action_label"])

    state = torch.tensor([
        float(row["congestion_norm"]),
        float(row["active_trains_norm"]),
        float(row["priority_norm"]),
        float(row["time_of_day_norm"]),
    ], dtype=torch.float32)

    # DQN action
    with torch.no_grad():
        dqn_action = int(model(state.unsqueeze(0)).argmax().item())

    rand_action   = random.randint(0, 1)
    always_action = 1
    never_action  = 0

    dqn_score    += outcome_reward(dqn_action,    delay, congestion)
    random_score += outcome_reward(rand_action,   delay, congestion)
    always_score += outcome_reward(always_action, delay, congestion)
    never_score  += outcome_reward(never_action,  delay, congestion)

    dqn_correct  += (dqn_action  == label)
    rand_correct += (rand_action == label)

# ── 5. Metrics ───────────────────────────────────────────────────────────────
dqn_acc  = dqn_correct  / N_EVAL * 100
rand_acc = rand_correct / N_EVAL * 100

vs_random = ((dqn_score - random_score) / max(abs(random_score), 1)) * 100
vs_always = ((dqn_score - always_score) / max(abs(always_score), 1)) * 100
vs_never  = ((dqn_score - never_score)  / max(abs(never_score),  1)) * 100

print("\n" + "=" * 60)
print("DQN BENCHMARK RESULTS (outcome-based evaluation)")
print("=" * 60)
print(f"  Evaluation scenarios    : {N_EVAL}")
print(f"  Data source             : {data_source}")
print()
print(f"  Policy          | Score    | Acc%   | vs DQN")
print(f"  ─────────────── | ──────── | ────── | ──────────")
print(f"  DQN (ours)      | {dqn_score:+8.0f} | {dqn_acc:5.1f}% | baseline")
print(f"  Random          | {random_score:+8.0f} | {rand_acc:5.1f}% | DQN {vs_random:+.1f}%")
print(f"  Always intervene| {always_score:+8.0f} |   N/A  | DQN {vs_always:+.1f}%")
print(f"  Never intervene | {never_score:+8.0f} |   N/A  | DQN {vs_never:+.1f}%")
print("=" * 60)

if dqn_score > random_score and dqn_score > always_score and dqn_score > never_score:
    print(f"\n✅ DQN dominates ALL baselines on outcome-based evaluation.")
elif dqn_score > random_score:
    print(f"\n✅ DQN outperforms random policy by {vs_random:.1f}%.")
    print(f"   But doesn't beat always/never baselines — needs more training.")
else:
    print(f"\nℹ️  DQN underperforms random policy — increase training episodes.")

print(f"\n📋 Intervention accuracy: DQN correctly classifies "
      f"{dqn_acc:.1f}% of scenarios (random = {rand_acc:.1f}%)")

# ── 6. Save for frontend ──────────────────────────────────────────────────────
benchmark = {
    "eval_scenarios":        N_EVAL,
    "data_source":           data_source,
    "dqn_score":             dqn_score,
    "random_score":          random_score,
    "always_intervene_score":always_score,
    "never_intervene_score": never_score,
    "outperformance_pct":    round(vs_random, 2),
    "dqn_accuracy_pct":      round(dqn_acc, 2),
    "random_accuracy_pct":   round(rand_acc, 2),
    "congestion_trigger":    0.30,
}
with open(BENCH_OUT, "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"\n✅ Benchmark saved → {BENCH_OUT}")
print("   The /model_benchmarks endpoint will now serve these results.")