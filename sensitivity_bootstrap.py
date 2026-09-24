#!/usr/bin/env python3
"""Cross-classified sensitivity analysis (Reviewer R, major comment 5).

Re-estimates the examiner-vs-LLM difference against the expert majority consensus
using a two-way cluster bootstrap over students and examiners (2,000 replicates,
resampled-cluster multiplicities as weights), and per-case estimates.

Reads item_level_scores_anonymized.csv (v2, includes examiner_id) produced by the
same anonymization pipeline as the manuscript. Run after run_analysis.py:

    python3 sensitivity_bootstrap.py

Requires: pandas, numpy. The companion crossed random-effects models
(sensitivity_glmm.R) require R with lme4.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("item_level_scores_anonymized.csv")
LLM = "deepseek_v4_pro"
HUMAN = "examiner_score"
EXPERTS = ["expert_wst", "expert_fj", "expert_gxx"]

disc = df[LLM] != df[HUMAN]
maj = df[EXPERTS].mode(axis=1)[0]
has_maj = disc & (df[EXPERTS].nunique(axis=1) <= 2)
sub = df.loc[has_maj].copy()
ref = maj[has_maj].values
sub["d_llm"] = (sub[LLM].values == ref).astype(float)
sub["d_ex"] = (sub[HUMAN].values == ref).astype(float)
diff = (sub["d_llm"] - sub["d_ex"]).values
mae_diff = (np.abs(sub[LLM].values - ref) - np.abs(sub[HUMAN].values - ref))

print(f"n = {len(sub)} adjudicated items with expert majority consensus")
print(f"crude agreement difference: {diff.mean()*100:.1f} pp "
      f"(LLM {sub['d_llm'].mean()*100:.1f}% vs examiner {sub['d_ex'].mean()*100:.1f}%)")
print(f"crude MAE difference (LLM - examiner): {mae_diff.mean():.3f}")

# two-way cluster bootstrap: resample student clusters and examiner clusters
# independently with replacement; weight each item by the product of the
# multiplicities of its student and examiner clusters (Cameron-Gelbach-Miller
# style two-way cluster bootstrap).
rng = np.random.default_rng(20251110)
stu = sub["student_id"].values
exa = sub["examiner_id"].values
stu_list = np.unique(stu)
exa_list = np.unique(exa)
ns, ne = len(stu_list), len(exa_list)
si = np.searchsorted(stu_list, stu)
ei = np.searchsorted(exa_list, exa)

B = 2000
boot_d = np.empty(B)
boot_m = np.empty(B)
for b in range(B):
    cs = np.bincount(rng.integers(0, ns, ns), minlength=ns).astype(float)
    ce = np.bincount(rng.integers(0, ne, ne), minlength=ne).astype(float)
    w = cs[si] * ce[ei]
    boot_d[b] = np.average(diff, weights=w)
    boot_m[b] = np.average(mae_diff, weights=w)

lo, hi = np.percentile(boot_d, [2.5, 97.5])
lom, him = np.percentile(boot_m, [2.5, 97.5])
print("\n=== two-way cluster bootstrap (student x examiner, B=2000) ===")
print(f"agreement difference: {boot_d.mean()*100:.1f} pp  95% CI [{lo*100:.1f}, {hi*100:.1f}]")
print(f"MAE difference: {boot_m.mean():.3f}  95% CI [{lom:.3f}, {him:.3f}]")

print("\nper-case agreement differences (pp):")
for c, g in sub.groupby("sp_case"):
    print(f"  {c}: {(g['d_llm'].mean() - g['d_ex'].mean())*100:+.1f} (n={len(g)})")
