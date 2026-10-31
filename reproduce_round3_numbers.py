#!/usr/bin/env python3
"""Reproduce the statistics added in the round-3 revision from the public CSVs alone.
Reviewer P, round 3 (comments 2-5, 8-9, 13b, 13e). Requires pandas, numpy, scipy.

Outputs:
  - student-cluster bootstrap CI for the whole-examination accuracy difference
  - development-set discordance rate and conditional match rates
  - Wilson CI and paired McNemar test for grade-band reclassification (70/240 vs 46/240)
  - half-point (non-integer) score counts per rater
  - Wilcoxon p-values for the single-expert reference rows of Table 3
  - expert disagreement among model-consensus errors
"""
import collections
import math

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("item_level_scores_anonymized.csv")
LLM, HUM = "deepseek_v4_pro", "examiner_score"
EXP = ["expert_wst", "expert_fj", "expert_gxx"]

def majority(row):
    vs = [v for v in row if pd.notna(v)]
    c = collections.Counter(vs)
    if not c:
        return None
    v, k = c.most_common(1)[0]
    return v if k >= 2 else None

df["consensus"] = df[EXP].apply(majority, axis=1)
disc = df[LLM] != df[HUM]

# ---- half-point counts (comment 8) -------------------------------------------
print("== non-integer (half-point) scores ==")
for c in [HUM, "deepseek_v4_pro_raw", LLM, "deepseek_v4_flash_0731",
          "qwen3_next_80b_a3b_instruct"]:
    s = df[c].dropna()
    print(f"  {c}: {int((s % 1 != 0).sum())}/{len(s)}")
esc = pd.concat([df[c].dropna() for c in EXP])
print(f"  individual expert scores: {int((esc % 1 != 0).sum())}/{len(esc)}")
adj = df[disc & df.consensus.notna()]
print(f"  majority consensus on the 818: {int((adj.consensus % 1 != 0).sum())}/{len(adj)}")

# ---- three pre-cap discordant items (comment 9) ------------------------------
odd = df[(df.deepseek_v4_pro_raw != df.deepseek_v4_pro)]
print(f"\n== items changed by boundary correction: {len(odd)} ==")
print(odd[["encounter_id", "item_seq", "item_max", "examiner_score",
           "deepseek_v4_pro_raw", "deepseek_v4_pro"]].to_string())

# ---- bootstrap CI for the whole-examination difference (comment 3) -----------
rng = np.random.default_rng(20251110)
df["llm_match"] = np.where(df.consensus.notna(), (df[LLM] == df.consensus).astype(float), 0.0)
df["hum_match"] = np.where(df.consensus.notna(), (df[HUM] == df.consensus).astype(float), 0.0)
students = df.student_id.unique()
diffs = []
for _ in range(2000):
    pick = rng.choice(students, len(students), replace=True)
    b = pd.concat([df[df.student_id == s] for s in pick])
    diffs.append((b.llm_match.sum() - b.hum_match.sum()) / len(b) * 100)
diffs = np.array(diffs)
pt = (536 - 228) / 8405 * 100
print(f"\n== whole-examination difference ==\n  point {pt:.2f} pp, "
      f"student-cluster bootstrap 95% CI {np.percentile(diffs, 2.5):.1f} to "
      f"{np.percentile(diffs, 97.5):.1f}")

# ---- grade-band reclassification (comment 5) ---------------------------------
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round((c - h) * 100, 1), round((c + h) * 100, 1)

def band(t, mx):
    p = t / mx * 100
    return 2 if p >= 90 else (1 if p >= 80 else 0)

flags = {}
for enc, g in df.groupby("encounter_id"):
    mx = g.item_max.sum()
    scored = g[g[EXP].notna().any(axis=1)]
    if len(scored):
        cm = scored[EXP].mode(axis=1).iloc[:, 0]
        hm = scored[EXP].nunique(axis=1) <= 2
        cmap = cm[hm]
    else:
        cmap = pd.Series(dtype=float)
    d = g[LLM] != g[HUM]
    for route in [HUM, LLM]:
        corr = g[route].astype(float).copy()
        corr[d & g.index.isin(cmap.index)] = cmap
        flags[(enc, route)] = band(g[route].sum(), mx) != band(corr.sum(), mx)
encs = sorted({k[0] for k in flags})
fe = np.array([flags[(e, HUM)] for e in encs])
fl = np.array([flags[(e, LLM)] for e in encs])
b11 = int((fe & fl).sum()); b10 = int((fe & ~fl).sum())
b01 = int((~fe & fl).sum()); b00 = int((~fe & ~fl).sum())
print(f"\n== band reclassification ==\n  examiner {b11+b10}/240 ({(b11+b10)/240*100:.1f}%), "
      f"Wilson 95% CI {wilson(b11+b10, 240)}")
print(f"  LLM {b11+b01}/240 ({(b11+b01)/240*100:.1f}%), Wilson 95% CI {wilson(b11+b01, 240)}")
print(f"  cross-tab: both {b11}, examiner-only {b10}, LLM-only {b01}, neither {b00}")
print(f"  McNemar exact ({b10} vs {b01}): p = {stats.binomtest(b10, b10+b01, 0.5).pvalue:.2e}")

# ---- development set (comment 4) ----------------------------------------------
dv = pd.read_csv("development_set_scores.csv")
dd = dv.examiner_score != dv.llm_score_jul9
dv["consensus"] = dv[["expert_wst", "expert_fj", "expert_gxx"]].apply(majority, axis=1)
ddm = dv[dd & dv.consensus.notna()]
print(f"\n== development set ==\n  items {len(dv)}; discordant {int(dd.sum())} ({dd.mean()*100:.1f}%)")
print(f"  discordant with majority {len(ddm)}: examiner "
      f"{(ddm.examiner_score == ddm.consensus).mean()*100:.1f}%, LLM "
      f"{(ddm.llm_score_jul9 == ddm.consensus).mean()*100:.1f}%")

# ---- single-expert Wilcoxon p-values (comment 13b) ----------------------------
sub = df[disc & df[EXP].notna().any(axis=1)]
print("\n== Table 3 single-expert rows, Wilcoxon p (LLM vs examiner absolute error) ==")
for e in EXP:
    a = np.abs(sub[HUM] - sub[e])
    b = np.abs(sub[LLM] - sub[e])
    print(f"  {e}: p = {stats.wilcoxon(b, a, zero_method='wilcox').pvalue:.2e}")

# ---- expert disagreement among model-consensus errors (comment 13e) -----------
m3 = df[["deepseek_v4_pro", "deepseek_v4_flash_0731",
         "qwen3_next_80b_a3b_instruct"]].nunique(axis=1) == 1
err = df[m3 & df.consensus.notna() & (df[LLM] != df.consensus)]
unan = err[EXP].nunique(axis=1) == 1
print(f"\n== model-consensus errors (n={len(err)}) ==\n  experts split (not unanimous): "
      f"{int((~unan).sum())} ({(~unan).mean()*100:.0f}%), unanimous: {int(unan.sum())}")
