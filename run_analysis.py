#!/usr/bin/env python3
"""Reproduce the headline analyses of the manuscript from the anonymized item-level dataset.

Dataset: item_level_scores_anonymized.csv (8,405 checklist items from 240 SP encounters;
92 students; six SP cases). One row per checklist item. Expert columns are empty for the
7,523 concordant items that were not adjudicated. Expert scores include half-points
(e.g., 0.5, 1.5); the expert majority consensus is therefore a float and comparisons
are exact (no rounding).

Reference: "Transcript-based large language model scoring as a low-cost second rater for
standardized patient history-taking examinations: a retrospective adjudication study"
(JMIR Medical Education, under review).

Run:  python3 run_analysis.py
Requires: pandas, numpy.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("item_level_scores_anonymized.csv")

LLM = "deepseek_v4_pro"
HUMAN = "examiner_score"
EXPERTS = ["expert_wst", "expert_fj", "expert_gxx"]

print(f"Items: {len(df)} | Encounters: {df['encounter_id'].nunique()} | "
      f"Students: {df['student_id'].nunique()} | Cases: {df['sp_case'].nunique()}")

# ---- Concordance and discordance -------------------------------------------------
disc_mask = df[LLM] != df[HUMAN]
n_disc = int(disc_mask.sum())
print(f"\nExaminer-LLM concordance: {1 - disc_mask.mean():.3f} "
      f"({len(df) - n_disc}/{len(df)}); discordant: {n_disc} ({disc_mask.mean()*100:.1f}%)")

enc_with_disc = df.loc[disc_mask, 'encounter_id'].nunique()
print(f"Encounters with >=1 discordant item: {enc_with_disc}/{df['encounter_id'].nunique()}")

# ---- Expert majority consensus ----------------------------------------------------
expert_nunique = df[EXPERTS].nunique(axis=1)
has_majority = disc_mask & (expert_nunique <= 2)
unresolved = disc_mask & (expert_nunique > 2)
print(f"Discordant items with expert majority: {int(has_majority.sum())}; "
      f"without majority (unresolved): {int(unresolved.sum())}")

sub = df.loc[has_majority]
maj = df.loc[has_majority, EXPERTS].mode(axis=1)[0]  # float; half-point consensus possible

def acc(col):
    return float((sub[col].values == maj.values).mean())

def mae(col):
    return float(np.abs(sub[col].values - maj.values).mean())

print("\n--- Accuracy vs expert majority consensus (n = %d) ---" % len(sub))
for col, label in [(LLM, "DeepSeek-V4-Pro"), ("deepseek_v4_flash_0731", "DeepSeek-V4-Flash-0731"),
                   ("qwen3_next_80b_a3b_instruct", "Qwen3-Next-80B-A3B-Instruct"),
                   (HUMAN, "On-site examiner")]:
    print(f"{label:32s} exact agreement {acc(col)*100:5.1f}%   MAE {mae(col):.3f}")

# ---- Zero-credit conditional agreement ---------------------------------------------
zero = maj.values == 0
print(f"\nItems where consensus awarded zero: {int(zero.sum())} "
      f"({zero.mean()*100:.1f}% of adjudicated items)")
for col, label in [(HUMAN, "examiner"), (LLM, "DeepSeek-V4-Pro")]:
    agree = (sub.loc[zero, col].values == 0).mean()
    print(f"  {label} also awarded zero: {agree*100:.1f}%")

# ---- Decision-level consequences (illustrative grade bands) ------------------------
# Bands were defined by the authors for illustrative purposes (excellent >=90%,
# good 80-89%, below 80% of the checklist maximum), NOT official institutional cut-points.
# Corrected total = discordant items replaced by the expert majority consensus where one
# exists; unresolved discordant items keep the route's own original score.
def band(total, mx):
    pct = total / mx * 100
    return np.where(pct >= 90, 2, np.where(pct >= 80, 1, 0))  # 2=excellent, 1=good, 0=below80

rows = []
for enc, g in df.groupby("encounter_id"):
    mx = g["item_max"].sum()
    scored = g[g[EXPERTS].notna().any(axis=1)]
    if len(scored):
        maj_all = scored[EXPERTS].mode(axis=1)[0]
        has_maj_row = scored[EXPERTS].nunique(axis=1) <= 2
        corr_map = maj_all[has_maj_row]
    else:
        corr_map = pd.Series(dtype=float)  # encounter with no discordant items
    discordant = g[LLM] != g[HUMAN]
    for route in [HUMAN, LLM]:
        tot = g[route].sum()
        corr = g[route].astype(float).copy()
        replace = discordant & g.index.isin(corr_map.index)
        corr[replace] = corr_map
        rows.append({"encounter_id": enc, "route": route, "max": mx,
                     "raw_total": tot, "corr_total": corr.sum(),
                     "raw_band": band(tot, mx), "corr_band": band(corr.sum(), mx),
                     "has_unresolved": bool((discordant & ~g.index.isin(corr_map.index)).any())})
R = pd.DataFrame(rows)

print("\n--- Grade-band reclassification (illustrative bands: >=90% excellent, "
      "80-89% good, <80% below 80%) ---")
for route, label in [(HUMAN, "examiner"), (LLM, "DeepSeek-V4-Pro")]:
    r = R[R["route"] == route]
    changed = (r["raw_band"] != r["corr_band"]).sum()
    n_exc = (r["raw_band"] == 2).sum()
    print(f"{label:18s} reclassified: {changed}/{len(r)} = {changed/len(r)*100:.1f}% "
          f"(raw excellent: {n_exc})")

# sensitivity: exclude transcripts containing at least one unresolved discordant item
bad_enc = R.loc[R["has_unresolved"], "encounter_id"].unique()
Rs = R[~R["encounter_id"].isin(bad_enc)]
print(f"\nSensitivity excluding {len(bad_enc)} transcripts with >=1 unresolved discordant item "
      f"(n = {len(Rs[Rs['route'] == HUMAN])} transcripts):")
for route, label in [(HUMAN, "examiner"), (LLM, "DeepSeek-V4-Pro")]:
    r = Rs[Rs["route"] == route]
    changed = (r["raw_band"] != r["corr_band"]).sum()
    print(f"{label:18s} reclassified: {changed}/{len(r)} = {changed/len(r)*100:.1f}%")

print("\nDone. Values should match the manuscript's Abstract, Results, and Tables 3-5.")
