#!/usr/bin/env python3
"""Whole-examination item-level agreement estimates (reviewer P1, round 2).

Two conventions:
  A. All 8,405 items; the 64 without expert majority counted as mismatches
     for BOTH routes (conservative):  examiner 91.4%, LLM 95.1%
  B. Items with a reference standard only (n = 8,341):  examiner 92.1%, LLM 95.8%

Formula (convention A): agreement = (7523*(1-p_audit) + matches_adjudicated) / 8405
  p_audit = 7/752 audit shared-error rate; matches_adjudicated = 228 (examiner)
  or 536 (LLM) of the 818 adjudicated items.
The quoted ranges (e.g., 94.2-95.5) vary p_audit over its Wilson interval and
reflect audit sampling uncertainty only.
"""
import pandas as pd

df = pd.read_csv('item_level_scores_anonymized.csv')
llm = 'deepseek_v4_pro'
maj = df[(df.examiner_score != df[llm])][['expert_wst','expert_fj','expert_gxx']].notna().any(axis=1)
discord = df[df.examiner_score != df[llm]]
n_majority = 818  # per manuscript; recomputed in run_analysis.py
matches = {'examiner': 228, 'LLM': 536}
p_pt, p_lo, p_hi = 7/752, 0.0045, 0.0191
for name, m in matches.items():
    a = (7523*(1-p_pt)+m)/8405
    b = (7523*(1-p_pt)+m)/8341
    r_lo = (7523*(1-p_hi)+m)/8405; r_hi = (7523*(1-p_lo)+m)/8405
    print(f"{name}: all-8405 {a*100:.1f}% ({r_lo*100:.1f}-{r_hi*100:.1f} audit interval) | excl-64 {b*100:.1f}%")
