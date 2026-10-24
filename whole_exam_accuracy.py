#!/usr/bin/env python3
"""Whole-examination item-level agreement estimates (reviewer P1, round 2),
computed live from item_level_scores_anonymized.csv (no hardcoded counts).

Two conventions for the unresolved discordant items (no expert majority):
  A. counted as mismatches for BOTH routes (conservative, all 8,405 items):
     examiner 91.4%, LLM 95.1%
  B. excluded (items with a reference standard only, n = 8,341):
     examiner 92.1%, LLM 95.8%
The quoted ranges vary the audit shared-error rate (7/752) over its Wilson
interval and reflect audit sampling uncertainty only, not total uncertainty.

Note: the adjudicated set is the discordant items that received expert
re-scores. Three concordant rows in the CSV also carry expert scores (audit
artefacts) and are excluded here, consistent with the manuscript flow
(8,405 -> 7,523 concordant + 882 adjudicated -> 818 majority + 64 unresolved).
"""
import collections
import pandas as pd

df = pd.read_csv('item_level_scores_anonymized.csv')
EXPERTS = ['expert_wst', 'expert_fj', 'expert_gxx']
LLM, HUMAN = 'deepseek_v4_pro', 'examiner_score'

def majority(row):
    vs = [v for v in row[EXPERTS] if pd.notna(v)]
    c = collections.Counter(vs)
    if not c:
        return None
    v, k = c.most_common(1)[0]
    return v if k >= 2 else None

discordant = df[HUMAN] != df[LLM]
adjudicated = df[discordant & df[EXPERTS].notna().any(axis=1)].copy()
adjudicated['consensus'] = adjudicated.apply(majority, axis=1)

n_majority = int(adjudicated.consensus.notna().sum())
n_unresolved = len(adjudicated) - n_majority
n_concordant = int((~discordant).sum())
matches = {c: int((adjudicated[c] == adjudicated.consensus).sum())
           for c in [HUMAN, LLM]}

print(f"items {len(df)} | concordant {n_concordant} | adjudicated {len(adjudicated)} "
      f"| majority {n_majority} | unresolved {n_unresolved}")
print(f"adjudicated matches: examiner {matches[HUMAN]}/{n_majority}, "
      f"LLM {matches[LLM]}/{n_majority}")

audit_pt, audit_lo, audit_hi = 7/752, 0.0045, 0.0191
n_ref = n_concordant + n_majority  # items with a usable reference standard
for name, col in [('examiner', HUMAN), ('LLM', LLM)]:
    a = (n_concordant * (1 - audit_pt) + matches[col]) / len(df)
    b = (n_concordant * (1 - audit_pt) + matches[col]) / n_ref
    r_lo = (n_concordant * (1 - audit_hi) + matches[col]) / len(df)
    r_hi = (n_concordant * (1 - audit_lo) + matches[col]) / len(df)
    print(f"{name}: all-{len(df)} {a*100:.1f}% ({r_lo*100:.1f}-{r_hi*100:.1f} audit interval) "
          f"| excl-unresolved (n={n_ref}) {b*100:.1f}%")
