#!/usr/bin/env python3
"""Reproduce every October-stability number reported in the manuscript from the
two public CSVs alone (item_level_scores_anonymized.csv + october_rescore_analysis_set.csv).
Reviewer P4, round 2. Requires pandas, scipy.
"""
import pandas as pd
import collections
from scipy import stats

df = pd.read_csv('item_level_scores_anonymized.csv')
oc = pd.read_csv('october_rescore_analysis_set.csv')

def majority(r):
    vs = [v for v in [r.expert_wst, r.expert_fj, r.expert_gxx] if pd.notna(v)]
    c = collections.Counter(vs)
    if not c:
        return None
    v, k = c.most_common(1)[0]
    return v if k >= 2 else None

df['consensus'] = df.apply(majority, axis=1)
m = df.merge(oc, on=['encounter_id', 'item_seq'], how='inner')
print(f"matched items: {len(m)} across {m.encounter_id.nunique()} encounters (236 of 240 traceable)")

dj = m.examiner_score != m.deepseek_v4_pro
da = m.examiner_score != m.oct_run_a
db = m.examiner_score != m.oct_run_b
print(f"discordant with examiner: July {dj.sum()} ({dj.mean()*100:.2f}%), "
      f"OctA {da.sum()} ({da.mean()*100:.2f}%), OctB {db.sum()} ({db.mean()*100:.2f}%)")
print(f"flips out (discordant->concordant): A {(dj&~da).sum()}, B {(dj&~db).sum()}")
print(f"flips in (concordant->discordant):  A {(~dj&da).sum()}, B {(~dj&db).sum()}")
print(f"stable discordant: A {(dj&da).sum()} ({(dj&da).sum()/dj.sum()*100:.0f}%), "
      f"B {(dj&db).sum()} ({(dj&db).sum()/dj.sum()*100:.0f}%)")

ref = m[m.consensus.notna()]
print(f"reference items: {len(ref)} of 818")
jv = ref.deepseek_v4_pro == ref.consensus
av = ref.oct_run_a == ref.consensus
bv = ref.oct_run_b == ref.consensus
print(f"agreement with expert majority: July {jv.sum()}/{len(ref)} = {jv.mean()*100:.1f}%, "
      f"OctA {av.mean()*100:.1f}%, OctB {bv.mean()*100:.1f}%")
for lab, x in [('OctA', av), ('OctB', bv)]:
    b01 = int((jv & ~x).sum()); b10 = int((~jv & x).sum())
    p = stats.binomtest(b01, b01 + b10, 0.5).pvalue
    print(f"McNemar July vs {lab}: {b01} vs {b10} discordant pairs, exact p = {p:.2f}")
