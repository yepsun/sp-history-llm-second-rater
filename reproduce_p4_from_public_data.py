#!/usr/bin/env python3
"""Reproduce every October-stability number reported in the manuscript from the
two public CSVs alone (item_level_scores_anonymized.csv + october_rescore_analysis_set.csv).
Reviewer P4, round 2; updated in v5 after the four previously untraceable encounters
were resolved and added (see README). Requires pandas, scipy.

v5 note: the October file now covers all 240 analysis-set encounters (8,405 items).
The original release linked the October re-scoring to the July release by matching
score sequences, which left four encounters (two pairs with identical examiner and
July model score sequences) unassignable. The v5 release resolves all four with
three concordant internal signals (examination-room rosters, full secondary-model
score sequences, and expert adjudication records) and adds their October scores;
no student-identifying information is included in this repository.

v6 (2026-10-31, same day) fixes a NaN-comparison slip in the majority-vote section of
reproduce_p4_from_public_data.py: the 11 items with no three-run majority were being
counted as discordant. Corrected figures: 807/8,394 defined-vote discordances (9.6%),
818 items routed (versus 882 July, 827-829 October). Accuracy figures (66.4% on the 818
reference items) were unaffected.
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
print(f"matched items: {len(m)} across {m.encounter_id.nunique()} encounters (all 240)")

print(f"same-day October A vs B: {(m.oct_run_a == m.oct_run_b).mean()*100:.2f}%")
print(f"July vs OctA: {(m.deepseek_v4_pro == m.oct_run_a).mean()*100:.2f}%  "
      f"July vs OctB: {(m.deepseek_v4_pro == m.oct_run_b).mean()*100:.2f}%")

dj = m.examiner_score != m.deepseek_v4_pro
da = m.examiner_score != m.oct_run_a
db = m.examiner_score != m.oct_run_b
print(f"discordant with examiner: July {dj.sum()} ({dj.mean()*100:.2f}%), "
      f"OctA {da.sum()} ({da.mean()*100:.2f}%), OctB {db.sum()} ({db.mean()*100:.2f}%)")
print(f"flips out (discordant->concordant): A {(dj&~da).sum()}, B {(dj&~db).sum()}")
print(f"flips in (concordant->discordant):  A {(~dj&da).sum()}, B {(~dj&db).sum()}")
print(f"stable discordant: A {(dj&da).sum()} ({(dj&da).sum()/dj.sum()*100:.0f}%), "
      f"B {(dj&db).sum()} ({(dj&db).sum()/dj.sum()*100:.0f}%)")

# strict adjudicated reference: discordant AND expert majority available (818 items;
# three concordant items carry expert scores because their raw model output exceeded
# the item maximum, was capped to the maximum, and thereby became concordant)
ref = m[(m.consensus.notna()) & (m.deepseek_v4_pro != m.examiner_score)]
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

# three-run majority vote (July + two October passes), round-3 reviewer comment 7
def mvote(r):
    c = collections.Counter([r.deepseek_v4_pro, r.oct_run_a, r.oct_run_b])
    v, k = c.most_common(1)[0]
    return v if k >= 2 else None
m['mv'] = m.apply(mvote, axis=1)
print(f"\n3-run majority vote defined on {int(m.mv.notna().sum())}/{len(m)} items "
      f"({m.mv.notna().mean()*100:.1f}%)")
for col, lab in [('deepseek_v4_pro', 'July'), ('oct_run_a', 'OctA'), ('oct_run_b', 'OctB')]:
    print(f"  discordance with examiner ({lab}): {(m.examiner_score != m[col]).mean()*100:.2f}%")
# for the majority vote, discordance is only defined where a vote exists; comparing
# against NaN would silently count the 11 no-majority items as discordant
defined = m.mv.notna()
mv_disc = int((m.loc[defined, 'examiner_score'] != m.loc[defined, 'mv']).sum())
undefined = int((~defined).sum())
print(f"  discordance with examiner (3-run majority, defined votes only): "
      f"{mv_disc}/{int(defined.sum())} = {mv_disc / defined.sum() * 100:.2f}%")
print(f"  items routed to expert review (discordant or no majority): "
      f"{mv_disc + undefined} ({mv_disc} discordant + {undefined} with no run majority), "
      f"versus 882 July / 827 OctA / 829 OctB")
ok = ref.assign(mv=m.loc[ref.index, 'mv'])['mv'] == ref.consensus
print(f"  accuracy vs 818 reference (3-run majority, undefined votes count as non-matches): "
      f"{ok.sum()}/{len(ref)} = {ok.mean()*100:.1f}%")
