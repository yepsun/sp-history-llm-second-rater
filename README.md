# Item-level scoring data: LLM as a second rater for SP history-taking examinations

Dataset for: *Transcript-based large language model scoring as a second rater for
standardized patient history-taking examinations: a retrospective adjudication study*
(JMIR Medical Education, under review; manuscript ID 110683).

## Contents

| File | Description |
| --- | --- |
| `item_level_scores_anonymized.csv` | De-identified item-level scores, 8,405 rows (one row per checklist item) |
| `run_analysis.py` | Reproduces the headline analyses (agreement, expert consensus accuracy, grade-band reclassification) from the CSV |
| `sensitivity_bootstrap.py` | Two-way cluster bootstrap (student × examiner) and per-case estimates for the accuracy and MAE differences (reviewer comment R5) |
| `sensitivity_glmm.R` | Crossed random-effects models (logistic for agreement, Gaussian for absolute error; R/lme4) |
| `october_rescore_analysis_set.csv` | Item-level scores of two same-day re-scoring passes on 23 October 2026 (all 240 analysis-set encounters; 8,405 rows), used for the run-to-run stability and July-to-October change analyses (reviewer P4, round 2; completed to all 240 encounters in v5, see version history) |
| `development_set_scores.csv` | Complete item-level scores for the 36-encounter prompt-development set (35 encounters expert-scored; 1,257 rows): examiner, LLM (9 July snapshot), and three expert scores, anonymized (D01–D36) (reviewer P6, round 2) |
| `whole_exam_accuracy.py` | Whole-examination item-level agreement estimates under both conventions for the 64 unresolved items (reviewer P1, round 2) |
| `rescore_20261023.py` | Byte-faithful replica of the original scoring pipeline used for the October 2026 two-pass re-scoring |
| `reproduce_p4_from_public_data.py` | Reproduces every stability number reported in the manuscript (8,405 items / 240 encounters; 882 July-discordant; flips 199–212 out / 144–159 in; 76–77% stable; 65.5% / 64.8% / 66.0% on the 818 reference items; McNemar p = 0.72 / 0.83; three-run majority vote 66.4%) from the two public CSVs alone |
| `reproduce_round3_numbers.py` | Reproduces the statistics added in the round-3 revision: whole-examination difference bootstrap CI, development-set conditional match rates, grade-band McNemar test, half-point score counts, single-expert Wilcoxon p-values, and expert-disagreement counts among model-consensus errors |
| `analyze_rerun_20261023.py` | Internal stability analysis over the authors’ archived working files, provided for transparency only; it uses local paths and links the October files to this dataset by score-sequence matching, so it predates the v5 resolution of the last four encounters |
| `README.md` | This file |

## Study and data summary

Retrospective diagnostic-accuracy-style study of a real high-stakes diagnostic
history-taking final examination (10 November 2025) at a single Chinese medical school.
92 students each completed three standardized-patient (SP) encounters (six cases;
276 recordings). One encounter from each of 36 students was reserved for LLM prompt
development (six per case, randomly selected subject to preserving the case
distribution); the remaining 240 encounters form the analysis set. Every checklist
item was scored by the on-site examiner and independently by an LLM
(DeepSeek-V4-Pro) from the verified transcript; the 882 discordant items (10.5%)
were re-scored by three blinded faculty experts, and two further models
(DeepSeek-V4-Flash-0731; Qwen3-Next-80B-A3B-Instruct) scored the same transcripts for secondary
comparison.

Ethics approval: Peking Union Medical College Hospital Ethics Committee
(Approval No. I-26PJ0511); the requirement for individual informed consent was waived
(retrospective use of pre-existing de-identified educational data).

## Data dictionary (`item_level_scores_anonymized.csv`)

| Column | Description |
| --- | --- |
| `encounter_id` | Anonymous encounter identifier (E001–E240); random assignment (seed 20251110) |
| `student_id` | Anonymous student identifier (S001–S092); assigned in order of first appearance. 56 students contribute 3 encounters and 36 contribute 2 (one encounter each reserved for prompt development) |
| `sp_case` | SP case (SP01–SP06); not identifying |
| `examiner_id` | Anonymous on-site examiner identifier (E01–E12), derived from the examination-room numbers used as examiner identifiers in the manuscript; 12 examiners, each examining one morning and/or one afternoon case |
| `item_seq` | Checklist item number within the case |
| `item_max` | Maximum score for the item (1–4) |
| `examiner_score` | Score assigned by the on-site examiner during the encounter |
| `deepseek_v4_pro_raw` | DeepSeek-V4-Pro score before capping at `item_max` |
| `deepseek_v4_pro` | DeepSeek-V4-Pro score after capping (index LLM route) |
| `deepseek_v4_flash_0731` | DeepSeek-V4-Flash-0731 score (official 31 July 2026 release), same prompt |
| `qwen3_next_80b_a3b_instruct` | Qwen3-Next-80B-A3B-Instruct score (on-premises, llama.cpp UD-Q4_K_XL), same prompt |
| `expert_wst`, `expert_fj`, `expert_gxx` | Independent blinded expert re-scores (half-points allowed, e.g. 0.5, 1.5); empty for the 7,523 concordant items that were not adjudicated |

## Reproducing the analyses

```bash
python3 run_analysis.py   # requires pandas, numpy
```

v2 (2026-09-24) adds the anonymized examiner identifier (E01–E12) and the two cross-classified sensitivity analysis scripts above.

v3 (2026-10-23) adds the October 2026 two-pass re-scoring data and stability scripts, the
development-set expert scores, and the whole-examination accuracy script (second-round
revision).

v4 (2026-10-24) fixes a column-labelling error in `development_set_scores.csv` (the
`sp_case` column previously held the source file extension instead of the case code; it
now reads SP01–SP06; scores and all other columns are unchanged) and rewrites
`whole_exam_accuracy.py` to compute every count live from the CSV instead of relying on
hard-coded numbers.

v5 (2026-10-31) completes `october_rescore_analysis_set.csv` to all 240 analysis-set
encounters (8,405 rows). The original v3 file linked the October re-scoring to this
dataset by matching examiner-plus-model score sequences, which left four encounters
(two pairs of encounters with identical sequences) unassignable and therefore absent.
All four have now been resolved and added, using three concordant internal signals:
the examination-room rosters (which assign each encounter to an examiner, and the two
members of one pair had different examiners), the full secondary-model score sequences,
and the expert adjudication records. Every previously reported stability figure is
unchanged or unchanged to the reported precision (same-day October agreement 97.4%;
July-to-October agreement 95.5% and 95.1%; 76–77% of the July-discordant set stable;
accuracy against the 818 reference items 65.5% July versus 64.8% and 66.0% in October;
McNemar p = 0.72 and 0.83). v5 also adds `reproduce_round3_numbers.py` (round-3
reviewer statistics) and updates `reproduce_p4_from_public_data.py` to the
complete-linkage figures and the strict 818-item reference definition.

v6 (31 October 2026, later the same day) fixes a NaN-comparison slip in the
majority-vote section of `reproduce_p4_from_public_data.py`: the 11 items with no
three-run majority had been counted as discordant. Corrected figures: 807 of 8,394
defined-vote discordances (9.6%), 818 items routed to expert review (versus 882 in
July and 827–829 in October). The accuracy figures (66.4% on the 818 reference items)
were unaffected.

Reproduced headline values: examiner–LLM exact agreement 89.5% (7,523/8,405); 882
discordant items, of which 818 have an expert majority consensus and 64 are unresolved;
against the majority consensus the LLM matches on 65.5% of items (MAE 0.37) versus
27.9% (MAE 0.75) for the examiner; DeepSeek-V4-Flash-0731 49.9% and
Qwen3-Next-80B-A3B-Instruct 50.6%; grade-band reclassification after partial expert
correction 29.2% (examiner) versus 19.2% (LLM) under illustrative bands defined by the
authors (excellent ≥90%, good 80–89%, below 80% of the checklist maximum — these are
**not** official institutional cut-points).

## Privacy and availability boundaries

This dataset contains no student names, no audio, and no transcripts; the mapping
between anonymous and original identifiers is retained only by the authors and is not
published. Raw audio and identified transcripts are not publicly available to protect
student privacy. SP case labels (SP01–SP06) are retained because the case checklists
are described in the manuscript's prompts appendix.

## Citation and archive

Archived on Zenodo: **https://doi.org/10.5281/zenodo.22927341** (DOI 10.5281/zenodo.22927341).
GitHub repository: https://github.com/yepsun/sp-history-llm-second-rater

Authors: Jun Feng¹*, Shaoting Wang²*, Xiaoxing Gao²*, Luo Wang², Xiaoming Huang³,
Xuefeng Sun²,⁴ (*equal contribution). ¹Department of Hematology; ²Department of Respiratory
and Critical Care Medicine; ³Department of General Medicine, Department of Medical
Diagnostics; ⁴Department of Internal Medicine — Peking Union Medical College Hospital,
Chinese Academy of Medical Sciences and Peking Union Medical College, Beijing, China.
Corresponding author: Xuefeng Sun, sunxfer@sina.com, ORCID 0000-0002-5355-3405.

## License

CC-BY-4.0 (see LICENSE). Cite the manuscript (above) when reusing these data.
