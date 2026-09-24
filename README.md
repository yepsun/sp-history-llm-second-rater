# Item-level scoring data: LLM as a second rater for SP history-taking examinations

Dataset for: *Transcript-based large language model scoring as a low-cost second rater for
standardized patient history-taking examinations: a retrospective adjudication study*
(JMIR Medical Education, under review; manuscript ID 110683).

## Contents

| File | Description |
| --- | --- |
| `item_level_scores_anonymized.csv` | De-identified item-level scores, 8,405 rows (one row per checklist item) |
| `run_analysis.py` | Reproduces the headline analyses (agreement, expert consensus accuracy, grade-band reclassification) from the CSV |
| `sensitivity_bootstrap.py` | Two-way cluster bootstrap (student × examiner) and per-case estimates for the accuracy and MAE differences (reviewer comment R5) |
| `sensitivity_glmm.R` | Crossed random-effects models (logistic for agreement, Gaussian for absolute error; R/lme4) |
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
(DeepSeek-V4-Flash-0731; Qwen3-Next-80B-A3B-Instruct) replicated the LLM scoring.

Ethics approval: Peking Union Medical College Hospital Ethics Committee
(Approval No. I-26PJ0511); the requirement for individual informed consent was waived
(retrospective use of pre-existing anonymized educational data).

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

Archived on Zenodo: **https://doi.org/10.5281/zenodo.22927342** (DOI 10.5281/zenodo.22927342).
GitHub repository: https://github.com/yepsun/sp-history-llm-second-rater

Authors: Jun Feng¹*, Shaoting Wang²*, Xiaoxing Gao²*, Luo Wang², Xiaoming Huang³,
Xuefeng Sun²,⁴ (*equal contribution). ¹Department of Hematology; ²Department of Respiratory
and Critical Care Medicine; ³Department of General Medicine, Department of Medical
Diagnostics; ⁴Department of Internal Medicine — Peking Union Medical College Hospital,
Chinese Academy of Medical Sciences and Peking Union Medical College, Beijing, China.
Corresponding author: Xuefeng Sun, sunxfer@sina.com, ORCID 0000-0002-5355-3405.

## License

CC-BY-4.0 (see LICENSE). Cite the manuscript (above) when reusing these data.
