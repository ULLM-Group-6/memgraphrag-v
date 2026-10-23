# Evaluate the completed WebQA test

Standalone Python 3 script. No packages, GPU, model files or repository installation required.
Nothing is pushed to GitHub and no pipeline scripts need changing.

## Local use (Windows PowerShell)

Copy the ENTIRE `webqa-tuned-v1-test-single` run folder from Snellius to your machine.
Extract this bundle. From the extracted bundle directory, run (replace the result path):

```powershell
py -3 .\evaluate_webqa.py "C:\Users\Antonio\Downloads\webqa-tuned-v1-test-single"
```

If `py` is unavailable, use your Python executable or `python`.
Required files in the run folder: `evaluation_inputs.jsonl`, `answers.jsonl`,
`labels.jsonl`, `retrieval.jsonl`, `run.json`, `completed.json`.
Keep the other run files too, especially reader inputs and tuning provenance.
The default expects 120 unique questions and refuses partial or smoke runs.
Do not override the expected count to disguise an incomplete 120-question run.

The script validates saved answer and contract hashes, question coverage,
labels, answer consistency and retrieval ranking consistency. It does not revalidate
the original graph/model files. It records all input hashes in its output for audit.

## Output

Files go to `RUN_FOLDER/evaluation_report/` (rerunning overwrites these report files,
not the source results). `--output PATH` selects a different directory.

- `report.md`: ready-to-copy percentage table with 95% intervals, sample counts,
  selected beta, answer completion and graph/fallback counts.
- `summary.json`: full-precision scores, counts, bootstrap settings and source hashes.
- `per_question.csv`: scores, IDs, answers and review categories; opens in Excel.
- `review_examples.md`: up to eight examples printed in the terminal too.
- `review_candidates.jsonl`: every failed-containment/failed-answer candidate,
  including the supplied document records for manual inspection.

Use `--examples 15` to print more examples. These are selected deterministically,
round-robin across review categories; they are illustrative, not a representative sample.

## Metric definitions and reporting

Document-ID Recall@5 is the mean across questions of
`len(set(gold_doc_ids) & set(top_k_doc_ids)) / len(set(gold_doc_ids))`.
It matches the tuning metric. It is not binary any-hit accuracy for multi-document questions.
All expected questions are retained, even if their answer generation failed.

Supplied-image Recall@5 uses `supplied_image_ids`, not simply selected image IDs.
Only questions with nonempty gold image labels enter this metric.

Answer containment is a HEURISTIC: Unicode NFKC, lowercase, remove Unicode punctuation,
remove English articles a/an/the, collapse whitespace, and match any normalized reference
as a word-bounded substring of the prediction. Failed/empty predictions count as zero.
It is neither semantic correctness nor LLM-judge accuracy: paraphrases may fail and
contradictory statements containing the reference may pass. Review candidates are NOT
confirmed wrong answers. Evidence-ID absence is not proof of causal attribution.

Confidence intervals use 2,000 percentile bootstrap resamples, seed 42. Questions sharing
gold document IDs form connected components, including transitive overlaps. Entire components
are sampled with replacement and a question-weighted mean is computed each time.
The saved `group_id` field is not used: inspect the reported cluster count and use the SAME
grouping algorithm for baseline comparisons. A single cluster produces no interval.
Few clusters limit uncertainty estimation and generalization. Individual system intervals
do not provide a paired difference test. This script does not evaluate a baseline or judge answers.

Suggested wording, after filling the actual numbers:
"On 120 test questions, MemGraphRAG-V achieved mean Document-ID Recall@5 of X%
(95% grouped-bootstrap interval [L%, U%]; G shared-gold-document groups).
Supplied-image recall was Y%. Answer containment accuracy was Z%; this is a lexical
heuristic rather than a semantic answer-quality judgement."

## Read tuning results on Snellius (no GPU)

```bash
export MGRV_ARTIFACTS="/scratch-shared/$USER/mgrv-webqa-full-v1"
"$HOME/venvs/memgraph/bin/python" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['MGRV_ARTIFACTS']) / 'runs/webqa-tuned-v1/tuning_summary.json'
if not p.is_file():
    raise SystemExit(f'Final tuning summary is not present yet: {p}')
s = json.loads(p.read_text())
print('Eligible dev questions:', s['eligible_questions'], '/', s['dev_questions'])
for row in s['scores']:
    print(f"beta={row['beta']:.2f}: Recall@5={row['mean_document_id_recall_at_5']:.6f} "
          f"({100*row['mean_document_id_recall_at_5']:.2f}%), N={row['questions']}")
print('SELECTED BETA:', s['selected_beta'])
print('Tie rule:', s['tie_break'])
PY
```

The full decision is `runs/webqa-tuned-v1/tuning_summary.json`;
the fixed test configuration is `runs/webqa-tuned-v1/test_config.yaml`;
per-question dev scores are `runs/webqa-tuned-v1/dev_recall_by_question.jsonl`.
These tuning outputs use 425 eligible development questions when the documented
55-question exclusion applies; test evaluation uses all 120 test questions.

## Optional execution on Snellius

No GPU job is needed for this small evaluation:

```bash
"$HOME/venvs/memgraph/bin/python" evaluate_webqa.py \
  "/scratch-shared/$USER/mgrv-webqa-full-v1/runs/webqa-tuned-v1-test-single"
```

## Tests

```powershell
py -3 .\test_evaluate_webqa.py
```

Four tests cover macro recall, multi-gold/transitive grouping, reproducibility,
containment boundaries, execution failures, unavailable image labels, single-cluster
intervals, CLI output and invalid/inconsistent source files. Tested with synthetic data;
actual test results were not available when this bundle was created.
