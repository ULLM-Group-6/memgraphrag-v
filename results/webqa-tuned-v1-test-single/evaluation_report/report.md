# WebQA test results

Questions: 120; beta: 0.25; gold-document groups: 17.

| Metric | Score (%) | 95% grouped-bootstrap CI (%) | N |
|---|---:|---|---:|
| Document-ID Recall@5 | 67.50 | [44.44, 81.35] | 120 |
| Supplied-image Recall@5 | 63.75 | [44.44, 75.24] | 120 |
| Answer containment accuracy (heuristic) | 0.83 | [0.00, 3.33] | 120 |

Recall is the mean per-question fraction of gold IDs recovered, not any-hit accuracy.
Image recall uses images actually supplied to the reader; only questions with gold images are included.
Containment is a heuristic, NOT judged semantic accuracy. Paraphrases can fail it; incorrect answers can pass it.
Failed/empty answers count as zero containment matches; their saved retrieval is still scored.
Bootstrap clusters are connected components of shared gold document IDs, including transitive overlap.
The stored group_id is not used. Match this grouping in baseline evaluation before comparing intervals.
Intervals describe this selected sample; few clusters limit generalization. No paired system comparison is performed.

Successful nonempty answers: 120/120.
Routes: {'graph': 120}. Statuses: {'ok': 120}.
