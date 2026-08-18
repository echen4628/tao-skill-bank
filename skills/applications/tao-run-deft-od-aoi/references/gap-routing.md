# Gap matching and routing

## Exact matching algorithm

Run the same KPI predictions through `tao-analyze-gaps-od-map` twice. Both
passes use one class (`defect`), confidence filtering before matching, IoU
threshold 0.5, minimum area zero, and same-class greedy one-to-one matching in
descending prediction-confidence order.

- A matched prediction/GT pair at IoU 0.5 or above is a true positive.
- An unmatched retained prediction is a false positive and records its best
  IoU with any opposite-side box.
- An unmatched GT is a false negative and records its best IoU with any
  opposite-side prediction.

Per-image AP50 is computed from the full prediction ranking, because AP sweeps
confidence internally. Weak-image selection is recall below 1.0 or AP50 below
0.5; precision-based weak selection is disabled. DEFT OD AOI routing consumes box
gaps, not the weak-image list.

## Dual confidence passes

| Pass | Confidence | Consumed rows | Purpose |
|---|---:|---|---|
| Loose | 0.3 | FP only | Precision: near-miss positives and clean negatives |
| Strict | 0.8 | FN only | Recall: real positives and synthetic requests |

The strict pass is not a high-confidence FP search. It discards predictions
below 0.8 before matching, so a GT covered only by a lower-confidence box
becomes an FN. This deliberately requests data that should move a tentative
detection above the operating threshold.

## Confidence-by-IoU decision table

IoU applies to a retained prediction's best overlap. A GT with no retained
match is handled by the FN rule in the last column.

| Prediction confidence | Best IoU below 0.05 | Best IoU 0.05 to below 0.5 | IoU 0.5 or above | Unmatched GT |
|---|---|---|---|---|
| 0.8 or above | Loose FP: request clean negatives | Loose FP near miss: request real defects | TP: no route | Not possible for the same matched pair; other unmatched GTs are strict FNs |
| 0.3 to below 0.8 | Loose FP: request clean negatives | Loose FP near miss: request real defects | Loose TP; no FP route | The corresponding GT is a strict FN after this prediction is filtered out |
| Below 0.3 | Ignored by loose and strict matching | Ignored by loose and strict matching | Ignored by both passes; its GT is an FN in both | Loose FN and strict FN, but DEFT OD AOI consumes the strict FN route |

Boundary rules are exact: IoU below 0.05 is background-like; IoU from 0.05
inclusive to 0.5 exclusive is a near miss; IoU 0.5 or above is a match.

## Doses

- Group strict FNs by `(benchmark, texture, defect_type)` pocket.
- Compare stable `(image, rounded GT box)` identities from two prior strict
  gap sets. Mine about `1 / conversion_rate` real examples, clipped to 1–6
  times the current FN count. Before a trackable rate exists, use the frozen
  0.33 prior.
- When fewer than 5% of at least four old boxes convert, freeze synthesis for
  that pocket and keep real mining at the minimum factor.
- Rank real candidates by DCT similarity between source defect crops and KPI
  FN crops. KPI pixels remain queries only.
- Each near-miss loose FP requests two real defect images, capped at 20 per
  pocket per iteration.
- Each background-like loose FP requests two clean images from the same
  benchmark/texture. Cumulative clean negatives cannot exceed cumulative
  admitted real defect images.
- Synthetic requests are half the admitted strict-FN real dose plus bounded
  shortage fill. Requests are capped at 1,500 per iteration and admitted
  synthetic images remain at or below 25% of cumulative defective data.
- Uniform mining is independent of gaps. Its value comes only from the frozen
  profile; zero disables it.

## Important clean-KPI limitation

The current gap implementation iterates classes present in each image's ground
truth. Predictions on a KPI image with no GT classes therefore are normally
not emitted as FP rows. DEFT OD AOI can route low-IoU loose FPs that are emitted on
defect-containing KPI images, but it does not recover missing clean-image false
alarms after the fact. Treat this as a measured coverage limitation, not proof
that the model has no false alarms on clean KPI images.

## Threshold rationale

- Confidence 0.3 keeps the FP search broad enough to expose spurious and
  poorly localized boxes without allowing the very long low-confidence tail to
  dominate mining cost.
- Confidence 0.8 turns uncertain coverage into recall work at the intended
  high-confidence operating point.
- IoU 0.5 is the AP50/localization success boundary.
- IoU 0.05 distinguishes a box with almost no spatial evidence for the GT from
  a box plausibly aimed at the defect. It is a routing heuristic, not a new
  definition of true positive.
