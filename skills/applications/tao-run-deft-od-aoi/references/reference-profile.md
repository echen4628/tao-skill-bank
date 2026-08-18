# `deft_od_aoi_reference` profile

Use this profile only when reproducing the reference v4 policy rather
than adapting the algorithm to a new run.

| Setting | Frozen value |
|---|---:|
| Iterations | 10 |
| Loose confidence | 0.3 |
| Strict confidence | 0.8 |
| Match IoU | 0.5 |
| Background-like IoU | below 0.05 |
| Uniform top-up | 12 per pocket in iterations 1–2; 0 afterward |
| Real FN multiplier | about `1/conversion`, clipped 1–6 |
| Near-miss real dose | 2 per FP, cap 20 per pocket |
| Clean dose | 2 per background-like FP |
| Cumulative clean cap | 1 per cumulative admitted real image |
| Synthetic base dose | 0.5 per admitted strict-FN real image plus shortage fill |
| Synthetic freeze | conversion below 0.05 with at least 4 trackable boxes |
| Cumulative synthetic cap | 25% of defective data |
| Synthetic request cap | 1,500 per iteration |
| Fixed training | 36 epochs in iterations 1–2 |
| Later probes | 3 independent runs of 10 epochs |
| Main epoch budget | `round(36 * sqrt(10000 / training_images))`, clipped 24–48 |
| Late-best extension | 12 epochs when best is in the final 3 |

The profile freezes policy, not cluster paths, container tags, or private
artifacts. The operator must still supply validated pools, a warehouse
checkpoint, AnomalyGenNext mappings, and a selected platform.

The submitted driver carried a default top-up of 12, but the realized run
froze a runtime override to zero after iteration 2. This profile follows the
effective artifact history rather than the dormant driver default.
