# Historical v3 pilot audit

Audited published dataset commit `74acdd0f22190a40a043b65c67946f6d904ee650`. The remote HEAD matched the local source checkout during this run. Original Parquet shards and published splits were preserved. Generated views remain local for the pilot.

| Split | Source rows | Eligible rows | Excluded rows | Zero-temperature readings |
|---|---:|---:|---:|---:|
| train | 569,344 | 559,829 | 9,515 | 13 |
| validation | 118,784 | 116,928 | 1,856 | 0 |
| test | 117,653 | 103,864 | 13,789 | 11,997 |

## Observed distributions

Means include the original suspect zeros so the published split difference remains visible. These results do not rebalance or modify test membership.

| Split | Mean temperature (C) | Mean power (W) | Mean historical sm_clock_mhz | Mean memory clock (MHz) |
|---|---:|---:|---:|---:|
| train | 32.897 | 16.184 | 555.639 | 1794.653 |
| validation | 32.847 | 14.916 | 459.090 | 971.198 |
| test | 30.775 | 31.517 | 692.803 | 2538.502 |

All state/outcome joins and duplicate-key/split-membership checks passed. Episode counts are 139/29/29 with zero overlap. All 805,781 rows retain missing timestamps, rewards, and action labels. One test row has zero values in all four required historical sensor fields. The separate exclusions artifact records every excluded source key and all reasons.

The audit reports distributions for all 37 numeric sensor columns. Forecast eligibility validates the four required historical sensor fields and exact 64-sample temperature outcomes. Source indices and intermediate gaps are preserved; historical sample horizons are never interpreted as seconds. Historical `sm_clock_mhz` is preserved as a source column, without equating it to the fresh collector graphics-clock feature.

Reproduce with `spikenaut-etl audit-v3 --input DATASET_CHECKOUT --output NEW_AUDIT_DIRECTORY`. Output manifests hash all nine source state/outcome/action shards and the eligible/exclusion files.
