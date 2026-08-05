# XAIShiftBench data and methodological sources

## Raw datasets

- Heart Disease, UCI Machine Learning Repository, DOI 10.24432/C52P4X. The analysis uses the Cleveland, Hungarian, Switzerland, and VA Long Beach processed source files; their SHA-256 hashes are listed in the input registry.
- Student Performance, UCI Machine Learning Repository, Dataset 320, DOI 10.24432/C5TG7T.
- Open University Learning Analytics Dataset, UCI Machine Learning Repository, DOI 10.24432/C5KK69, and Kuzilek et al., Scientific Data 4, 170171 (2017).
- South German Credit, UCI Machine Learning Repository, Dataset 573 (2020), DOI 10.24432/C5QG88.
- 2018 and 2024 ACS 1-Year PUMS national person files, U.S. Census Bureau.

## ACS income adjustment and weighting scope

The ACS PUMS documentation defines `ADJINC` as an income adjustment factor with six implied decimal places. XAIShiftBench applies

```text
PINCP_ADJ = PINCP * ADJINC / 1_000_000
```

before the income filter and target construction. CPI-U annual averages are used only for the documented between-year real-dollar threshold construction. The primary explanation-distribution endpoint describes the empirical filtered PUMS person-record distribution and is not survey-weighted. A PWGTP-weighted sensitivity is evaluated on the same matched rows; it is not claimed to be a full design-based estimator using replicate weights.

## Methodological sources

- Szekely and Rizzo, Energy statistics: A class of statistics based on distances, Journal of Statistical Planning and Inference 143 (2013) 1249-1272.
- Lundberg and Lee, A Unified Approach to Interpreting Model Predictions, NeurIPS 2017.
- Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NeurIPS 2017.
- The published XAI evaluation, explanation-shift, distribution-shift, concept-shift, and attribution-stability studies cited in the accompanying manuscript.

## Reference outputs

Four immutable primary result tables used for raw-rerun parity are bundled under `reference_outputs/` and protected by `reference_outputs/MANIFEST_SHA256_REFERENCE.txt`. They are reference outputs, not substitute raw data. The executable parity step validates all four hashes at runtime before comparison.
