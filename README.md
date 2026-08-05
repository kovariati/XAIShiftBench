# XAIShiftBench

XAIShiftBench is a tabular benchmark and audit toolkit for predictive performance and **class-composition-adjusted matched-null shifts in row-wise L1-normalized, signed, semantically aggregated SHAP attribution composition** under distribution shift.

## Scientific scope

The accompanying article is **Benchmarking Predictive Performance, Interpretability, Reliability, and Robustness under Distribution Shift for Trustworthy XAI**. In the benchmark, *interpretability* denotes the audited signed-SHAP representation, *reliability* denotes repeatability relative to the matched sampling-and-refit baseline, *robustness* denotes sensitivity under the release-defined shifts, and *trustworthy XAI* is the application motivation. The benchmark does not claim human comprehensibility, causal correctness, fairness, decision utility, or complete trustworthy-AI compliance.

The primary high-level results are dataset-specific raw-scale summaries. Any arithmetic equal-dataset summary is a supporting descriptive index only because the energy scale is not established as a common cross-dataset effect-size scale.

## Matched-sampling estimand

For each split/refit pair:

```text
source_shift S_A : source prevalence
target_shift T_A : target prevalence
source_null  S_B : target prevalence, disjoint source sample
```

The target-prevalence source-null matching is part of the estimand. It intentionally treats much of a pure class-prior change as nuisance variation. The endpoint is therefore not a generic detector of every explanation-distribution, label, or concept shift. The release records source-target movement, a same-model matched-sample null, a refit null, and their matched-null-adjusted excess.

A three-case synthetic sanity check verifies that pure prior shift is removed on average, a covariate/representation shift is retained, and pure relabeling with an unchanged fixed-model explanation marginal remains near zero.

## Reproduction

Install the locked environment and package:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Run the software/regression checks:

```bash
python reproduce.py test
```

Print the complete scientific raw-to-results command sequence without executing it:

```bash
python reproduce.py full-plan
```

Run the complete scientific reproduction after obtaining the external public archives:

```bash
python reproduce.py full \
  --oulad-zip "/path/open university learning analytics dataset.zip" \
  --acs-2018-a /path/psam_pusa_2018.zip \
  --acs-2018-b /path/psam_pusb_2018.zip \
  --acs-2024-a /path/psam_pusa_2024.zip \
  --acs-2024-b /path/psam_pusb_2024.zip
```

The full workflow verifies raw-input hashes, prepares data, executes the primary and sensitivity analyses, validates the immutable reference-output manifest at runtime, produces the raw-rerun parity audit, runs the estimand sanity simulation, finalizes result checks, and then runs the test suite. Expected hashes are in `data_checksums/expected_raw_inputs.json`.

The optional Scopus title-term audit is editorial only and is not a scientific reproduction dependency:

```bash
python reproduce.py scopus-audit --scopus /path/scopus_export.csv
```

## Tests

In the redistributable code-only repository, the regression suite reports **56 passed and 2 skipped**. The skips are raw-data-dependent integration tests for Heart Disease and South German Credit; the raw datasets are intentionally not redistributed. Test success is software/regression evidence, not proof that the estimand is universally valid.

## Data

Raw datasets are not included. Obtain them from the sources listed in `SOURCES.md` and place the small local files in the documented `data/*/raw/` directories. The large OULAD and ACS archives are passed to `reproduce.py full` by command-line argument.

## Environment and container

`requirements-lock.txt` is the normative dependency lock for the released calculations. The provided Dockerfile pins the Python image by tag and supports functional, not bit-identical, base-image reproduction.

## Citation

If you use XAIShiftBench, its code, benchmark protocol, or results in academic or scientific work, please cite the software and the associated article when available. GitHub can generate a software citation directly from `CITATION.cff`.

Software citation:

```bibtex
@software{kovari2026xaishiftbench,
  author  = {Kovari, Attila},
  title   = {XAIShiftBench},
  year    = {2026},
  version = {1.0.0}
}
```

Associated article:

> Attila Kovari. *Benchmarking Predictive Performance, Interpretability, Reliability, and Robustness under Distribution Shift for Trustworthy XAI*.

The bibliographic details and DOI of the article should be added here after publication.

## License

Source code is released under the MIT License. Dataset licenses remain with their original providers; see `DATA_LICENSE_NOTICE.md`.
