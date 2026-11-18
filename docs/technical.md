# How it works

```
footprints ──► mlss split ──► mlss preprocess ──► mlss train ──► mlss test ──► mlss assess
               features,       encoding            CV search,      test set      agreement,
               blocks, split                       final fit,                    importance
                                                   thresholds
```

`mlss report` runs train, test and assess in one go and writes `summary.txt`, `test_metrics.csv`, `generalizability.csv` and `feature_importance.csv`.

## 1. Split (`mlss split`)

**Features.** Each source file is loaded and the footprints are cleaned. Then:

- Footprint features listed in `feature_columns` but missing from the file are computed with [footprint_attributes](https://github.com/GeomaticsCaminosUPM/footprint_attributes). This covers 18 plan-shape indices from Eurocode 8, the American Society of Civil Engineers standard ASCE 7, the Italian National Group for Earthquake Defence form (GNDT-II), the Costa Rican seismic code CSCR-2010 and the Mexico City standards NTC-23, plus 5 neighbor descriptors. The neighbor descriptors are contact forces, confinement, angular acceleration, contact angle and position in the block.
- `code_quality` is derived from `year` with the years each national seismic code came into force:

| `code_type` | low_code | medium_code | high_code |
|---|---|---|---|
| `costa_rica` | 1974 | 1986 | 2003 |
| `guatemala` | 1996 | | |
| `dominican_republic` | 1979 | 2011 | |

- Footprints closer than 5 cm are joined into blocks (`block_id`) when the file has none.

Buildings missing a feature are dropped, survey codes are mapped to classes and rare classes are removed.

**Split.** The labeled buildings of each source file are split into training (65%), validation (15%) and test (20%) sets. The split is stratified by class (and by `group`, if set), and whole blocks go to one partition. Neighbors share contact features, so a building-level random split would leak information between partitions.

## 2. Preprocess (`mlss preprocess`)

Numeric features get a Yeo-Johnson transform, ordinal features (`code_quality`) an ordinal code and categorical features a one-hot code with the first category dropped. Everything is then standardized. The encoder is fitted on the training set only.

## 3. Train (`mlss train`)

**Model selection.** For each model, a grid search with stratified k-fold cross-validation on the training set compares every hyperparameter set with every imbalance strategy. It keeps the highest mean macro F1, and every run is logged to MLflow.

| Strategy | What it does |
|---|---|
| `baseline` | Plain fit |
| `class_weight` | Each class weighted by `max_count / class_count` |
| `smote` | Synthetic Minority Over-sampling Technique (SMOTE) |
| `smote_tomek` | SMOTE, then removal of Tomek links |
| `cost_sensitive` | Only with a `cost_matrix`: weights from misclassification costs |

| Model | Grid | Notes |
|---|---|---|
| `LogisticRegression` | C ∈ {0.01, 0.1, 1, 10} | L2 regularization |
| `RandomForest` | depth ∈ {6, 12, None}, min split ∈ {2, 8}, max features ∈ {sqrt, 0.5} | Balanced random forest, which under-samples each tree, so no SMOTE. 200 trees in cross-validation, 500 in the final fit. |
| `HistGradientBoosting` | learning rate ∈ {0.05, 0.1}, depth ∈ {4, 6, None}, min leaf ∈ {10, 30} | Early stopping on an internal 10% split |
| `XGBoost` | depth ∈ {4, 6}, learning rate ∈ {0.05, 0.1}, min child weight ∈ {1, 5}, subsample 0.8 | Early stopping on the held-out fold. GPU if available. |
| `CatBoost` | depth ∈ {4, 6}, learning rate ∈ {0.05, 0.1}, L2 ∈ {1, 10} | Early stopping on the held-out fold. GPU if available. |

**Final fit.** The selected configuration is refitted on the whole training set. XGBoost and CatBoost use the mean best iteration of the folds as their number of trees.

**Decision rule.** Most buildings belong to the majority class, so taking the highest probability rarely predicts the others. Each class probability $p_c$ is divided by a class divisor $d_c$, and the class with the largest $p_c / d_c$ wins. The divisors are tuned on the validation set to maximize macro F1, by coordinate-wise search over 120 log-spaced values between 0.05 and 5, in two passes. The validation set is used for nothing else.

## 4. Test (`mlss test`)

Each final model is scored once on the test set. The metrics are macro F1, accuracy, Cohen's κ, the macro-averaged True Skill Statistic (TSS, recall + specificity − 1), per-class F1 and the confusion matrix, broken down per source file or group.

## 5. Assess (`mlss assess`)

The models are compared on the same test buildings:

- per-building agreement between models, and Fleiss' κ
- bootstrap confidence intervals
- the gap between cross-validated and test scores
- feature importance (SHAP for most models, permutation importance otherwise) with a consensus ranking across models

## Predict

[`predict`](api.md#ml_structural_system.predict.predict) applies the trained models to any buildings. They go through the same feature computation and the saved encoder, and each model applies its tuned divisors.

## GPU

XGBoost and CatBoost train on a CUDA GPU when one is found ([`cuda_available`](api.md#ml_structural_system.gpu.cuda_available)), and predict on the CPU. The other models run on the CPU.
