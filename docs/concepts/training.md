# Model selection and imbalance

## Encoding

Numeric features get a Yeo-Johnson transform, ordinal features (`code_quality`) an ordinal code and categorical features a one-hot code with the first category dropped. Everything is then standardized. The encoder is fitted on the training set only and saved for prediction.

## Search

For each model, a grid search with stratified k-fold cross-validation (`n_splits`, default 5) on the training set compares every hyperparameter set with every imbalance strategy, and keeps the one with the highest mean macro F1. Every run is logged to MLflow.

| Model | Hyperparameter grid | Notes |
|---|---|---|
| `LogisticRegression` | C ∈ {0.01, 0.1, 1, 10} | L2 regularization, linear baseline |
| `RandomForest` | max depth ∈ {6, 12, None}, min samples split ∈ {2, 8}, max features ∈ {sqrt, 0.5} | Balanced random forest (imbalanced-learn), which under-samples the majority class in each tree. 200 trees in cross-validation, 500 in the final fit |
| `HistGradientBoosting` | learning rate ∈ {0.05, 0.1}, max depth ∈ {4, 6, None}, min samples leaf ∈ {10, 30} | Early stopping on an internal 10% split |
| `XGBoost` | max depth ∈ {4, 6}, learning rate ∈ {0.05, 0.1}, min child weight ∈ {1, 5}, subsample 0.8, column sample 0.8 | Early stopping on the held-out fold. GPU if available |
| `CatBoost` | depth ∈ {4, 6}, learning rate ∈ {0.05, 0.1}, L2 leaf regularization ∈ {1, 10} | Early stopping on the held-out fold. GPU if available |

## Imbalance strategies

| Strategy | What it does |
|---|---|
| `baseline` | Plain fit |
| `class_weight` | Each class weighted by `max_count / class_count` |
| `smote` | Synthetic Minority Over-sampling Technique (SMOTE): new minority examples interpolated between neighbors |
| `smote_tomek` | SMOTE, then removal of Tomek links (pairs of nearest neighbors from different classes) |
| `cost_sensitive` | Only with a `cost_matrix`: weights from the misclassification costs |

The balanced random forest already under-samples each tree, so it only compares `baseline` and `class_weight` (and `cost_sensitive`).

## Final fit

The selected configuration is refitted on the whole training set. XGBoost and CatBoost use the mean best iteration of the folds, plus one, as their number of trees. The model is saved compressed in `trained_models/<Model>/`.
