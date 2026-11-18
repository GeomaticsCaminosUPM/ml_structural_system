# Configuration

One `config.yaml` per experiment. Paths are relative to the file. A config can start with `extends: ../main.yaml` to inherit shared settings and override only what differs. [examples/ensanche_quisquella/config.yaml](https://github.com/GeomaticsCaminosUPM/ml_structural_system/blob/main/examples/ensanche_quisquella/config.yaml) is a complete, commented example.

## Top level

| Key | Meaning |
|---|---|
| `random_state` | Seed for splits, folds and models |
| `n_splits` | Number of cross-validation folds |
| `label` | Label column |
| `dataset` | Where the preprocessed CSVs (`train_path`, `val_path`, `test_path`) and raw splits (`train_gpkg`, `val_gpkg`, `test_gpkg`) are written and read |
| `models_dir` | Where trained models are saved |
| `mlflow.db_path` | MLflow database that logs every cross-validation run |
| `models.enabled` | Any of `LogisticRegression`, `RandomForest`, `HistGradientBoosting`, `XGBoost`, `CatBoost` |
| `cost_matrix` | Optional. Adds a cost-sensitive strategy to the search (see [How it works](technical.md)) |

## `preprocessing`

| Key | Meaning |
|---|---|
| `label_values` | Allowed classes after `label_replacements` |
| `label_replacements` | Survey code to class, `None` drops the building |
| `min_class_fraction`, `min_class_size_per_split` | Classes rarer than this are dropped |
| `feature_columns` | Features used by the models |
| `compute_missing_features` | Compute missing footprint features with footprint_attributes |
| `id_col` | Block column used to keep touching buildings in one partition (`block_id`) |
| `group` | Optional columns (e.g. `city`) added to the stratification |
| `data.source` | One file or a list of files, split automatically |
| `data.train_source`, `data.test_source` | Alternatively, separate training and test areas (transfer experiments) |
| `data.val_size`, `data.test_size` | Shares of the labeled buildings |
| `data.code_type` | `costa_rica`, `guatemala` or `dominican_republic`, to derive `code_quality` from `year` |
| `ordinal_feature_mappings` | Ordered categories, e.g. `code_quality` |
| `categorical_feature_mappings` | Allowed categories of one-hot features |
| `numeric_feature_ranges` | Allowed range of numeric features (values are rounded to integers) |
| `ignore_features` | Columns never used as features |
