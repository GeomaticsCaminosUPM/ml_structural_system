# Configuration

Everything is set in one YAML file per experiment. Relative paths are resolved from the file's folder.

!!! tip "Sharing settings"
    A config can start with `extends: ../main.yaml`. It inherits every key of that file and overrides only what it sets itself. Dictionaries are merged key by key. A list can remove items from the inherited one with `{exclude: [...]}`, for example `feature_columns: {exclude: [roof_material, year]}`.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `extends` | | Parent configuration to inherit from |
| `label` | required | Label column of the preprocessed data |
| `random_state` | `42` | Seed of folds, resampling and models |
| `n_splits` | `5` | Cross-validation folds |
| `models_dir` | `trained_models` | Where trained models are saved, one folder per model |
| `calibration_objective` | see meaning | What the decision rule optimizes: `f1_macro` (maximize macro F1) or `expected_cost` (minimize the cost matrix). If unset: `f1_macro`, or `expected_cost` when a `cost_matrix` is configured |

### `dataset`

Where `mlss preprocess` writes, and the other steps read, the data.

| Key | Meaning |
|---|---|
| `train_path`, `test_path` | Preprocessed training and test CSVs (required) |
| `val_path` | Preprocessed validation CSV. When set, the decision rule is tuned on it. Otherwise on a 20% holdout of the training set |
| `train_gpkg`, `val_gpkg`, `test_gpkg` | Raw splits with geometry, used for per-group test scores and maps |

### `models`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | | Any of `LogisticRegression`, `RandomForest`, `HistGradientBoosting`, `XGBoost`, `CatBoost` |
| `skip_completed` | `true` | Skip models that already have a final model on disk |
| `start_from_model` | | Start the run from this model |

### `mlflow`

Every cross-validation run is logged to [MLflow](https://mlflow.org).

| Key | Default | Meaning |
|---|---|---|
| `db_path` | `mlflow.db` | SQLite database of the runs |
| `tracking_uri` | from `db_path` | Use another MLflow server instead |
| `experiment_cv`, `experiment_final` | `multiclass_model_selection`, `final_evaluation` | MLflow experiment names |
| `hyperparam_db_path`, `hyperparam_tracking_uri`, `hyperparam_experiment_cv` | | Take the selected hyperparameters from another experiment's runs |

### `assessment`

| Key | Default | Meaning |
|---|---|---|
| `bootstrap.n_samples` | `1000` | Bootstrap resamples of the confidence intervals |
| `bootstrap.confidence` | `0.95` | Confidence level |
| `bootstrap.models` | all | Models to bootstrap |

### `cost_matrix`

Optional. Adds a cost-sensitive imbalance strategy, and enables `calibration_objective: expected_cost`.

| Key | Default | Meaning |
|---|---|---|
| `fragility_rank` | | Rank of each class by seismic fragility (higher is more fragile) |
| `pessimism_factor` | `3.0` | How much more it costs to predict a class less fragile than the true one |
| `label_encoding` | | Fixed class-to-code mapping, also used to name the classes in reports |

## `preprocessing`

| Key | Default | Meaning |
|---|---|---|
| `label` | required | Label column of the raw data |
| `seed` | `42` | Seed of the split |
| `label_values` | | Allowed classes after the replacements |
| `label_replacements` | | Survey code to class. `None` drops the building |
| `min_class_fraction` | `0.04` | Classes rarer than this share of labeled buildings are dropped |
| `min_class_size_per_split` | `0` | Classes with fewer buildings in any split are dropped |
| `feature_columns` | | Features used by the models |
| `compute_missing_features` | `true` | Compute missing footprint features with footprint_attributes |
| `ignore_features` | | Columns never used as features |
| `id_col` | | Column whose groups stay in one partition, normally `block_id` |
| `group` | | Columns added to the stratification of the split, e.g. `city` |
| `ordinal_feature_mappings` | | Ordered categories, e.g. `code_quality: [pre_code, low_code, medium_code, high_code]` |
| `categorical_feature_mappings` | | Allowed categories of one-hot features |
| `numeric_feature_ranges` | | Allowed range of numeric features, e.g. `year: [1900, 2050]`. Values are rounded to whole numbers |

### `preprocessing.data`

| Key | Default | Meaning |
|---|---|---|
| `source` | | One file or a list, split automatically into training, validation and test |
| `train_source`, `test_source` | | Alternatively, separate training and test areas (transfer to a new area) |
| `test_size` | `0.2` | Share of labeled buildings for test |
| `val_size` | | Share of labeled buildings for validation |
| `code_type` | | `costa_rica`, `guatemala` or `dominican_republic`, to derive `code_quality` from `year`. Can be a mapping from file to country |
| `max_per_group`, `min_per_label` | | Cap the buildings per group and split, keeping at least `min_per_label` (default 5) per class |

## Complete example

```yaml
--8<-- "examples/ensanche_quisquella/config.yaml"
```
