# Quickstart

## 1. A footprint file

One GeoPackage with every building footprint of the area, surveyed or not. Surveyed buildings carry their structural system in a label column. See [Preparing data](../guide/data.md).

## 2. A configuration

A `config.yaml` names the data, the label, the features and the models. A minimal one:

```yaml
label: survey_structural_system
dataset:
  train_path: preprocessed_splits/train_preprocessed.csv
  val_path: preprocessed_splits/val_preprocessed.csv
  test_path: preprocessed_splits/test_preprocessed.csv
  test_gpkg: splits/test_df.gpkg
models:
  enabled: [LogisticRegression, RandomForest, XGBoost]
preprocessing:
  label: survey_structural_system
  id_col: block_id
  data:
    source: buildings.gpkg
    val_size: 0.15
    test_size: 0.20
  ignore_features: [id, dataset, geometry, block_id]
  feature_columns: [height, polsby_popper, convex_hull_irregularity, contact_force, blockPosition]
```

Every key is described in [Configuration](../guide/configuration.md). The [Ensanche Quisqueya example](../examples/ensanche-quisquella.md) has a complete one.

## 3. Run

=== "Command line"

    ```bash
    mlss split      --config config.yaml --output-dir splits
    mlss preprocess --config config.yaml --input-dir splits --output-dir preprocessed_splits
    mlss report     --config config.yaml --output-dir report
    ```

=== "Python"

    ```python
    import subprocess, sys

    def mlss(*args):
        subprocess.run([sys.executable, "-m", "ml_structural_system.cli", *args], check=True)

    mlss("split", "--config", "config.yaml", "--output-dir", "splits")
    mlss("preprocess", "--config", "config.yaml", "--input-dir", "splits", "--output-dir", "preprocessed_splits")
    mlss("report", "--config", "config.yaml", "--output-dir", "report")
    ```

`report` trains every model, tests it and writes `report/test_metrics.csv`, `report/summary.txt` and the feature importance. See [Output files](../guide/outputs.md).

## 4. Predict every building

```python
from ml_structural_system.predict import predict

buildings = predict("config.yaml", "buildings.gpkg")
buildings[["id", "pred_XGBoost"]].head()
```

## 5. Look at the results

```python
from ml_structural_system.figures import plot_prediction_maps, plot_confusion_matrices, test_scores
```

See [Predicting new buildings](../guide/prediction.md).
