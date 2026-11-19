# Output files

With the default paths of the [Quickstart](../getting-started/quickstart.md):

```text
splits/                      mlss split
  train_df.gpkg              raw training buildings (with geometry and block_id)
  val_df.gpkg                raw validation buildings
  test_df.gpkg               raw test buildings
preprocessed_splits/         mlss preprocess
  train_preprocessed.csv     encoded features + label
  val_preprocessed.csv
  test_preprocessed.csv
  preprocessor.pkl           fitted encoder (needed to predict new buildings)
  label_encoder.pkl          class names and codes
  preprocessing.json         settings, feature names and class mapping
trained_models/<Model>/      mlss train, mlss test
  model_final_<Model>.pkl    final model (compressed joblib)
  thresholds_<Model>.npy     tuned class-probability divisors
  confusion_matrix_test.csv  test confusion matrix
  classification_report_test.txt
  group_f1_test.csv          test scores per source file or group
report/                      mlss report
  summary.txt                readable summary of the run
  test_metrics.csv           macro F1, accuracy, Cohen's kappa, TSS, per-class F1
  generalizability.csv       cross-validation versus test scores
  feature_importance.csv     consensus feature ranking
mlflow.db                    every cross-validation run (MLflow)
```

To apply the models elsewhere, keep `preprocessed_splits/preprocessor.pkl`, `label_encoder.pkl`, `preprocessing.json` and `trained_models/`. The rest can be regenerated.
