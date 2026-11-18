# Example: Ensanche Quisqueya

[`examples/ensanche_quisquella/`](https://github.com/GeomaticsCaminosUPM/ml_structural_system/tree/main/examples/ensanche_quisquella) trains and tests three fast models on a survey of 758 labeled buildings in Ensanche Quisqueya, Santo Domingo (Dominican Republic). It takes a few minutes on a laptop.

```bash
uv run python examples/ensanche_quisquella/run_example.py
```

The data file, `data/ensanche_quisquella.gpkg`, holds 1,400 footprints with only `id`, `height`, `year`, `roof_material` and `survey_structural_system`. Everything else is computed:

- the 23 footprint features, with footprint_attributes
- `code_quality`, from `year` with the Dominican seismic-code years (`code_type: dominican_republic`)
- the blocks of touching buildings

All settings are in [`config.yaml`](https://github.com/GeomaticsCaminosUPM/ml_structural_system/blob/main/examples/ensanche_quisquella/config.yaml).

## 1. Split, train and test

```python
mlss("split", "--config", CONFIG, "--output-dir", OUT / "splits")
mlss("preprocess", "--config", CONFIG, "--input-dir", OUT / "splits", "--output-dir", OUT / "preprocessed_splits")
mlss("report", "--config", CONFIG, "--output-dir", OUT / "report")   # train, test and assess
```

The two steel buildings are too few to learn from and are dropped. The split keeps whole blocks together:

| | Train | Validation | Test |
|---|---|---|---|
| Masonry (M) | 344 | 75 | 100 |
| Reinforced concrete (CR) | 154 | 35 | 46 |

## 2. Predict every building

```python
from ml_structural_system.predict import predict

buildings = predict(CONFIG, DATA)   # surveyed or not, one pred_<model> column per model
```

On the test buildings, `predict` gives exactly the scores of `mlss report`:

| Model | Macro F1 | Accuracy | Cohen's κ | True Skill Statistic |
|---|---|---|---|---|
| XGBoost | 0.936 | 0.945 | 0.872 | 0.861 |
| LogisticRegression | 0.928 | 0.938 | 0.856 | 0.851 |
| RandomForest | 0.879 | 0.904 | 0.761 | 0.707 |
| Majority baseline | 0.407 | 0.685 | 0.000 | 0.000 |

![Scores](assets/scores.jpg)

![Confusion matrices](assets/confusion_matrices.jpg)

## 3. Maps

```python
plot_prediction_maps(buildings, test, "class", "XGBoost")
```

Left: survey labels. Center: XGBoost predictions for every footprint, including the unsurveyed ones. Right: test buildings, correct or wrong.

![Maps](assets/maps.jpg)

The predictions are saved to `output/predictions.gpkg` for use in a geographic information system (GIS).
