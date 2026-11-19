# Predicting new buildings

[`predict`][ml_structural_system.predict.predict] applies trained models to any buildings. They go through the same preparation as the training data, the saved encoder, and each model's tuned decision rule.

```python
import geopandas as gpd
from ml_structural_system.predict import predict

buildings = predict("config.yaml", "new_area.gpkg")                 # a file or a list of files
buildings = predict("config.yaml", gpd.read_file("new_area.gpkg"))  # or a GeoDataFrame
buildings = predict("config.yaml", "new_area.gpkg", models=["XGBoost"])
```

The result is a GeoDataFrame with one `pred_<model>` column per model, holding the class name. Buildings missing a feature are left out and logged.

!!! warning "Same features, same meaning"
    New buildings need the same features as the training data. Footprint features are computed when missing, but height, year and roof material must come with the data, from comparable sources.

## Scores and figures

[`ml_structural_system.figures`](../reference/figures.md) turns predictions into tables, confusion matrices and maps:

```python
from ml_structural_system.figures import (
    test_scores, plot_scores, plot_confusion_matrices, plot_prediction_maps,
)

models = ["LogisticRegression", "RandomForest", "XGBoost"]
test = buildings[buildings["id"].isin(test_ids)]       # test buildings, with the true class in "class"
scores = test_scores(test, "class", models)            # macro F1, accuracy, kappa, TSS
plot_scores(scores).savefig("scores.jpg")
plot_confusion_matrices(test, "class", models).savefig("confusion_matrices.jpg")
plot_prediction_maps(buildings, test, "class", "XGBoost").savefig("maps.jpg")
```

The [Ensanche Quisqueya example](../examples/ensanche-quisquella.md) runs all of this end to end.
