# ml_structural_system

Predict the structural system of individual buildings (adobe, reinforced concrete, masonry, steel...) from their footprint, their neighbors and a few low-cost attributes such as height.

It is the code of the paper *How far can footprint data go? Inferring structural systems for seismic exposure models across three Latin American cities*. The experiments, data and figures of the paper are in [ml_structural_system_paper](https://github.com/GeomaticsCaminosUPM/ml_structural_system_paper), and its results can be explored with maps in a [Google Colab notebook](https://colab.research.google.com/github/GeomaticsCaminosUPM/ml_structural_system_paper/blob/main/paper_results.ipynb).

## Installation

```bash
git clone https://github.com/GeomaticsCaminosUPM/ml_structural_system.git
cd ml_structural_system
uv sync
```

Or with pip: `pip install git+https://github.com/GeomaticsCaminosUPM/ml_structural_system.git`. Python 3.12 or newer.

XGBoost and CatBoost train on the GPU when CUDA is available and on the CPU otherwise. Their standard wheels already include GPU support on Linux and Windows, so nothing else needs installing.

## Quick start

Everything is set in a `config.yaml` (see [Configuration](configuration.md)). Then:

```bash
mlss split      --config config.yaml --output-dir splits
mlss preprocess --config config.yaml --input-dir splits --output-dir preprocessed_splits
mlss report     --config config.yaml --output-dir report      # train, test and assess every model
```

and, to apply the trained models to any buildings, labeled or not:

```python
from ml_structural_system.predict import predict

buildings = predict("config.yaml", "buildings.gpkg")   # one pred_<model> column per model
```

## Input data

A GeoPackage with one footprint polygon per building, covering every building of the area (neighbor features depend on the buildings around each one).

| Column | Required | Notes |
|---|---|---|
| `geometry` | yes | Footprint polygons, any coordinate reference system |
| label column | for training | Surveyed structural system, mapped to classes by `label_replacements` |
| `height` | if used as a feature | Meters |
| `year`, `roof_material` | no | Only if listed in `feature_columns`. Nothing estimates them. |
| `code_quality` | no | Derived from `year` when `code_type` is set |
| footprint features | no | Computed with [footprint_attributes](https://github.com/GeomaticsCaminosUPM/footprint_attributes) when missing |
| `block_id` | no | Computed from touching footprints when missing |

See [Example](examples.md) for a complete run and [How it works](technical.md) for the details.

## License

[European Union Public Licence v1.2 (EUPL-1.2)](https://github.com/GeomaticsCaminosUPM/ml_structural_system/blob/main/LICENSE).
