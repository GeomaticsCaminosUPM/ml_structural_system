"""Train and test three fast models on the Ensanche Quisqueya survey (Santo Domingo).

The survey is downloaded once from the paper repository
(GeomaticsCaminosUPM/ml_structural_system_paper), keeping only the footprints,
building height, construction year, roof material and the surveyed structural
system. The footprint features are computed with footprint_attributes and the
seismic-code level is derived from the construction year. Every setting is in
config.yaml.

Run from the repository root:

    uv run python examples/ensanche_quisquella/run_example.py

Everything is written to examples/ensanche_quisquella/output/.
"""
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import yaml

from ml_structural_system.figures import plot_confusion_matrices, plot_prediction_maps, plot_scores, test_scores
from ml_structural_system.predict import predict

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
OUT = HERE / "output"
DATA = HERE / "data" / "ensanche_quisquella.gpkg"
URL = ("https://raw.githubusercontent.com/GeomaticsCaminosUPM/ml_structural_system_paper/"
       "main/data/santo_domingo/ensanche_quisquella.gpkg")
cfg = yaml.safe_load(CONFIG.read_text())
LABEL = cfg["label"]
MODELS = cfg["models"]["enabled"]


def download_data() -> None:
    """Download the survey and keep only the raw columns, so the example computes the rest."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "survey.gpkg"
        urllib.request.urlretrieve(URL, path)
        survey = gpd.read_file(path)[["id", "height", "year", "roof_material", LABEL, "geometry"]]
    survey["year"] = survey["year"].astype("Int64")
    DATA.parent.mkdir(exist_ok=True)
    survey.to_file(DATA)


def mlss(*args: str) -> None:
    """Run one step of the mlss command line."""
    subprocess.run([sys.executable, "-m", "ml_structural_system.cli", *args], check=True)


# 0. Data
if not DATA.exists():
    download_data()

# 1. Split, prepare features, train, test (from scratch)
shutil.rmtree(OUT, ignore_errors=True)
mlss("split", "--config", str(CONFIG), "--output-dir", str(OUT / "splits"))
mlss("preprocess", "--config", str(CONFIG), "--input-dir", str(OUT / "splits"),
     "--output-dir", str(OUT / "preprocessed_splits"))
mlss("report", "--config", str(CONFIG), "--output-dir", str(OUT / "report"))

# 2. Predict every building of the area, surveyed or not
buildings = predict(CONFIG, DATA)
replacements = cfg["preprocessing"]["label_replacements"]
buildings["class"] = buildings[LABEL].map(
    lambda v: None if not isinstance(v, str) or replacements.get(v, v) in (None, "None") else replacements.get(v, v))
buildings.drop(columns=["block_id"]).to_file(OUT / "predictions.gpkg")

# 3. Test scores (the test buildings were never used for training or tuning)
test_ids = gpd.read_file(OUT / "splits" / "test_df.gpkg")["id"]
test = buildings[buildings["id"].isin(test_ids)]
scores = test_scores(test, "class", MODELS)
print(scores.round(3).to_string(index=False))
scores.round(3).to_csv(OUT / "test_scores.csv", index=False)

# 4. Figures and maps
best = scores.loc[scores["model"] != "Majority baseline", "model"].iloc[0]
plot_scores(scores).savefig(OUT / "scores.jpg", dpi=200, pil_kwargs={"quality": 95})
plot_confusion_matrices(test, "class", MODELS).savefig(OUT / "confusion_matrices.jpg", dpi=200, pil_kwargs={"quality": 95})
plot_prediction_maps(buildings, test, "class", best).savefig(OUT / "maps.jpg", dpi=200, pil_kwargs={"quality": 95})
plt.close("all")
print(f"Best model: {best}. Outputs in {OUT}")
