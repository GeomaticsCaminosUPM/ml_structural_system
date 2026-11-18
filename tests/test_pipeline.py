"""Tests on synthetic footprints: data preparation, splitting and a full small run."""

import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import box

from ml_structural_system.dataset_prep.blocks import add_block_ids
from ml_structural_system.dataset_prep.code_quality import year_series_to_code_quality
from ml_structural_system.dataset_prep.splitting import stratified_split

FEATURES = ["height", "polsby_popper", "contact_force"]


def synthetic(n_blocks: int = 60, per_block: int = 4, seed: int = 0) -> gpd.GeoDataFrame:
    """Blocks of touching 10 m squares. Taller buildings are reinforced concrete."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        x0, y0 = (b % 10) * 100.0, (b // 10) * 100.0
        for k in range(per_block):
            cr = rng.random() < 0.35
            rows.append({
                "id": len(rows),
                "geometry": box(x0 + 10 * k, y0, x0 + 10 * (k + 1), y0 + 10),
                "height": float(rng.normal(18 if cr else 6, 2)),
                "polsby_popper": float(rng.normal()),
                "contact_force": float(rng.normal()),
                "survey_structural_system": "CR" if cr else "MR",
            })
    return gpd.GeoDataFrame(rows, crs="EPSG:32619")


def test_code_quality_from_year():
    out = year_series_to_code_quality(pd.Series([1970, 1990, 2010, np.nan]), "costa_rica")
    assert out.iloc[:3].tolist() == ["pre_code", "medium_code", "high_code"]
    assert pd.isna(out.iloc[3])
    with pytest.raises(ValueError):
        year_series_to_code_quality(pd.Series([2000]), "atlantis")


def test_add_block_ids_groups_touching_footprints():
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)], crs="EPSG:32619")
    ids = add_block_ids(gdf, prefix="a_")["block_id"].tolist()
    assert ids[0] == ids[1] != ids[2]
    assert ids[0].startswith("a_")


def test_stratified_split_keeps_blocks_together():
    gdf = add_block_ids(synthetic())
    train, test, _ = stratified_split(gdf, "survey_structural_system", None, 0.25, 42, id_col="block_id")
    assert not set(train["block_id"]) & set(test["block_id"])
    assert len(train) + len(test) == len(gdf)


def test_full_run_and_predict(tmp_path):
    synthetic().to_file(tmp_path / "buildings.gpkg")
    config = {
        "random_state": 42, "n_splits": 3, "label": "survey_structural_system",
        "dataset": {"train_path": "prep/train_preprocessed.csv", "val_path": "prep/val_preprocessed.csv",
                    "test_path": "prep/test_preprocessed.csv", "train_gpkg": "splits/train_df.gpkg",
                    "val_gpkg": "splits/val_df.gpkg", "test_gpkg": "splits/test_df.gpkg"},
        "models_dir": "models", "mlflow": {"db_path": "mlflow.db"},
        "models": {"enabled": ["LogisticRegression"]},
        "preprocessing": {
            "label": "survey_structural_system", "label_values": ["CR", "M"],
            "label_replacements": {"MR": "M"}, "id_col": "block_id",
            "compute_missing_features": False, "feature_columns": FEATURES,
            "ignore_features": ["id", "dataset", "geometry", "block_id"],
            "data": {"source": "buildings.gpkg", "val_size": 0.15, "test_size": 0.2},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    def mlss(*args):
        subprocess.run([sys.executable, "-m", "ml_structural_system.cli", *args], cwd=tmp_path, check=True)

    mlss("split", "--config", "config.yaml", "--output-dir", "splits")
    mlss("preprocess", "--config", "config.yaml", "--input-dir", "splits", "--output-dir", "prep")
    mlss("train", "--config", "config.yaml")
    mlss("test", "--config", "config.yaml")

    from ml_structural_system.predict import predict

    out = predict(tmp_path / "config.yaml", tmp_path / "buildings.gpkg")
    assert len(out) == 240
    truth = out["survey_structural_system"].replace({"MR": "M"})
    assert (out["pred_LogisticRegression"] == truth).mean() > 0.9
