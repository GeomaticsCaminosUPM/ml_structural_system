"""Apply trained models to new buildings, labeled or not."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .dataset_prep import load_preprocessing_config
from .dataset_prep.blocks import add_block_ids
from .dataset_prep.io import clean_geometries, ensure_feature_columns, load_tabular_source

logger = logging.getLogger(__name__)


def predict(
    config_path: str | Path,
    data: str | Path | Sequence[str | Path] | gpd.GeoDataFrame,
    models: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """Predict the structural system of every building in ``data``.

    The buildings go through the same preparation as the training data:
    missing footprint features are computed with footprint_attributes and
    ``code_quality`` is derived from ``year`` when the config sets a
    ``code_type``. They are then encoded with the preprocessor fitted on the
    training split, and each model applies its tuned class-probability
    divisors. Run ``mlss preprocess`` and ``mlss train`` (or ``mlss report``)
    on the config first.

    Args:
        config_path: The experiment's ``config.yaml``.
        data: A GeoPackage or CSV path, a list of paths, or a GeoDataFrame.
            Pass every footprint of the area, since neighbor features depend
            on the buildings around each one.
        models: Models to apply. Defaults to the config's enabled models.

    Returns:
        The buildings that have every feature, with one ``pred_<model>``
        column per model holding the predicted class name. Buildings with a
        missing feature are left out and logged.
    """
    cfg = load_config(config_path)
    prep = load_preprocessing_config(config_path)
    features = list(prep.feature_columns)

    if isinstance(data, gpd.GeoDataFrame):
        gdf = clean_geometries(data)
        if "block_id" not in gdf.columns:
            gdf = add_block_ids(gdf)
        gdf = ensure_feature_columns(gdf, features, code_type=prep.code_type, source_label="data",
                                     compute_missing_features=prep.compute_missing_features)
    else:
        paths = [data] if isinstance(data, (str, Path)) else list(data)
        gdf = load_tabular_source([str(p) for p in paths], features, prep.code_type,
                                  compute_missing_features=prep.compute_missing_features)

    x = gdf[features].copy()
    for col in prep.numeric_feature_ranges:  # rounded to whole numbers, as in training
        if col in x.columns:
            x[col] = x[col].round()

    prep_dir = Path(cfg.train_path).parent
    preprocessor = joblib.load(prep_dir / "preprocessor.pkl")
    label_encoder = joblib.load(prep_dir / "label_encoder.pkl")
    meta_path = prep_dir / "preprocessing.json"
    if meta_path.exists():
        columns = json.loads(meta_path.read_text())["feature_names"]
    else:
        columns = [c for c in pd.read_csv(cfg.train_path, nrows=0).columns if c != cfg.label]
    x_enc = pd.DataFrame(preprocessor.transform(x), columns=columns, index=gdf.index)

    out = gdf.copy()
    for name in models or cfg.enabled_models:
        work_dir = Path(cfg.models_dir) / name
        model = joblib.load(work_dir / f"model_final_{name}.pkl")
        thresholds_path = work_dir / f"thresholds_{name}.npy"
        if thresholds_path.exists():
            codes = (model.predict_proba(x_enc) / np.load(thresholds_path)).argmax(axis=1)
        else:
            codes = np.asarray(model.predict(x_enc)).ravel().astype(int)
        out[f"pred_{name}"] = label_encoder.inverse_transform(codes)
    return out
