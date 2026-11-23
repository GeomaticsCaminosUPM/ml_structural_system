"""Tests of the label sanity check: voting, flagging and the HTML map."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from ml_structural_system.sanity import flag_suspects, make_sanity_map, vote_table


def buildings() -> gpd.GeoDataFrame:
    """Four buildings, three models: A is unanimous, B 2 of 3, C split, D unlabeled."""
    return gpd.GeoDataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "survey": ["M", "M", "M", None],
            "height": [6.0, 9.0, 12.0, 3.0],
            "pred_X": ["CR", "CR", "M", "CR"],
            "pred_Y": ["CR", "CR", "CR", "M"],
            "pred_Z": ["CR", "M", "ADO", "M"],
        },
        geometry=[box(10 * i, 0, 10 * i + 8, 8) for i in range(4)],
        crs="EPSG:32619",
    )


def test_vote_table():
    votes = vote_table(buildings())
    assert votes["consensus"].tolist() == ["CR", "CR", "ADO", "M"]
    assert votes["agreement"].round(2).tolist() == [1.0, 0.67, 0.33, 0.67]
    assert votes["entropy"].iloc[0] == 0 and votes["entropy"].iloc[2] == pytest.approx(1.0)


def test_flag_suspects_threshold_and_unlabeled():
    b = buildings()
    assert flag_suspects(b, "survey", agreement_threshold=0.9).index.tolist() == [0]
    assert flag_suspects(b, "survey", agreement_threshold=0.6).index.tolist() == [0, 1]  # d has no label
    assert flag_suspects(b, "survey", ["X", "Y"]).index.tolist() == [0, 1]
    with pytest.raises(ValueError):
        flag_suspects(b, "survey", ["nope"])


def test_flag_suspects_skips_labels_models_cannot_predict():
    b = buildings()
    b.loc[0, "survey"] = "S"
    assert flag_suspects(b, "survey", classes=["CR", "M", "ADO"]).index.tolist() == [1]


def test_sanity_map_html(tmp_path):
    pytest.importorskip("folium")
    b = buildings()
    suspects = flag_suspects(b, "survey")
    out = make_sanity_map(suspects, "survey", context=b, output_path=tmp_path / "map.html")
    html = out.read_text()
    assert "Street View" in html and "Export CSV" in html and "maps.google.com" in html
    with pytest.raises(ValueError):
        make_sanity_map(suspects.iloc[0:0], "survey", output_path=tmp_path / "empty.html")
