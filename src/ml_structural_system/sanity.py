"""Find surveyed buildings whose label the models contradict, and review them on a map.

A building is a suspect when most models agree on a class different from its
survey label. Either the label is wrong or the models share a blind spot, so
a person has to look. The interactive map (needs the ``sanity`` extra, see the
README) shows each suspect with every model's vote and a Street View link, and
lets a reviewer correct labels in the browser and export them as CSV.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from shapely.geometry.base import BaseGeometry

from .figures import CLASS_COLORS

logger = logging.getLogger(__name__)

_PREFIX = "pred_"
#: Fill of buildings that are not suspects.
_CONTEXT = "#C8C8C8"


def _model_columns(buildings: pd.DataFrame, models: Sequence[str] | None) -> list[str]:
    """Return the ``pred_<model>`` columns to vote with, all of them if ``models`` is None."""
    if models is None:
        columns = [c for c in buildings.columns if c.startswith(_PREFIX)]
    else:
        columns = [_PREFIX + m for m in models]
    missing = [c for c in columns if c not in buildings.columns]
    if missing or not columns:
        raise ValueError(f"Prediction columns not found: {missing or 'no pred_<model> column'}")
    return columns


def vote_table(
    buildings: pd.DataFrame, models: Sequence[str] | None = None, classes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize the models' votes for every building.

    Args:
        buildings: Output of :func:`ml_structural_system.predict.predict`, with
            one ``pred_<model>`` column per model.
        models: Model names to vote with. Defaults to every ``pred_`` column.
        classes: All class names, used to normalize the entropy. Defaults to the
            distinct classes predicted.

    Returns:
        A frame with the index of ``buildings`` and the columns ``consensus``
        (majority class, ties go to the alphabetically first), ``agreement``
        (share of models voting for it) and ``entropy`` (0 when unanimous, 1
        when the votes are spread evenly over all classes).
    """
    votes = buildings[_model_columns(buildings, models)].astype(str)
    n_classes = len(classes) if classes is not None else votes.stack().nunique()
    rows = []
    for row in votes.itertuples(index=False):
        counts = Counter(row)
        top = min(counts, key=lambda c: (-counts[c], c))
        p = np.array(list(counts.values())) / len(row)
        entropy = abs(float(-(p * np.log(p)).sum() / math.log(n_classes))) if n_classes > 1 else 0.0
        rows.append((top, counts[top] / len(row), entropy))
    return pd.DataFrame(rows, columns=pd.Index(["consensus", "agreement", "entropy"]), index=buildings.index)


def flag_suspects(
    buildings: pd.DataFrame,
    label_column: str,
    models: Sequence[str] | None = None,
    *,
    agreement_threshold: float = 0.6,
    classes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the labeled buildings that the models agree are another class.

    Args:
        buildings: Output of :func:`ml_structural_system.predict.predict`.
        label_column: Column with the surveyed class name. Missing values mark
            buildings without a label, which are never flagged.
        models: Model names to vote with. Defaults to every ``pred_`` column.
        agreement_threshold: Minimum share of models that must vote for the
            same class.
        classes: Classes the models can predict. Buildings labeled with any
            other class (for example one dropped from training) are skipped.

    Returns:
        The suspects, most agreed first, with the extra columns ``consensus``,
        ``agreement`` and ``entropy``.
    """
    votes = vote_table(buildings, models, classes)
    label = buildings[label_column]
    mask = label.notna() & (votes["consensus"] != label.astype(str)) & (votes["agreement"] >= agreement_threshold)
    if classes is not None:
        mask &= label.isin(list(classes))
    out = buildings[mask].join(votes[mask]).sort_values(["agreement", "entropy"], ascending=[False, True])
    logger.info("Sanity suspects: %d of %d labeled buildings (agreement >= %.0f%%)",
                len(out), int(label.notna().sum()), 100 * agreement_threshold)
    return out


def _class_colors(classes: Sequence[str]) -> dict[str, str]:
    """Return a hex color per class: the paper's colors, or a categorical palette for new classes."""
    extra = iter(plt.get_cmap("tab10")(range(10)))
    return {c: CLASS_COLORS.get(c) or to_hex(next(extra)[:3]) for c in sorted(classes)}


def _street_view_url(geom_metric: BaseGeometry) -> str:
    """Return a Google Street View URL looking at a building from outside its longest wall.

    ``geom_metric`` is the footprint in a metric CRS (EPSG:3857). The camera
    stands 20 m beyond the middle of the longest edge and looks at the centroid.
    """
    from pyproj import Transformer

    poly = geom_metric if geom_metric.geom_type == "Polygon" else max(geom_metric.geoms, key=lambda g: g.area)
    cx, cy = poly.centroid.x, poly.centroid.y
    pts = list(poly.exterior.coords)
    (x1, y1), (x2, y2) = max(zip(pts, pts[1:]), key=lambda s: math.dist(*s))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    d = math.dist((mx, my), (cx, cy)) or 1.0
    cam_x, cam_y = mx + (mx - cx) / d * 20, my + (my - cy) / d * 20
    heading = math.degrees(math.atan2(cx - cam_x, cy - cam_y)) % 360
    lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(cam_x, cam_y)
    return f"https://maps.google.com/maps?q=&layer=c&cbll={lat:.6f},{lon:.6f}&cbp=12,{heading:.1f},,0,0"


def _popup(row: pd.Series, label_column: str, vote_columns: list[str], colors: dict[str, str],
           height_column: str | None, street_view: str) -> str:
    """Return the HTML of a suspect's popup: label, consensus, votes and the Street View link."""
    label, consensus = str(row[label_column]), str(row["consensus"])
    paint = lambda c: colors.get(c, "#888")  # noqa: E731
    html = [
        '<div style="font:12px/1.7 Arial,sans-serif;min-width:250px">',
        f'<span style="font-size:13px"><b>Survey:</b> <b style="color:{paint(label)}">{label}</b>'
        f'&nbsp;&nbsp;<b>Models:</b> <b style="color:{paint(consensus)}">{consensus}</b></span><br>',
    ]
    if height_column is not None and height_column in row.index and bool(pd.notna(row[height_column])):
        html.append(f'<span style="color:#444">Height: {float(row[height_column]):.1f} m</span><br>')
    html.append(f'<span style="color:#666;font-size:11px">Agreement {row["agreement"]:.0%}'
                f' | Entropy {row["entropy"]:.2f}</span>'
                '<hr style="margin:5px 0;border:none;border-top:1px solid #ddd">')
    for col in vote_columns:
        vote = str(row[col])
        color = paint(vote) if vote == consensus else "#888"
        html.append(f'<span style="display:inline-block;width:10px;height:10px;background:{color};'
                    f'border-radius:2px;margin-right:5px"></span>{col.removeprefix(_PREFIX)}: <b style="color:{color}">{vote}</b><br>')
    html.append(f'<a href="{street_view}" target="_blank" style="color:#1565C0;text-decoration:none">'
                'Street View &rarr;</a></div>')
    return "".join(html)


_REVIEW_JS = """
<script>
var KEY = '__KEY__', CLASSES = __CLASSES__, COLORS = __COLORS__, LAYER = '__LAYER__', TOTAL = __TOTAL__;
var state = {}; try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
var markers = {};
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function btn(idx, value, text, bg, extra) {
  var on = state[idx] === value ? '2px solid #000' : '1px solid transparent';
  return '<button onclick="setReview(' + idx + ',' + (value === null ? 'null' : "'" + value + "'") + ')" '
    + 'style="background:' + bg + ';color:' + (extra || '#fff') + ';padding:4px 9px;border:' + on
    + ';border-radius:4px;margin:2px;cursor:pointer;font-size:12px;font-weight:bold">' + text + '</button>';
}
function popup(f) {
  var i = f.properties._idx, cur = state[i];
  var status = cur === 'ignore' ? '<i style="color:#888">ignored</i>'
    : cur ? '<b style="color:#2e7d32">&#10003; ' + cur + '</b>' : '<span style="color:#aaa">not reviewed</span>';
  var b = CLASSES.map(function (c) { return btn(i, c, c, COLORS[c]); }).join('')
    + btn(i, 'ignore', 'Ignore', '#777') + (cur ? btn(i, null, '&times; Clear', '#eee', '#555') : '');
  return f.properties._popup + '<hr style="margin:6px 0;border:none;border-top:1px solid #ddd">'
    + '<div style="font:12px Arial,sans-serif"><b>Review:</b> ' + status + '<br>' + b + '</div>';
}
function paint(l) {
  var p = l.feature.properties;
  l.setStyle(state[p._idx] ? {fillColor: '#aaaaaa', fillOpacity: 0.65, color: '#666', weight: 1.5}
                           : {fillColor: p._fill, fillOpacity: 0.5 + 0.45 * p.agreement, color: p._border, weight: 3});
}
function progress() {
  var done = Object.keys(state).length, ign = Object.values(state).filter(function (v) { return v === 'ignore'; }).length;
  document.getElementById('progress').innerHTML = (done - ign) + ' relabeled &middot; ' + ign + ' ignored &middot; '
    + (TOTAL - done) + ' pending';
}
window.setReview = function (i, v) {
  if (v === null) delete state[i]; else state[i] = v;
  save(); paint(markers[i]); markers[i].setPopupContent(popup(markers[i].feature)); progress();
};
function exportCsv() {
  var rows = [['index', 'id', 'survey_label', 'consensus', 'agreement', 'reviewed_label', 'lat', 'lon']];
  Object.keys(markers).forEach(function (i) {
    var p = markers[i].feature.properties;
    rows.push([i, p.id, p.label, p.consensus, p.agreement, state[i] || '', p._lat, p._lon]);
  });
  var csv = rows.map(function (r) { return r.map(function (v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(','); }).join('\\n');
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8;'}));
  a.download = KEY + '_reviewed.csv'; document.body.appendChild(a); a.click(); document.body.removeChild(a);
}
function resetAll() {
  if (!confirm('Delete all reviews?')) return;
  state = {}; save(); Object.values(markers).forEach(paint); progress();
}
window.addEventListener('load', function () {
  window[LAYER].eachLayer(function (l) {
    markers[l.feature.properties._idx] = l; paint(l);
    l.bindPopup(popup(l.feature), {maxWidth: 420});
    l.on('popupopen', function () { l.setPopupContent(popup(l.feature)); });
  });
  progress();
});
</script>
<div style="position:fixed;bottom:30px;right:30px;z-index:1000;background:#fff;padding:12px 16px;border-radius:8px;
 border:1px solid #bbb;font:13px Arial,sans-serif;box-shadow:2px 2px 8px rgba(0,0,0,.25);min-width:200px">
 <b>Label review</b>
 <span id="progress" style="display:block;color:#666;font-size:11px;margin:3px 0 6px"></span>
 <button onclick="exportCsv()" style="width:100%;padding:6px;background:#1565C0;color:#fff;border:none;
  border-radius:4px;cursor:pointer;font-weight:bold">Export CSV</button>
 <button onclick="resetAll()" style="width:100%;padding:4px;margin-top:4px;background:#fff;color:#c62828;
  border:1px solid #c62828;border-radius:4px;cursor:pointer;font-size:11px">Clear all</button>
 <div style="margin-top:5px;font-size:10px;color:#aaa;text-align:center">Saved in this browser</div>
</div>
"""


def make_sanity_map(
    suspects: gpd.GeoDataFrame,
    label_column: str,
    models: Sequence[str] | None = None,
    *,
    classes: Sequence[str] | None = None,
    context: gpd.GeoDataFrame | None = None,
    output_path: str | Path = "sanity_check.html",
    agreement_threshold: float = 0.6,
    height_column: str | None = "height",
    id_column: str | None = "id",
    max_context: int = 20000,
) -> Path:
    """Write an interactive HTML map of the suspect buildings.

    Each suspect is filled with the models' consensus color and outlined with
    its survey color. Clicking it shows the votes of every model and a Street
    View link, and buttons to record the right class. Reviews are saved in the
    browser and can be exported as CSV.

    Args:
        suspects: Output of :func:`flag_suspects`, with geometry and CRS.
        label_column: Column with the surveyed class name.
        models: Model names shown in the popups. Defaults to every ``pred_`` column.
        classes: All class names, for the legend and review buttons. Defaults to
            those seen in the suspects.
        context: Other buildings drawn in gray under the suspects (hidden by
            default in the layer control). Skipped if it has more than
            ``max_context`` rows, to keep the file light.
        output_path: HTML file to write.
        agreement_threshold: Threshold used to flag, shown in the legend.
        height_column: Column with the building height in meters, or None.
        id_column: Column with a building identifier, or None.
        max_context: Largest ``context`` that is drawn.

    Returns:
        The path of the HTML file.

    Raises:
        ImportError: If the ``sanity`` extra (folium) is not installed.
        ValueError: If ``suspects`` is empty.
    """
    try:
        import folium
    except ImportError as exc:
        raise ImportError("The sanity map needs folium: install it with `uv sync --extra sanity` "
                          "or `pip install ml-structural-system[sanity]`.") from exc

    if suspects.empty:
        raise ValueError("There are no suspect buildings to map.")
    vote_columns = _model_columns(suspects, models)
    seen = set(suspects[label_column].astype(str)) | set(suspects["consensus"].astype(str))
    colors = _class_colors(list(classes) if classes is not None else seen)
    colors.update({c: "#888888" for c in seen if c not in colors})

    wgs = suspects.to_crs(4326)
    metric = suspects.to_crs(3857).geometry
    keep = [label_column, "consensus", "agreement", "entropy", *vote_columns]
    keep += [c for c in (height_column, id_column) if c and c in suspects.columns]
    layer = wgs[keep + ["geometry"]].reset_index(drop=True).copy()
    centroids = metric.centroid.to_crs(4326)
    popups, props = [], []
    for i, (_, row) in enumerate(layer.iterrows()):
        popups.append(_popup(row, label_column, vote_columns, colors, height_column,
                             _street_view_url(metric.iloc[i])))
        props.append((str(row["consensus"]), str(row[label_column])))
    layer["_idx"] = range(len(layer))
    layer["_popup"] = popups
    layer["_fill"] = [colors[c] for c, _ in props]
    layer["_border"] = [colors[g] for _, g in props]
    layer["_lat"] = centroids.y.values.round(6)
    layer["_lon"] = centroids.x.values.round(6)
    layer["label"] = layer[label_column].astype(str)
    layer["id"] = layer[id_column].astype(str) if id_column and id_column in layer.columns else layer["_idx"].astype(str)
    layer = layer[["_idx", "id", "label", "consensus", "agreement", "_popup", "_fill", "_border", "_lat", "_lon", "geometry"]]

    minx, miny, maxx, maxy = wgs.total_bounds
    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])

    if context is not None and 0 < len(context) <= max_context:
        gray = context.to_crs(4326)[["geometry"]].copy()
        gray["geometry"] = gray.geometry.simplify(2e-6)
        folium.GeoJson(gray.to_json(), name="Other buildings", show=False,
                       style_function=lambda _: {"fillColor": _CONTEXT, "color": _CONTEXT, "weight": 0.5,
                                                 "fillOpacity": 0.6}).add_to(fmap)
    elif context is not None:
        logger.warning("Context has %d buildings (limit %d): not drawn.", len(context), max_context)

    suspect_layer = folium.GeoJson(
        layer.to_json(), name=f"Suspects ({len(layer)})",
        tooltip=folium.GeoJsonTooltip(fields=["id", "label", "consensus", "agreement"],
                                      aliases=["Building", "Survey", "Models", "Agreement"], sticky=True),
    )
    suspect_layer.add_to(fmap)

    output_path = Path(output_path)
    script = (_REVIEW_JS.replace("__KEY__", f"sanity_review_{output_path.stem}")
              .replace("__CLASSES__", json.dumps(sorted(colors))).replace("__COLORS__", json.dumps(colors))
              .replace("__LAYER__", suspect_layer.get_name()).replace("__TOTAL__", str(len(layer))))
    fmap.get_root().html.add_child(folium.Element(script))

    rows = "".join(f'<span style="display:inline-block;width:14px;height:14px;background:{c};border-radius:3px;'
                   f'margin-right:6px;vertical-align:middle"></span>{n}<br>' for n, c in sorted(colors.items()))
    legend = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:#fff;padding:14px 18px;'
              f'border-radius:8px;border:1px solid #bbb;font:13px/1.8 Arial,sans-serif;box-shadow:2px 2px 8px rgba(0,0,0,.25)">'
              f'<b>Label suspects</b> <span style="font-size:11px;color:#888">(agreement &ge; {agreement_threshold:.0%})</span><br>'
              f'{rows}<span style="font-size:11px;color:#666">Fill = model consensus<br>Border = survey label</span></div>')
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(fmap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    logger.info("Sanity map saved to %s (%d suspects)", output_path, len(layer))
    return output_path
