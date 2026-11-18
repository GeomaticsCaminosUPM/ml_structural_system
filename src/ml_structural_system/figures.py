"""Test scores, confusion matrices and maps of an experiment's results."""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

#: Map colors of the default classes: adobe, reinforced concrete, masonry and steel.
CLASS_COLORS: dict[str, str] = {"ADO": "#B24A2A", "CR": "#9AA5AE", "M": "#F0D78C", "S": "#3D4F63"}
#: Full names of the default classes, used in legends.
CLASS_NAMES: dict[str, str] = {"ADO": "Adobe", "CR": "Reinforced concrete", "M": "Masonry", "S": "Steel"}
UNLABELED = "#E3E3E3"
CORRECT = "#7FB77E"
WRONG = "#C0392B"


def _color(cls: object) -> str:
    """Return the map color of a class, light gray for a missing class."""
    if not isinstance(cls, str):
        return UNLABELED
    return CLASS_COLORS.get(cls, plt.get_cmap("tab10")(hash(cls) % 10))


def _name(cls: str) -> str:
    """Return a legend label such as "Masonry (M)", or the code itself if it has no known name."""
    return f"{CLASS_NAMES[cls]} ({cls})" if cls in CLASS_NAMES else cls


def _tss(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence) -> float:
    """Return the True Skill Statistic (recall + specificity - 1) averaged over ``labels``."""
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    total, out = cm.sum(), []
    for c in range(len(labels)):
        tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        tn = total - tp - fn - fp
        if tp + fn > 0 and tn + fp > 0:
            out.append(tp / (tp + fn) + tn / (tn + fp) - 1)
    return float(np.mean(out))


def test_scores(predictions: pd.DataFrame, label_column: str, models: Sequence[str]) -> pd.DataFrame:
    """Score every model and a majority-class baseline on the test buildings.

    Args:
        predictions: Test buildings with the true class in ``label_column``
            and one ``pred_<model>`` column per model.
        label_column: Column with the true class.
        models: Model names.

    Returns:
        One row per model, sorted by macro F1, with macro F1 (over the classes
        present in the test set), accuracy, Cohen's kappa and the True Skill
        Statistic (TSS).
    """
    y = predictions[label_column].to_numpy()
    labels = sorted(set(y))
    preds = {m: predictions[f"pred_{m}"].to_numpy() for m in models}
    preds["Majority baseline"] = np.full(len(y), pd.Series(y).mode()[0])
    rows = [{"model": name,
             "macro_f1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
             "accuracy": accuracy_score(y, p),
             "kappa": cohen_kappa_score(y, p),
             "tss": _tss(y, p, labels)} for name, p in preds.items()]
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False, ignore_index=True)


def plot_scores(scores: pd.DataFrame, metric: str = "macro_f1") -> Figure:
    """Draw one metric per model as horizontal bars.

    Args:
        scores: Output of :func:`test_scores`.
        metric: Column to plot.

    Returns:
        The figure.
    """
    res = scores.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(res) + 1))
    colors = ["#BBBBBB" if m == "Majority baseline" else "#3E6F9E" for m in res["model"]]
    ax.barh(res["model"], res[metric], color=colors)
    for y, v in enumerate(res[metric]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric.replace("_", " "))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(predictions: pd.DataFrame, label_column: str, models: Sequence[str]) -> Figure:
    """Draw the row-normalized confusion matrix of each model, with building counts.

    Args:
        predictions: Test buildings with the true class and ``pred_<model>`` columns.
        label_column: Column with the true class.
        models: Model names.

    Returns:
        The figure.
    """
    classes = sorted(set(predictions[label_column]) | {c for m in models for c in predictions[f"pred_{m}"]})
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        cm = confusion_matrix(predictions[label_column], predictions[f"pred_{model}"], labels=classes)
        pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 60 else "black")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(model)
    named = [_name(c) for c in classes if c in CLASS_NAMES]
    if named:
        fig.text(0.5, 0.0, ", ".join(named), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.05 if named else 0, 1, 1))
    return fig


def plot_class_map(gdf: gpd.GeoDataFrame, column: str, ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes:
    """Map buildings colored by class. Buildings without a class are light gray.

    Args:
        gdf: Buildings.
        column: Class column, e.g. the survey label or ``pred_XGBoost``.
        ax: Axes to draw on. A new figure is created if None.
        title: Axes title. Defaults to ``column``.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, color=[_color(c) for c in gdf[column]], edgecolor="k", linewidth=0.1)
    present = sorted(c for c in gdf[column].dropna().unique())
    handles = [Patch(facecolor=_color(c), edgecolor="k", linewidth=0.3, label=_name(c)) for c in present]
    if gdf[column].isna().any():
        handles.append(Patch(facecolor=UNLABELED, edgecolor="k", linewidth=0.3, label="Unlabeled"))
    ax.legend(handles=handles, loc="lower left", fontsize=7)
    ax.set_title(title or column)
    ax.set_axis_off()
    return ax


def plot_prediction_maps(buildings: gpd.GeoDataFrame, test: gpd.GeoDataFrame, label_column: str, model: str) -> Figure:
    """Draw three maps: survey labels, predictions for every building and test errors.

    Args:
        buildings: Every building of the area with a ``pred_<model>`` column
            (output of :func:`ml_structural_system.predict.predict`).
        test: Test buildings with the true class and ``pred_<model>``.
        label_column: Column with the true class.
        model: Model whose predictions are drawn.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    plot_class_map(buildings, label_column, axes[0], "Survey label")
    plot_class_map(buildings, f"pred_{model}", axes[1], f"Prediction ({model})")
    ok = test[label_column].to_numpy() == test[f"pred_{model}"].to_numpy()
    buildings.plot(ax=axes[2], color=UNLABELED, linewidth=0)
    test.to_crs(buildings.crs).plot(ax=axes[2], color=np.where(ok, CORRECT, WRONG), edgecolor="k", linewidth=0.1)
    axes[2].legend(handles=[Patch(color=CORRECT, label=f"Correct ({ok.sum()})"),
                            Patch(color=WRONG, label=f"Wrong ({(~ok).sum()})")], loc="lower left", fontsize=7)
    axes[2].set_title("Test buildings")
    axes[2].set_axis_off()
    fig.tight_layout()
    return fig
