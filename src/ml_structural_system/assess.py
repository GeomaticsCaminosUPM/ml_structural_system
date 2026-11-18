"""Assess several trained models together on the same buildings.

Computes per-building agreement, Fleiss' kappa, bootstrap confidence intervals,
generalizability statistics and an optional soft ensemble.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import ExperimentConfig
from .explainability import ExplainabilityResult, FeatureImportanceResult, compute_explainability
from .mlflow_utils import get_best_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """Percentile bootstrap confidence interval (CI) for a scalar metric.

    Attributes:
        lower: Lower bound.
        upper: Upper bound.
        confidence: Confidence level, for example 0.95.
        n_samples: Number of bootstrap resamples.
    """
    lower: float
    upper: float
    confidence: float   # e.g. 0.95
    n_samples: int      # number of bootstrap resamples


@dataclass
class ModelInferenceResult:
    """Predictions of one model.

    Attributes:
        model_name: Name of the model.
        status: ``"completed"``, ``"failed"`` or ``"skipped"``.
        predictions: Predicted class codes, shape ``(n_samples,)``.
        probabilities: Class scores divided by the calibrated per-class
            thresholds when available, shape ``(n_samples, n_classes)``.
        has_proba: Whether the model exposes ``predict_proba``.
        error: Error message when inference failed.
    """
    model_name: str
    status: str                         # "completed" | "failed" | "skipped"
    predictions: np.ndarray | None      # (n_samples,) int
    probabilities: np.ndarray | None    # (n_samples, n_classes), threshold-adjusted
    has_proba: bool
    error: str | None = None


@dataclass
class SampleAgreementStats:
    """Per-sample agreement between models.

    Attributes:
        majority_vote: Most voted class per sample. Ties go to the lowest code.
        agreement_ratio: Fraction of models that voted for the majority class.
        normalized_entropy: Shannon entropy of the vote distribution divided
            by ``log(n_classes)``. 0 means unanimous.
        is_contested: Whether ``agreement_ratio`` is below the threshold.
        contested_threshold: Agreement ratio below which a sample is contested.
    """
    majority_vote: np.ndarray           # (n_samples,) int
    agreement_ratio: np.ndarray         # (n_samples,) float  [1/n_models, 1.0]
    normalized_entropy: np.ndarray      # (n_samples,) float  [0.0, 1.0]
    is_contested: np.ndarray            # (n_samples,) bool
    contested_threshold: float


@dataclass
class GlobalAgreementStats:
    """Agreement between models over the whole dataset.

    Attributes:
        fleiss_kappa: Fleiss' kappa across models.
        mean_agreement_ratio: Mean of the per-sample agreement ratio.
        mean_normalized_entropy: Mean of the per-sample normalized entropy.
        pct_contested: Percentage (0 to 100) of contested samples.
        n_models: Number of models that produced predictions.
        n_samples: Number of samples.
        fleiss_kappa_ci: Bootstrap CI of Fleiss' kappa, if enabled.
    """
    fleiss_kappa: float
    mean_agreement_ratio: float
    mean_normalized_entropy: float
    pct_contested: float                # percentage (0-100)
    n_models: int
    n_samples: int
    fleiss_kappa_ci: BootstrapCI | None = None


@dataclass
class ModelGeneralizabilityStats:
    """Comparison of a model's cross-validation (CV) and test macro F1.

    Attributes:
        model_name: Name of the model.
        cv_mean_f1: Mean macro F1 of the best CV run, if available.
        cv_std_f1: Standard deviation of macro F1 of the best CV run.
        test_f1: Macro F1 on the known labels, or None without labels.
        generalizability_gap: ``test_f1 - cv_mean_f1``, or None if either is missing.
        is_low_generalizability: Whether the gap is below ``-gap_threshold``.
        test_f1_ci: Bootstrap CI of the test F1, if enabled for this model.
    """
    model_name: str
    cv_mean_f1: float | None
    cv_std_f1: float | None
    test_f1: float | None               # None when y is not provided
    generalizability_gap: float | None  # test_f1 - cv_mean_f1; None if either missing
    is_low_generalizability: bool | None
    test_f1_ci: BootstrapCI | None = None


@dataclass
class SoftEnsembleResult:
    """Soft-voting ensemble of the models that expose ``predict_proba``.

    Each model's probabilities are weighted by its CV macro F1. Without any
    such model, the ensemble falls back to the majority vote.

    Attributes:
        predictions: Ensemble class codes, shape ``(n_samples,)``.
        mean_proba: Weighted mean of the model scores, shape
            ``(n_samples, n_classes)``. These are not calibrated
            probabilities. All zeros in the majority-vote fallback.
        f1: Macro F1 on the known labels, or None without labels.
        models_included: Models in the ensemble.
        models_excluded: Models left out because they lack ``predict_proba``.
        model_weights: Model name mapped to its normalized weight.
        f1_ci: Bootstrap CI of the ensemble F1, if bootstrap is enabled.
    """
    predictions: np.ndarray             # (n_samples,) int
    mean_proba: np.ndarray              # (n_samples, n_classes), NOT calibrated probabilities
    f1: float | None                    # None when y is not provided
    models_included: list[str] = field(default_factory=list)
    models_excluded: list[str] = field(default_factory=list)
    model_weights: dict[str, float] = field(default_factory=dict)   # model_name → normalised weight
    f1_ci: BootstrapCI | None = None


@dataclass
class AssessmentResult:
    """Full output of :func:`run`.

    Attributes:
        model_results: Inference result of each model.
        sample_agreement: Per-sample agreement statistics.
        global_agreement: Dataset-level agreement statistics.
        generalizability: Generalizability statistics of each completed model.
        cross_model_f1_std: Standard deviation of test F1 across models, or
            None with fewer than two scored models.
        soft_ensemble: Soft ensemble result.
        sample_dataframe: Features plus per-sample predictions and statistics,
            ready for CSV export.
        explainability: Feature importances, if requested and successful.
    """
    model_results: list[ModelInferenceResult]
    sample_agreement: SampleAgreementStats
    global_agreement: GlobalAgreementStats
    generalizability: list[ModelGeneralizabilityStats]
    cross_model_f1_std: float | None    # std of test F1 across models; None when y missing
    soft_ensemble: SoftEnsembleResult
    sample_dataframe: pd.DataFrame      # per-sample output ready for CSV export
    explainability: ExplainabilityResult | None = None

# Re-export for callers that import from assess directly.
__all__ = [
    "AssessmentResult",
    "BootstrapCI",
    "ExplainabilityResult",
    "FeatureImportanceResult",
    "GlobalAgreementStats",
    "ModelGeneralizabilityStats",
    "ModelInferenceResult",
    "SampleAgreementStats",
    "SoftEnsembleResult",
    "run",
    "compute_explainability",
]


# ---------------------------------------------------------------------------
# Private helpers, inference
# ---------------------------------------------------------------------------

def _infer_model(
    model_name: str,
    cfg: ExperimentConfig,
    X: pd.DataFrame,
) -> ModelInferenceResult:
    """Load a model's final artifact and run inference with it.

    Probabilities are divided by the saved per-class thresholds when present.
    A missing model gives status ``"skipped"`` and an error gives ``"failed"``.
    """
    work_dir = cfg.models_dir / model_name
    model_path = work_dir / f"model_final_{model_name}.pkl"

    if not model_path.exists():
        logger.warning("No final model at %s, skipping %s.", model_path, model_name)
        return ModelInferenceResult(model_name, "skipped", None, None, has_proba=False)

    try:
        artifact = joblib.load(model_path)

        model = artifact
        thresholds_path = work_dir / f"thresholds_{model_name}.npy"

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if thresholds_path.exists():
                thresholds = np.load(thresholds_path)
                proba = proba / thresholds
            preds = proba.argmax(axis=1)
            return ModelInferenceResult(
                model_name, "completed", preds, probabilities=proba, has_proba=True
            )
        else:
            preds = model.predict(X)
            return ModelInferenceResult(
                model_name, "completed", preds, probabilities=None, has_proba=False
            )

    except Exception as exc:  # noqa: BLE001, one model failing must not crash the run
        logger.exception("Inference failed for %s.", model_name)
        return ModelInferenceResult(
            model_name, "failed", None, None, has_proba=False, error=str(exc)
        )


# ---------------------------------------------------------------------------
# Private helpers, statistics
# ---------------------------------------------------------------------------

def _compute_sample_agreement(
    pred_matrix: np.ndarray,
    n_classes: int,
    contested_threshold: float,
) -> SampleAgreementStats:
    """Compute per-sample agreement stats from an ``(n_samples, n_models)`` prediction matrix."""
    n_samples, n_models = pred_matrix.shape

    # Vote counts: table[i, k] = number of models predicting class k for sample i.
    table = np.zeros((n_samples, n_classes), dtype=int)
    for k in range(n_classes):
        table[:, k] = (pred_matrix == k).sum(axis=1)

    # Majority vote: class with the most votes (ties → lowest index via argmax).
    majority_vote = table.argmax(axis=1)

    # Agreement ratio: fraction of models that voted for the majority class.
    agreement_ratio = table[np.arange(n_samples), majority_vote].astype(float) / n_models

    # Normalized Shannon entropy of the vote distribution.
    # H(p) / log(n_classes): 0 = unanimous, 1 = maximally spread across all classes.
    log_n = np.log(max(n_classes, 2))
    p = table.astype(float) / n_models
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    normalized_entropy = -(p * log_p).sum(axis=1) / log_n

    return SampleAgreementStats(
        majority_vote=majority_vote,
        agreement_ratio=agreement_ratio,
        normalized_entropy=normalized_entropy,
        is_contested=agreement_ratio < contested_threshold,
        contested_threshold=contested_threshold,
    )


def _compute_fleiss_kappa(pred_matrix: np.ndarray, n_classes: int) -> float:
    """Compute Fleiss' kappa for inter-model agreement.

    Returns 1.0 with fewer than two models or when expected agreement is 1.
    """
    n_samples, n_models = pred_matrix.shape
    if n_models < 2:
        return 1.0

    table = np.zeros((n_samples, n_classes), dtype=float)
    for k in range(n_classes):
        table[:, k] = (pred_matrix == k).sum(axis=1)

    p_j = table.sum(axis=0) / (n_samples * n_models)    # marginal rate per class
    P_e = float((p_j ** 2).sum())

    if P_e >= 1.0 - 1e-12:
        return 1.0

    # Per-subject observed agreement
    P_i = ((table ** 2).sum(axis=1) - n_models) / (n_models * (n_models - 1))
    P_bar = float(P_i.mean())

    return (P_bar - P_e) / (1.0 - P_e)


def _bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Compute a percentile bootstrap confidence interval for macro-averaged F1."""
    n = len(y_true)
    scores = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(scores, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapCI(lower=float(lo), upper=float(hi), confidence=confidence, n_samples=n_samples)


def _bootstrap_kappa(
    pred_matrix: np.ndarray,
    n_classes: int,
    n_samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Compute a percentile bootstrap confidence interval for Fleiss' kappa by resampling rows."""
    n = pred_matrix.shape[0]
    scores = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        scores[i] = _compute_fleiss_kappa(pred_matrix[idx], n_classes)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(scores, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapCI(lower=float(lo), upper=float(hi), confidence=confidence, n_samples=n_samples)


def _get_cv_stats(
    cfg: ExperimentConfig,
    model_name: str,
) -> tuple[float | None, float | None]:
    """Return the best cross-validation run's mean and std macro F1 from MLflow.

    Returns ``(None, None)`` if MLflow is unavailable or has no runs.
    """
    try:
        run = get_best_run(cfg.experiment_cv, model_name)
        return float(run["metrics.mean_f1_macro"]), float(run["metrics.std_f1_macro"])
    except Exception:  # noqa: BLE001, MLflow unavailable or no runs: degrade gracefully
        return None, None


def _proba_column_names(cfg: ExperimentConfig) -> list[str] | None:
    """Return bare class names in code order, used to name one ``proba_<name>`` column per class."""
    n = cfg.num_classes
    if not n:
        return None
    label_map = cfg.label_map
    names = []
    for code in range(n):
        display = label_map.get(code)
        names.append(display.rsplit(" (", 1)[0] if display else str(code))
    return names


def _build_sample_dataframe(
    X: pd.DataFrame,
    model_results: list[ModelInferenceResult],
    sample_agreement: SampleAgreementStats,
    soft_ensemble: SoftEnsembleResult,
    y: pd.Series | None,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the per-sample DataFrame for export or downstream use."""
    df = X.copy()
    for m in model_results:
        if m.status == "completed" and m.predictions is not None:
            df[f"pred_{m.model_name}"] = m.predictions
    df["majority_vote"] = sample_agreement.majority_vote
    df["agreement_ratio"] = sample_agreement.agreement_ratio
    df["normalized_entropy"] = sample_agreement.normalized_entropy
    df["is_contested"] = sample_agreement.is_contested
    df["ensemble_pred"] = soft_ensemble.predictions
    if class_names is not None and soft_ensemble.mean_proba.shape[1] == len(class_names):
        # mean_proba is all-zeros (see run()'s no-proba-model fallback)
        # when no model exposes predict_proba; still written in that case
        #, a column of exact 0.0s is honest about "no soft-voted
        # probability available", not a reason to omit the columns and
        # change predictions.csv's shape between runs.
        for i, name in enumerate(class_names):
            df[f"proba_{name}"] = soft_ensemble.mean_proba[:, i]
    if y is not None:
        df["y_true"] = y.values
    return df


def _log_to_mlflow(
    experiment: str,
    global_agreement: GlobalAgreementStats,
    generalizability: list[ModelGeneralizabilityStats],
    soft_ensemble: SoftEnsembleResult,
    cross_model_f1_std: float | None,
    explainability: ExplainabilityResult | None = None,
) -> None:
    """Log global agreement stats, ensemble metrics and per-model generalizability to MLflow."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="assessment"):
        mlflow.log_metric("fleiss_kappa", global_agreement.fleiss_kappa)
        if global_agreement.fleiss_kappa_ci is not None:
            mlflow.log_metric("fleiss_kappa_ci_lower", global_agreement.fleiss_kappa_ci.lower)
            mlflow.log_metric("fleiss_kappa_ci_upper", global_agreement.fleiss_kappa_ci.upper)
        mlflow.log_metric("mean_agreement_ratio", global_agreement.mean_agreement_ratio)
        mlflow.log_metric("mean_normalized_entropy", global_agreement.mean_normalized_entropy)
        mlflow.log_metric("pct_contested", global_agreement.pct_contested)
        mlflow.log_param("n_models", global_agreement.n_models)
        mlflow.log_param("n_samples", global_agreement.n_samples)
        mlflow.log_param("models_in_ensemble", ",".join(soft_ensemble.models_included))
        mlflow.log_param(
            "models_excluded_from_ensemble", ",".join(soft_ensemble.models_excluded)
        )
        if soft_ensemble.f1 is not None:
            mlflow.log_metric("ensemble_f1", soft_ensemble.f1)
        if soft_ensemble.f1_ci is not None:
            mlflow.log_metric("ensemble_f1_ci_lower", soft_ensemble.f1_ci.lower)
            mlflow.log_metric("ensemble_f1_ci_upper", soft_ensemble.f1_ci.upper)
        if cross_model_f1_std is not None:
            mlflow.log_metric("cross_model_f1_std", cross_model_f1_std)
        if explainability is not None and len(explainability.consensus_ranking) > 0:
            for feat, rank in explainability.consensus_ranking.head(20).items():
                safe_name = str(feat).replace(" ", "_")
                mlflow.log_metric(f"consensus_rank_{safe_name}", float(rank))
        for g in generalizability:
            p = g.model_name
            if g.cv_mean_f1 is not None:
                mlflow.log_metric(f"{p}_cv_mean_f1", g.cv_mean_f1)
            if g.cv_std_f1 is not None:
                mlflow.log_metric(f"{p}_cv_std_f1", g.cv_std_f1)
            if g.test_f1 is not None:
                mlflow.log_metric(f"{p}_test_f1", g.test_f1)
            if g.generalizability_gap is not None:
                mlflow.log_metric(f"{p}_gap", g.generalizability_gap)
            if g.test_f1_ci is not None:
                mlflow.log_metric(f"{p}_test_f1_ci_lower", g.test_f1_ci.lower)
                mlflow.log_metric(f"{p}_test_f1_ci_upper", g.test_f1_ci.upper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    cfg: ExperimentConfig,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    *,
    models: list[str] | None = None,
    contested_threshold: float = 0.5,
    gap_threshold: float = 0.05,
    skip_mlflow_cv: bool = False,
    log_to_mlflow: bool = True,
    mlflow_experiment: str = "model_agreement",
    explain: bool = False,
    compute_shap: bool = True,
    compute_bootstrap: bool = True,
) -> AssessmentResult:
    """Run the quality assessment across all trained models.

    Runs inference with each model, then computes agreement statistics, a
    soft ensemble weighted by cross-validation (CV) macro F1, per-model
    generalizability and, optionally, feature importances. Labels that are
    NaN are ignored in the metrics.

    Args:
        cfg: Experiment configuration.
        X: Features of the samples to assess.
        y: Integer labels, or None for an unlabeled assessment. May contain NaN
            for unknown labels.
        models: Model names to include. Defaults to ``cfg.enabled_models``.
        contested_threshold: Agreement ratio below which a sample is contested.
        gap_threshold: F1 drop from CV to test above which a model is flagged
            as low generalizability.
        skip_mlflow_cv: Do not query MLflow for CV stats. Ensemble weights are
            then equal and generalizability gaps are not computed.
        log_to_mlflow: Log the results to MLflow. Failures are only logged.
        mlflow_experiment: MLflow experiment to log the results to.
        explain: Compute feature importances and the consensus ranking.
        compute_shap: Compute SHAP values when ``explain`` is True.
        compute_bootstrap: Compute bootstrap confidence intervals of per-model
            test F1 for the models enabled in ``cfg.bootstrap``.

    Returns:
        The full assessment result.

    Raises:
        RuntimeError: If no model produced predictions.
    """
    model_names = models if models is not None else cfg.enabled_models
    n_classes = cfg.num_classes
    bs = cfg.bootstrap
    rng = np.random.default_rng(cfg.random_state)

    # y may contain NaN where the label is unknown. Metrics use the known
    # labels only; y_true keeps its NaN entries in sample_dataframe.
    y_arr: np.ndarray | None = np.asarray(y, dtype=float) if y is not None else None
    y_known_mask: np.ndarray | None = ~np.isnan(y_arr) if y_arr is not None else None
    has_known_labels = bool(y_known_mask.any()) if y_known_mask is not None else False

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    # ------------------------------------------------------------------
    # 1. Inference
    # ------------------------------------------------------------------
    model_results: list[ModelInferenceResult] = []
    for name in model_names:
        logger.info("Running inference: %s", name)
        result = _infer_model(name, cfg, X)
        model_results.append(result)
        if result.status == "completed":
            logger.info("  %s OK (has_proba=%s)", name, result.has_proba)
        else:
            logger.warning(
                "  %s %s%s",
                name,
                result.status,
                f", {result.error}" if result.error else "",
            )

    completed = [
        m for m in model_results if m.status == "completed" and m.predictions is not None
    ]
    if not completed:
        raise RuntimeError(
            "No models produced predictions. Cannot compute agreement stats. "
            "Ensure at least one model has been trained (run_final completed)."
        )

    n_samples = len(X)

    # ------------------------------------------------------------------
    # 2. Agreement statistics
    # ------------------------------------------------------------------
    pred_matrix = np.column_stack([m.predictions for m in completed])  # (n_samples, n_completed)

    sample_agreement = _compute_sample_agreement(pred_matrix, n_classes, contested_threshold)

    kappa = _compute_fleiss_kappa(pred_matrix, n_classes)
    kappa_ci = (
        _bootstrap_kappa(pred_matrix, n_classes, bs.n_samples, bs.confidence, rng)
        if bs.any_enabled
        else None
    )
    global_agreement = GlobalAgreementStats(
        fleiss_kappa=kappa,
        mean_agreement_ratio=float(sample_agreement.agreement_ratio.mean()),
        mean_normalized_entropy=float(sample_agreement.normalized_entropy.mean()),
        pct_contested=float(sample_agreement.is_contested.mean() * 100),
        n_models=len(completed),
        n_samples=n_samples,
        fleiss_kappa_ci=kappa_ci,
    )

    # ------------------------------------------------------------------
    # 3. Soft ensemble weighted by CV macro-F1
    #
    # Each model's probability matrix is weighted by its mean CV macro-F1
    # so that stronger models (higher CV F1) dominate the ensemble.
    # When CV stats are unavailable for a model (for example MLflow is down) it falls back to a weight
    # equal to the mean of the available weights, so it still participates
    # but neither dominates nor is excluded.
    # ------------------------------------------------------------------
    proba_models = [m for m in completed if m.has_proba and m.probabilities is not None]
    excluded = [m.model_name for m in completed if not m.has_proba]

    if proba_models:
        # Fetch CV F1 for each proba-capable model
        raw_weights: list[float | None] = []
        for m in proba_models:
            cv_mean, _ = (None, None) if skip_mlflow_cv else _get_cv_stats(cfg, m.model_name)
            raw_weights.append(cv_mean)

        known = [w for w in raw_weights if w is not None]
        fallback = float(np.mean(known)) if known else 1.0
        weights = np.array([w if w is not None else fallback for w in raw_weights], dtype=float)
        # Clip to [0, inf), a model with negative CV F1 (degenerate) shouldn't pull the ensemble
        weights = np.clip(weights, 0.0, None)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(len(proba_models)) / len(proba_models)

        stacked = np.stack(
            [m.probabilities for m in proba_models], axis=0
        )  # (n_proba, n_samples, n_classes)
        mean_proba = (stacked * weights[:, np.newaxis, np.newaxis]).sum(axis=0)  # (n_samples, n_classes)
        ensemble_preds = mean_proba.argmax(axis=1)
        ensemble_f1 = (
            float(f1_score(
                y_arr[y_known_mask].astype(int), ensemble_preds[y_known_mask],
                average="macro", zero_division=0,
            ))
            if has_known_labels
            else None
        )
        ensemble_f1_ci = (
            _bootstrap_f1(
                y_arr[y_known_mask].astype(int), ensemble_preds[y_known_mask],
                bs.n_samples, bs.confidence, rng,
            )
            if ensemble_f1 is not None and bs.any_enabled
            else None
        )
        model_weights = {m.model_name: float(w) for m, w in zip(proba_models, weights)}
        soft_ensemble = SoftEnsembleResult(
            predictions=ensemble_preds,
            mean_proba=mean_proba,
            f1=ensemble_f1,
            models_included=[m.model_name for m in proba_models],
            models_excluded=excluded,
            model_weights=model_weights,
            f1_ci=ensemble_f1_ci,
        )
    else:
        # Fallback: no model has predict_proba, use majority vote.
        soft_ensemble = SoftEnsembleResult(
            predictions=sample_agreement.majority_vote.copy(),
            mean_proba=np.zeros((n_samples, n_classes)),
            f1=None,
            models_included=[],
            models_excluded=[m.model_name for m in completed],
        )

    # ------------------------------------------------------------------
    # 4. Per-model test F1 and generalizability
    # ------------------------------------------------------------------
    generalizability: list[ModelGeneralizabilityStats] = []
    test_f1_values: dict[str, float] = {}

    for m in completed:
        test_f1: float | None = None
        if has_known_labels:
            test_f1 = float(
                f1_score(
                    y_arr[y_known_mask].astype(int), m.predictions[y_known_mask],
                    average="macro", zero_division=0,
                )
            )
            test_f1_values[m.model_name] = test_f1

        cv_mean, cv_std = (
            (None, None) if skip_mlflow_cv else _get_cv_stats(cfg, m.model_name)
        )

        gap: float | None = None
        is_low: bool | None = None
        if test_f1 is not None and cv_mean is not None:
            gap = test_f1 - cv_mean
            is_low = gap < -gap_threshold

        test_f1_ci: BootstrapCI | None = None
        if compute_bootstrap and test_f1 is not None and has_known_labels and bs.enabled_for(m.model_name):
            logger.info("  Bootstrap CI for %s (%d resamples)…", m.model_name, bs.n_samples)
            test_f1_ci = _bootstrap_f1(
                y_arr[y_known_mask].astype(int), m.predictions[y_known_mask],
                bs.n_samples, bs.confidence, rng,
            )

        generalizability.append(
            ModelGeneralizabilityStats(
                model_name=m.model_name,
                cv_mean_f1=cv_mean,
                cv_std_f1=cv_std,
                test_f1=test_f1,
                generalizability_gap=gap,
                is_low_generalizability=is_low,
                test_f1_ci=test_f1_ci,
            )
        )

    cross_model_f1_std: float | None = (
        float(np.std(list(test_f1_values.values())))
        if len(test_f1_values) >= 2
        else None
    )

    # ------------------------------------------------------------------
    # 5. Sample DataFrame
    # ------------------------------------------------------------------
    sample_df = _build_sample_dataframe(
        X, model_results, sample_agreement, soft_ensemble, y, class_names=_proba_column_names(cfg)
    )

    # ------------------------------------------------------------------
    # 6. Explainability (feature importances + SHAP)
    # ------------------------------------------------------------------
    explainability: ExplainabilityResult | None = None
    if explain:
        logger.info("Computing feature importances and SHAP values …")
        try:
            explainability = compute_explainability(
                cfg, X, y, models=model_names, compute_shap=compute_shap
            )
        except Exception:  # noqa: BLE001, explainability failure must not abort the run
            logger.warning("Explainability computation failed.", exc_info=True)

    # ------------------------------------------------------------------
    # 7. Optional MLflow logging
    # ------------------------------------------------------------------
    if log_to_mlflow:
        try:
            _log_to_mlflow(
                mlflow_experiment,
                global_agreement,
                generalizability,
                soft_ensemble,
                cross_model_f1_std,
                explainability,
            )
        except Exception:  # noqa: BLE001, logging failure must not abort the assessment
            logger.warning("MLflow logging failed (results are still returned).", exc_info=True)

    return AssessmentResult(
        model_results=model_results,
        sample_agreement=sample_agreement,
        global_agreement=global_agreement,
        generalizability=generalizability,
        cross_model_f1_std=cross_model_f1_std,
        soft_ensemble=soft_ensemble,
        sample_dataframe=sample_df,
        explainability=explainability,
    )
