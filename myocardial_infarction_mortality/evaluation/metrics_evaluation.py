from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from imblearn.metrics import (
    geometric_mean_score,
    specificity_score,
)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def average_cost_score(y_true, y_pred, cost_matrix: np.ndarray) -> float:
    """
    Compute the average misclassification cost using a 2x2 cost matrix.

    This function evaluates predictions by computing the confusion matrix with an explicit
    label order ``[0, 1]`` and then applying a user-provided 2x2 cost matrix via element-wise
    multiplication. The resulting total cost is divided by the number of samples to return a
    **positive** average cost (lower is better).

    The confusion matrix produced by scikit-learn with ``labels=[0, 1]`` follows:

    ``[[TN, FP],``
    `` [FN, TP]]``

    With the project convention:
    - ``0`` = ALIVE (negative / majority class)
    - ``1`` = DEAD  (positive / minority class)

    Therefore, the cost matrix is interpreted as:
    - ``cost_matrix[0, 0]``: cost of TN (true ALIVE predicted ALIVE)
    - ``cost_matrix[0, 1]``: cost of FP (true ALIVE predicted DEAD)
    - ``cost_matrix[1, 0]``: cost of FN (true DEAD predicted ALIVE)
    - ``cost_matrix[1, 1]``: cost of TP (true DEAD predicted DEAD)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels. Expected values are ``0`` and ``1``.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels. Expected values are ``0`` and ``1``.
    cost_matrix : numpy.ndarray of shape (2, 2)
        Misclassification cost matrix aligned with the confusion-matrix layout for
        ``labels=[0, 1]``. Costs must be finite. Typically non-negative.

    Returns
    -------
    avg_cost : float
        Positive average cost per sample. Returns ``0.0`` if ``y_true`` is empty.

    Raises
    ------
    ValueError
        If ``cost_matrix`` does not have shape ``(2, 2)``.
    ValueError
        If ``cost_matrix`` contains non-finite values (NaN/Inf).

    Notes
    -----
    - This function returns a **positive cost** (lower is better). When integrating into
      scikit-learn hyperparameter search, use a scorer configured with
      ``greater_is_better=False``.
    - The computation is vectorized as:
      ``total_cost = sum(confusion_matrix * cost_matrix)``, which is equivalent to:
      ``TN*CM[0,0] + FP*CM[0,1] + FN*CM[1,0] + TP*CM[1,1]``.
    - The explicit ``labels=[0, 1]`` ensures a stable layout even if a fold contains only
      one class.

    Examples
    --------
    Basic usage with asymmetric FN/FP costs::

        >>> import numpy as np
        >>> y_true = [0, 0, 1, 1]
        >>> y_pred = [0, 1, 0, 1]
        >>> cost = np.array([[0.0, 1.0],
        ...                  [10.0, 0.0]])
        >>> average_cost_score(y_true, y_pred, cost)
        2.75
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.shape != (2, 2):
        raise ValueError("cost_matrix must have shape (2, 2).")
    if not np.isfinite(cost_matrix).all():
        raise ValueError("cost_matrix must contain only finite values.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    total_cost = float(np.sum(cm * cost_matrix))
    n_samples = int(len(y_true))
    return total_cost / n_samples if n_samples > 0 else 0.0


def get_avg_cost_scorer(cost_matrix: np.ndarray):
    """
    Create a scikit-learn scorer that minimizes average misclassification cost.

    This helper wraps :func:`average_cost_score` into a scorer object compatible with
    scikit-learn model selection utilities (e.g., :class:`~sklearn.model_selection.GridSearchCV`,
    :class:`~sklearn.model_selection.RandomizedSearchCV`).

    The returned scorer:
    - calls the estimator's ``predict`` method to obtain hard labels (0/1),
    - computes the **positive** average cost via the provided ``cost_matrix``,
    - is configured with ``greater_is_better=False`` so that **lower cost is better**.

    Parameters
    ----------
    cost_matrix : numpy.ndarray of shape (2, 2)
        Misclassification cost matrix aligned with ``confusion_matrix(..., labels=[0, 1])``.
        See :func:`average_cost_score` for interpretation.

    Returns
    -------
    scorer : callable
        Scorer object suitable for scikit-learn hyperparameter search. It can be passed
        directly as the ``scoring`` argument.

    Raises
    ------
    ValueError
        If ``cost_matrix`` does not have shape ``(2, 2)`` or contains non-finite values.

    Notes
    -----
    - Since this scorer uses ``predict``, it evaluates cost under the estimator's default
      decision policy (typically a probability threshold of 0.5 for binary classifiers with
      probabilistic outputs).
    - If you want a cost-sensitive *threshold-moving* evaluation (e.g., MEC), you need a
      probability-based scorer that uses ``predict_proba`` and applies a custom threshold.

    Examples
    --------
    Use the scorer in a grid search::

        >>> import numpy as np
        >>> from sklearn.model_selection import GridSearchCV
        >>> cost = np.array([[0.0, 1.0],
        ...                  [10.0, 0.0]])
        >>> scorer = get_avg_cost_scorer(cost)
        >>> # GridSearchCV(..., scoring=scorer)
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.shape != (2, 2):
        raise ValueError("cost_matrix must have shape (2, 2).")
    if not np.isfinite(cost_matrix).all():
        raise ValueError("cost_matrix must contain only finite values.")

    custom_metric = partial(average_cost_score, cost_matrix=cost_matrix)
    return make_scorer(custom_metric, greater_is_better=False)


def min_expected_cost_predict(y_proba: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Predict class labels by minimizing expected misclassification cost.

    This function implements the **Minimum Expected Cost (MEC)** decision rule for
    multi-class classification. Given class-probability estimates for each sample and
    a user-defined cost matrix, it predicts the class that yields the smallest expected
    cost.

    For each sample ``i`` and candidate predicted class ``c``, the expected cost is:

    ``E[cost | predict=c] = sum_k P(true=k | x_i) * cost_matrix[k, c]``

    The predicted label is:

    ``y_pred[i] = argmin_c E[cost | predict=c]``

    With the project binary convention (optional specialization):
    - ``0`` = ALIVE (negative / majority class)
    - ``1`` = DEAD  (positive / minority class)

    and the cost matrix aligned as ``[[TN, FP], [FN, TP]]`` (rows = true class,
    columns = predicted class), a large ``cost_matrix[1, 0]`` penalizes missed deaths
    (FN), thus moving the decision boundary away from the default 0.5 threshold.

    Parameters
    ----------
    y_proba : numpy.ndarray of shape (n_samples, n_classes)
        Predicted class probabilities for each sample. Each row must sum to 1 (within
        numerical tolerance) and contain finite values. Column order must match the
        class indexing used by ``cost_matrix``.
    cost_matrix : numpy.ndarray of shape (n_classes, n_classes)
        Cost matrix where ``cost_matrix[true_class, predicted_class]`` is the cost
        incurred by predicting ``predicted_class`` when the true class is
        ``true_class``. Values must be finite. Costs are typically non-negative, but
        the MEC rule also works with arbitrary real-valued costs.

    Returns
    -------
    y_pred : numpy.ndarray of shape (n_samples,)
        Predicted class labels (integer indices in ``[0, n_classes - 1]``) that minimize
        expected cost under ``cost_matrix``.

    Raises
    ------
    ValueError
        If ``y_proba`` is not 2D, if ``cost_matrix`` is not 2D square, if dimensions are
        incompatible (``y_proba.shape[1] != cost_matrix.shape[0]``), or if either input
        contains non-finite values.

    Notes
    -----
    - This function assumes ``y_proba`` provides calibrated probabilities. If probabilities
      are poorly calibrated, MEC decisions may be suboptimal; consider calibration (e.g.,
      Platt scaling / isotonic regression) before applying MEC.
    - In binary classification, MEC is equivalent to using a cost-derived threshold **only**
      under specific assumptions and when costs are defined purely on FP/FN (with TN/TP cost 0).
      The general matrix formulation used here is more robust and extends naturally to
      multi-class settings.

    Examples
    --------
    Binary MEC with asymmetric FN/FP costs::

        >>> import numpy as np
        >>> y_proba = np.array([[0.90, 0.10],
        ...                     [0.40, 0.60],
        ...                     [0.70, 0.30]])
        >>> cost = np.array([[0.0, 1.0],
        ...                  [10.0, 0.0]])  # FN cost >> FP cost
        >>> min_expected_cost_predict(y_proba, cost)
        array([0, 1, 0])

    Multi-class MEC::

        >>> y_proba = np.array([[0.2, 0.5, 0.3]])
        >>> cost = np.array([[0, 1, 2],
        ...                  [3, 0, 1],
        ...                  [1, 2, 0]])
        >>> min_expected_cost_predict(y_proba, cost)
        array([1])
    """
    y_proba = np.asarray(y_proba, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)

    if y_proba.ndim != 2:
        raise ValueError(
            f"y_proba must be a 2D array of shape (n_samples, n_classes). Got {y_proba.ndim}D."
        )
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError(
            "cost_matrix must be a 2D square array of shape (n_classes, n_classes). "
            f"Got shape {cost_matrix.shape}."
        )
    if y_proba.shape[1] != cost_matrix.shape[0]:
        raise ValueError(
            "Incompatible shapes: y_proba has n_classes="
            f"{y_proba.shape[1]} but cost_matrix is {cost_matrix.shape}."
        )
    if not np.isfinite(y_proba).all():
        raise ValueError("y_proba must contain only finite values (no NaN/Inf).")
    if not np.isfinite(cost_matrix).all():
        raise ValueError("cost_matrix must contain only finite values (no NaN/Inf).")

    expected_costs = np.dot(y_proba, cost_matrix)
    return np.argmin(expected_costs, axis=1)


def apply_decision_policy(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    policy_mode: str,
    cost_matrix: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a decision policy to produce hard predictions and positive-class probabilities.

    This helper standardizes inference under two decision-policy modes:

    - ``"standard"``: uses the estimator's default decision rule (typically a 0.5 threshold
      for probabilistic binary classifiers) via ``estimator.predict(X)``.
    - ``"mec"``: applies the **Minimum Expected Cost (MEC)** decision rule using a user-provided
      cost matrix and the estimator's class-probability estimates via
      :func:`min_expected_cost_predict`.

    The function returns both:
    1) hard labels ``y_pred`` (always),
    2) positive-class probabilities ``y_pred_prob`` when available.

    Project binary convention
    -------------------------
    - ``0`` = ALIVE (negative / majority class)
    - ``1`` = DEAD  (positive / minority class)

    MEC requires probabilities
    --------------------------
    MEC mathematically requires class probabilities to compute expected costs. Therefore,
    when ``policy_mode="mec"``, the estimator (or final pipeline step) **must** implement
    ``predict_proba`` and ``cost_matrix`` must be provided.

    Parameters
    ----------
    estimator : Any
        Fitted estimator or pipeline implementing ``predict(X)``. If ``policy_mode="mec"``,
        it must also implement ``predict_proba(X)`` returning an array of shape
        ``(n_samples, n_classes)``.
    X : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_features)
        Feature matrix to score.
    policy_mode : {'standard', 'mec'}
        Decision policy mode:

        - ``'standard'``: return ``estimator.predict(X)`` and, if available,
          ``estimator.predict_proba(X)[:, 1]``.
        - ``'mec'``: compute probabilities via ``predict_proba`` and derive hard predictions
          by minimizing expected cost using ``cost_matrix`` (see :func:`min_expected_cost_predict`).
    cost_matrix : numpy.ndarray of shape (2, 2), optional
        Misclassification cost matrix aligned with the class index order used by
        ``predict_proba``. For the project binary convention (classes ``[0, 1]``), the
        matrix is interpreted as ``cost_matrix[true_class, predicted_class]`` and is typically
        arranged as::

            [[TN_cost, FP_cost],
             [FN_cost, TP_cost]]

        Required when ``policy_mode='mec'``. Ignored when ``policy_mode='standard'``.

    Returns
    -------
    y_pred : numpy.ndarray of shape (n_samples,)
        Hard class predictions.

        - For ``policy_mode='standard'``: output of ``estimator.predict(X)``.
        - For ``policy_mode='mec'``: output of :func:`min_expected_cost_predict`.
    y_pred_prob : numpy.ndarray of shape (n_samples,) or None
        Positive-class (class ``1``) probability estimates, extracted as
        ``predict_proba(X)[:, 1]`` when available. If the estimator does not expose
        ``predict_proba`` and ``policy_mode='standard'`` is used, this is returned as ``None``.

    Raises
    ------
    ValueError
        If ``policy_mode`` is not one of ``{'standard', 'mec'}``.
    ValueError
        If ``policy_mode='mec'`` and ``cost_matrix`` is ``None``.
    AttributeError
        If ``policy_mode='mec'`` but the estimator does not implement ``predict_proba``.
    Exception
        Any exception raised by the estimator during prediction/probability estimation may
        propagate.

    Notes
    -----
    - When ``policy_mode='mec'``, this function assumes that the class order in
      ``predict_proba`` matches the indexing assumed by ``cost_matrix``. In scikit-learn,
      the probability columns are ordered by ``estimator.classes_``. If your estimator uses
      a different ordering (e.g., classes ``[1, 0]``), you must reorder either probabilities
      or the cost matrix accordingly before applying MEC.
    - The returned probabilities are always the model's native probabilities. Under MEC, the
      hard labels may correspond to a non-0.5 implicit threshold, but ``y_pred_prob`` remains
      the raw positive-class probability.

    Examples
    --------
    Standard policy (with probabilities)::

        >>> y_pred, y_prob = apply_decision_policy(clf, X_test, policy_mode="standard")
        >>> y_pred.shape == y_prob.shape
        True

    MEC policy with FN >> FP::

        >>> import numpy as np
        >>> cost = np.array([[0.0, 1.0],
        ...                  [10.0, 0.0]])
        >>> y_pred, y_prob = apply_decision_policy(clf, X_test, policy_mode="mec", cost_matrix=cost)
        >>> y_pred.shape == y_prob.shape
        True
    """

    # Check if the estimator can output probabilities
    has_proba = hasattr(estimator, "predict_proba")

    if policy_mode == "mec":
        # MEC mathematically REQUIRES probabilities to calculate expected costs.
        # If the model can't output them, we must crash loudly and clearly.
        if not has_proba:
            raise AttributeError(
                f"Cannot apply 'mec' policy. The estimator '{type(estimator).__name__}' "
                f"(or the final step in the pipeline) does not have a 'predict_proba' method."
            )

        # 1. Get the full probability array (needed for MEC math)
        y_proba_full = estimator.predict_proba(X)

        # 2. Calculate hard predictions (y_pred) using the cost matrix
        y_pred = min_expected_cost_predict(y_proba_full, cost_matrix)

        # 3. Extract positive class probabilities for standard metrics
        y_pred_prob = y_proba_full[:, 1]

    else:  # standard
        # 1. Use the standard scikit-learn hard prediction (0.5 threshold)
        y_pred = estimator.predict(X)

        # If probabilities are available, grab them for ROC-AUC/AP metrics.
        # If not, return None safely.
        if has_proba:
            # 2. Use the standard scikit-learn probability extraction
            y_pred_prob = estimator.predict_proba(X)[:, 1]
        else:
            y_pred_prob = None

    return y_pred, y_pred_prob


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_pred_proba: Sequence[float] | np.ndarray | None = None,
    cost_matrix: np.ndarray | None = None,
) -> Dict[str, Optional[float] | int]:
    """
    Compute a standard set of binary classification metrics (optionally including cost).

    This helper aggregates:

    - Confusion-matrix counts (TP/TN/FP/FN) using a fixed label order ``labels=[0, 1]``.
    - Threshold-based metrics: accuracy, precision, recall, F1.
    - Imbalance-oriented metrics: specificity, false positive rate (FPR), balanced accuracy,
      geometric mean (G-mean), MCC, Cohen's kappa.
    - Optional ranking/probability metrics: ROC-AUC and average precision, computed only
      when ``y_pred_proba`` is provided and both classes are present in ``y_true``.
    - Optional cost metric: average misclassification cost computed from a 2x2 cost matrix
      aligned with the confusion matrix layout when ``cost_matrix`` is provided.

    Project label convention (binary classification)
    -----------------------------------------------
    - ``0`` = ALIVE (negative / majority class)
    - ``1`` = DEAD  (positive / minority class)

    Confusion matrix layout enforced by ``labels=[0, 1]``:
    ``[[TN, FP], [FN, TP]]``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels. Expected values are ``0`` and ``1``.
    y_pred : array-like of shape (n_samples,)
        Predicted hard labels (``0`` or ``1``), typically produced by ``estimator.predict``.
    y_pred_proba : array-like of shape (n_samples,) or (n_samples, n_classes), optional
        Predicted probability or score for the positive class (label ``1``).

        - If 1D, it is interpreted as the positive-class probability/score.
        - If 2D with at least 2 columns, column 1 (``[:, 1]``) is interpreted as the
          positive-class probability/score.
        - If ``None``, probability-based metrics (ROC-AUC, average precision) are not
          computed and are returned as ``None``.
    cost_matrix : numpy.ndarray of shape (2, 2), optional
        Misclassification cost matrix aligned with the confusion-matrix layout produced by
        ``confusion_matrix(..., labels=[0, 1])`` i.e. ``[[TN, FP], [FN, TP]]``.

        Interpreted as:
        - ``cost_matrix[0, 0]``: cost of TN (true 0 predicted 0)
        - ``cost_matrix[0, 1]``: cost of FP (true 0 predicted 1)
        - ``cost_matrix[1, 0]``: cost of FN (true 1 predicted 0)
        - ``cost_matrix[1, 1]``: cost of TP (true 1 predicted 1)

        If provided, the function computes ``average_cost`` via :func:`average_cost_score`
        using ``(y_true, y_pred)`` (hard labels). If ``None``, ``average_cost`` is returned
        as ``None``.

    Returns
    -------
    metrics : dict[str, float | int | None]
        Dictionary of metrics with keys:

        Counts
        - ``"tp"``, ``"tn"``, ``"fp"``, ``"fn"`` : int

        Threshold-based
        - ``"accuracy"``, ``"precision"``, ``"recall"``, ``"f1"`` : float

        Imbalance-oriented
        - ``"specificity"``, ``"fpr"``, ``"balanced_accuracy"``, ``"geometric_mean"`` : float
        - ``"mcc"``, ``"kappa"`` : float

        Probability / ranking (optional)
        - ``"roc_auc"``, ``"average_precision"`` : float or None

        Cost (optional)
        - ``"average_cost"`` : float or None
          Average misclassification cost per sample (lower is better), computed only when
          ``cost_matrix`` is provided.

    Raises
    ------
    ValueError
        If ``cost_matrix`` is provided but is invalid (e.g., not shape ``(2, 2)`` or contains
        non-finite values). This is raised by :func:`average_cost_score`.

    Notes
    -----
    - Confusion-matrix counts are computed with ``labels=[0, 1]`` to enforce a stable 2×2
      layout even when one class is absent in predictions.
    - ``precision``, ``recall``, and ``f1`` use ``zero_division=0`` to handle degenerate
      cases (e.g., no positive predictions) without raising.
    - ``fpr`` is computed as ``1.0 - specificity``.
    - ROC-AUC and average precision require both classes to be present in ``y_true``; when
      undefined (e.g., single-class fold), they are returned as ``None``.
    - ``average_cost`` is computed from hard predictions (``y_pred``). If you want
      probability-based threshold moving (e.g., MEC), compute a custom ``y_pred`` from
      probabilities and then call this function.

    Examples
    --------
    Basic usage (no probabilities)::

        >>> y_true = [0, 0, 0, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 0]
        >>> metrics = compute_classification_metrics(y_true, y_pred)
        >>> (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])
        (1, 1, 1, 2)

    With probabilities (ROC-AUC / average precision)::

        >>> import numpy as np
        >>> y_true = [0, 0, 0, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 0]
        >>> y_pred_proba = np.array([0.10, 0.35, 0.60, 0.80, 0.40])
        >>> metrics = compute_classification_metrics(y_true, y_pred, y_pred_proba)
        >>> metrics["roc_auc"] is not None
        True

    With an asymmetric cost matrix (FN >> FP)::

        >>> import numpy as np
        >>> cost = np.array([[0.0, 1.0],
        ...                  [1000.0, 0.0]])
        >>> metrics = compute_classification_metrics(y_true, y_pred, cost_matrix=cost)
        >>> metrics["average_cost"] >= 0.0
        True
    """

    # Convert to numpy arrays for safety
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # 1. Confusion Matrix (forced labels to ensure 2x2 shape)
    tn, fp, fn, tp = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    ).ravel()

    # 2. Imbalanced-learn optimized metrics
    #    Specificity: TN / (TN + FP)
    spec = specificity_score(y_true_arr, y_pred_arr, average="binary")

    #    Geometric mean: sqrt(Sensitivity * Specificity)
    g_mean = geometric_mean_score(y_true_arr, y_pred_arr, average="binary")

    # 3. Probabilistic metrics
    roc_auc = None
    avg_precision = None
    if y_pred_proba is not None:
        proba_arr = np.asarray(y_pred_proba)

        # Accept both (n_samples,) and (n_samples, 2) formats
        if proba_arr.ndim == 2:
            if proba_arr.shape[1] < 2:
                # Fallback: treat as (n_samples,)
                proba_pos = proba_arr.ravel()
            else:
                # Assume positive class is column 1
                proba_pos = proba_arr[:, 1]
        else:
            proba_pos = proba_arr.ravel()

        try:
            roc_auc = roc_auc_score(y_true_arr, proba_pos)
            avg_precision = average_precision_score(y_true_arr, proba_pos)
        except ValueError:
            # Edge cases where y_true has only one class → metrics undefined
            roc_auc = None
            avg_precision = None

    # 4. Calculate Average Cost if matrix is provided
    avg_cost = None
    if cost_matrix is not None:
        avg_cost = average_cost_score(y_true_arr, y_pred_arr, cost_matrix)

    results_dict = {
        # --- Raw counts ---
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        # --- Standard classification metrics ---
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        ),
        # Sensitivity / True Positive Rate
        "recall": recall_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true_arr,
            y_pred_arr,
            zero_division=0,
        ),
        # --- Imbalance & robustness ---
        "specificity": spec,
        "fpr": 1.0 - spec,
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "geometric_mean": g_mean,
        "mcc": matthews_corrcoef(y_true_arr, y_pred_arr),
        "kappa": cohen_kappa_score(y_true_arr, y_pred_arr),
        # --- Probabilistic / ranking ---
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        # --- Custom Business Metrics ---
        "average_cost": avg_cost,
    }

    return results_dict


def collect_single_report_one_fold(
    store: List[Dict[str, Any]],
    *,
    experiment_name: str,
    iteration: int,
    fold: int,
    model: str,
    metrics: Mapping[str, Optional[float] | int],
    data_split: str,
    **extra: Any,
) -> None:
    """
    Append one standardized metrics row for a single CV fold.

    This helper builds a single dictionary row with common identifiers
    (experiment name, iteration, fold, model, and split label) and merges them
    with scalar metrics plus any additional user-provided fields. The constructed
    row is appended to ``store`` in-place.

    Parameters
    ----------
    store : list of dict
        Mutable list that receives the constructed row (side effect: append).
    experiment_name : str
        Experiment identifier written into the row.
    iteration : int
        Outer repetition index for the current run.
    fold : int
        Fold index within the current iteration.
    model : str
        Model (or pipeline) identifier written into the row.
    metrics : Mapping[str, float or int or None]
        Mapping of metric names to scalar values (e.g., ``{"f1": 0.81, "tp": 10}``).
    data_split : str
        Split label written into the row under the ``"split"`` key (e.g.,
        ``"resubstitution"``, ``"test"``).
    **extra : Any
        Additional key-value pairs to merge into the row (e.g., hyperparameters,
        timings, selected feature metadata).

    Returns
    -------
    None
        This function operates by side effect.

    Notes
    -----
    The row is created via ``{**metrics, **extra}`` merging. To prevent accidental
    overwrites of the identifier fields (``experiment_name``, ``iteration``, ``fold``,
    ``model``, ``split``), consider validating that ``metrics`` and ``extra`` do not
    contain these reserved keys.

    Examples
    --------
    >>> rows = []
    >>> collect_single_report_one_fold(
    ...     rows,
    ...     experiment_name="baseline-v1",
    ...     iteration=1,
    ...     fold=3,
    ...     model="RandomForest",
    ...     metrics={"f1": 0.812, "roc_auc": 0.972},
    ...     data_split="test",
    ...     best_params={"n_estimators": 200},
    ...     tuning_time=12.4,
    ... )
    >>> len(rows)
    1
    """

    row = {
        "experiment_name": experiment_name,
        "iteration": iteration,
        "fold": fold,
        "model": model,
        "split": data_split,
        **metrics,
        **extra,
    }
    store.append(row)


def collect_fold_reports(
    *,
    resubstitution_rows: List[Dict[str, Any]],
    generalization_rows: List[Dict[str, Any]],
    experiment_name: str,
    iteration: int,
    fold: int,
    model_name: str,
    resubstitution_metrics: Mapping[str, Optional[float] | int],
    test_metrics: Mapping[str, Optional[float] | int],
    fold_size_train: int,
    fold_size_test: int,
    selected_features_names: Sequence[str],
    tuning_results: Dict[str, Any] | None = None,
) -> None:
    """
    Append per-fold report rows for resubstitution and generalization splits.

    This helper writes two standardized rows into the provided output buffers:
    one row for metrics computed on the training (resubstitution) split and one
    row for metrics computed on the held-out test (generalization) split. Row
    construction is delegated to :func:`collect_single_report_one_fold`, and
    feature-selection metadata is recorded consistently for both splits. If
    provided, ``tuning_results`` is attached only to the resubstitution row to
    avoid duplicating tuning metadata.

    Parameters
    ----------
    resubstitution_rows : list of dict
        Output buffer that receives the resubstitution row.
    generalization_rows : list of dict
        Output buffer that receives the generalization row.
    experiment_name : str
        Experiment identifier stored in both rows.
    iteration : int
        Outer repetition index stored in both rows.
    fold : int
        Outer fold index stored in both rows.
    model_name : str
        Model identifier stored in both rows.
    resubstitution_metrics : Mapping[str, float or int or None]
        Metrics computed on the training split.
    test_metrics : Mapping[str, float or int or None]
        Metrics computed on the outer test split.
    fold_size_train : int
        Number of samples in the training split; stored as ``fold_size`` in the
        resubstitution row.
    fold_size_test : int
        Number of samples in the test split; stored as ``fold_size`` in the
        generalization row.
    selected_features_names : Sequence[str]
        Names of selected features for this fold, aligned with
        ``selected_features_indices``. Stored in both rows.
    tuning_results : dict or None, optional
        Optional tuning summary to attach to the resubstitution row (e.g., best
        parameters, inner-CV scores, tuning time). If ``None``, no tuning fields
        are added.

    Returns
    -------
    None
        This function operates by side effect (appends to both output buffers).

    Notes
    -----
    - ``tuning_results`` is attached only to the resubstitution row to avoid
      duplicating tuning metadata in the generalization row.
    - Both rows always include selected feature indices and names to support
      downstream feature-selection stability analyses.
    - The final row schema and merge behavior are governed by
      :func:`collect_single_report_one_fold`.

    Examples
    --------
    >>> resub_rows, gen_rows = [], []
    >>> collect_fold_reports(
    ...     resubstitution_rows=resub_rows,
    ...     generalization_rows=gen_rows,
    ...     experiment_name="COST_SENSITIVE_NO_RESAMPLING",
    ...     iteration=1,
    ...     fold=3,
    ...     model_name="SVC",
    ...     resubstitution_metrics={"f1": 0.95, "roc_auc": 0.99},
    ...     test_metrics={"f1": 0.82, "roc_auc": 0.93},
    ...     fold_size_train=18_000,
    ...     fold_size_test=2_000,
    ...     selected_features_names=["F1", "F2", "F3"],
    ...     tuning_results={"best_params": {"classifier__C": 1.0}, "tuning_time": 12.4},
    ... )
    >>> (len(resub_rows), len(gen_rows))
    (1, 1)
    """

    tuning_kwargs = tuning_results or {}

    collect_single_report_one_fold(
        resubstitution_rows,
        experiment_name=experiment_name,
        iteration=iteration,
        fold=fold,
        model=model_name,
        metrics=resubstitution_metrics,
        data_split="resubstitution",
        fold_size=fold_size_train,
        **tuning_kwargs,
        selected_features_names=selected_features_names,
    )

    collect_single_report_one_fold(
        generalization_rows,
        experiment_name=experiment_name,
        iteration=iteration,
        fold=fold,
        model=model_name,
        metrics=test_metrics,
        data_split="generalization",
        fold_size=fold_size_test,
        selected_features_names=selected_features_names,
    )
