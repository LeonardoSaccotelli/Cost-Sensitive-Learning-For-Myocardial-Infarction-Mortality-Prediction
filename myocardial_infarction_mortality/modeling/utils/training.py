from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from myocardial_infarction_mortality.data_preparation.feature_selection import (
    get_selected_feature_names,
)
from myocardial_infarction_mortality.evaluation.metrics_evaluation import (
    apply_decision_policy,
    collect_fold_reports,
    compute_classification_metrics,
    get_avg_cost_scorer,
)
from myocardial_infarction_mortality.modeling.utils.models import (
    get_des_model,
    get_static_ensemble_model_and_search_space,
    get_static_model_and_search_space,
)
from myocardial_infarction_mortality.modeling.utils.pipeline import (
    build_model_pipeline,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def run_randomized_search_cv(
    estimator: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    *,
    n_iter: int,
    val_cv_split: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
    cost_matrix: Optional[np.ndarray] = None,
    verbose: int = 3,
) -> tuple[Union[ImbPipeline, Pipeline, BaseEstimator], Dict[str, Any]]:
    """
    Run a randomized hyperparameter search (inner CV) and return the refit best estimator.

    This helper runs :class:`sklearn.model_selection.RandomizedSearchCV` over
    ``search_space`` using an inner stratified K-fold splitter:

    ``StratifiedKFold(n_splits=val_cv_split, shuffle=True, random_state=random_state)``.

    The search is fit on ``(X_train, y_train)`` and the best configuration is refit on the
    full training set (``refit=True``). A compact tuning summary is extracted from
    ``search.cv_results_`` at ``search.best_index_``.

    The function supports both standard scikit-learn scorers (via their string identifiers)
    and a project-specific cost-sensitive tuning mode triggered by ``scoring="Average_Cost"``.
    In that mode, a custom scorer is created via :func:`get_avg_cost_scorer`, which wraps
    :func:`average_cost_score` and minimizes the **average misclassification cost** computed
    from a 2x2 cost matrix aligned with ``confusion_matrix(..., labels=[0, 1])``.

    Parameters
    ----------
    estimator : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Estimator (or pipeline) to tune. Must implement ``fit`` and expose hyperparameters
        via ``get_params`` so RandomizedSearchCV can route parameters.
    search_space : dict[str, Any]
        Hyperparameter search space forwarded as ``param_distributions`` to
        :class:`sklearn.model_selection.RandomizedSearchCV`.

        Keys must match valid parameter names for ``estimator`` (e.g., ``"C"`` for a bare
        estimator, or ``"classifier__C"`` for a pipeline step named ``"classifier"``).
        Values may be candidate lists and/or SciPy distribution objects.
    X_train : pandas.DataFrame or numpy.ndarray
        Training features of shape ``(n_samples, n_features)``.
    y_train : pandas.Series or numpy.ndarray
        Training labels of shape ``(n_samples,)``. Expected binary values are ``0`` and ``1``.
    n_iter : int
        Number of hyperparameter configurations to sample. Must be >= 1.
    val_cv_split : int
        Number of folds for inner CV. Must be >= 2.
    scoring : str
        Scoring identifier. Two modes are supported:

        1. Any standard scikit-learn scoring name (e.g., ``"f1"``, ``"roc_auc"``,
           ``"average_precision"``). In this case, ``scoring`` is passed directly to
           RandomizedSearchCV.

        2. ``"Average_Cost"`` to enable cost-sensitive tuning using
           :func:`get_avg_cost_scorer` and the provided ``cost_matrix``. Lower average cost
           is better.
    random_state : int
        Random seed used for CV shuffling and randomized hyperparameter sampling.
    n_jobs : int
        Number of parallel jobs used by RandomizedSearchCV. Use ``-1`` for all available cores.
    cost_matrix : numpy.ndarray of shape (2, 2), optional
        Misclassification cost matrix used **only** when ``scoring="Average_Cost"``.
        The matrix must be aligned with ``confusion_matrix(..., labels=[0, 1])``:

        ``[[TN, FP],``
        `` [FN, TP]]``

        With the project convention:
        - ``0`` = ALIVE (negative / majority class)
        - ``1`` = DEAD  (positive / minority class)

        Therefore, it is interpreted as:
        - ``cost_matrix[0, 0]``: cost of TN
        - ``cost_matrix[0, 1]``: cost of FP
        - ``cost_matrix[1, 0]``: cost of FN
        - ``cost_matrix[1, 1]``: cost of TP

        If ``scoring="Average_Cost"``, this parameter is required.
    verbose : int, default=3
        Verbosity level forwarded to RandomizedSearchCV.

    Returns
    -------
    best_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Best estimator found by RandomizedSearchCV, refit on the full training set.
    tuning_results : dict[str, Any]
        Standardized tuning summary for the best candidate with keys:

        - ``"cv_tuning_mean_train_score"`` : float
            Mean inner-CV training score for the best configuration.
            If ``scoring="Average_Cost"``, this is returned as a **positive average cost**
            (lower is better).
        - ``"cv_tuning_std_train_score"`` : float
            Standard deviation of the inner-CV training score for the best configuration.
        - ``"cv_tuning_mean_val_score"`` : float
            Mean inner-CV validation score for the best configuration.
            If ``scoring="Average_Cost"``, this is returned as a **positive average cost**
            (lower is better).
        - ``"cv_tuning_std_val_score"`` : float
            Standard deviation of the inner-CV validation score for the best configuration.
        - ``"best_params"`` : dict[str, Any]
            Best hyperparameter configuration found.
        - ``"tuning_time"`` : float
            Wall-clock time in seconds spent inside ``search.fit``.

    Raises
    ------
    ValueError
        If ``n_iter < 1`` or ``val_cv_split < 2``.
    ValueError
        If ``scoring="Average_Cost"`` and ``cost_matrix`` is ``None``.
    Exception
        Any exception raised during fitting is propagated. Because ``error_score="raise"``
        is set, failures during CV are not masked.

    Notes
    -----
    - The reported train/validation scores refer to the **inner** CV used by the randomized
      search (not the outer evaluation loop).
    - When ``scoring="Average_Cost"``, the scorer returned by :func:`get_avg_cost_scorer`
      is built with ``greater_is_better=False``. As a consequence, RandomizedSearchCV
      stores **negative** values in ``cv_results_["mean_train_score"]`` and
      ``cv_results_["mean_test_score"]`` (higher is better in scikit-learn’s convention).
      This function converts them back to **positive costs** when filling ``tuning_results``.
    - The custom cost scorer uses hard predictions from ``predict``. If you need
      cost-sensitive threshold moving, you must implement a probability-based scorer
      using ``predict_proba`` (or ``decision_function``) and an explicit threshold rule.
    - To avoid nested parallelism, keep estimator-level parallelism disabled (e.g., model
      ``n_jobs=1``) when RandomizedSearchCV runs with ``n_jobs > 1``.

    Examples
    --------
    Standard metric tuning::

        >>> best_model, tuning = run_randomized_search_cv(
        ...     estimator=clf,
        ...     search_space=space,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     n_iter=30,
        ...     val_cv_split=5,
        ...     scoring="f1",
        ...     random_state=42,
        ...     n_jobs=-1,
        ...     verbose=2,
        ... )

    Cost-sensitive tuning (Average_Cost)::

        >>> import numpy as np
        >>> cost = np.array([[0.0, 1.0],
        ...                  [1000.0, 0.0]])
        >>> best_model, tuning = run_randomized_search_cv(
        ...     estimator=clf,
        ...     search_space=space,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     n_iter=30,
        ...     val_cv_split=5,
        ...     scoring="Average_Cost",
        ...     random_state=42,
        ...     n_jobs=-1,
        ...     cost_matrix=cost,
        ...     verbose=2,
        ... )
        >>> tuning["cv_tuning_mean_val_score"] >= 0.0
        True
    """

    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1. Got {n_iter}.")
    if val_cv_split < 2:
        raise ValueError(f"val_cv_split must be >= 2. Got {val_cv_split}.")

    # Fix the scoring function to be used during tuning phase
    if scoring == "Average_Cost":
        if cost_matrix is None:
            raise ValueError("A 'cost_matrix' must be provided when scoring='Average_Cost'.")

        print("[RANDOMIZED SEARCH SETTINGS]: Using custom Average_Cost scorer.")
        print(f"                               Cost Matrix:\n{cost_matrix}")
        print("                               Note: Optimization scores will appear negative.")

        active_scorer = get_avg_cost_scorer(cost_matrix=cost_matrix)
    else:
        print(
            f"[RANDOMIZED SEARCH SETTINGS]: scoring: {scoring}, random_state: {random_state}, n_jobs: {n_jobs}"
        )
        active_scorer = scoring

    splitter = StratifiedKFold(
        n_splits=val_cv_split,
        random_state=random_state,
        shuffle=True,
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=active_scorer,
        n_jobs=n_jobs,
        refit=True,
        cv=splitter,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )

    start_tuning_time = time.time()
    search.fit(X_train, y_train)
    end_tuning_time = time.time()

    best_model = search.best_estimator_

    tuning_train_mean = (
        -search.cv_results_["mean_train_score"][search.best_index_]
        if scoring == "Average_Cost"
        else search.cv_results_["mean_train_score"][search.best_index_]
    )
    tuning_val_mean = (
        -search.cv_results_["mean_test_score"][search.best_index_]
        if scoring == "Average_Cost"
        else search.cv_results_["mean_test_score"][search.best_index_]
    )

    tuning_results = {
        "cv_tuning_mean_train_score": tuning_train_mean,
        "cv_tuning_std_train_score": search.cv_results_["std_train_score"][search.best_index_],
        "cv_tuning_mean_val_score": tuning_val_mean,
        "cv_tuning_std_val_score": search.cv_results_["std_test_score"][search.best_index_],
        "best_params": search.best_params_,
        "tuning_time": end_tuning_time - start_tuning_time,
    }

    print(f"[RANDOMIZED SEARCH BEST PARAMS]: {tuning_results['best_params']}")

    return best_model, tuning_results


def train_and_evaluate_one_fold_static_model(
    experiment_setting: dict[str, Any],
    base_model: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    logger: Any,
    n_iter: int,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[
    Union[ImbPipeline, Pipeline, BaseEstimator],
    Dict[str, Any],
    Dict[str, Optional[float] | int],
    Dict[str, Optional[float] | int],
]:
    """
    Tune and evaluate a static (non-DES) classifier on a single outer CV fold.

    The function performs inner-CV hyperparameter tuning on the provided outer training
    split using :func:`run_randomized_search_cv`, refits the best configuration on the
    full outer training data, and then evaluates the refit model on both:

    - the outer training split (resubstitution), and
    - the outer test split (generalization),

    using :func:`compute_classification_metrics`.

    Decision policy (standard vs MEC)
    ---------------------------------
    Hard predictions used for metric computation are produced by :func:`apply_decision_policy`
    using ``experiment_setting["decision_policy_mode"]``:

    - ``"standard"``: uses the estimator's default decision rule via ``predict``.
      If ``predict_proba`` is available, positive-class probabilities are also returned and
      probability-based metrics (ROC-AUC / average precision) can be computed; otherwise
      they are returned as ``None``.
    - ``"mec"``: uses the **Minimum Expected Cost (MEC)** decision rule via
      :func:`min_expected_cost_predict`, which requires ``predict_proba`` and a valid
      ``experiment_setting["costs_matrix"]``. Hard labels are derived by minimizing expected
      cost; returned probabilities remain the raw positive-class probability ``P(y=1|x)``.

    Cost-sensitive tuning (Average_Cost)
    ------------------------------------
    When ``scoring="Average_Cost"``, tuning uses a custom scorer built from
    ``experiment_setting["costs_matrix"]`` via :func:`get_avg_cost_scorer`. Because the scorer
    is configured with ``greater_is_better=False``, scikit-learn stores negative scores in
    ``cv_results_``; :func:`run_randomized_search_cv` converts them back to **positive costs**
    in the returned ``tuning_results``.

    Test-time inference latency is measured on ``X_test`` around the decision-policy calls
    (prediction + probability extraction when applicable) and returned as ``"score_time"``
    within the test metrics.

    Parameters
    ----------
    experiment_setting : dict[str, Any]
        Experiment configuration describing the approach, optional imbalance strategy, decision
        policy, and misclassification costs. A typical schema includes::

            {
                "experiment_name": "baseline__mec_fp1_fn10",
                "description": "...",
                "approach": "baseline",  # baseline | cost_sensitive_learning | data_level
                "tags": [...],
                "class_weight": None,    # None | "balanced" | {0: w0, 1: w1}
                "resampling_method": None,
                "resampling_params": None,
                "decision_policy_mode": "standard",  # standard | mec
                "costs_matrix": COST_MATRIX,
            }

        This function uses:
        - ``experiment_setting["decision_policy_mode"]`` to choose between standard vs MEC
          hard-labeling (via :func:`apply_decision_policy`).
        - ``experiment_setting["costs_matrix"]``:
            * for tuning when ``scoring="Average_Cost"``,
            * for MEC hard predictions when ``decision_policy_mode="mec"``,
            * for reporting ``average_cost`` inside :func:`compute_classification_metrics`.

        Cost matrix alignment (project convention):
        - ``0`` = ALIVE (negative / majority class)
        - ``1`` = DEAD  (positive / minority class)

        The cost matrix is interpreted as ``costs_matrix[true_class, predicted_class]`` and
        typically arranged as::

            [[TN_cost, FP_cost],
             [FN_cost, TP_cost]]

        **Important:** MEC assumes the column order of ``predict_proba`` matches the class
        indexing used by the cost matrix (in scikit-learn this is given by ``estimator.classes_``).

    base_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Estimator or pipeline to tune and evaluate. Must implement ``fit`` and ``predict``.
        If ``experiment_setting["decision_policy_mode"] == "mec"``, it must also implement
        ``predict_proba``. Under the standard policy, ``predict_proba`` is optional; when absent,
        probability-based metrics are returned as ``None``.
    search_space : dict[str, Any]
        Hyperparameter search space passed to the tuning routine. Keys must match valid
        parameter names of ``base_model`` (e.g., ``"classifier__C"`` for a pipeline
        step named ``"classifier"``).
    X_train : pandas.DataFrame or numpy.ndarray
        Training features for the current outer fold of shape ``(n_train, n_features)``.
    y_train : pandas.Series or numpy.ndarray
        Training labels for the current outer fold of shape ``(n_train,)``.
    X_test : pandas.DataFrame or numpy.ndarray
        Test features for the current outer fold of shape ``(n_test, n_features)``.
    y_test : pandas.Series or numpy.ndarray
        Test labels for the current outer fold of shape ``(n_test,)``.
    logger : Any
        Logger-like object exposing an ``info(str)`` method used for progress messages.
    n_iter : int
        Number of parameter configurations sampled during randomized search.
    val_cv_split : int, default=5
        Number of inner stratified CV folds used during tuning.
    scoring : str, default="f1"
        Scoring identifier used to select the best configuration during tuning.

        - If a standard scikit-learn scorer name is provided (e.g., ``"f1"``, ``"roc_auc"``,
          ``"average_precision"``), it is passed directly to RandomizedSearchCV.
        - If ``"Average_Cost"``, the tuning objective is the **average misclassification cost**
          computed with ``experiment_setting["costs_matrix"]`` (lower is better).
    random_state : int, default=42
        Random seed forwarded to the tuning routine (inner splitter and parameter sampling).
    n_jobs : int, default=-1
        Number of parallel jobs used during tuning. To avoid nested parallelism, set
        estimator-level ``n_jobs`` appropriately (often ``1``).

    Returns
    -------
    best_model : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Best estimator found by tuning, refit on the full outer training split.
    tuning_results : dict[str, Any]
        Standardized tuning summary returned by :func:`run_randomized_search_cv`
        (e.g., inner-CV mean/std scores, best params, tuning time). If ``scoring="Average_Cost"``,
        the reported mean train/val scores are **positive costs** (lower is better).
    resubstitution_metrics : dict[str, float | int | None]
        Metrics computed on the outer training split via :func:`compute_classification_metrics`.
        If a cost matrix is provided, includes ``"average_cost"``; if probabilities are not
        available (standard policy without ``predict_proba``), ROC-AUC and average precision
        are returned as ``None``.
    test_metrics : dict[str, float | int | None]
        Metrics computed on the outer test split via :func:`compute_classification_metrics`,
        with an additional key:

        - ``"score_time"`` : float
          Wall-clock time in seconds measured around decision-policy prediction on ``X_test``.

    Raises
    ------
    KeyError
        If ``experiment_setting["decision_policy_mode"] == "mec"`` and ``"costs_matrix"`` is missing
        or ``None``.
    AttributeError
        If ``experiment_setting["decision_policy_mode"] == "mec"`` but the estimator/pipeline does
        not expose ``predict_proba`` (raised by :func:`apply_decision_policy`).
    ValueError
        If tuning fails due to invalid ``search_space`` keys, incompatible ``scoring``,
        or an invalid CV configuration (raised by scikit-learn inside the tuning helper).
    Exception
        Any exception raised by the underlying estimator or pipeline during tuning,
        fitting, or prediction may propagate.

    Notes
    -----
    - Leakage safety depends on encapsulating preprocessing, feature selection, and
      resampling inside ``base_model`` when the function is used within an outer CV loop.
    - Under MEC, hard labels are derived from probabilities and the cost matrix; the returned
      probability vector remains the raw positive-class probability.
    - The ``average_cost`` metric (when enabled) is computed from hard predictions, consistent
      with :func:`average_cost_score`.

    Examples
    --------
    Standard policy evaluation::

        >>> best_model, tuning_results, resub_metrics, test_metrics = train_and_evaluate_one_fold_static_model(
        ...     experiment_setting=experiment_setting,
        ...     base_model=base_model,
        ...     search_space=search_space,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ...     logger=logger,
        ...     n_iter=30,
        ...     val_cv_split=5,
        ...     scoring="average_precision",
        ...     random_state=42,
        ...     n_jobs=-1,
        ... )

    MEC policy with cost-based tuning::

        >>> import numpy as np
        >>> experiment_setting = {
        ...     "experiment_name": "baseline__mec_fp1_fn10",
        ...     "description": "MEC policy with FP=1, FN=10.",
        ...     "approach": "baseline",
        ...     "tags": ["baseline", "mec_policy"],
        ...     "class_weight": None,
        ...     "resampling_method": None,
        ...     "resampling_params": None,
        ...     "decision_policy_mode": "mec",
        ...     "costs_matrix": np.array([[0.0, 1.0],
        ...                              [10.0, 0.0]]),
        ... }
        >>> best_model, tuning_results, resub_metrics, test_metrics = train_and_evaluate_one_fold_static_model(
        ...     experiment_setting=experiment_setting,
        ...     base_model=base_model,
        ...     search_space=search_space,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ...     logger=logger,
        ...     n_iter=30,
        ...     val_cv_split=5,
        ...     scoring="Average_Cost",
        ...     random_state=42,
        ...     n_jobs=-1,
        ... )
        >>> test_metrics["average_cost"] is not None
        True
    """

    costs_matrix = experiment_setting.get("costs_matrix")
    decision_policy_mode = experiment_setting.get("decision_policy_mode", "standard")

    if decision_policy_mode == "mec" and costs_matrix is None:
        raise KeyError("Cannot use 'mec' decision policy without a 'costs_matrix'.")

    # Run RandomizedSearchCV
    best_model, tuning_results = run_randomized_search_cv(
        estimator=base_model,
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        n_iter=n_iter,
        val_cv_split=val_cv_split,
        scoring=scoring,
        random_state=random_state,
        cost_matrix=costs_matrix,
        n_jobs=n_jobs,
    )

    # --- Evaluate on the training set (resubstitution error)
    logger.info("[COMPUTING RESUBSTITUTION METRICS]...")

    # Apply decision policy on the training set
    y_train_pred, y_train_pred_prob = apply_decision_policy(
        estimator=best_model,
        X=X_train,
        policy_mode=decision_policy_mode,
        cost_matrix=costs_matrix,
    )

    resubstitution_metrics = compute_classification_metrics(
        y_true=y_train,
        y_pred=y_train_pred,
        y_pred_proba=y_train_pred_prob,
        cost_matrix=costs_matrix,
    )
    logger.info(f"[RESUBSTITUTION METRICS]: {resubstitution_metrics}")

    # --- Evaluate on the test set (generalization error)
    logger.info("[COMPUTING GENERALIZATION METRICS]...")

    start_score_time = time.time()

    # Apply decision policy on the test set
    y_test_pred, y_test_pred_prob = apply_decision_policy(
        estimator=best_model,
        X=X_test,
        policy_mode=decision_policy_mode,
        cost_matrix=costs_matrix,
    )

    end_score_time = time.time()

    test_metrics = compute_classification_metrics(
        y_true=y_test, y_pred=y_test_pred, y_pred_proba=y_test_pred_prob, cost_matrix=costs_matrix
    )
    test_metrics["score_time"] = end_score_time - start_score_time
    logger.info(f"[GENERALIZATION METRICS]: {test_metrics}")

    return best_model, tuning_results, resubstitution_metrics, test_metrics


def train_and_evaluate_one_fold_des_model(
    experiment_setting: dict[str, Any],
    des_model: BaseEstimator,
    des_conf: Dict[str, Any],
    pool_classifiers: Union[ImbPipeline, Pipeline, BaseEstimator],
    search_space: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    logger: Any,
    n_iter: int,
    dsel_size: float = 0.2,
    val_cv_split: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[
    Pipeline,
    Dict[str, Any],
    Dict[str, Optional[float] | int],
    Dict[str, Optional[float] | int],
]:
    """
    Train and evaluate a Dynamic Ensemble Selection (DES) model on a single outer CV fold.

    This helper implements a leakage-safe two-stage DES workflow within one outer split:

    1) Split the outer training fold into a pool-training subset and a DSEL subset using a
       stratified ``train_test_split`` controlled by ``random_state``.

    2) Tune and refit the pool pipeline on the pool-training subset via
       :func:`run_randomized_search_cv`. The returned best pipeline is then evaluated on the
       pool-training subset (pool resubstitution).

    3) Fit the DES model on DSEL:
       - extract the fitted preprocessing part of the tuned pool pipeline by slicing it up to
         (but excluding) the step named ``"resampling"``,
       - transform ``X_dsel`` with the fitted preprocessing,
       - extract the fitted pool estimator from the tuned pipeline step named ``"classifier"``,
       - inject the fitted pool into a local copy of ``des_conf`` under ``"pool_classifiers"``,
         then call ``des_model.set_params(**...)`` and ``des_model.fit(...)`` on transformed DSEL.

    4) Build an inference pipeline (preprocessing -> DES) and evaluate on the outer test fold.
       Test-time predictions are timed and stored as ``"score_time"`` inside the returned
       ``test_metrics``.

    Decision policy (standard vs MEC)
    ---------------------------------
    Hard predictions are produced by :func:`apply_decision_policy` using
    ``experiment_setting["decision_policy_mode"]``:

    - ``"standard"``: use ``predict`` (default model decision rule).
    - ``"mec"``: use **Minimum Expected Cost (MEC)** via :func:`min_expected_cost_predict`,
      which requires ``predict_proba`` and a valid ``costs_matrix``.

    When probabilities are available, positive-class probabilities are returned and used to
    compute probability-based metrics (ROC-AUC, average precision). If probabilities are not
    available under the standard policy, probability-based metrics are returned as ``None``.

    Cost-sensitive tuning
    ---------------------
    When ``scoring="Average_Cost"``, pool tuning uses a custom scorer built from
    ``experiment_setting["costs_matrix"]`` via :func:`get_avg_cost_scorer`. Cost-sensitivity
    in this function applies to **pool tuning** and to **hard-labeling under MEC**; it does
    not otherwise change the DES model unless encoded in ``des_conf`` / estimator itself.

    Parameters
    ----------
    experiment_setting : dict[str, Any]
        Experiment configuration describing the approach, optional imbalance handling, decision
        policy, and the cost matrix. A typical schema includes::

            {
                "experiment_name": "baseline__mec_fp1_fn10",
                "description": "No resampling, no class_weight. Decision policy: MEC with costs FP=1, FN=10.",
                "approach": "baseline",  # baseline | cost_sensitive_learning | data_level
                "tags": ["baseline", "mec_policy"],
                "class_weight": None,    # None | "balanced" | {0: w0, 1: w1}
                "resampling_method": None,
                "resampling_params": None,
                "decision_policy_mode": "mec",  # standard | mec
                "costs_matrix": COST_MATRIX,
            }

        This function uses:
        - ``experiment_setting["decision_policy_mode"]`` to choose between standard vs MEC
          hard-labeling (via :func:`apply_decision_policy`).
        - ``experiment_setting["costs_matrix"]``:
            * for pool tuning when ``scoring="Average_Cost"``,
            * for MEC hard predictions when ``decision_policy_mode="mec"``,
            * optionally for reporting ``average_cost`` inside
              :func:`compute_classification_metrics`.

        Cost matrix alignment (project convention):
        - ``0`` = ALIVE (negative / majority class)
        - ``1`` = DEAD  (positive / minority class)

        The cost matrix is interpreted as ``costs_matrix[true_class, predicted_class]`` and
        typically arranged as::

            [[TN_cost, FP_cost],
             [FN_cost, TP_cost]]

        **Important:** MEC assumes the column order of ``predict_proba`` matches the class
        indexing used by the cost matrix (in scikit-learn this is given by ``estimator.classes_``).

    des_model : sklearn.base.BaseEstimator
        Unfitted DES estimator (typically from DESlib) implementing ``set_params``, ``fit``,
        and ``predict``. For MEC, the final inference pipeline must also implement
        ``predict_proba``.
    des_conf : dict[str, Any]
        Configuration forwarded to ``des_model.set_params(**des_conf_local)``. This function
        creates a local shallow copy to inject the fitted pool under the ``"pool_classifiers"``
        key before calling ``set_params``.
    pool_classifiers : imblearn.pipeline.Pipeline or sklearn.pipeline.Pipeline or sklearn.base.BaseEstimator
        Pool estimator to tune on the pool-training subset. Expected to behave like a pipeline
        after tuning (i.e., expose ``named_steps`` and support slicing), and to contain:
        - a step named ``"resampling"`` (excluded from inference; may be a real resampler or a
          ``"passthrough"`` placeholder),
        - a step named ``"classifier"`` (the fitted pool injected into the DES model).
    search_space : dict[str, Any]
        Hyperparameter distributions or candidate lists used to tune ``pool_classifiers``.
        Keys must match valid parameter names of the pool pipeline using the double-underscore
        convention (e.g., ``"classifier__n_estimators"``, ``"feature_selection_filter__k"``).
    X_train : pandas.DataFrame or numpy.ndarray
        Features for the outer training fold of shape ``(n_train_samples, n_features)``.
    y_train : pandas.Series or numpy.ndarray
        Labels for the outer training fold of shape ``(n_train_samples,)``.
    X_test : pandas.DataFrame or numpy.ndarray
        Features for the outer test fold of shape ``(n_test_samples, n_features)``.
    y_test : pandas.Series or numpy.ndarray
        Labels for the outer test fold of shape ``(n_test_samples,)``.
    logger : Any
        Logger exposing ``.info(str)``.
    n_iter : int
        Number of hyperparameter configurations sampled during pool tuning.
    dsel_size : float, default=0.2
        Fraction of ``X_train`` reserved for DSEL (must satisfy ``0 < dsel_size < 1``).
        The split is stratified by ``y_train``.
    val_cv_split : int, default=5
        Number of folds for the inner CV used during pool tuning.
    scoring : str, default="f1"
        Scoring identifier used by the randomized CV search to select the best pool
        configuration.

        - If a standard scikit-learn scorer name is provided (e.g., ``"f1"``, ``"roc_auc"``,
          ``"average_precision"``), it is passed directly to RandomizedSearchCV.
        - If ``"Average_Cost"``, the tuning objective is the **average misclassification cost**
          computed with ``experiment_setting["costs_matrix"]`` (lower is better). Note that
          internal CV scores stored by scikit-learn appear negative due to
          ``greater_is_better=False``; the tuning helper converts them back to positive costs
          in the returned ``tuning_results``.

    random_state : int, default=42
        Random seed used for:
        - the pool-train vs DSEL split,
        - the inner CV splitter shuffling,
        - the randomized hyperparameter sampling.
    n_jobs : int, default=-1
        Number of parallel jobs used by the randomized CV search. To avoid nested parallelism,
        configure underlying pool estimators with ``n_jobs=1`` when appropriate.

    Returns
    -------
    final_des_pipeline : sklearn.pipeline.Pipeline
        Fitted inference pipeline used for test evaluation. It contains the fitted preprocessing
        steps extracted from the tuned pool pipeline (all steps before ``"resampling"``),
        followed by the fitted DES estimator as the final step named ``"classifier"``.
    tuning_results : dict[str, Any]
        Tuning summary returned by :func:`run_randomized_search_cv` for the best pool candidate
        (e.g., CV mean/std scores, best params, tuning time). If ``scoring="Average_Cost"``,
        the reported mean train/val scores are **positive costs** (lower is better).
    pool_resubstitution_metrics : dict[str, Optional[float] | int]
        Metrics computed on the pool-training subset using:
        - hard labels produced by :func:`apply_decision_policy` (standard or MEC),
        - probabilities when available,
        and optionally including ``average_cost`` when a cost matrix is provided to
        :func:`compute_classification_metrics`.
    test_metrics : dict[str, Optional[float] | int]
        Metrics computed on the outer test fold using:
        - hard labels produced by :func:`apply_decision_policy` (standard or MEC),
        - probabilities when available,
        and optionally including ``average_cost``. The returned dictionary also includes
        ``"score_time"`` (seconds), measured around the prediction calls.

    Raises
    ------
    ValueError
        If ``dsel_size`` is not in ``(0, 1)``, if stratified splitting fails, if tuning fails
        due to invalid configuration (e.g., incompatible ``scoring`` or invalid parameter names
        in ``search_space``), or if the tuned pool pipeline does not include a step named
        ``"resampling"`` (required for preprocessing extraction).
    KeyError
        If required keys are missing from ``experiment_setting`` (e.g., ``"costs_matrix"`` when
        needed, or ``"decision_policy_mode"`` in strict setups).
    AttributeError
        If ``experiment_setting["decision_policy_mode"] == "mec"`` but the estimator/pipeline
        used for inference does not implement ``predict_proba`` (raised by
        :func:`apply_decision_policy`).
    Exception
        Any exception raised by the underlying estimators/pipeline during fitting, transformation,
        or prediction may propagate.

    Notes
    -----
    - No outer-test leakage: the outer test fold is used only for final evaluation; pool tuning
      and DES fitting occur exclusively within the outer training fold.
    - Train-time-only resampling: the pool pipeline step named ``"resampling"`` is excluded from
      the final inference pipeline.
    - Binary classification convention: when available, the positive-class probability is taken
      as ``predict_proba(X)[:, 1]``.
    - Under MEC, hard labels are computed from the full probability matrix and ``costs_matrix``;
      the returned ``y_pred_prob`` remains the raw positive-class probability.

    Examples
    --------
    Standard policy evaluation::

        >>> experiment_setting = {
        ...     "experiment_name": "baseline__standard",
        ...     "description": "Standard policy.",
        ...     "approach": "baseline",
        ...     "tags": ["baseline", "standard_policy"],
        ...     "class_weight": None,
        ...     "resampling_method": None,
        ...     "resampling_params": None,
        ...     "decision_policy_mode": "standard",
        ...     "costs_matrix": None,
        ... }
        >>> final_pipe, tuning, pool_resub, test_metrics = train_and_evaluate_one_fold_des_model(
        ...     experiment_setting=experiment_setting,
        ...     des_model=des_model,
        ...     des_conf=des_conf,
        ...     pool_classifiers=pool_pipeline,
        ...     search_space=pool_space,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ...     logger=logger,
        ...     n_iter=30,
        ...     dsel_size=0.2,
        ...     val_cv_split=5,
        ...     scoring="average_precision",
        ...     random_state=42,
        ...     n_jobs=-1,
        ... )

    MEC policy evaluation (FN >> FP)::

        >>> import numpy as np
        >>> experiment_setting = {
        ...     "experiment_name": "baseline__mec_fp1_fn10",
        ...     "description": "MEC policy with FP=1, FN=10.",
        ...     "approach": "baseline",
        ...     "tags": ["baseline", "mec_policy"],
        ...     "class_weight": None,
        ...     "resampling_method": None,
        ...     "resampling_params": None,
        ...     "decision_policy_mode": "mec",
        ...     "costs_matrix": np.array([[0.0, 1.0],
        ...                              [10.0, 0.0]]),
        ... }
        >>> final_pipe, tuning, pool_resub, test_metrics = train_and_evaluate_one_fold_des_model(
        ...     experiment_setting=experiment_setting,
        ...     des_model=des_model,
        ...     des_conf=des_conf,
        ...     pool_classifiers=pool_pipeline,
        ...     search_space=pool_space,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ...     logger=logger,
        ...     n_iter=30,
        ...     dsel_size=0.2,
        ...     val_cv_split=5,
        ...     scoring="Average_Cost",
        ...     random_state=42,
        ...     n_jobs=-1,
        ... )
        >>> test_metrics["average_cost"] is not None
        True
    """

    costs_matrix = experiment_setting.get("costs_matrix")
    decision_policy_mode = experiment_setting.get("decision_policy_mode", "standard")

    # Split TRAIN into pool-training and DSEL
    X_train_pool, X_dsel, y_train_pool, y_dsel = train_test_split(
        X_train,
        y_train,
        test_size=dsel_size,
        stratify=y_train,
        random_state=random_state,
    )

    # Run RandomizedSearchCV
    best_pipe_pool_classifiers, tuning_results = run_randomized_search_cv(
        estimator=pool_classifiers,
        search_space=search_space,
        X_train=X_train_pool,
        y_train=y_train_pool,
        n_iter=n_iter,
        val_cv_split=val_cv_split,
        scoring=scoring,
        random_state=random_state,
        cost_matrix=costs_matrix,
        n_jobs=n_jobs,
    )

    # --- Pool resubstitution metrics (only for the tuned pool pipeline) ---
    logger.info("[COMPUTING POOL RESUBSTITUTION METRICS]...")

    # Apply decision policy on the training set
    y_train_pool_pred, y_train_pool_pred_prob = apply_decision_policy(
        estimator=best_pipe_pool_classifiers,
        X=X_train_pool,
        policy_mode=decision_policy_mode,
        cost_matrix=costs_matrix,
    )

    pool_resubstitution_metrics = compute_classification_metrics(
        y_true=y_train_pool,
        y_pred=y_train_pool_pred,
        y_pred_proba=y_train_pool_pred_prob,
        cost_matrix=costs_matrix,
    )
    logger.info(f"[POOL RESUBSTITUTION METRICS]: {pool_resubstitution_metrics}")

    # Extract the preprocessing pipeline (fitted)
    # We skip resampling step since it is required just at training time
    resampling_idx = list(best_pipe_pool_classifiers.named_steps.keys()).index("resampling")
    fitted_preproc = best_pipe_pool_classifiers[:resampling_idx]

    # Apply the preprocessing steps on the DSEL dataset
    X_dsel_trans = fitted_preproc.transform(X_dsel)

    # Extract the fitted pool of classifiers
    fitted_pool = best_pipe_pool_classifiers.named_steps["classifier"]

    # Add the trained pool of classifiers to the DES model config
    des_conf_local = dict(des_conf)  # shallow copy is enough
    des_conf_local["pool_classifiers"] = fitted_pool

    # Fit DES model on DSEL in transformed space
    logger.info("[FITTING DSEL METHOD]...")
    des_model.set_params(**des_conf_local)
    des_model.fit(X_dsel_trans, y_dsel)

    # Final inference pipeline: preprocessing -> DES
    final_des_pipeline = Pipeline(fitted_preproc.steps + [("classifier", des_model)])

    # Evaluate on the test set (generalization error)
    logger.info("[COMPUTING GENERALIZATION METRICS]...")
    start_score_time = time.time()

    # Apply decision policy on the test set
    y_test_pred, y_test_pred_proba = apply_decision_policy(
        estimator=final_des_pipeline,
        X=X_test,
        policy_mode=decision_policy_mode,
        cost_matrix=costs_matrix,
    )

    end_score_time = time.time()

    test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        y_pred_proba=y_test_pred_proba,
        cost_matrix=costs_matrix,
    )
    test_metrics["score_time"] = end_score_time - start_score_time
    logger.info(f"[GENERALIZATION METRICS]: {test_metrics}")

    return final_des_pipeline, tuning_results, pool_resubstitution_metrics, test_metrics


def train_and_evaluate_one_fold_all_models(
    run_id: int,
    iteration_idx: int,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    experiment_setting: dict[str, Any],
    config_preprocessing_features: dict[str, list[Any]],
    static_models: Sequence[str],
    static_ensemble_models: Sequence[str],
    static_ensemble_pools: Sequence[str],
    des_models: Sequence[str],
    fs_k_best_to_keep: int | str,
    fs_k_best_candidates: Sequence[int | str] | None,
    tuning_n_iter: int,
    tuning_cv_inner_n_splits: int,
    tuning_scoring: str,
    tuning_n_jobs: int,
    dsel_size: float,
    random_state: int,
    logger: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train and evaluate all STATIC, STATIC-ENSEMBLE, and DES models on a single outer CV fold.

    This orchestrator executes the full workflow for **one** outer split of a repeated
    stratified cross-validation experiment. Given the full dataset (``X``, ``y``) and the
    current outer ``train_idx`` / ``test_idx`` indices, it:

    1) Builds outer-fold train/test datasets and logs class distributions.
    2) Trains and evaluates each STATIC model listed in ``static_models``.
    3) Trains and evaluates each STATIC-ENSEMBLE model listed in ``static_ensemble_models``,
       using the shared base-model pool ``static_ensemble_pools``.
    4) Trains and evaluates each DES model listed in ``des_models`` using a two-stage DES
       workflow (pool tuning on a pool-training subset + DES fitting on a DSEL subset).

    For each trained model, it collects standardized reporting rows for:
    - training-side evaluation (resubstitution / pool-resubstitution), and
    - test-side evaluation (generalization metrics on the outer test fold),

    by delegating row construction to ``collect_fold_reports``.

    The behavior of models/pipelines is driven by ``experiment_setting`` (e.g., class weights,
    resampling strategy, and an optional cost matrix used when ``tuning_scoring="Average_Cost"``).
    The experiment may also specify a decision policy mode (e.g., MEC), which is treated as
    experiment metadata unless thresholding logic is implemented in downstream helpers.

    Parameters
    ----------
    run_id : int
        Global counter for the current outer split, typically produced by
        ``enumerate(cv_outer.split(X, y))``. Used to diversify tuning randomness as
        ``random_state + run_id``.
    iteration_idx : int
        0-based repetition index. Stored as 1-based in output rows via ``iteration_idx + 1``.
    fold_idx : int
        0-based fold index within the repetition. Stored as 1-based in output rows via
        ``fold_idx + 1``.
    train_idx : numpy.ndarray of shape (n_train,)
        Positional indices for samples used as training data for this outer fold.
        These are expected to index rows of ``X`` / ``y`` (e.g., via ``.iloc``).
    test_idx : numpy.ndarray of shape (n_test,)
        Positional indices for samples used as test data for this outer fold.
        These are expected to index rows of ``X`` / ``y`` (e.g., via ``.iloc``).
    X : pandas.DataFrame of shape (n_samples, n_features)
        Full feature matrix. Must be row-aligned with ``y`` and compatible with
        ``config_preprocessing_features`` column-name lists.
    y : pandas.Series of shape (n_samples,)
        Full target vector. Must be row-aligned with ``X``.
    experiment_setting : dict[str, Any]
        Experiment configuration describing the approach, optional imbalance strategy,
        optional decision policy, and the cost matrix used for cost-based tuning/evaluation.
        A typical schema includes::

            {
                "experiment_name": "baseline__mec_fp1_fn10",
                "description": "No resampling, no class_weight. Decision policy: MEC with costs FP=1, FN=10.",
                "approach": "baseline",  # baseline | cost_sensitive_learning | data_level
                "tags": ["baseline", "mec_policy"],
                "class_weight": None,    # None | "balanced" | {0: w0, 1: w1}
                "resampling_method": None,
                "resampling_params": None,
                "decision_policy_mode": "mec",  # standard | mec
                "costs_matrix": COST_MATRIX,
            }

        This function uses, at minimum:
        - ``experiment_setting["experiment_name"]`` for row labeling.
        - ``experiment_setting["class_weight"]`` to configure model factories (STATIC, ENSEMBLE,
          DES pool) when building estimators.
        - ``experiment_setting["resampling_method"]`` and ``experiment_setting["resampling_params"]``
          when building pipelines via ``build_model_pipeline``.
        - ``experiment_setting["costs_matrix"]`` when ``tuning_scoring="Average_Cost"`` is used
          by downstream tuning (see :func:`run_randomized_search_cv`) and/or when downstream
          evaluation includes ``average_cost`` (see :func:`compute_classification_metrics`).

        Notes on ``decision_policy_mode``:
        - ``"standard"`` typically implies the estimator's default decision rule (e.g., 0.5 threshold
          for probabilistic classifiers).
        - ``"mec"`` (minimum expected cost) implies thresholding based on FP/FN costs; however,
          MEC threshold moving is **not applied by this function** unless implemented in downstream
          helpers (e.g., by converting probabilities to hard labels before metric computation).

    config_preprocessing_features : dict[str, list[Any]]
        Column configuration passed to ``build_model_pipeline`` / preprocessing builders.
        Expected keys mirror the preprocessing setup (example):
        - ``"num_log1p_standard_scaler"``, ``"num_standard_scaler"``,
        - ``"cat_nominal"``, ``"cat_ordinal"``, ``"cat_ordinal_order"``,
        - ``"cat_partial_ordinal"``, ``"cat_binary"``.
        Values should be lists of column names (recommended for pandas inputs).
    static_models : Sequence[str]
        Names of static (single-estimator) models to train
        (e.g., ``["SVC", "RandomForestClassifier"]``).
    static_ensemble_models : Sequence[str]
        Names of static ensemble types to train
        (e.g., ``["VotingClassifier", "StackingClassifier"]``).
    static_ensemble_pools : Sequence[str]
        Pool of base-model names used to build each static ensemble in this fold
        (e.g., ``["SVC", "XGBClassifier"]``).
    des_models : Sequence[str]
        Names of DES models to train (e.g., ``["KNORAU", "METADES"]``).
    fs_k_best_to_keep : int or {'all'}
        Default ``k`` used when constructing the ``SelectKBest(k=...)`` step.
        If tuning explores ``"feature_selection_filter__k"``, the fitted value may differ.
    fs_k_best_candidates : Sequence[int | str] or None
        Optional candidate values for ``SelectKBest.k`` to be explored during tuning.
        When provided, candidates are injected into the relevant search spaces as
        ``"feature_selection_filter__k": list(fs_k_best_candidates)``.
    tuning_n_iter : int
        Number of parameter settings sampled in the randomized hyperparameter search.
    tuning_cv_inner_n_splits : int
        Number of stratified folds for the inner CV used during hyperparameter tuning.
    tuning_scoring : str
        Scoring metric used by hyperparameter search.

        - If a standard scikit-learn scorer name is provided (e.g., ``"f1"``,
          ``"average_precision"``, ``"roc_auc"``), it is forwarded to the tuning helpers.
        - If ``"Average_Cost"``, downstream tuning uses the cost-sensitive scorer built from
          ``experiment_setting["costs_matrix"]`` (lower cost is better). Note: internal CV
          scores inside scikit-learn appear negative due to ``greater_is_better=False``, but
          the tuning helper converts them back to positive costs in the returned summaries.
    tuning_n_jobs : int
        Number of parallel jobs used during hyperparameter tuning (inner randomized search).
    dsel_size : float
        Fraction of the outer training set reserved for the DSEL subset used to fit DES
        competence models (must satisfy ``0 < dsel_size < 1``). Used only for DES models.
    random_state : int
        Base random seed forwarded to model factories and splitters. Tuning uses
        ``random_state + run_id`` to diversify randomized search sampling across outer folds.
    logger : Any
        Logger instance exposing at least ``.info(str)`` (and optionally ``.warning(str)``,
        ``.error(str)``).

    Returns
    -------
    resubstitution_rows : list[dict[str, Any]]
        Metrics rows on the training side of the current outer fold.

        - STATIC models: resubstitution metrics on ``(X_train, y_train)``.
        - STATIC-ENSEMBLE models: resubstitution metrics on ``(X_train, y_train)``.
        - DES models: pool-resubstitution metrics computed for the tuned/fitted pool on the
          pool-training subset used inside :func:`train_and_evaluate_one_fold_des_model`.
    generalization_rows : list[dict[str, Any]]
        Metrics rows on the test side of the current outer fold.

        - STATIC models: test metrics on ``(X_test, y_test)``.
        - STATIC-ENSEMBLE models: test metrics on ``(X_test, y_test)``.
        - DES models: test metrics produced by the final DES inference pipeline on
          ``(X_test, y_test)``.

    Raises
    ------
    KeyError
        If required keys are missing from ``experiment_setting`` (e.g., ``"experiment_name"``,
        ``"class_weight"``, ``"resampling_method"``, ``"resampling_params"``, ``"decision_policy_mode"``,
        or ``"costs_matrix"`` when ``tuning_scoring="Average_Cost"``), or if expected pipeline
        step names are missing during feature-name extraction.
    ValueError
        If downstream helpers raise due to invalid CV configuration, incompatible scoring,
        invalid parameter names, or invalid ``dsel_size``.
    AttributeError
        If downstream evaluation requires probability estimates but a fitted estimator/pipeline
        does not expose ``predict_proba`` (depending on your metric implementation).
    Exception
        Any exception raised by estimators, samplers, or scikit-learn model-selection routines
        may propagate.

    Notes
    -----
    - Output rows use 1-based ``iteration`` and ``fold`` indices (``iteration_idx + 1``,
      ``fold_idx + 1``).
    - Feature-selection tuning via ``"feature_selection_filter__k"`` assumes the pipeline step
      is named exactly ``"feature_selection_filter"``.
    - Selected feature names should be extracted from the **fitted best estimator pipeline**
      (e.g., ``best_estimator.named_steps["feature_selection_filter"].get_feature_names_out()``).
    - Any decision-policy logic (e.g., MEC thresholding) is not applied here unless it is
      embedded in the estimator/pipeline or implemented inside the downstream train/evaluate
      helpers.

    Examples
    --------
    Typical usage inside an outer CV loop::

        >>> from sklearn.model_selection import RepeatedStratifiedKFold
        >>> cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
        >>> resub_rows_all, gen_rows_all = [], []
        >>>
        >>> for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
        ...     iteration_idx, fold_idx = divmod(run_id, 10)
        ...     res_rows, gen_rows = train_and_evaluate_one_fold_all_models(
        ...         run_id=run_id,
        ...         iteration_idx=iteration_idx,
        ...         fold_idx=fold_idx,
        ...         train_idx=train_idx,
        ...         test_idx=test_idx,
        ...         X=X,
        ...         y=y,
        ...         experiment_setting=experiment_setting,
        ...         config_preprocessing_features=CONFIG_PREPROCESSING_FEATURES,
        ...         static_models=["SVC", "RandomForestClassifier"],
        ...         static_ensemble_models=["VotingClassifier"],
        ...         static_ensemble_pools=["SVC", "XGBClassifier"],
        ...         des_models=["KNORAU"],
        ...         fs_k_best_to_keep=20,
        ...         fs_k_best_candidates=[10, 20, 30, "all"],
        ...         tuning_n_iter=35,
        ...         tuning_cv_inner_n_splits=5,
        ...         tuning_scoring="average_precision",
        ...         tuning_n_jobs=-1,
        ...         dsel_size=0.2,
        ...         random_state=42,
        ...         logger=logger,
        ...     )
        ...     resub_rows_all.extend(res_rows)
        ...     gen_rows_all.extend(gen_rows)
    """

    resubstitution_rows: List[Dict[str, Any]] = []
    generalization_rows: List[Dict[str, Any]] = []

    # Split the data into training set (9 training folds) and test set (1 test fold)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Report class balance statistics for iteration
    for name, target_arr in zip(["train dataset", "test dataset"], [y_train, y_test]):
        unique, frequency = np.unique(target_arr, return_counts=True)
        logger.info(
            f"Class distribution ({name} statistics) [class, frequency]: {(unique, frequency)}"
        )

    logger.info(f"[ITERATION {iteration_idx + 1:2} - FOLD {fold_idx + 1:2} - RUN_ID {run_id:3}]")

    # ----- Start training STATIC MODELS -----
    for static_model_name in static_models:
        print("-" * 165)
        logger.info(f"Training STATIC model: {static_model_name}")

        # Get the static model estimator and with its hyperparameter search space
        static_model_estimator, static_model_search_space = get_static_model_and_search_space(
            static_model_name,
            random_state=random_state,
            class_weight=experiment_setting["class_weight"],
            y_train=y_train,
        )

        # Add the k candidates for SelectKBest to be tuned with the model
        if fs_k_best_candidates is not None:
            static_model_search_space["feature_selection_filter__k"] = list(fs_k_best_candidates)

        # Build the final pipeline: Preprocessing + Feature Selection + Resampling (OPTIONAL) + Classifier
        static_model_pipeline = build_model_pipeline(
            estimator=static_model_estimator,
            config_preprocessing_features=config_preprocessing_features,
            fs_k_best_to_keep=fs_k_best_to_keep,
            resampling_method=experiment_setting["resampling_method"],
            resampling_params=experiment_setting["resampling_params"],
        )

        # Tune the static model, fit on the training folds and evaluate on the test fold
        best_static_model, tuning_results, resubstitution_metrics, test_metrics = (
            train_and_evaluate_one_fold_static_model(
                experiment_setting=experiment_setting,
                base_model=static_model_pipeline,
                search_space=static_model_search_space,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_iter=tuning_n_iter,
                val_cv_split=tuning_cv_inner_n_splits,
                scoring=tuning_scoring,
                random_state=random_state + run_id,
                n_jobs=tuning_n_jobs,
                logger=logger,
            )
        )

        # Extract selected feature names
        selected_names = get_selected_feature_names(pipeline=best_static_model)

        # Collect resubstitution and generalization metrics
        collect_fold_reports(
            resubstitution_rows=resubstitution_rows,
            generalization_rows=generalization_rows,
            experiment_name=experiment_setting["experiment_name"],
            iteration=iteration_idx + 1,
            fold=fold_idx + 1,
            model_name=static_model_name,
            resubstitution_metrics=resubstitution_metrics,
            test_metrics=test_metrics,
            fold_size_train=len(X_train),
            fold_size_test=len(X_test),
            tuning_results=tuning_results,  # keep tuning info on resub row
            selected_features_names=selected_names,
        )

    # ----- Start training STATIC ENSEMBLE MODELS -----
    for static_ensemble_model_name in static_ensemble_models:
        print("-" * 165)
        logger.info(f"Training STATIC ENSEMBLE model: {static_ensemble_model_name}")

        # Get the static ensemble model estimator with its hyperparameter search space
        static_ensemble_model_estimator, static_ensemble_model_search_space = (
            get_static_ensemble_model_and_search_space(
                ensemble_type=static_ensemble_model_name,
                model_pool=static_ensemble_pools,
                random_state=random_state,
                class_weight=experiment_setting["class_weight"],
                y_train=y_train,
            )
        )

        # Add the k candidates for SelectKBest to be tuned with the model
        if fs_k_best_candidates is not None:
            static_ensemble_model_search_space["feature_selection_filter__k"] = list(
                fs_k_best_candidates
            )

        # Build the final pipeline: Preprocessing + Feature Selection + Resampling (OPTIONAL) + Classifier
        static_ensemble_model_pipeline = build_model_pipeline(
            estimator=static_ensemble_model_estimator,
            config_preprocessing_features=config_preprocessing_features,
            fs_k_best_to_keep=fs_k_best_to_keep,
            resampling_method=experiment_setting["resampling_method"],
            resampling_params=experiment_setting["resampling_params"],
        )

        # Tune the static model, fit on the training folds and evaluate on the test fold
        best_static_ensemble_model, tuning_results, resubstitution_metrics, test_metrics = (
            train_and_evaluate_one_fold_static_model(
                experiment_setting=experiment_setting,
                base_model=static_ensemble_model_pipeline,
                search_space=static_ensemble_model_search_space,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_iter=tuning_n_iter,
                val_cv_split=tuning_cv_inner_n_splits,
                scoring=tuning_scoring,
                random_state=random_state + run_id,
                n_jobs=tuning_n_jobs,
                logger=logger,
            )
        )

        # Extract selected feature names
        selected_names = get_selected_feature_names(pipeline=best_static_ensemble_model)

        # Collect resubstitution and generalization metrics
        collect_fold_reports(
            resubstitution_rows=resubstitution_rows,
            generalization_rows=generalization_rows,
            experiment_name=experiment_setting["experiment_name"],
            iteration=iteration_idx + 1,
            fold=fold_idx + 1,
            model_name=static_ensemble_model_name,
            resubstitution_metrics=resubstitution_metrics,
            test_metrics=test_metrics,
            fold_size_train=len(X_train),
            fold_size_test=len(X_test),
            tuning_results=tuning_results,  # keep tuning info on resub row
            selected_features_names=selected_names,
        )

    # ----- Start training DES MODELS -----
    for des_model_name in des_models:
        print("-" * 165)
        logger.info(f"Training DES model: {des_model_name}")

        # Get the des model estimator and its configuration, with the
        # pool of classifiers and its hyperparameter search space
        pool_classifiers, pool_search_space, des_model_estimator, des_model_conf = get_des_model(
            des_model_name,
            random_state=random_state,
            class_weight=experiment_setting["class_weight"],
            y_train=y_train,
        )

        # Add the k candidates for SelectKBest to be tuned with the model
        if fs_k_best_candidates is not None:
            pool_search_space["feature_selection_filter__k"] = list(fs_k_best_candidates)

        # Build the final pipeline: Preprocessing + Feature Selection + Resampling (OPTIONAL) + Classifier
        pool_classifiers_pipeline = build_model_pipeline(
            estimator=pool_classifiers,
            config_preprocessing_features=config_preprocessing_features,
            fs_k_best_to_keep=fs_k_best_to_keep,
            resampling_method=experiment_setting["resampling_method"],
            resampling_params=experiment_setting["resampling_params"],
        )

        # Tune the des model, fit on the training folds and evaluate on the test fold
        best_des_model, tuning_results, resubstitution_metrics, test_metrics = (
            train_and_evaluate_one_fold_des_model(
                experiment_setting=experiment_setting,
                des_model=des_model_estimator,
                des_conf=des_model_conf,
                pool_classifiers=pool_classifiers_pipeline,
                search_space=pool_search_space,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_iter=tuning_n_iter,
                dsel_size=dsel_size,
                val_cv_split=tuning_cv_inner_n_splits,
                scoring=tuning_scoring,
                random_state=random_state + run_id,
                n_jobs=tuning_n_jobs,
                logger=logger,
            )
        )

        # Extract selected feature names
        selected_names = get_selected_feature_names(pipeline=best_des_model)

        # Collect resubstitution and generalization metrics
        collect_fold_reports(
            resubstitution_rows=resubstitution_rows,
            generalization_rows=generalization_rows,
            experiment_name=experiment_setting["experiment_name"],
            iteration=iteration_idx + 1,
            fold=fold_idx + 1,
            model_name=des_model_name,
            resubstitution_metrics=resubstitution_metrics,
            test_metrics=test_metrics,
            fold_size_train=len(X_train),
            fold_size_test=len(X_test),
            tuning_results=tuning_results,
            selected_features_names=selected_names,
        )

    logger.info(
        f"Completed [ITERATION {iteration_idx + 1} - FOLD {fold_idx + 1}] - RUN_ID {run_id}]"
    )

    return resubstitution_rows, generalization_rows
