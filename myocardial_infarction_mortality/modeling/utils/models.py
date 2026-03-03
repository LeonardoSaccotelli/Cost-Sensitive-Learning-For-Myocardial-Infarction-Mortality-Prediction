from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

from deslib.dcs import LCA, MLA, OLA, APosteriori, APriori
from deslib.des import DESKNN, DESP, KNOP, KNORAE, KNORAU, METADES, DESClustering
from deslib.des.probabilistic import DESKL, RRC, Exponential, Logarithmic
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def _tree_common_param_space(
    prefix: str = "classifier__",
    max_depth_min: int = 3,
    max_depth_max: int = 20,
    min_samples_split_min: int = 2,
    min_samples_split_max: int = 10,
    min_samples_leaf_min: int = 1,
    min_samples_leaf_max: int = 10,
    max_leaf_nodes_min: int = 2,
    max_leaf_nodes_max: int = 20,
    min_impurity_decrease_min: float = 0.0,
    min_impurity_decrease_max: float = 0.1,
    ccp_alpha_min: float = 0.0,
    ccp_alpha_max: float = 0.01,
    max_features_choices: Iterable[str] = ("sqrt", "log2"),
) -> Dict[str, Any]:
    """
    Build a shared hyperparameter search space for tree-based classifiers.

    This helper returns a ``param_distributions`` dictionary suitable for
    ``RandomizedSearchCV`` (or compatible search utilities). Integer-valued
    parameters are sampled using ``scipy.stats.randint(a, b)`` (support ``[a, b)``),
    while continuous parameters are sampled using ``scipy.stats.uniform(loc, scale)``
    (support ``[loc, loc + scale)``). Parameter keys are prefixed (typically with a
    pipeline step name such as ``'classifier__'``).

    Parameters
    ----------
    prefix : str, default 'classifier__'
        Prefix prepended to every parameter name (e.g., pipeline step name plus ``'__'``).
    max_depth_min : int, default 3
        Minimum value for ``max_depth`` (inclusive).
    max_depth_max : int, default 20
        Maximum value for ``max_depth`` (exclusive).
    min_samples_split_min : int, default 2
        Minimum value for ``min_samples_split`` (inclusive).
    min_samples_split_max : int, default 10
        Maximum value for ``min_samples_split`` (exclusive).
    min_samples_leaf_min : int, default 1
        Minimum value for ``min_samples_leaf`` (inclusive).
    min_samples_leaf_max : int, default 10
        Maximum value for ``min_samples_leaf`` (exclusive).
    max_leaf_nodes_min : int, default 2
        Minimum value for ``max_leaf_nodes`` (inclusive).
    max_leaf_nodes_max : int, default 20
        Maximum value for ``max_leaf_nodes`` (exclusive).
    min_impurity_decrease_min : float, default 0.0
        Lower bound for ``min_impurity_decrease`` (inclusive).
    min_impurity_decrease_max : float, default 0.1
        Upper bound for ``min_impurity_decrease`` (exclusive).
    ccp_alpha_min : float, default 0.0
        Lower bound for ``ccp_alpha`` (inclusive).
    ccp_alpha_max : float, default 0.01
        Upper bound for ``ccp_alpha`` (exclusive).
    max_features_choices : Iterable[str], default ('sqrt', 'log2')
        Categorical choices for ``max_features``.

    Returns
    -------
    param_distributions : dict[str, Any]
        Mapping from prefixed parameter names to SciPy distributions / categorical lists:
        - ``<prefix>max_depth`` : ``randint``
        - ``<prefix>min_samples_split`` : ``randint``
        - ``<prefix>min_samples_leaf`` : ``randint``
        - ``<prefix>max_features`` : ``list[str]``
        - ``<prefix>max_leaf_nodes`` : ``randint``
        - ``<prefix>min_impurity_decrease`` : ``uniform``
        - ``<prefix>ccp_alpha`` : ``uniform``

    Notes
    -----
    - For ``randint(a, b)``, SciPy samples integers in ``[a, b)`` (upper bound excluded).
    - For ``uniform(loc, scale)``, SciPy samples continuous values in
      ``[loc, loc + scale)``.
    - The returned dictionary is designed to be merged into a larger search space
      for composite estimators (e.g., pipelines).

    Examples
    --------
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> space = _tree_common_param_space(prefix="classifier__")
    >>> # search = RandomizedSearchCV(pipe, param_distributions=space, n_iter=30, cv=5)
    """

    return {
        f"{prefix}max_depth": randint(max_depth_min, max_depth_max),
        f"{prefix}min_samples_split": randint(min_samples_split_min, min_samples_split_max),
        f"{prefix}min_samples_leaf": randint(min_samples_leaf_min, min_samples_leaf_max),
        f"{prefix}max_features": list(max_features_choices),
        f"{prefix}max_leaf_nodes": randint(max_leaf_nodes_min, max_leaf_nodes_max),
        f"{prefix}min_impurity_decrease": uniform(
            min_impurity_decrease_min,
            min_impurity_decrease_max - min_impurity_decrease_min,
        ),
        f"{prefix}ccp_alpha": uniform(
            ccp_alpha_min,
            ccp_alpha_max - ccp_alpha_min,
        ),
    }


def _boosting_core_param_space(
    prefix: str = "classifier__",
    n_estimators_min: int = 100,
    n_estimators_max: int = 1000,
    learning_rate_min: float = 1e-3,
    learning_rate_max: float = 1.0,
) -> Dict[str, Any]:
    """
    Build a shared hyperparameter search space for boosting core parameters.

    This helper returns a ``param_distributions`` dictionary for the two
    fundamental hyperparameters used by most boosting estimators: the number of
    boosting stages (``n_estimators``) and the shrinkage factor
    (``learning_rate``). Keys are prefixed (typically with a pipeline step name
    such as ``'classifier__'``) so the result can be used directly in
    scikit-learn model-selection utilities.

    Parameters
    ----------
    prefix : str, default 'classifier__'
        Prefix prepended to every parameter name (e.g., pipeline step name plus ``'__'``).
    n_estimators_min : int, default 100
        Minimum value for ``n_estimators`` (inclusive).
    n_estimators_max : int, default 1000
        Maximum value for ``n_estimators`` (exclusive).
    learning_rate_min : float, default 1e-3
        Lower bound for ``learning_rate`` (strictly positive).
    learning_rate_max : float, default 1.0
        Upper bound for ``learning_rate`` (must be greater than ``learning_rate_min``).

    Returns
    -------
    param_distributions : dict[str, Any]
        Mapping from prefixed parameter names to SciPy distributions:
        - ``<prefix>n_estimators`` : ``scipy.stats.randint``
        - ``<prefix>learning_rate`` : ``scipy.stats.loguniform``

    Notes
    -----
    - ``randint(a, b)`` samples integers in ``[a, b)`` (upper bound excluded).
    - ``loguniform(a, b)`` samples positive values in ``[a, b)`` on a log scale.
    - Smaller ``learning_rate`` typically requires larger ``n_estimators`` for
      comparable training error, but can improve regularization.

    Examples
    --------
    >>> space = _boosting_core_param_space(prefix="classifier__")
    >>> sorted(space.keys())
    ['classifier__learning_rate', 'classifier__n_estimators']
    """

    return {
        f"{prefix}n_estimators": randint(n_estimators_min, n_estimators_max),
        f"{prefix}learning_rate": loguniform(learning_rate_min, learning_rate_max),
    }


def compute_xgboost_weight_policy(
    class_weight: dict[int, float] | str | None,
    y_train: pd.Series,
    *,
    pos_label: int = 1,
    neg_label: int = 0,
    set_max_delta_step: bool = True,
) -> Dict[str, Any]:
    """
    Compute XGBoost imbalance parameters from a sklearn-like ``class_weight`` setting.

    This function maps the experiment-level ``class_weight`` policy to XGBoost parameters
    without using ``sample_weight`` in ``fit``. It is designed for the "option 1" approach:
    compute the policy once using the outer-fold ``y_train`` and keep it fixed during inner CV.

    Mapping
    -------
    - ``class_weight is None``:
      Baseline. Returns ``scale_pos_weight=1.0`` and (optionally) ``max_delta_step=0``.
    - ``class_weight == "balanced"``:
      Prevalence-based weighting from ``y_train``:
      ``scale_pos_weight = n_neg / n_pos``.
    - ``class_weight is dict``:
      Business weights:
      ``scale_pos_weight = w_pos / w_neg`` where weights come from the dict (keys are labels).

    Parameters
    ----------
    class_weight : dict[int, float] or {'balanced'} or None
        Sklearn-like class weight specification:
        - None: no weighting
        - 'balanced': automatic prevalence-based ratio (n_neg / n_pos) computed from ``y_train``
        - dict: explicit weights, expected keys include ``neg_label`` and ``pos_label``
    y_train : pandas.Series
        Target labels for the current outer training fold.
    pos_label : int, default 1
        Label treated as the positive class.
    neg_label : int, default 0
        Label treated as the negative class.
    set_max_delta_step : bool, default True
        If True, also return ``max_delta_step``:
        - 0 for baseline
        - 1 for balanced/dict policies (stability aid)

    Returns
    -------
    xgb_params : dict[str, Any]
        Parameters to inject into ``XGBClassifier`` via ``set_params``:
        - ``scale_pos_weight`` : float
        - optionally ``max_delta_step`` : int

    Raises
    ------
    ValueError
        If the 'balanced' policy cannot be computed due to missing classes, or if dict
        weights are invalid/missing required labels.

    Examples
    --------
    >>> import pandas as pd
    >>> y = pd.Series([0, 0, 0, 1])
    >>> compute_xgboost_weight_policy("balanced", y)["scale_pos_weight"]
    3.0

    >>> y = pd.Series([0, 1, 0, 1])
    >>> compute_xgboost_weight_policy({0: 2.0, 1: 10.0}, y)["scale_pos_weight"]
    5.0
    """
    y_arr = y_train.to_numpy()

    # Baseline
    if class_weight is None:
        out: Dict[str, Any] = {"scale_pos_weight": 1.0}
        if set_max_delta_step:
            out["max_delta_step"] = 0
        return out

    # Balanced heuristic: n_neg / n_pos
    if class_weight == "balanced":
        n_pos = int(np.sum(y_arr == pos_label))
        n_neg = int(np.sum(y_arr == neg_label))

        if n_pos <= 0:
            raise ValueError(
                f"Cannot compute scale_pos_weight: no positive samples (label={pos_label}) in y_train."
            )
        if n_neg <= 0:
            raise ValueError(
                f"Cannot compute scale_pos_weight: no negative samples (label={neg_label}) in y_train."
            )

        out = {"scale_pos_weight": float(n_neg) / float(n_pos)}
        if set_max_delta_step:
            out["max_delta_step"] = 1
        return out

    # Business weights: w_pos / w_neg
    if isinstance(class_weight, dict):
        if neg_label not in class_weight or pos_label not in class_weight:
            raise ValueError(
                f"class_weight dict must contain keys {{{neg_label}, {pos_label}}} "
                f"(got keys={sorted(class_weight.keys())})."
            )

        w_neg = float(class_weight[neg_label])
        w_pos = float(class_weight[pos_label])
        if w_neg <= 0.0 or w_pos <= 0.0:
            raise ValueError("class_weight values must be positive.")

        out = {"scale_pos_weight": w_pos / w_neg}
        if set_max_delta_step:
            out["max_delta_step"] = 1
        return out

    raise ValueError("class_weight must be None, 'balanced', or a dict of weights.")


def get_static_model_and_search_space(
    model_name: str,
    y_train: pd.Series,
    random_state: int | None = None,
    class_weight: dict[int, float] | str | None = None,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    """
    Instantiate a static classifier and its estimator-level hyperparameter search space.

    This factory returns:
    1) an unfitted estimator instance, and
    2) an estimator-only ``param_distributions`` dictionary suitable for CV-based hyperparameter
       search (e.g., ``RandomizedSearchCV`` / ``HalvingRandomSearchCV``).

    Returned parameter names are prefixed for a pipeline step named ``"classifier"``
    (e.g., ``classifier__C``). The returned search space intentionally excludes pipeline-level
    parameters (e.g., ``feature_selection_filter__k``); those must be added by the orchestration
    layer that builds the full pipeline.

    Class-weight policy
    -------------------
    The argument ``class_weight`` mirrors the experiment configuration and can be:
    - ``None``: baseline (no class weighting),
    - ``"balanced"``: scikit-learn heuristic class weighting (where supported),
    - ``{0: w0, 1: w1}``: explicit business-rule weights (where supported).

    XGBoost special handling
    ------------------------
    ``XGBClassifier`` does not support ``class_weight`` directly in the same way as scikit-learn
    estimators. To emulate the same policy without passing ``sample_weight`` to ``fit``, this
    factory injects an XGBoost weighting policy derived from ``class_weight`` and the provided
    outer-fold ``y_train``:

    - ``class_weight is None``:
      ``scale_pos_weight = 1.0`` and ``max_delta_step = 0``.
    - ``class_weight == "balanced"``:
      ``scale_pos_weight = n_neg / n_pos`` computed from ``y_train`` and ``max_delta_step = 1``.
    - ``class_weight is dict`` (e.g., ``{0: w0, 1: w1}``):
      ``scale_pos_weight = w1 / w0`` and ``max_delta_step = 1``.

    The XGBoost parameters ``scale_pos_weight`` and ``max_delta_step`` are intentionally not
    included in the returned search space, so that hyperparameter tuning cannot override the
    experiment policy.

    Parameters
    ----------
    model_name : str
        Canonical model key identifying which estimator to build. Supported keys are those
        defined by the internal registry (e.g., ``"LogisticRegression"``, ``"SVC"``,
        ``"DecisionTreeClassifier"``, ``"RandomForestClassifier"``, ``"BaggingDecisionTreeClassifier"``,
        ``"XGBClassifier"``).
    y_train : pandas.Series
        Target labels for the current **outer-fold training set**. This is used to compute the
        prevalence ratio for XGBoost when ``class_weight="balanced"``. It is otherwise ignored
        for non-XGBoost models.
    random_state : int or None, default None
        Random seed forwarded to estimators that support it.
    class_weight : dict[int, float] or {'balanced'} or None, default None
        Class-weight policy:
        - ``None``: baseline (no weighting),
        - ``"balanced"``: scikit-learn heuristic weighting (where supported),
        - ``{0: w0, 1: w1}``: explicit business-rule weights (where supported).

    Returns
    -------
    estimator : sklearn.base.BaseEstimator
        Unfitted classifier instance configured according to ``model_name``, ``random_state``,
        and (where applicable) the provided weighting policy.
    param_dist : dict[str, Any]
        Estimator-level hyperparameter search space (SciPy distributions / categorical lists).
        Keys are prefixed with ``"classifier__"`` to match a pipeline step named ``"classifier"``.

    Raises
    ------
    ValueError
        If ``model_name`` is not supported, or if the XGBoost weight policy cannot be computed
        (e.g., missing classes in ``y_train`` for ``class_weight="balanced"`` or invalid dict weights).

    Notes
    -----
    - The returned ``param_dist`` is estimator-only by design; add pipeline-level parameters
      (e.g., feature selection ``k``) outside this function.
    - XGBoost imbalance handling is enforced deterministically through ``scale_pos_weight`` and
      is computed once from the outer-fold ``y_train`` (kept fixed during inner CV).

    Examples
    --------
    Baseline (no class weighting)::

        >>> import pandas as pd
        >>> y_tr = pd.Series([0, 1, 0, 1])
        >>> clf, space = get_static_model_and_search_space(
        ...     model_name="RandomForestClassifier",
        ...     y_train=y_tr,
        ...     random_state=42,
        ...     class_weight=None,
        ... )

    Sklearn heuristic weighting::

        >>> y_tr = pd.Series([0, 0, 0, 1])
        >>> clf, space = get_static_model_and_search_space(
        ...     model_name="SVC",
        ...     y_train=y_tr,
        ...     random_state=42,
        ...     class_weight="balanced",
        ... )

    XGBoost business-rule weighting::

        >>> y_tr = pd.Series([0, 0, 1, 0, 1])
        >>> clf, space = get_static_model_and_search_space(
        ...     model_name="XGBClassifier",
        ...     y_train=y_tr,
        ...     random_state=42,
        ...     class_weight={0: 1.0, 1: 10.0},
        ... )
    """

    model_configurations = {
        "LogisticRegression": {
            "model_class": LogisticRegression,
            "model_args": {
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "tol": 1e-4,
                "fit_intercept": True,
                "random_state": random_state,
                "n_jobs": 1,  # avoid nested parallelism
                "class_weight": class_weight,  # None | "balanced" | {0:w0, 1:w1}
            },
            "param_dist": {
                # Inverse regularization strength (smaller -> stronger regularization)
                "classifier__C": loguniform(1e-4, 1e4),
                # Optionally tune tolerance / intercept
                "classifier__tol": loguniform(1e-5, 1e-2),
                "classifier__fit_intercept": [True, False],
            },
        },
        "SVC": {
            "model_class": SVC,
            "model_args": {
                "coef0": 0.0,
                "shrinking": True,
                "probability": True,
                "tol": 1e-3,
                "cache_size": 200,
                "verbose": False,
                "max_iter": -1,
                "random_state": random_state,
                "class_weight": class_weight,
            },
            "param_dist": {
                # Regularization parameter. Smaller values specify stronger regularization.
                "classifier__C": loguniform(1e-3, 1e3),
                # Kernel coefficient for 'rbf' and 'poly' kernels.
                "classifier__gamma": loguniform(1e-4, 1e0),
                # Kernel type
                "classifier__kernel": ["rbf", "poly", "linear"],
                # Degree of the polynomial kernel function (only relevant if kernel='poly').
                "classifier__degree": randint(2, 5),
            },
        },
        "DecisionTreeClassifier": {
            "model_class": DecisionTreeClassifier,
            "model_args": {
                # The function to measure the quality of a split.
                "criterion": "gini",
                "splitter": "best",
                "random_state": random_state,
                "class_weight": class_weight,
            },
            "param_dist": _tree_common_param_space(),
        },
        "RandomForestClassifier": {
            "model_class": RandomForestClassifier,
            "model_args": {
                # The function to measure the quality of a split.
                "criterion": "gini",
                # Bootstrapping (sampling with replacement) enabled.
                "bootstrap": True,
                "oob_score": False,
                "n_jobs": 1,
                "random_state": random_state,
                "class_weight": class_weight,
            },
            "param_dist": {
                # Number of trees in the forest.
                "classifier__n_estimators": randint(100, 1000),
                # Controls the size of the bootstrap sample (the subset of data)
                # used to train each individual decision tree in the forest: [0.5, 0.5 + 0.4] = [0.5, 0.9]
                "classifier__max_samples": uniform(0.5, 0.4),
                **_tree_common_param_space(),
            },
        },
        "BaggingDecisionTreeClassifier": {
            "model_class": BaggingClassifier,
            "model_args": {
                "estimator": DecisionTreeClassifier(
                    # The function to measure the quality of a split.
                    criterion="gini",
                    splitter="best",
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            },
            "param_dist": {
                # --- Bagging-level hyperparameters ---
                # Number of trees in the ensemble
                "classifier__n_estimators": randint(100, 1000),
                # Fraction of samples used per base estimator: [0.5, 1.0)
                "classifier__max_samples": uniform(0.5, 0.5),
                # Fraction of features used per base estimator: [0.5, 1.0)
                "classifier__max_features": uniform(0.5, 0.5),
                # --- Internal DecisionTree hyperparameters ---
                # Reuse the common tree space but for the *internal* estimator
                # i.e. "classifier__estimator__max_depth", etc.
                **_tree_common_param_space(prefix="classifier__estimator__"),
            },
        },
        "XGBClassifier": {
            "model_class": XGBClassifier,
            "model_args": {
                # Binary classification with logistic loss.
                "objective": "binary:logistic",
                # Consistent with binary:logistic.
                "eval_metric": "logloss",
                "n_jobs": 1,
                "random_state": random_state,
                # NOTE: scale_pos_weight / max_delta_step injected below (deterministic)
            },
            "param_dist": {
                **_boosting_core_param_space(
                    n_estimators_min=200,
                    n_estimators_max=800,
                    learning_rate_min=1e-2,
                    learning_rate_max=0.2,
                ),
                # Maximum tree depth — lower = less overfitting.
                "classifier__max_depth": randint(3, 10),
                # Fraction of samples per tree. Helps generalization: [0.6, 0.6 + 0.4] = [0.6, 1.0]
                "classifier__subsample": uniform(0.6, 0.4),
                # Fraction of features per tree. Avoids co-adaptation.
                "classifier__colsample_bytree": uniform(0.6, 0.4),
                # Minimum loss reduction for a split. Acts as regularization.
                "classifier__gamma": uniform(0.0, 5.0),
                # L1 regularization on weights.
                "classifier__reg_alpha": loguniform(1e-4, 10.0),
                # L2 regularization on weights.
                "classifier__reg_lambda": loguniform(1e-4, 10.0),
                # Minimum sum of instance weight (hessian) in child.
                "classifier__min_child_weight": randint(1, 10),
            },
        },
    }

    if model_name not in model_configurations:
        raise ValueError(f"Unknown model name: {model_name}")

    config = model_configurations[model_name]

    # XGBoost-only: inject weight policy derived from class_weight + y_train
    if model_name == "XGBClassifier":
        xgb_policy = compute_xgboost_weight_policy(
            class_weight=class_weight,
            y_train=y_train,
        )
        config["model_args"].update(xgb_policy)

    model = config["model_class"](**config["model_args"])
    param_dist: Dict[str, Any] = dict(config["param_dist"])
    return model, param_dist


def get_static_ensemble_model_and_search_space(
    ensemble_type: str,
    y_train: pd.Series,
    model_pool: Sequence[str],
    random_state: int | None = None,
    class_weight: dict[int, float] | str | None = None,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    """
    Instantiate a static ensemble (voting/stacking) and a merged nested search space.

    This factory builds an unfitted ensemble estimator from a list of base-model identifiers
    and returns a single merged ``param_distributions`` dictionary that targets each nested
    sub-estimator using scikit-learn's parameter routing (e.g., ``classifier__svc_0__C``).

    The function:
    1) calls :func:`get_static_model_and_search_space` for each entry in ``model_pool``,
    2) assigns a unique name to each estimator instance (duplicates allowed), and
    3) rewrites base search-space keys from ``classifier__<param>`` into
       ``classifier__<est_name>__<param>`` so the result can be used to tune the ensemble
       when it is placed under a pipeline step named ``"classifier"``.

    Class-weight policy
    -------------------
    The argument ``class_weight`` mirrors the experiment configuration and can be:
    - ``None``: baseline (no class weighting),
    - ``"balanced"``: scikit-learn heuristic class weighting (where supported),
    - ``{0: w0, 1: w1}``: explicit business-rule weights (where supported).

    This policy is forwarded to all base estimators via the base factory. For
    ``StackingClassifier``, the same ``class_weight`` is also applied to the meta-learner
    (``LogisticRegression``).

    XGBoost special handling
    ------------------------
    If ``"XGBClassifier"`` is included in ``model_pool``, the base factory uses ``y_train`` to
    inject XGBoost imbalance parameters (e.g., ``scale_pos_weight`` / ``max_delta_step``)
    according to the same ``class_weight`` policy. Therefore, ``y_train`` must correspond to
    the current outer-fold training labels.

    Parameters
    ----------
    ensemble_type : str
        Ensemble type to instantiate. Supported values are:
        - ``"VotingClassifier"``: soft voting over probabilities,
        - ``"StackingClassifier"``: stacked generalization with a logistic-regression meta-learner.
    y_train : pandas.Series
        Target labels for the current **outer-fold training set**. This is forwarded to
        :func:`get_static_model_and_search_space` so XGBoost base estimators can compute a
        deterministic weighting policy when ``class_weight="balanced"``.
    model_pool : Sequence[str]
        Base-model identifiers to include (e.g., ``["SVC", "XGBClassifier"]``).
        Duplicates are allowed; each occurrence becomes a distinct estimator instance with a
        unique internal name (e.g., ``"svc_0"``, ``"svc_1"``).
    random_state : int or None, default None
        Random seed forwarded to base estimators (via the base factory) and to the stacking
        meta-learner when applicable.
    class_weight : dict[int, float] or {'balanced'} or None, default None
        Class-weight policy forwarded to base estimators (and to the stacking meta-learner).
        See "Class-weight policy" above.

    Returns
    -------
    estimator : sklearn.base.BaseEstimator
        Unfitted ensemble estimator instance:
        - ``sklearn.ensemble.VotingClassifier`` when ``ensemble_type="VotingClassifier"``,
        - ``sklearn.ensemble.StackingClassifier`` when ``ensemble_type="StackingClassifier"``.
    param_dist : dict[str, Any]
        Merged nested hyperparameter search space for all base estimators. Keys target the
        sub-estimators inside the ensemble assuming the ensemble is mounted under a pipeline
        step named ``"classifier"``. Example rewrite:
        - input key: ``"classifier__C"``
        - output key: ``"classifier__svc_0__C"``

    Raises
    ------
    ValueError
        If ``model_pool`` is empty or if ``ensemble_type`` is not supported.

    Notes
    -----
    - Soft voting requires each base estimator to implement ``predict_proba``; ensure your
      base factory configures probabilistic outputs where needed (e.g., ``SVC(probability=True)``).
    - Key rewriting assumes the base factory emits keys prefixed with ``"classifier__"``.
      If you change that prefix, update the rewriting logic accordingly.
    - The ensemble ``n_jobs`` value is set in code; ensure it matches your parallelism strategy
      to avoid nested parallelism on multi-core/HPC environments.

    Examples
    --------
    Build a soft-voting ensemble and obtain the nested search space::

        >>> import pandas as pd
        >>> y_tr = pd.Series([0, 1, 0, 1])
        >>> ens, space = get_static_ensemble_model_and_search_space(
        ...     ensemble_type="VotingClassifier",
        ...     y_train=y_tr,
        ...     model_pool=["SVC", "XGBClassifier"],
        ...     random_state=42,
        ...     class_weight="balanced",
        ... )
        >>> any(k.startswith("classifier__svc_0__") for k in space)
        True
    """

    if not model_pool:
        raise ValueError("model_pool list cannot be empty.")

    estimators = []
    ensemble_param_dist = {}

    # 1. Build Base Estimators and Merge Spaces
    for idx, model_name in enumerate(model_pool):
        # Retrieve the base model and its specific search space
        base_model, base_space = get_static_model_and_search_space(
            model_name,
            random_state=random_state,
            class_weight=class_weight,
            y_train=y_train,
        )

        # Create a unique name for this estimator instance (e.g., 'xgbclassifier_0')
        # This name is crucial for the scikit-learn parameter routing.
        est_name = f"{model_name.lower()}_{idx}"
        estimators.append((est_name, base_model))

        # Rewrite search space keys.
        # Original: "classifier__max_depth"
        # Target (inside Pipeline > Ensemble): "classifier__<est_name>__max_depth"
        for key, distribution in base_space.items():
            # Remove the standard prefix provided by the factory function
            # We assume the factory returns keys starting with "classifier__"
            clean_param = key.removeprefix("classifier__")

            # Construct the new nested key
            new_key = f"classifier__{est_name}__{clean_param}"
            ensemble_param_dist[new_key] = distribution

    # 2. Construct the Ensemble
    if ensemble_type == "VotingClassifier":
        # Soft voting returns the class label as argmax of the sum of predicted probabilities.
        # This requires 'probability=True' in SVC (handled in base factory).
        model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=len(model_pool))

    elif ensemble_type == "StackingClassifier":
        # Define the meta-learner
        final_estimator = LogisticRegression(
            random_state=random_state,
            solver="lbfgs",
            penalty="l2",
            max_iter=1000,
            tol=1e-4,
            fit_intercept=True,
            class_weight=class_weight,
        )

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=len(model_pool),
            passthrough=False,  # 'passthrough': False -> Train meta-model only on predictions of base models
            cv=5,  # Internal CV for training the meta-model
        )
    else:
        raise ValueError(
            f"Unknown ensemble_type: {ensemble_type}. Use 'VotingClassifier' or 'StackingClassifier'."
        )

    return model, ensemble_param_dist


def get_des_model(
    model_name: str,
    y_train: pd.Series,
    random_state: int | None = None,
    class_weight: dict[int, float] | str | None = None,
) -> Tuple[BaseEstimator, Dict[str, Any], BaseEstimator, Dict[str, Any]]:
    """
    Instantiate the pool (bagging) model and a DESlib estimator configuration for DES.

    This factory returns two coupled components required by the Dynamic Ensemble Selection (DES)
    workflow used in this project:

    1) **Pool model (bagging) + its hyperparameter search space**
       The pool is intended to be tuned as the ``"classifier"`` step of the standard
       training pipeline:

       ``preprocessing → feature_selection → resampling → pool``

       In this implementation, the pool model is a ``BaggingDecisionTreeClassifier`` obtained
       via :func:`get_static_model_and_search_space`.

    2) **DESlib model (unfitted) + a dict of default kwargs**
       The DESlib estimator is returned unfitted, while its default configuration is returned
       separately as a dictionary. The DES training routine is expected to:
       - inject the tuned pool via ``pool_classifiers=...`` (or ``fitted_pool.estimators_`` when required),
       - apply the remaining DES parameters via ``des_model.set_params(**des_kwargs)``,
       - fit the DES model on the DSEL subset.

    Parameters
    ----------
    model_name : str
        DES method identifier. Supported values are the keys of the internal DES registry
        (e.g., ``"KNORAE"``, ``"KNORAU"``, ``"DESP"``, ``"METADES"``, etc.).
    y_train : pandas.Series
        Target labels for the current **outer-fold training set**. This is forwarded to
        :func:`get_static_model_and_search_space` when building the pool, enabling any
        fold-aware imbalance policy needed by specific base models (notably XGBoost, if ever
        used inside the pool factory).
    random_state : int or None, default None
        Random seed forwarded to the pool factory. The DESlib estimator itself is instantiated
        without constructor kwargs here; if a DES method exposes randomness control, apply it
        later through ``des_kwargs`` (and/or explicit overrides).
    class_weight : dict[int, float] or {'balanced'} or None, default None
        Class-weight policy forwarded to the pool factory. This mirrors the experiment setting:
        - ``None``: baseline (no weighting),
        - ``"balanced"``: heuristic weighting (where supported),
        - ``{0: w0, 1: w1}``: explicit business-rule weights (where supported).

    Returns
    -------
    pool_estimator : sklearn.base.BaseEstimator
        Unfitted bagging ensemble used as the pool of classifiers. Intended to be placed under a
        pipeline step named ``"classifier"`` during the pool-tuning stage.
    pool_param_dist : dict[str, Any]
        Hyperparameter search space for the pool, compatible with CV search over a pipeline where
        the pool sits in the ``"classifier"`` step (e.g., ``classifier__n_estimators``,
        ``classifier__max_samples``, ``classifier__estimator__max_depth``).
    des_model : sklearn.base.BaseEstimator
        Unfitted DESlib estimator instance corresponding to ``model_name``.
    des_kwargs : dict[str, Any]
        Default keyword arguments for the DES model (e.g., ``k``, ``DFP``, ``IH_rate``, ``voting``,
        ``n_jobs``). This dictionary does **not** include ``pool_classifiers``; callers typically
        add it before fitting. If you mutate this dictionary, copy it first to avoid unintended
        cross-call side effects.

    Raises
    ------
    ValueError
        If ``model_name`` is not a supported DES identifier.

    Notes
    -----
    - This function returns an **estimator-only** pool search space. Pipeline-level search keys
      (e.g., ``feature_selection_filter__k``) must be added by the orchestration layer that
      constructs the full pool pipeline.
    - DESlib methods may require either the fitted bagging object or a list/array of base
      estimators as ``pool_classifiers``. If needed, pass ``fitted_pool.estimators_`` instead
      of the bagger instance.
    - Typical two-stage workflow:
      (i) tune/fit the pool on TRAIN,
      (ii) fit the DES model on DSEL with ``pool_classifiers`` injected and ``des_kwargs`` applied.

    Examples
    --------
    Instantiate the pool configuration and a DESlib method configuration::

        >>> import pandas as pd
        >>> y_tr = pd.Series([0, 1, 0, 1])
        >>> pool_est, pool_space, des, des_kwargs = get_des_model(
        ...     model_name="KNORAE",
        ...     y_train=y_tr,
        ...     random_state=42,
        ...     class_weight="balanced",
        ... )
        >>> "classifier__n_estimators" in pool_space
        True
    """

    # Pool: BaggingDecisionTreeClassifier + its search space
    pool_estimator, pool_param_dist = get_static_model_and_search_space(
        model_name="BaggingDecisionTreeClassifier",
        random_state=random_state,
        class_weight=class_weight,
        y_train=y_train,
    )

    # DES model configuration (class + default kwargs)
    des_model_configurations = {
        "APriori": {
            "model_class": APriori,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "APosteriori": {
            "model_class": APosteriori,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "LCA": {
            "model_class": LCA,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "MLA": {
            "model_class": MLA,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "OLA": {
            "model_class": OLA,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "selection_method": "best",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
            },
        },
        "KNORAE": {
            "model_class": KNORAE,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "KNORAU": {
            "model_class": KNORAU,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESP": {
            "model_class": DESP,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESKNN": {
            "model_class": DESKNN,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "pct_accuracy": 0.5,
                "pct_diversity": 0.3,
                "more_diverse": True,
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESClustering": {
            "model_class": DESClustering,
            "model_args": {
                "pct_accuracy": 0.5,
                "pct_diversity": 0.3,
                "more_diverse": True,
                "metric_performance": "accuracy_score",
                "n_clusters": 5,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "KNOP": {
            "model_class": KNOP,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "knn_classifier": "knn",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
        "DESKL": {
            "model_class": DESKL,
            "model_args": {
                "k": 8,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "Exponential": {
            "model_class": Exponential,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "Logarithmic": {
            "model_class": Logarithmic,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "RRC": {
            "model_class": RRC,
            "model_args": {
                "k": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "voting": "soft",
                "n_jobs": 1,
            },
        },
        "METADES": {
            "model_class": METADES,
            "model_args": {
                "k": 8,
                "Kp": 8,
                "DFP": True,
                "IH_rate": 0.3,
                "mode": "selection",
                "knn_classifier": "knn",
                "knn_metric": "minkowski",
                "knne": True,
                "n_jobs": 1,
                "voting": "soft",
            },
        },
    }

    if model_name not in des_model_configurations:
        raise ValueError(f"Unknown DES model name: {model_name}")

    des_config = des_model_configurations[model_name]
    model = des_config["model_class"]()
    model_args = dict(des_config["model_args"])

    return pool_estimator, pool_param_dist, model, model_args
