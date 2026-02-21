from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from imblearn.pipeline import Pipeline as ImbPipeline
import sklearn
from sklearn.base import BaseEstimator

from myocardial_infarction_mortality.data_preparation.data_construct import (
    get_preprocessing_pipeline,
)
from myocardial_infarction_mortality.data_preparation.feature_selection import (
    get_feature_selection,
)
from myocardial_infarction_mortality.data_preparation.sampling import get_resampling_pipeline


def build_model_pipeline(
    estimator: BaseEstimator,
    config_preprocessing_features: dict[str, list[Any]],
    fs_k_best_to_keep: Union[int, str],
    resampling_method: Optional[str],
    resampling_params: Optional[Mapping[str, Any]],
) -> ImbPipeline:
    """
    Build a leakage-safe imbalanced-learn pipeline for binary classification.

    The returned :class:`imblearn.pipeline.Pipeline` is intended for cross-validation and
    hyperparameter tuning. All data-dependent steps (preprocessing, feature selection, and
    optional resampling) are fit/applied **only on the training split** within each fold,
    reducing the risk of data leakage.

    The pipeline enforces the following train-time ordering:

    1) **Preprocessing** (step name: ``"preprocessor"``)
       - casts configured numeric columns to ``float64`` to ensure consistent NaN handling,
       - imputes missing values,
       - applies configured transformations (e.g., ``log1p`` and ``StandardScaler``),
       - encodes categorical variables (e.g., one-hot encoding for nominal, ordinal encoding
         for ordinal features, and custom mapping for partial-ordinal features such as ``ZSN_A``),
       as defined by ``config_preprocessing_features``.
    2) **Filter-based feature selection** (step name: ``"feature_selection_filter"``)
       - selects the top-``k`` features using :class:`sklearn.feature_selection.SelectKBest`.
    3) **Optional resampling** (step name: ``"resampling"``)
       - applies an imbalanced-learn sampler (e.g., SMOTE / undersampling / hybrid) during
         ``fit`` only, to address class imbalance.
       - if disabled, the step is set to the literal string ``"passthrough"``.
       - ``resampling_method`` is normalized with ``strip()`` and the values
         ``None``, ``"none"``, and ``"passthrough"`` disable resampling.
    4) **Final estimator** (step name: ``"classifier"``)
       - the provided classifier is trained on the transformed (and possibly resampled)
         training data.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Final classifier used as the last pipeline step (named ``"classifier"``).
        Must implement ``fit`` and ``predict`` (and optionally ``predict_proba`` if required
        by downstream evaluation).
    config_preprocessing_features : dict[str, list[Any]]
        Column configuration forwarded to :func:`get_preprocessing_pipeline`, defining the
        feature groups and their transformations/encodings.
    fs_k_best_to_keep : int or {'all'}
        Default number of top features to keep in the ``SelectKBest`` step. This value can be
        overridden during tuning via ``feature_selection_filter__k``.
    resampling_method : str or None
        Resampling strategy name forwarded to :func:`get_resampling_pipeline`. If ``None`` or
        one of ``{"none", "passthrough"}`` (case-insensitive, after stripping), resampling is
        disabled and the resampling step becomes a no-op.
    resampling_params : Mapping[str, Any] or None
        Optional keyword arguments forwarded to the sampler constructor via
        :func:`get_resampling_pipeline` (e.g., ``sampling_strategy``, ``random_state``,
        ``k_neighbors``). Ignored when resampling is disabled.

    Returns
    -------
    imblearn.pipeline.Pipeline
        A pipeline with steps:
        - ``('preprocessor', sklearn.compose.ColumnTransformer)``
        - ``('feature_selection_filter', sklearn.feature_selection.SelectKBest)``
        - ``('resampling', imblearn.base.BaseSampler or 'passthrough')``
        - ``('classifier', sklearn.base.BaseEstimator)``

    Raises
    ------
    ValueError
        If ``resampling_method`` is not supported by :func:`get_resampling_pipeline` or if
        invalid sampler parameters are provided via ``resampling_params``.

    Notes
    -----
    - This function calls ``sklearn.set_config(transform_output="pandas")``. This is a **global**
      scikit-learn setting that affects how transformers output data across the entire Python
      process. The motivation is to preserve feature names (as pandas columns) after the
      preprocessing step, so the downstream selector can expose selected feature names via
      ``get_feature_names_out()`` (scikit-learn 1.6.1).
    - Resampling is applied only during ``fit`` and is not applied during ``predict``.
    - Step names are chosen to support parameter routing for tuning, e.g.:
      ``classifier__C``, ``feature_selection_filter__k``, ``resampling__sampling_strategy``.

    Examples
    --------
    Build a pipeline with SMOTE resampling::

        >>> from sklearn.svm import SVC
        >>> svc = SVC(probability=True)
        >>> pipe = build_model_pipeline(
        ...     estimator=svc,
        ...     config_preprocessing_features=config_preprocessing_features,
        ...     fs_k_best_to_keep=20,
        ...     resampling_method="SMOTE",
        ...     resampling_params={"sampling_strategy": 0.2, "random_state": 42},
        ... )

    Disable resampling explicitly::

        >>> pipe = build_model_pipeline(
        ...     estimator=svc,
        ...     config_preprocessing_features=config_preprocessing_features,
        ...     fs_k_best_to_keep=20,
        ...     resampling_method=None,
        ...     resampling_params=None,
        ... )
    """

    sklearn.set_config(transform_output="pandas")

    # Step 1: Preprocessing
    # -- Cast to float64
    # -- SimpleImputer
    # -- Feature Transformation
    # ----> Log1p, StandardScaler, One-hot-encoding, OrdinalEncoding
    preprocessor = get_preprocessing_pipeline(config_cols=config_preprocessing_features)

    # Step 2: Feature selection
    fs_filter = get_feature_selection(k=fs_k_best_to_keep)

    # Step 3: Resampling (Optional)
    resampling_method = None if resampling_method is None else resampling_method.strip()

    # If you consider None/"none" as "no resampling", avoid passing kwargs
    if resampling_method is None or resampling_method.lower() in {"none", "passthrough"}:
        resampling = get_resampling_pipeline(strategy_name=None)
    else:
        safe_resampling_params: dict[str, Any] = dict(resampling_params or {})
        resampling = get_resampling_pipeline(
            strategy_name=resampling_method, **safe_resampling_params
        )

    # Step 4: Preprocessing (Step 1 - Step 3) + Estimator
    final_pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection_filter", fs_filter),
            ("resampling", resampling),
            ("classifier", estimator),
        ]
    )

    return final_pipeline
