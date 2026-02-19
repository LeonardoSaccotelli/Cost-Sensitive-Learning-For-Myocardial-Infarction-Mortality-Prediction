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

    The returned :class:`imblearn.pipeline.Pipeline` is designed for cross-validation and
    hyperparameter tuning. All data-dependent steps (preprocessing, feature selection, and
    resampling) are fit **only on the training split** within each fold, reducing the risk
    of data leakage.

    The pipeline enforces the following train-time ordering:

    1) **Preprocessing** (``'preprocessor'``)
       - casts numeric columns to ``float64`` to ensure consistent NaN handling,
       - imputes missing values,
       - applies configured transformations (e.g., ``log1p`` and ``StandardScaler`` for
         selected numeric features),
       - encodes categorical variables (e.g., one-hot encoding for nominal, ordinal encoding
         for ordinal features, and custom mapping for partial-ordinal features such as ``ZSN_A``),
       as defined by ``config_preprocessing_features``.
    2) **Filter-based feature selection** (``'feature_selection_filter'``)
       - selects the top-``k`` features using :class:`sklearn.feature_selection.SelectKBest`.
    3) **Optional resampling** (``'resampling'``)
       - applies a sampler (e.g., SMOTE / undersampling / hybrid) during ``fit`` only, to handle
         class imbalance.
       - if disabled, the step is set to ``'passthrough'``.
    4) **Final estimator** (``'classifier'``)
       - the provided classifier is trained on the transformed (and possibly resampled) training
         data.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Final classifier used as the last pipeline step (named ``"classifier"``).
    config_preprocessing_features : dict[str, list[Any]]
        Column configuration passed to :func:`get_preprocessing_pipeline`.
    fs_k_best_to_keep : int or {'all'}
        Number of top features to keep in the ``SelectKBest`` step.
    resampling_method : str or None
        Resampling strategy name passed to :func:`get_resampling_pipeline`.
    resampling_params : Mapping[str, Any] or None
        Optional keyword arguments forwarded to the sampler factory.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline with steps:
        - ``('preprocessor', sklearn.compose.ColumnTransformer)``
        - ``('feature_selection_filter', sklearn.feature_selection.SelectKBest)``
        - ``('resampling', imblearn.base.BaseSampler or 'passthrough')``
        - ``('classifier', sklearn.base.BaseEstimator)``

    Notes
    -----
    - This function calls ``sklearn.set_config(transform_output="pandas")``. This is a **global**
      scikit-learn setting that affects how transformers output data across the entire Python
      process. The motivation is to preserve feature names (as pandas columns) after the
      ``ColumnTransformer``, so the downstream selector can expose the selected feature names.
    - Resampling is applied only during ``fit`` and is not applied during ``predict``.
    - Step names are chosen to support parameter routing for tuning, e.g.:
      ``classifier__C``, ``feature_selection_filter__k``, ``resampling__sampling_strategy``.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> svc = SVC(probability=True)
    >>> pipe = build_model_pipeline(
    ...     estimator=svc,
    ...     config_preprocessing_features=config_preprocessing_features,
    ...     fs_k_best_to_keep=20,
    ...     resampling_method="SMOTE",
    ...     resampling_params={"sampling_strategy": 0.2, "random_state": 42},
    ... )
    >>> search_space = {
    ...     "feature_selection_filter__k": [10, 20, 30, "all"],
    ...     "classifier__C": [0.1, 1.0, 10.0],
    ... }
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
    safe_resampling_params: dict[str, Any] = dict(resampling_params or {})

    # If you consider None/"none" as "no resampling", avoid passing kwargs
    if resampling_method is None or resampling_method == "none":
        resampling = get_resampling_pipeline(strategy_name=None)
    else:
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
