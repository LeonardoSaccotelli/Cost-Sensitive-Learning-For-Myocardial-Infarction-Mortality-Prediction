from __future__ import annotations

from typing import Union, List

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from imblearn.pipeline import Pipeline as ImbPipeline


def get_feature_selection(
    *,
    k: Union[int, str] = 20,
    score: str = "mutual_info_classif",
) -> SelectKBest:
    """
    Build a filter-based feature selector using ``SelectKBest``.

    This factory returns a :class:`sklearn.feature_selection.SelectKBest` configured
    with a chosen univariate scoring function. It is intended to be used as a pipeline
    step named ``"feature_selection_filter"`` so the hyperparameter ``k`` can be tuned
    via ``feature_selection_filter__k`` in cross-validation searches.

    Parameters
    ----------
    k : int or {'all'}, default 20
        Number of top-ranked features to keep.
        - If ``int``, keep the top-``k`` features.
        - If ``'all'``, keep all features (no filtering).
    score : {'f_classif', 'mutual_info_classif', 'chi2'}, default 'mutual_info_classif'
        Scoring function used to rank features:
        - ``'f_classif'``: ANOVA F-test (assumes linear separability to some extent).
        - ``'mutual_info_classif'``: mutual information (non-linear dependency).
        - ``'chi2'``: chi-square test (requires **non-negative** feature values).

    Returns
    -------
    sklearn.feature_selection.SelectKBest
        A ``SelectKBest`` selector configured with the chosen scoring function and ``k``.

    Raises
    ------
    ValueError
        If ``score`` is not one of ``{'chi2', 'f_classif', 'mutual_info_classif'}``.

    Notes
    -----
    - When using ``score='chi2'``, ensure all input features are non-negative (e.g., via
      MinMax scaling). Standardization around zero is typically incompatible with chi-square
      feature selection.
    - ``mutual_info_classif`` is stochastic unless you pass a fixed ``random_state`` to the
      scorer (not done here). If you need full determinism, consider wrapping MI with a
      fixed random state or using a deterministic scorer.

    Examples
    --------
    Create a selector and use it as a pipeline step::

        >>> from sklearn.pipeline import Pipeline
        >>> skb = get_feature_selection(k=20, score="f_classif")
        >>> pipe = Pipeline([("feature_selection_filter", skb)])

    Tune ``k`` via a parameter grid::

        >>> param_grid = {"feature_selection_filter__k": [10, 20, "all"]}
    """

    if score == "chi2":
        score_func = chi2
    elif score == "f_classif":
        score_func = f_classif
    elif score == "mutual_info_classif":
        score_func = mutual_info_classif
    else:
        raise ValueError(
            f"Unsupported score '{score}'. "
            "Choose from {'chi2', 'f_classif', 'mutual_info_classif'}."
        )

    return SelectKBest(score_func=score_func, k=k)


def get_selected_feature_names(pipe: ImbPipeline) -> List[str]:
    """
    Return selected feature names from the fitted ``feature_selection_filter`` (SelectKBest) step.

    Parameters
    ----------
    pipe : imblearn.pipeline.Pipeline
        Fitted pipeline with a ``feature_selection_filter`` step.

    Returns
    -------
    selected_feature_names : list[str]
        Selected feature names in the post-preprocessing feature space.

    Raises
    ------
    KeyError
        If the step name is missing.
    ValueError
        If the selector is not fitted or feature names are unavailable.

    Examples
    --------
    >>> # best_pipe = grid_search.best_estimator_
    >>> # names = get_selected_feature_names(best_pipe)  # doctest: +SKIP
    """
    selector = pipe.named_steps["feature_selection_filter"]
    try:
        return [str(x) for x in selector.get_feature_names_out()]
    except Exception as e:
        raise ValueError(
            "Cannot retrieve selected feature names. Ensure the pipeline is fitted and that "
            "feature names are preserved (e.g., sklearn.set_config(transform_output='pandas'))."
        ) from e
