from datetime import datetime
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
from loguru import logger
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import typer

from myocardial_infarction_mortality.config import (
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER,
    CONFIG_PREPROCESSING_FEATURES,
    CV_OUTER_N_REPEATS,
    CV_OUTER_N_SPLITS,
    CV_OUTER_PARALLEL_N_JOBS,
    DES_MODELS,
    DSEL_SIZE,
    EXPERIMENT_ID,
    EXPERIMENTS,
    FILENAME_BASE,
    FLOAT_FEATURES,
    FS_K_BEST_CANDIDATES,
    FS_K_BEST_TO_KEEP,
    INT_FEATURES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SELECTED_TIME_SLOT,
    STATIC_ENSEMBLE_MODELS,
    STATIC_ENSEMBLE_POOLS,
    STATIC_MODELS,
    TUNING_CV_INNER_N_SPLITS,
    TUNING_N_ITER,
    TUNING_N_JOBS,
    TUNING_SCORING,
)
from myocardial_infarction_mortality.modeling.utils.training import (
    train_and_evaluate_one_fold_all_models,
)
from myocardial_infarction_mortality.utils.io_utils import load_dataset_with_schema, save_dict_json

app = typer.Typer()


@app.command()
def main(
    experiment_id: str = EXPERIMENT_ID,
    input_path: Path = PROCESSED_DATA_DIR
    / f"{FILENAME_BASE}_{SELECTED_TIME_SLOT}_LET_IS_BINARY_features.csv",
    model_path: Path = MODELS_DIR,
    target: str = "CLASS",
    outer_n_jobs: int = CV_OUTER_PARALLEL_N_JOBS,
):
    """
    Run the full training workflow for the myocardial infarction mortality project
    (STATIC, STATIC-ENSEMBLE, DES) under a configured experiment setting.

    This Typer CLI entry point loads the processed dataset, configures the experiment
    (including optional cost matrix and decision policy metadata), executes a repeated
    stratified outer cross-validation loop, and persists per-model results to disk.

    The workflow executed is:

    1) Load the processed dataset (CSV) from ``input_path`` and log dataset shape and target checks.
    2) Select the active experiment configuration as ``experiment_setting = EXPERIMENTS[experiment_id]``.
    3) Perform a single global shuffle using ``RANDOM_STATE`` to randomize row order.
    4) Split the dataset into features and target using ``target``.
    5) Build the outer evaluation loop using
       :class:`sklearn.model_selection.RepeatedStratifiedKFold`.
    6) Execute each outer fold:
       - sequentially when ``outer_n_jobs == 1``,
       - in parallel using :class:`joblib.Parallel` when ``outer_n_jobs > 1``.

       Each fold delegates the end-to-end orchestration (STATIC, STATIC-ENSEMBLE, DES)
       to :func:`myocardial_infarction_mortality.modeling.utils.training.train_and_evaluate_one_fold_all_models`,
       which:
       - builds pipelines (preprocessing + SelectKBest + optional resampling + estimator),
       - performs inner-CV hyperparameter tuning,
       - evaluates resubstitution and generalization metrics,
       - for DES: tunes the pool, fits the DES model on DSEL, and evaluates the inference pipeline,
       - applies the experiment decision policy (``standard`` vs ``mec``) and cost reporting when enabled,
       - returns fold-level rows for both training-side and test-side reporting.

    7) Aggregate fold-level rows across all outer folds and persist results **per model** under::

           <model_path>/<experiment_id>/<MODEL_NAME>/

       Each model subfolder includes:
       - ``generalization_metrics_summary.csv`` (outer-test metrics),
       - ``resubstitution_metrics_summary.csv`` (train-side metrics; for DES this refers to pool-resubstitution),
       - ``experiment_config.json`` (experiment tracking metadata replicated per model).

    Parameters
    ----------
    experiment_id : str, optional
        Key used to select the experiment configuration from ``EXPERIMENTS``.
        The folder name created under ``model_path`` is also ``experiment_id``.
        Defaults to ``EXPERIMENT_ID``.
    input_path : pathlib.Path, optional
        Path to the processed dataset CSV containing engineered features and the target column.
        Defaults to::

            PROCESSED_DATA_DIR / f"{FILENAME_BASE}_{SELECTED_TIME_SLOT}_LET_IS_BINARY_features.csv"

    model_path : pathlib.Path, optional
        Root directory where the experiment folder is created and results are saved.
        Defaults to ``MODELS_DIR``.
    target : str, optional
        Name of the target column in the input dataset. Expected to exist in the loaded CSV.
        Defaults to ``"CLASS"``.
    outer_n_jobs : int, optional
        Number of outer CV folds executed in parallel.

        - ``1`` executes folds sequentially.
        - ``> 1`` parallelizes folds using :class:`joblib.Parallel`.

        To avoid nested parallelism and CPU oversubscription, it is generally recommended to
        keep inner-tuning parallelism low (often ``TUNING_N_JOBS = 1``) when ``outer_n_jobs > 1``.
        Defaults to ``CV_OUTER_PARALLEL_N_JOBS``.

    Returns
    -------
    None
        Side effects only (training, logging, and persistence of outputs).

    Raises
    ------
    typer.Exit
        Raised with code ``1`` if ``input_path`` does not exist or if ``target`` is missing.
    KeyError
        If ``experiment_id`` is not a key in ``EXPERIMENTS``.
    ValueError
        If the aggregated results DataFrames are missing mandatory columns (e.g., ``"model"``),
        or if no models are found in results (nothing to persist).
    RuntimeError
        If a model is missing expected persisted rows (generalization and/or resubstitution).
    pandas.errors.EmptyDataError
        If the input CSV is empty or has no columns to parse.
    pandas.errors.ParserError
        If the input CSV is malformed and cannot be parsed.
    PermissionError
        If experiment/model folders or output files cannot be created/written due to insufficient
        permissions.
    OSError
        If an OS-related error occurs during directory creation or file writing.

    Notes
    -----
    Experiment setting schema
        ``experiment_setting`` is expected to follow a schema such as::

            {
                "experiment_name": "baseline__mec_fp1_fn10",
                "description": "...",
                "approach": "baseline",  # baseline | cost_sensitive_learning | data_level
                "tags": ["baseline", "mec_policy"],
                "class_weight": None,  # None | "balanced" | {0: w0, 1: w1}
                "resampling_method": None,
                "resampling_params": None,
                "decision_policy_mode": "mec",  # standard | mec
                "costs_matrix": COST_MATRIX,
            }

        Decision policy (e.g., MEC) and the cost matrix are applied downstream by the fold
        training/evaluation helpers (e.g., via ``apply_decision_policy`` and cost-aware metrics).

    Parallel execution
        When ``outer_n_jobs > 1``, joblib may use process-based parallelism. Ensure objects
        captured by fold workers are picklable. If passing the logger causes issues, prefer
        fold-local logging or passing a lightweight logger proxy.

    Feature selection
        ``SelectKBest`` is included in modeling pipelines. Candidate values for ``k`` may be
        tuned when ``FS_K_BEST_CANDIDATES`` is provided and injected by the fold orchestrator.

    Examples
    --------
    Run with defaults::

        python myocardial_infarction_mortality/train.py

    Run sequentially (no outer parallelism)::

        python myocardial_infarction_mortality/train.py --outer-n-jobs 1

    Run parallelizing outer folds (ensure inner tuning does not also saturate CPUs)::

        python myocardial_infarction_mortality/train.py --outer-n-jobs 10
    """

    logger.info("Running myocardial_infarction_mortality/train.py ...")

    # --- Set experiment folder and experiment tracking
    experiment_setting: dict[str, Any] = EXPERIMENTS[EXPERIMENT_ID]

    experiment_path = model_path / experiment_id
    experiment_path.mkdir(parents=True, exist_ok=True)

    experiment_tracking = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_setting["experiment_name"],
        "experiment_description": experiment_setting["description"],
        "experiment_approach": experiment_setting["approach"],
        "experiment_tags": experiment_setting["tags"],
        "experiment_start_time": datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),
        "feature_selection_KBest_candidates": FS_K_BEST_CANDIDATES,
        "outer_evaluation_loop": f"RepeatedStratifiedKFold_{CV_OUTER_N_REPEATS}_times_{CV_OUTER_N_SPLITS}_folds",
        "DSEL_size": DSEL_SIZE,
        "tuning_hyperparameters_n_iter": TUNING_N_ITER,
        "tuning_hyperparameters_cv_splits": TUNING_CV_INNER_N_SPLITS,
        "tuning_hyperparameters_scoring": TUNING_SCORING,
        "tuning_hyperparameters_n_jobs": TUNING_N_JOBS,
        "static_models_to_train": STATIC_MODELS,
        "static_ensemble_models_to_train": STATIC_ENSEMBLE_MODELS,
        "static_ensemble_pools": STATIC_ENSEMBLE_POOLS,
        "des_models_to_train": DES_MODELS,
    }

    logger.info(f"Initialized experiment: {experiment_id}")
    logger.info(experiment_tracking)

    ################################# INITIAL CHECKS #################################
    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error(
            "Run the feature engineering step first, e.g.: `python myocardial_infarction_mortality/features.py`."
        )
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading FEATURES dataset at path:\n\t{input_path}")
    df: pd.DataFrame = load_dataset_with_schema(
        file_path=input_path,
        delimiter=",",
        header=0,
        index_col=0,
        float_features=FLOAT_FEATURES,
        int_features=INT_FEATURES,
        categorical_nominal_features=CATEGORICAL_NOMINAL_FEATURES,
        categorical_ordinal_features=CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER,
    )
    logger.info(f"Input dataset shape: {df.shape} (rows, cols)")

    if target not in df.columns:
        logger.error(f"Target column '{target}' not found.")
        raise typer.Exit(code=1)

    ################################# PREPARE DATASET FOR TRAINING #################################
    # Initial shuffle of the data (frac=1.0 means that all rows will be kept but shuffled)
    logger.info(f"Shuffling dataset with random_state={RANDOM_STATE}")
    df = df.sample(frac=1.0, random_state=RANDOM_STATE)

    # Split data into features and labels
    X, y = df.drop([target], axis=1), df[target]
    logger.info(
        f"Splitting dataset into features and labels. "
        f"Shape of X: {X.shape} - Shape of y: {y.shape}"
    )

    ################################# PREPARE SETTINGS FOR TRAINING ##################################
    # Fix the evaluation strategy: RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    cv_outer = RepeatedStratifiedKFold(
        n_splits=CV_OUTER_N_SPLITS, n_repeats=CV_OUTER_N_REPEATS, random_state=RANDOM_STATE
    )

    # List to store all the resubstitution (train) and generalization (test) metrics
    # for each iteration of the RepeatedStratifiedKFold
    resubstitution_metrics_summary = []
    generalization_metrics_summary = []

    ################################# START TRAINING PHASE ###########################################
    logger.info("Starting model training over all outer folds...")

    if outer_n_jobs == 1:
        logger.info("Training all outer folds sequentially")

        # ---- Sequential execution (current behaviour) ----
        for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
            iteration_idx, fold_idx = divmod(run_id, CV_OUTER_N_SPLITS)

            print("=" * 165)

            resubstitution_rows, generalization_rows = train_and_evaluate_one_fold_all_models(
                run_id=run_id,
                iteration_idx=iteration_idx,
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                experiment_setting=experiment_setting,
                config_preprocessing_features=CONFIG_PREPROCESSING_FEATURES,
                static_models=STATIC_MODELS,
                static_ensemble_models=STATIC_ENSEMBLE_MODELS,
                static_ensemble_pools=STATIC_ENSEMBLE_POOLS,
                des_models=DES_MODELS,
                fs_k_best_to_keep=FS_K_BEST_TO_KEEP,
                fs_k_best_candidates=FS_K_BEST_CANDIDATES,
                tuning_n_iter=TUNING_N_ITER,
                tuning_cv_inner_n_splits=TUNING_CV_INNER_N_SPLITS,
                tuning_scoring=TUNING_SCORING,
                tuning_n_jobs=TUNING_N_JOBS,
                dsel_size=DSEL_SIZE,
                random_state=RANDOM_STATE,
                logger=logger,
            )

            resubstitution_metrics_summary.extend(resubstitution_rows)
            generalization_metrics_summary.extend(generalization_rows)

    else:
        # ---- Parallel execution of outer folds ----
        logger.info(f"Parallelizing outer folds with outer_n_jobs={outer_n_jobs}")

        # Run the 10x10 CV in parallel
        parallel_results = Parallel(n_jobs=outer_n_jobs, verbose=10)(
            delayed(train_and_evaluate_one_fold_all_models)(
                run_id=run_id,
                iteration_idx=divmod(run_id, CV_OUTER_N_SPLITS)[0],
                fold_idx=divmod(run_id, CV_OUTER_N_SPLITS)[1],
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                experiment=experiment_setting,
                config_preprocessing_features=CONFIG_PREPROCESSING_FEATURES,
                static_models=STATIC_MODELS,
                static_ensemble_models=STATIC_ENSEMBLE_MODELS,
                static_ensemble_pools=STATIC_ENSEMBLE_POOLS,
                des_models=DES_MODELS,
                fs_k_best_to_keep=FS_K_BEST_TO_KEEP,
                fs_k_best_candidates=FS_K_BEST_CANDIDATES,
                tuning_n_iter=TUNING_N_ITER,
                tuning_cv_inner_n_splits=TUNING_CV_INNER_N_SPLITS,
                tuning_scoring=TUNING_SCORING,
                tuning_n_jobs=TUNING_N_JOBS,
                dsel_size=DSEL_SIZE,
                random_state=RANDOM_STATE,
                logger=logger,
            )
            for run_id, (train_idx, test_idx) in enumerate(cv_outer.split(X, y))
        )

        # ---- Aggregation (Post-Processing) ----
        # Parallel returns a list of tuples: [(res_rows, gen_rows), (res_rows, gen_rows), ...]
        # We must flatten this back into your summary lists.
        for resubstitution_rows, generalization_rows in parallel_results:
            resubstitution_metrics_summary.extend(resubstitution_rows)
            generalization_metrics_summary.extend(generalization_rows)

    experiment_tracking["experiment_end_time"] = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")

    ############################### STORE EXPERIMENTAL RESULTS #####################################
    resubstitution_metrics_summary_df = pd.DataFrame(resubstitution_metrics_summary)
    generalization_metrics_summary_df = pd.DataFrame(generalization_metrics_summary)

    # Basic validation
    if (not resubstitution_metrics_summary_df.empty) and (
        "model" not in resubstitution_metrics_summary_df.columns
    ):
        raise ValueError("Missing 'model' column in resubstitution_metrics_summary.")
    if (not generalization_metrics_summary_df.empty) and (
        "model" not in generalization_metrics_summary_df.columns
    ):
        raise ValueError("Missing 'model' column in generalization_metrics_summary.")

    # Union of models across both summaries (covers DES-only runs too)
    models_in_results: set[str] = set()
    if not resubstitution_metrics_summary_df.empty:
        models_in_results |= set(resubstitution_metrics_summary_df["model"].astype(str).unique())
    if not generalization_metrics_summary_df.empty:
        models_in_results |= set(generalization_metrics_summary_df["model"].astype(str).unique())

    if not models_in_results:
        raise RuntimeError("No models found in results; nothing to persist.")

    for model_name in sorted(models_in_results):
        model_dir = experiment_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Filter rows for the model
        resubstitution_metrics_model = (
            resubstitution_metrics_summary_df[
                resubstitution_metrics_summary_df["model"].astype(str) == model_name
            ].copy()
            if not resubstitution_metrics_summary_df.empty
            else None
        )
        generalization_metrics_model = (
            generalization_metrics_summary_df[
                generalization_metrics_summary_df["model"].astype(str) == model_name
            ].copy()
            if not generalization_metrics_summary_df.empty
            else None
        )

        # Generalization expected for both STATIC and DES
        if generalization_metrics_model is None or generalization_metrics_model.empty:
            raise RuntimeError(f"No generalization rows found for model='{model_name}'.")
        generalization_metrics_model.to_csv(
            model_dir / "generalization_metrics_summary.csv", index=False, sep=","
        )

        # Resubstitution now expected for both STATIC and DES
        if resubstitution_metrics_model is None or resubstitution_metrics_model.empty:
            raise RuntimeError(f"No resubstitution rows found for model='{model_name}'.")
        resubstitution_metrics_model.to_csv(
            model_dir / "resubstitution_metrics_summary.csv", index=False, sep=","
        )

        # Store experiment settings
        save_dict_json(
            data=experiment_tracking, path=model_dir / "experiment_config.json", mode="w"
        )

    logger.success("Running myocardial_infarction_mortality/train.py COMPLETED!")


if __name__ == "__main__":
    app()
