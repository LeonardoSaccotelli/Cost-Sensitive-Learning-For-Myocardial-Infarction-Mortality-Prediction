from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from myocardial_infarction_mortality.config import (
    CATEGORICAL_NOMINAL_FEATURES,
    CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER,
    EXCLUDE_FEATURES_BY_SLOT,
    EXTERNAL_DATA_DIR,
    EXTERNAL_FILENAME,
    FILENAME_BASE,
    FLOAT_FEATURES,
    INT_FEATURES,
    RAW_DATA_DIR,
    SELECTED_TIME_SLOT,
)
from myocardial_infarction_mortality.utils.io_utils import load_dataset_with_schema

app = typer.Typer()


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / EXTERNAL_FILENAME,
    output_path: Path = RAW_DATA_DIR,
    filename_base: str = FILENAME_BASE,
    selected_time_slot: str = SELECTED_TIME_SLOT,
) -> None:
    """
    Construct a time-slot-specific dataset by dropping unavailable features and adding a binary target.

    The original dataset supports multiple ICU time points (e.g., admission, 24h, 48h, 72h).
    For a given ``selected_time_slot``, this script drops the features listed in
    ``EXCLUDE_FEATURES_BY_SLOT[selected_time_slot]`` and creates a binary target:

    - ``LET_IS_BINARY`` = 0 if ``LET_IS`` == 0 (alive)
    - ``LET_IS_BINARY`` = 1 if ``LET_IS`` >= 1 (dead)

    The resulting dataset is saved as a CSV under ``output_path`` using the filename:
    ``{filename_base}_{selected_time_slot}.csv``.

    Parameters
    ----------
    input_path : pathlib.Path, optional
        Path to the source dataset CSV. Defaults to ``EXTERNAL_DATA_DIR / EXTERNAL_FILENAME``.
        If this path does not exist, the script will also try the fallback
        ``RAW_DATA_DIR / RAW_FILENAME``.
    output_path : pathlib.Path, optional
        Directory where the constructed dataset will be written. Defaults to ``RAW_DATA_DIR``.
    filename_base : str, optional
        Base filename (without extension) used to build the output filename.
    selected_time_slot : str, optional
        Time slot key used to select which features to drop. Must be a key of
        ``EXCLUDE_FEATURES_BY_SLOT`` (e.g., "admission", "day1", "day2", "day3").

    Returns
    -------
    None
        Side effects only (read CSV, transform, write CSV, logging).

    Raises
    ------
    typer.Exit
        Raised with exit code ``1`` if the input dataset is missing, if the time slot is invalid,
        if ``LET_IS`` is missing, if ``LET_IS`` contains non-numeric values, or if writing fails.

    Examples
    --------
    Build the "admission" dataset (explicit paths shown as examples)::

        python myocardial_infarction_mortality/dataset_time_split.py \
            --input-path data/external/myocardial_infarction.csv \
            --output-path data/raw \
            --filename-base myocardial_infarction \
            --selected-time-slot admission

    Build the 48h dataset (``day2``) using defaults::

        python myocardial_infarction_mortality/dataset_time_split.py \
            --selected-time-slot day2
    """

    logger.info("Running myocardial_infarction_mortality/dataset_time_split.py ...")

    # Preconditions
    if not input_path.exists():
        logger.error(f"Dataset not found at path:\n\t{input_path}")
        logger.error(
            "Run the downloader first, e.g.: `python myocardial_infarction_mortality/dataset.py`."
        )
        raise typer.Exit(code=1)

    if selected_time_slot not in EXCLUDE_FEATURES_BY_SLOT:
        valid = ", ".join(sorted(EXCLUDE_FEATURES_BY_SLOT.keys()))
        logger.error(f"Invalid selected_time_slot='{selected_time_slot}'. Valid options: {valid}")
        raise typer.Exit(code=1)

    # Load and basic checks
    logger.info(f"Loading EXTERNAL dataset at path:\n\t{input_path}")
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

    # Create LET_IS_BINARY target (0 -> ALIVE, >=1 -> DEAD)
    if "LET_IS" not in df.columns:
        logger.error("Target column 'LET_IS' not found in the dataset.")
        raise typer.Exit(code=1)

    df["LET_IS_BINARY"] = np.where(df["LET_IS"] != 0, 1, 0)
    df["LET_IS_BINARY"] = df["LET_IS_BINARY"].astype("category")

    # Drop excluded features for the selected time slot (ignore missing columns, but log them)
    to_drop = EXCLUDE_FEATURES_BY_SLOT[selected_time_slot]
    present_to_drop = [c for c in to_drop if c in df.columns]
    missing_to_drop = [c for c in to_drop if c not in df.columns]

    if missing_to_drop:
        logger.warning(
            "Some features listed for exclusion are not present in the dataset and will be ignored:\n\t"
            + ", ".join(missing_to_drop)
        )

    if present_to_drop:
        df = df.drop(columns=present_to_drop)
        logger.info(
            f"Dropped {len(present_to_drop)} features for time slot '{selected_time_slot}'."
        )
        logger.info(f"Dropped features: {present_to_drop}.")

    # Ensure RAW_DATA_DIR exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename_output_path = RAW_DATA_DIR / f"{filename_base}_{selected_time_slot}.csv"

    df.to_csv(filename_output_path, index=True, sep=",")

    # Store the sampling dataset
    logger.success(f"Wrote RAW dataset to path:\n\t{output_path}")
    logger.info(f"Output dataset shape: {df.shape} (rows, cols)")

    logger.success("Running myocardial_infarction_mortality/dataset_time_split.py COMPLETED!")


if __name__ == "__main__":
    app()
