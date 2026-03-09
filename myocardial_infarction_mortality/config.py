from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
import numpy as np

# Load environment variables from .env file if it exists
load_dotenv()

#################################################################
# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EXTERNAL_FILENAME = "myocardial_infarction_external.csv"
EXTERNAL_METADATA_FILENAME = "myocardial_infarction_metadata.txt"
EXTERNAL_VARIABLES_FILENAME = "myocardial_infarction_variables.csv"
EXTERNAL_DATASET_ID = 579
EXTERNAL_FORCE_DOWNLOAD = False

FILENAME_BASE = "myocardial_infarction"

#################################################################
# DATA TYPE FEATURES
FLOAT_FEATURES: list[str] = [
    "K_BLOOD",
    "NA_BLOOD",
    "ALT_BLOOD",
    "AST_BLOOD",
    "KFK_BLOOD",
    "L_BLOOD",
    "ROE",
]

INT_FEATURES: list[str] = [
    "AGE",
    "S_AD_KBRIG",
    "D_AD_KBRIG",
    "S_AD_ORIT",
    "D_AD_ORIT",
]

CATEGORICAL_NOMINAL_FEATURES: list[str] = [
    "SEX",
    "IBS_NASL",
    "IBS_POST",
    "SIM_GIPERT",
    "nr_11",
    "nr_01",
    "nr_02",
    "nr_03",
    "nr_04",
    "nr_07",
    "nr_08",
    "np_01",
    "np_04",
    "np_05",
    "np_07",
    "np_08",
    "np_09",
    "np_10",
    "endocr_01",
    "endocr_02",
    "endocr_03",
    "zab_leg_01",
    "zab_leg_02",
    "zab_leg_03",
    "zab_leg_04",
    "zab_leg_06",
    "O_L_POST",
    "K_SH_POST",
    "MP_TP_POST",
    "SVT_POST",
    "GT_POST",
    "FIB_G_POST",
    "IM_PG_P",
    "ritm_ecg_p_01",
    "ritm_ecg_p_02",
    "ritm_ecg_p_04",
    "ritm_ecg_p_06",
    "ritm_ecg_p_07",
    "ritm_ecg_p_08",
    "n_r_ecg_p_01",
    "n_r_ecg_p_02",
    "n_r_ecg_p_03",
    "n_r_ecg_p_04",
    "n_r_ecg_p_05",
    "n_r_ecg_p_06",
    "n_r_ecg_p_08",
    "n_r_ecg_p_09",
    "n_r_ecg_p_10",
    "n_p_ecg_p_01",
    "n_p_ecg_p_03",
    "n_p_ecg_p_04",
    "n_p_ecg_p_05",
    "n_p_ecg_p_06",
    "n_p_ecg_p_07",
    "n_p_ecg_p_08",
    "n_p_ecg_p_09",
    "n_p_ecg_p_10",
    "n_p_ecg_p_11",
    "n_p_ecg_p_12",
    "fibr_ter_01",
    "fibr_ter_02",
    "fibr_ter_03",
    "fibr_ter_05",
    "fibr_ter_06",
    "fibr_ter_07",
    "fibr_ter_08",
    "GIPO_K",
    "GIPER_NA",
    "NA_KB",
    "NOT_NA_KB",
    "LID_KB",
    "NITR_S",
    "LID_S_n",
    "B_BLOK_S_n",
    "ANT_CA_S_n",
    "GEPAR_S_n",
    "ASP_S_n",
    "TIKL_S_n",
    "TRENT_S_n",
    "FIBR_PREDS",
    "PREDS_TAH",
    "JELUD_TAH",
    "FIBR_JELUD",
    "A_V_BLOK",
    "OTEK_LANC",
    "RAZRIV",
    "DRESSLER",
    "ZSN",
    "REC_IM",
    "P_IM_STEN",
    "LET_IS",
    "LET_IS_BINARY",
    "CLASS",
]

CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER: dict[str, list[int]] = {
    "INF_ANAM": [0, 1, 2, 3],
    "STENOK_AN": [0, 1, 2, 3, 4, 5, 6],
    "FK_STENOK": [0, 1, 2, 3, 4],
    "GB": [0, 1, 2, 3],
    "DLIT_AG": [0, 1, 2, 3, 4, 5, 6, 7],
    "ZSN_A": [0, 1, 2, 3, 4],
    "TIME_B_S": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "R_AB_1_n": [0, 1, 2, 3],
    "R_AB_2_n": [0, 1, 2, 3],
    "R_AB_3_n": [0, 1, 2, 3],
    "NA_R_1_n": [0, 1, 2, 3, 4],
    "NA_R_2_n": [0, 1, 2, 3],
    "NA_R_3_n": [0, 1, 2],
    "NOT_NA_1_n": [0, 1, 2, 3, 4],
    "NOT_NA_2_n": [0, 1, 2, 3],
    "NOT_NA_3_n": [0, 1, 2],
    "ant_im": [0, 1, 2, 3, 4],
    "lat_im": [0, 1, 2, 3, 4],
    "inf_im": [0, 1, 2, 3, 4],
    "post_im": [0, 1, 2, 3, 4],
}

CATEGORICAL_ORDINAL_FEATURES: list[str] = list(CATEGORICAL_ORDINAL_FEATURES_WITH_ORDER.keys())

#################################################################
# ICU time interval and excluded features and targets
# Selected time slot for analysis
SELECTED_TIME_SLOT: str = "admission"

# Excluded feature names by time slot
EXCLUDE_FEATURES_BY_SLOT: dict[str, list[str]] = {
    "admission": [
        "R_AB_1_n",
        "R_AB_2_n",
        "R_AB_3_n",
        "NA_R_1_n",
        "NA_R_2_n",
        "NA_R_3_n",
        "NOT_NA_1_n",
        "NOT_NA_2_n",
        "NOT_NA_3_n",
    ],  # features exclusions at the admission
    "day1": [
        "R_AB_2_n",
        "R_AB_3_n",
        "NA_R_2_n",
        "NA_R_3_n",
        "NOT_NA_2_n",
        "NOT_NA_3_n",
    ],  # features exclusions at 24h
    "day2": [
        "R_AB_3_n",
        "NA_R_3_n",
        "NOT_NA_3_n",
    ],  # features exclusions at 48h
    "day3": [],  # no features exclusions at 72h
}

# Excluded targets
EXCLUDE_TARGETS: list[str] = [
    "FIBR_PREDS",
    "PREDS_TAH",
    "JELUD_TAH",
    "FIBR_JELUD",
    "A_V_BLOK",
    "OTEK_LANC",
    "RAZRIV",
    "DRESSLER",
    "ZSN",
    "REC_IM",
    "P_IM_STEN",
    "LET_IS",
]

#################################################################
# seed for reproducibility
RANDOM_STATE = 42

#################################################################
# train.py parameters

# ------- feature transformation
CONFIG_PREPROCESSING_FEATURES: dict[str, list[Any]] = {
    "num_log1p_standard_scaler": ["L_BLOOD", "ALT_BLOOD", "AST_BLOOD", "ROE", "K_BLOOD"],
    "num_standard_scaler": ["AGE", "S_AD_ORIT", "D_AD_ORIT", "NA_BLOOD"],
    "cat_nominal": [
        "IBS_POST",
    ],
    "cat_binary": [
        "SEX",
        "SIM_GIPERT",
        "nr_11",
        "nr_01",
        "nr_02",
        "nr_03",
        "nr_04",
        "nr_07",
        "nr_08",
        "np_01",
        "np_04",
        "np_05",
        "np_07",
        "np_08",
        "np_09",
        "np_10",
        "endocr_01",
        "endocr_02",
        "endocr_03",
        "zab_leg_01",
        "zab_leg_02",
        "zab_leg_03",
        "zab_leg_04",
        "zab_leg_06",
        "O_L_POST",
        "K_SH_POST",
        "MP_TP_POST",
        "SVT_POST",
        "GT_POST",
        "FIB_G_POST",
        "IM_PG_P",
        "ritm_ecg_p_01",
        "ritm_ecg_p_02",
        "ritm_ecg_p_04",
        "ritm_ecg_p_06",
        "ritm_ecg_p_07",
        "ritm_ecg_p_08",
        "n_r_ecg_p_01",
        "n_r_ecg_p_02",
        "n_r_ecg_p_03",
        "n_r_ecg_p_04",
        "n_r_ecg_p_05",
        "n_r_ecg_p_06",
        "n_r_ecg_p_08",
        "n_r_ecg_p_09",
        "n_r_ecg_p_10",
        "n_p_ecg_p_01",
        "n_p_ecg_p_03",
        "n_p_ecg_p_04",
        "n_p_ecg_p_05",
        "n_p_ecg_p_06",
        "n_p_ecg_p_07",
        "n_p_ecg_p_08",
        "n_p_ecg_p_09",
        "n_p_ecg_p_10",
        "n_p_ecg_p_11",
        "n_p_ecg_p_12",
        "fibr_ter_01",
        "fibr_ter_02",
        "fibr_ter_03",
        "fibr_ter_05",
        "fibr_ter_06",
        "fibr_ter_07",
        "fibr_ter_08",
        "GIPO_K",
        "GIPER_NA",
        "NITR_S",
        "LID_S_n",
        "B_BLOK_S_n",
        "ANT_CA_S_n",
        "GEPAR_S_n",
        "ASP_S_n",
        "TIKL_S_n",
        "TRENT_S_n",
    ],
    "cat_partial_ordinal": [
        "ZSN_A",
    ],
    "cat_ordinal": [
        "INF_ANAM",
        "STENOK_AN",
        "FK_STENOK",
        "GB",
        "DLIT_AG",
        "TIME_B_S",
        "ant_im",
        "lat_im",
        "inf_im",
        "post_im",
    ],
    "cat_ordinal_order": [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ],
}

# ------- feature selection
FS_K_BEST_TO_KEEP: int | str = 20

# Candidate values used by RandomizedSearchCV to tune SelectKBest.
# NOTE: ensure each int <= n_features AFTER preprocessing. "all" is allowed.
FS_K_BEST_CANDIDATES: list[int | str] = list(range(5, 91, 5)) + ["all"]

# ------- experiment settings
# The negative class (the majority one) is 0-ALIVE;
# The positive class (the minority one) is 1-DEAD;
# --> TN: true ALIVE (0) predicted ALIVE (0)
# --> FP: true ALIVE (0) predicted DEAD  (1)
# --> FN: true DEAD  (1) predicted ALIVE (0)
# --> TP: true DEAD  (1) predicted DEAD  (1)
# FN (missed death): predicted ALIVE but actually DEAD → usually the most costly
# FP (false alarm): predicted DEAD but actually ALIVE → costly (stress/resources) but often less costly than FN
#                              _______________________________
#                             |              |                |
#                   0 (ALIVE) |      TN      |       FP       |
# TRUE LABEL                  +--------------+----------------+
#                             |      FN      |       TP       |
#                   1 (DEAD)  |______________|________________|
#                                 0 (ALIVE)        1 (DEAD)
#                                      PREDICTED LABEL

COST_MATRIX = np.array(
    [
        [0, 1],  # True = 0 (ALIVE), Predict = 1 (DEAD)  --> FP = 1
        [10, 0],  # True = 1 (DEAD),  Predict = 0 (ALIVE) --> FN = 10
    ]
)

COST_SENSITIVE_CLASS_WEIGHT = {0: 1, 1: 10}

EXPERIMENT_ID: str = "smoteenn_auto__mec_fp1_fn10"
EXPERIMENTS: dict[str, dict[str, Any]] = {
    # ============================================================
    # BASELINE
    # ============================================================
    "baseline__standard": {
        "experiment_name": "baseline__standard",
        "description": "No resampling, no class_weight. Decision policy: standard (threshold=0.5 implicit).",
        "approach": "baseline",  # baseline | cost_sensitive_learning | data_level
        "tags": ["baseline", "standard_policy"],
        "class_weight": None,  # None | "balanced" | {0: w0, 1: w1}
        "resampling_method": None,
        "resampling_params": None,
        "decision_policy_mode": "standard",  # standard | mec
        "costs_matrix": COST_MATRIX,
    },
    "baseline__mec_fp1_fn10": {
        "experiment_name": "baseline__mec_fp1_fn10",
        "description": "No resampling, no class_weight. Decision policy: MEC with costs FP=1, FN=10.",
        "approach": "baseline",
        "tags": ["baseline", "mec_policy"],
        "class_weight": None,
        "resampling_method": None,
        "resampling_params": None,
        "decision_policy_mode": "mec",
        "costs_matrix": COST_MATRIX,
    },
    # ============================================================
    # COST-SENSITIVE LEARNING (class_weight)
    # ============================================================
    "csl_balanced__standard": {
        "experiment_name": "csl_balanced__standard",
        "description": "class_weight='balanced'. No resampling. Decision policy: standard (threshold=0.5 implicit).",
        "approach": "cost_sensitive_learning",
        "tags": ["cost_sensitive_learning", "class_weight_balanced", "standard_policy"],
        "class_weight": "balanced",
        "resampling_method": None,
        "resampling_params": None,
        "decision_policy_mode": "standard",
        "costs_matrix": COST_MATRIX,
    },
    "csl_balanced__mec_fp1_fn10": {
        "experiment_name": "csl_balanced__mec_fp1_fn10",
        "description": "class_weight='balanced'. No resampling. Decision policy: MEC with costs FP=1, FN=10.",
        "approach": "cost_sensitive_learning",
        "tags": ["cost_sensitive_learning", "class_weight_balanced", "mec_policy"],
        "class_weight": "balanced",
        "resampling_method": None,
        "resampling_params": None,
        "decision_policy_mode": "mec",
        "costs_matrix": COST_MATRIX,
    },
    "csl_fp1_fn10__standard": {
        "experiment_name": "csl_fp1_fn10__standard",
        "description": "class_weight={0:1, 1:10} (business rule). No resampling. Decision policy: standard (0.5 implicit).",
        "approach": "cost_sensitive_learning",
        "tags": [
            "cost_sensitive_learning",
            "class_weight_business",
            "fp1_fn10",
            "standard_policy",
        ],
        "class_weight": COST_SENSITIVE_CLASS_WEIGHT,
        "resampling_method": None,
        "resampling_params": None,
        "decision_policy_mode": "standard",
        "costs_matrix": COST_MATRIX,
    },
    "csl_fp1_fn10__mec_fp1_fn10": {
        "experiment_name": "csl_fp1_fn10__mec_fp1_fn10",
        "description": "class_weight={0:1, 1:10} (business rule). No resampling. Decision policy: MEC with costs FP=1, FN=10.",
        "approach": "cost_sensitive_learning",
        "tags": ["cost_sensitive_learning", "class_weight_business", "fp1_fn10", "mec_policy"],
        "class_weight": COST_SENSITIVE_CLASS_WEIGHT,
        "resampling_method": None,
        "resampling_params": None,
        "decision_policy_mode": "mec",
        "costs_matrix": COST_MATRIX,
    },
    # ============================================================
    # DATA-LEVEL (resampling)
    # ============================================================
    "smote_auto__standard": {
        "experiment_name": "smote_auto__standard",
        "description": "SMOTE sampling_strategy='auto'. No class_weight. Decision policy: standard (0.5 implicit).",
        "approach": "data_level",
        "tags": ["data_level", "smote", "auto", "standard_policy"],
        "class_weight": None,
        "resampling_method": "SMOTE",
        "resampling_params": {"sampling_strategy": "auto", "random_state": RANDOM_STATE},
        "decision_policy_mode": "standard",
        "costs_matrix": COST_MATRIX,
    },
    "smote_auto__mec_fp1_fn10": {
        "experiment_name": "smote_auto__mec_fp1_fn10",
        "description": "SMOTE sampling_strategy='auto'. No class_weight. Decision policy: MEC with costs FP=1, FN=10.",
        "approach": "data_level",
        "tags": ["data_level", "smote", "auto", "mec_policy"],
        "class_weight": None,
        "resampling_method": "SMOTE",
        "resampling_params": {"sampling_strategy": "auto", "random_state": RANDOM_STATE},
        "decision_policy_mode": "mec",
        "costs_matrix": COST_MATRIX,
    },
    "smoteenn_auto__standard": {
        "experiment_name": "smoteenn_auto__standard",
        "description": "SMOTEENN sampling_strategy='auto'. No class_weight. Decision policy: standard (0.5 implicit).",
        "approach": "data_level",
        "tags": ["data_level", "smoteenn", "auto", "standard_policy"],
        "class_weight": None,
        "resampling_method": "SMOTEENN",
        "resampling_params": {"sampling_strategy": "auto", "random_state": RANDOM_STATE},
        "decision_policy_mode": "standard",
        "costs_matrix": COST_MATRIX,
    },
    "smoteenn_auto__mec_fp1_fn10": {
        "experiment_name": "smoteenn_auto__mec_fp1_fn10",
        "description": "SMOTEENN sampling_strategy='auto'. No class_weight. Decision policy: MEC with costs FP=1, FN=10.",
        "approach": "data_level",
        "tags": ["data_level", "smoteenn", "auto", "mec_policy"],
        "class_weight": None,
        "resampling_method": "SMOTEENN",
        "resampling_params": {"sampling_strategy": "auto", "random_state": RANDOM_STATE},
        "decision_policy_mode": "mec",
        "costs_matrix": COST_MATRIX,
    },
}

# ------- evaluation protocol
# outer evaluation setting: RepeatedStratifiedKFold (10 x 10)
CV_OUTER_N_SPLITS = 10
CV_OUTER_N_REPEATS = 10
CV_OUTER_PARALLEL_N_JOBS = 1

# inner evaluation setting for dynamical ensemble models
DSEL_SIZE = 0.25

# inner evaluation setting for hyperparameters tuning: RandomizedSearchCV
TUNING_N_ITER = 30
TUNING_CV_INNER_N_SPLITS = 5
TUNING_SCORING = "Average_Cost"
TUNING_N_JOBS = -1

# ------- models to train
# classic ML model + static ensemble models
STATIC_MODELS = [
    "LogisticRegression",
    "SGDClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
]

STATIC_ENSEMBLE_MODELS = [
    "VotingClassifier",
    "StackingClassifier"
]
STATIC_ENSEMBLE_POOLS = [
    "SGDClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
]

# des model
DES_MODELS = [
    "MLA",
    "KNORAE",
    "DESKL",
    "Exponential",
    "METADES",
]

#################################################################
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
