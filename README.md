# Cost-Sensitive Learning for Myocardial Infarction Mortality Prediction: A Static and Dynamic Ensemble Framework

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter"  alt=""/>
</a>

## 📖 Project Description
This project addresses the critical challenge of predicting intensive care unit (ICU) admission-time mortality in myocardial infarction (MI) patients using highly **imbalanced tabular data**. In the context of MI mortality prediction, survival outcomes vastly outnumber lethal outcomes (approx. 84% vs. 16%), making standard machine learning approaches biased toward the majority class.

The core of this research is a unified comparative analysis between **Static Ensemble Learning** (e.g., Stacking, XGBoost) and **Dynamic Ensemble Selection (DES)** methods. By evaluating how these models behave under severe class imbalance and asymmetric misclassification costs (where predicting a false negative carries severe clinical consequences), this project aims to identify the most robust architecture for early risk stratification.

### 🧪 Handling Class Imbalance
#### Learning-Level Strategies (Training Phase)
The framework implements and compare three main strategies to handle the class distribution during the training phase:
* **Baseline:** A standard, cost-insensitive benchmark utilizing the original imbalanced dataset with no resampling or class weighting.
* **Algorithm-Level (Class-Frequency-Based Weighting):** Implements internal weighting mechanisms (e.g., `class_weight="balanced"`) to automatically adjust weights inversely proportional to class frequencies.
* **Cost-Sensitive Learning (Cost-Matrix-Driven):** Explicitly guides the training algorithm using an asymmetric business cost matrix, applying a heavy penalty to false negatives (`{0:1, 1:10`} where predicting ALIVE when the true class is DEAD costs 10x more than a false alarm).
* **Data-Level (Resampling):** Utilizes `imbalanced-learn` oversampling techniques, specifically `SMOTE`, to synthetically increase the representation of the minority (DEAD) class.

#### Decision-Level Rules (Evaluation Phase)
The framework implements two main strategies to handle the class distribution during the evaluation phase:
* **Standard Threshold:** Models output predictions using the default 0.5 probability threshold.
* **Minimum Expected Cost (MEC):** A strictly post-hoc decision mechanism. For each prediction, it selects the class that minimizes the expected misclassification cost based on the defined asymmetric cost matrix.

In addition to standard classification metrics (ROC-AUC, F1-score, etc.), the framework utilizes a domain-specific **Average Cost per Prediction (AvgCost)** metric. This directly quantifies the asymmetric clinical objective of the model, serving as the primary criterion for overall model evaluation.

---

## 🔬 Core Objectives
1.  **Benchmarking:** Comparing the predictive power of static ensembles (fixed at training) against dynamic ensembles (which adaptively tailor the classifier subset for each specific query instance) for admission-time MI mortality.
2.  **Strategy Comparison:** Evaluating the distinct impacts and trade-offs of modifying the underlying training objective (learning-level strategies) versus applying post-hoc thresholding mechanisms (decision-level rules like MEC).
3.  **Modular Scalability:** Providing a reproducible, pipelined codebase that explicitly prevents data leakage and can easily swap different techniques to find the optimal configuration for any clinically imbalanced dataset.

---

## 🛠 Installation & Environment Setup

Before installing the project, make sure that the following tools are installed and available from your terminal:

- **Python 3.10.x**
- **GNU Make**

This project uses a `Makefile` to automate the setup process. 

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/Cost-Sensitive-Learning-For-Myocardial-Infarction-Mortality-Prediction.git
cd Cost-Sensitive-Learning-For-Myocardial-Infarction-Mortality-Prediction
```

### 2. Create the Virtual Environment
```bash
make create_environment
```

### 3. Activate the Environment
Based on your operating system, run the activation command:
- Linux/macOS:
```bash
source venv/bin/activate
```

- Windows (Command Prompt):
```bash
.\venv\Scripts\activate.bat
```

- Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
Once the environment is activated, install the required libraries (including the dynamic ensemble dependencies):
```bash
make requirements
```

---
## 📁 Directory Structure

```text
├── .gitignore         <- Files and directories to be ignored by Git.
├── LICENSE            <- Open-source license file.
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`.
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- The original, immutable data dump (UCI Myocardial Infarction Complications).
│   ├── interim        <- The cleaned time-slot-specific RAW dataset (filtered missingness, pruned extra targets).
│   ├── processed      <- The final, canonical data sets ready for modeling.
│   └── raw            <- The time-slot-specific dataset (e.g., admission, 24h, 48h, 72h at the ICU).
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details.
│
├── models             <- Trained and serialized models (e.g., StackingClassifier, DESKL) and predictions.
│
├── notebooks          <- Jupyter notebooks for data quality inspection, EDA, and model evaluation.
│
├── pyproject.toml     <- Project configuration file with package metadata and tool configs.
│
├── references         <- Data dictionaries, clinical manuals, and explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics (e.g., Class_imbalance.png, correlation_matrix.png).
│   ├── latex          <- LaTeX source files for documentation.
│   │   ├── presentation
│   │   └── report
│   └── CRISP_DM_Process-Reports.pdf <- Compiled report following the CRISP-DM methodology.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│                         Generated with `pip freeze > requirements.txt` (includes scikit-learn, DESlib).
│
│
└── myocardial_infarction_mortality  <- Source code for use in this project.
    │
    ├── __init__.py
    ├── cleaning.py                  <- CLI script to clean the RAW dataset (filters missingness/targets, removes duplicates).
    ├── config.py                    <- Stores paths, feature schemas, cost matrices (FP=1, FN=10), and hyperparameter grids.
    ├── dataset.py                   <- CLI script to fetch the UCI dataset via `ucimlrepo` and save it to external.
    ├── dataset_time_split.py        <- CLI script to drop future features for specific ICU time slots and create the binary target.
    ├── features.py                  <- CLI script to finalize the PROCESSED dataset by standardizing the target name to `CLASS`.
    │
    ├── data_preparation             <- Modules for data transformations.
    │   ├── __init__.py
    │   ├── data_clean.py            <- Helper functions for data cleaning (e.g., `remove_duplicates`).
    │   ├── data_construct.py        <- Builds ColumnTransformer preprocessing pipelines (imputation, log1p, scaling, OHE, ZSN_A mapping).
    │   ├── feature_selection.py     <- Factory for filter-based SelectKBest (e.g., mutual_info_classif).
    │   └── sampling.py              <- Factory for imblearn resampling strategies (SMOTE, SMOTEENN, etc.).
    │
    ├── evaluation                   <- Modules for model evaluation and metrics.
    │   ├── __init__.py
    │   ├── metrics_evaluation.py    <- Computes standard metrics and applies the MEC decision policy using the cost matrix.
    │   ├── statistical_test_evaluation.py <- Implements the Nadeau-Bengio corrected resampled t-test.
    │   └── visual_evaluation.py     <- Generates pairwise significance heatmaps, ROC stability plots, and ranking charts.
    │
    ├── modeling                     <- Modules for model building, dynamic selection, and training.
    │   ├── __init__.py
    │   ├── train.py                 <- Main Typer CLI script to execute the 10x10 repeated cross-validation experiments.
    │   └── utils
    │       ├── __init__.py
    │       ├── models.py            <- Defines static models, static ensembles (Stacking/Voting), and DES configurations.
    │       ├── pipeline.py          <- Assembles the leakage-safe `imblearn` pipelines (preprocessing -> selection -> resampling -> model).
    │       └── training.py          <- Training loops, RandomizedSearchCV optimization, and fold evaluation logic.
    │
    └── utils                        <- General utility functions.
        ├── __init__.py
        ├── general_utils.py         <- Helpers like `subset_features_by_time_slot`.
        └── io_utils.py              <- File I/O helpers, including `load_dataset_with_schema` for robust dtype enforcement.
   
```
---

## 🧹 Maintenance Commands

The Makefile also includes utility commands for project maintenance:


| Command                       | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| ```make help ```              | Display a list of all available commands and their descriptions. |
| ```make lint ```              | Check code quality and formatting using Ruff.                    |
| ```make format ```            | Automatically fix linting issues and format code.                |
| ```make clean ```             | Remove __pycache__ and compiled Python files                     |
| ```make clean_environment ``` | Completely remove the venv directory.                            |
| ```make freeze ```            | Update the requirements.txt file with current environment state. |

---

## 📊 Data Workflow

The project manages data through a structured pipeline. Before running any analysis, you must retrieve the raw dataset.

### 1. External Dataset Acquisition
The first step is to download the "Myocardial infarction complications" dataset from UC Irvine Machine Learning Repository. This is handled automatically by the ```dataset.py``` script.

**Execution:**

```bash
make dataset
```

**What this command does:**
- Checks if the dataset already exists in ```data/external/```. 
- If missing, it uses ```ucimlrepo``` to download the ```fetch_ucirepo(id=579) ``` dataset. 
- Moves and renames the file to match the project's internal configuration.
 
**Related Configuration** (```myocardial_infarction_mortality/config.py```): The script relies on these path definitions. If you wish to change where data is stored, modify these variables:

| Variable                           | Default Value                             | Description                                                            |
|------------------------------------|-------------------------------------------|------------------------------------------------------------------------|
| ```EXTERNAL_DATA_DIR ```           | ```PROJ_ROOT / "data" / "external"```     | The directory where external raw data is stored.                       |
| ```EXTERNAL_FILENAME ```           | ```myocardial_infarction_external.csv```  | The final filename used by the project scripts.                        |
| ```EXTERNAL_METADATA_FILENAME ```  | ```myocardial_infarction_metadata.txt```  | The name of the metadata file.                                         |
| ```EXTERNAL_VARIABLES_FILENAME ``` | ```myocardial_infarction_variables.csv``` | The name of the file with variables list.                              |
| ```EXTERNAL_DATASET_ID ```         | ```579```                                 | UCI dataset ID to fetch.                                               |
| ```EXTERNAL_FORCE_DOWNLOAD ```     | ```False```                               | If True, re-download and overwrite outputs even if they already exist. |

--- 
## 📉 Data Sampling & Reduction

The second step in the pipeline is to transition from the **External** (full) data to a **Raw** (subsampled) dataset.
The dataset provides 4 different time windows for the training of the machine learning models, based on the ICU time points: 
- the time of admission to hospital
- the end of the first day (24 hours after admission to the hospital)
- the end of the second day (48 hours after admission to the hospital)
- the end of the third day (72 hours after admission to the hospital)

**1. Subsampling Execution**


Use the following command to generate your subsampled dataset based on the current configuration:
```bash
make dataset_time_split
````
**What this command does:**

- Loads the full dataset from `data/external/`.
- Construct a time-slot-specific dataset by dropping unavailable features and adding a binary target (`LET_IS_BINARY`)
- Saves the resulting subset to `data/raw/myocardial_infarction_<selected_time_slot>.csv`.

**2. Sampling Configuration**

You can control how the data is reduced by modifying these variables in `myocardial_infarction_mortality/config.py`.


| Variable                  | Default Value                            | Description                                                                                                                                    |
|---------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| ```EXTERNAL_DATA_DIR ```  | ```PROJ_ROOT / "data" / "external"```    | The directory where external raw data is stored.                                                                                               |
| ```EXTERNAL_FILENAME ```  | ```myocardial_infarction_external.csv``` | The final filename used by the project scripts.                                                                                                |
| ```RAW_DATA_DIR ```       | ```PROJ_ROOT / "data" / "raw"```         | The directory where raw data is stored.                                                                                                        |
| ```FILENAME_BASE ```      | ```myocardial_infarction```              | Base filename (without extension) used to build the output filename.                                                                           |
| ```SELECTED_TIME_SLOT ``` | ```admission```                          | Time slot key used to select which features to drop. Must be a key of``EXCLUDE_FEATURES_BY_SLOT`` (e.g., "admission", "day1", "day2", "day3"). |


---

## 🧹 Data Cleaning & Deduplication
Once the raw sample is created, the next phase of the pipeline is cleaning. This script focuses on creating a clean time-slot-specific RAW dataset by filtering missingness, pruning extra targets, and removing duplicates.

**1. Cleaning Execution**

To process your raw data into a cleaned format, run:
```bash
make cleaning
````
**What this command does:**

- Loads the subsampled data from `data/raw/`.
- Row filtering: drop rows whose missingness fraction is strictly greater than ``threshold_drop_missing_rows``.
- Column filtering: drop columns whose missingness fraction is strictly greater than ``threshold_drop_missing_cols``. The selected ``target`` is never dropped here. The script logs both the number of dropped columns and their names.
- Target pruning: drop any columns listed in ``EXCLUDE_TARGETS`` if present, except the selected ``target``.
- Duplicate removal: remove duplicated rows using ``remove_duplicates`` with the current policy ``subset=None`` and ``keep="first"``. The script logs how many rows were removed.
- Saves the result to `data/interim/myocardial_infarction_<selected_time_slot>_<target>_cleaned.csv`

**2. Cleaning Configuration**

This script relies on the logic defined in `myocardial_infarction_mortality/config.py`. While the core logic is automated, the following paths are used:

| Variable                           | Default Value                        | Description                                                                                                                                    |
|------------------------------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| ```RAW_DATA_DIR ```                | ```PROJ_ROOT / "data" / "raw"```     | The directory where raw data is stored.                                                                                                        |
| ```FILENAME_BASE ```               | ```myocardial_infarction```          | Base filename (without extension) used to build the output filename.                                                                           |
| ```SELECTED_TIME_SLOT ```          | ```admission```                      | Time slot key used to select which features to drop. Must be a key of``EXCLUDE_FEATURES_BY_SLOT`` (e.g., "admission", "day1", "day2", "day3"). |
| ```INTERIM_DATA_DIR ```            | ```PROJ_ROOT / "data" / "interim"``` | The directory where interim data is stored.                                                                                                    |
| ```target ```                      | ```LET_IS_BINARY```                  | Target column that must exist and must be preserved.                                                                                           |
| ```threshold_drop_missing_rows ``` | ```0.20```                           | Drop rows with missingness strictly greater than this fraction.                                                                                |
| ```threshold_drop_missing_cols ``` | ```0.30```                           | Drop columns with missingness strictly greater than this fraction.                                                                             |


---

## 🛠 Feature Engineering & Transformation
The final stage of the data pipeline converts cleaned data into a **Processed** dataset. Create the PROCESSED (features) dataset from the cleaned INTERIM dataset and standardize the target name.

**1. Feature Engineering Execution**

To generate the final features for your models, run:
```bash
make features
````

**What this command does:**

- The target column specified by ``target`` (default: ``"LET_IS_BINARY"``) is renamed to ``target_alias`` (default: ``"CLASS"``) to enforce a consistent target name for modeling.
- Saves the result to `data/processed/myocardial_infarction_<selected_time_slot>_<target>_features.csv`.

**2. Configuration & Parameters**

This script relies on the logic defined in `myocardial_infarction_mortality/config.py`.

| Variable                  | Default Value                          | Description                                                                                                                                    |
|---------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| ```INTERIM_DATA_DIR ```   | ```PROJ_ROOT / "data" / "interim"```   | The directory where interim data is stored.                                                                                                    |
| ```FILENAME_BASE ```      | ```myocardial_infarction```            | Base filename (without extension) used to build the output filename.                                                                           |
| ```SELECTED_TIME_SLOT ``` | ```admission```                        | Time slot key used to select which features to drop. Must be a key of``EXCLUDE_FEATURES_BY_SLOT`` (e.g., "admission", "day1", "day2", "day3"). |
| ```PROCESSED_DATA_DIR ``` | ```PROJ_ROOT / "data" / "processed"``` | The directory where processed data is stored.                                                                                                  |
| ```target ```             | ```LET_IS_BINARY```                    | Selected target column.                                                                                                                        |
| ```target_alias ```       | ```CLASS```                            | Standardized target name used inside the processed dataset (i.e., the column is renamed to this value).                                        |

---
## ⚙️ ️ Model Training & Evaluation
**1. The Cost Matrix & Decision Policy**

Traditional machine learning assumes all classification errors are equal. We override this using a specific clinical cost matrix where a **Missed Death (FN) is penalized 10x more heavily than a False Alarm (FP)**:
* **TN (True ALIVE predicted ALIVE):** Cost = 0
* **FP (True ALIVE predicted DEAD):** Cost = 1
* **FN (True DEAD predicted ALIVE):** Cost = 10
* **TP (True DEAD predicted DEAD):** Cost = 0

**2. Experimental Schemas**

The `EXPERIMENTS` dictionary defines multiple configurations to test which approach best handles the severe class imbalance:
* **Baseline:** Standard models using a default `0.5` decision threshold.
* **Cost-Sensitive Learning (CSL):** Adjusting `class_weight` (e.g., `{0: 1, 1: 10}`) directly within the model algorithms.
* **Data-Level Resampling:** Utilizing `SMOTE` or `SMOTEENN` to synthetically balance the training data distributions.
* **Minimum Expected Cost (MEC) Policy:** Shifting the final probability decision threshold to mathematically minimize the overall risk based on the cost matrix.

The training pipeline supports the following experiment configurations, selected through the `experiment_id` argument.

#### Baseline experiments

- `baseline__standard`
  - No resampling
  - No class weighting
  - Standard decision policy

- `baseline__mec_fp1_fn10`
  - No resampling
  - No class weighting
  - Minimum Expected Cost (MEC) decision policy with `FP=1`, `FN=10`

#### Cost-sensitive learning experiments

- `csl_balanced__standard`
  - `class_weight="balanced"`
  - No resampling
  - Standard decision policy

- `csl_balanced__mec_fp1_fn10`
  - `class_weight="balanced"`
  - No resampling
  - MEC decision policy with `FP=1`, `FN=10`

- `csl_fp1_fn10__standard`
  - `class_weight={0: 1, 1: 10}`
  - No resampling
  - Standard decision policy

- `csl_fp1_fn10__mec_fp1_fn10`
  - `class_weight={0: 1, 1: 10}`
  - No resampling
  - MEC decision policy with `FP=1`, `FN=10`

#### Data-level imbalance handling experiments

- `smote_auto__standard`
  - Resampling: `SMOTE`
  - `sampling_strategy="auto"`
  - Standard decision policy

- `smote_auto__mec_fp1_fn10`
  - Resampling: `SMOTE`
  - `sampling_strategy="auto"`
  - MEC decision policy with `FP=1`, `FN=10`

- `smoteenn_auto__standard`
  - Resampling: `SMOTEENN`
  - `sampling_strategy="auto"`
  - Standard decision policy

- `smoteenn_auto__mec_fp1_fn10`
  - Resampling: `SMOTEENN`
  - `sampling_strategy="auto"`
  - MEC decision policy with `FP=1`, `FN=10`

**3. Supported Models to Train**

The pipeline trains three model families: static single models, static ensemble models, and dynamic ensemble selection (DES) models.

#### Static single models

- `LogisticRegression`
- `SGDClassifier`
- `DecisionTreeClassifier`

#### Static ensemble models (homogeneous)

- `RandomForestClassifier`
- `XGBClassifier`

#### Static ensemble models (heterogeneous)

- `VotingClassifier`
- `StackingClassifier`


##### Heterogeneous static ensemble pool members

These models are used as the base pool for heterogeneous static ensembles:

- `SGDClassifier`
- `RandomForestClassifier`
- `XGBClassifier`

#### Dynamic ensemble selection (DES) models

- `MLA`
- `KNORAE`
- `DESKL`
- `Exponential`
- `METADES`

**4. Training**

The training pipeline uses a rigorous **Repeated Nested Stratified Cross-Validation** approach to ensure the models generalize reliably to unseen patients.
The **outer cross validation (k=10)** is used to assess the generalization ability of the model on the unseen dataset, while the **inner cross validation (k=5)** 
is used to evaluate the quality of each hyperparameters combination during the tuning phase. To tune each model the **Random Search** algorithm is adopted with the **number of combinations equal to 30**. The whole process is **repeated 10 times**. 

For non-ensemble and static ensemble models the process is the same. For DES models the training phase is divided into two steps: first, train the pool of classifiers; secondly, train the DES model. Thus, the outer training folds are split in a smaller training dataset (to tune and train the pool of classifiers, in the same way used for non-ensemble and static ensemble models)
and the DSEL dataset (0.25 % of the outer training folds) to train the DES models. 
 
**5. Execution**

To execute the training pipeline based on the currently active `EXPERIMENT_ID` (e.g., `smoteenn_auto__mec_fp1_fn10`), run the following command:

```bash
make train
```

---
## 📓 Notebooks & In-depth Analysis
The `notebooks/` directory contains the experimental logs and visual analyses of the project. These are organized chronologically to mirror the research workflow.

**Data Understanding (DU) & Preparation (DP)**
- `1.0-ls-DU-data-quality-inspection`: Initial assessment of the external dataset to identify missing values, data types, and potential inconsistencies.
- `2.0-ls-DU-data-exploration`: Comprehensive Exploratory Data Analysis (EDA) focused on feature distributions and the extreme skewness of the target class.
- `3.0-ls-DP-data-exploration-cleaned-dataset`:  Assessment of the dataset quality after the cleaning step.

**Evaluation (EV) & Experimental Results**
- `4.0-ls-EV-feature-selection-analysis`: A post-hoc analysis of the feature selection process, identifying which variables were most frequently selected during hyperparameter tuning across different models.
- `5.0-ls-EV-models-evaluation`: Detailed performance breakdown for individual models within a specific experimental configuration.
- `5.1-ls-EV-models-evaluation-comparison`: Comparison of different models within the same experiment setting, utilizing qualitative plots, quantitative metrics, and **Resampled Corrected t-tests** for statistical significance.
- `5.2-ls-EV-experimental-schemas-comparison`: An "All-vs-All" comparison across different experiment settings to determine the optimal pipeline.

---
## License

This project is released under the [LICENSE](LICENSE). See the LICENSE file for details.
