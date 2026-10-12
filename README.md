# Leakage-Controlled Verification of Machine Learning Models for Heart Disease Diagnosis and Heart Failure Mortality Prediction

## Overview

This repository documents a verification experiment that evaluates machine-learning models on two small, public cardiovascular datasets. The goal is **not to outperform the base study or claim a better model**. Instead, the experiment checks how reported performance changes when preprocessing, feature selection, and hyperparameter tuning are performed within the training data, and when performance is estimated using repeated nested cross-validation.

The experiment also investigates which variables contribute most to model predictions and evaluates how strongly heart-failure mortality models rely on `time`, the observed follow-up duration. Because follow-up duration is known only after observation has occurred, the configuration that includes `time` is a leakage-sensitivity comparison, not a model suitable for prospective prediction.

## Research questions

1. Do reported results on small public heart datasets remain similar when model selection and feature selection are evaluated using a leakage-controlled protocol?
2. Which features are prioritized by the rank-sum selector and by SHapley Additive exPlanations (SHAP)?
3. How much does the inclusion of follow-up duration change mortality-prediction performance on the Faisalabad heart-failure dataset?

## Datasets and experimental configurations

| Configuration | Dataset and task | Samples described in the manuscript | Predictors | Positive class |
|---|---|---:|---:|---|
| **A: Cleveland diagnosis** (`A_Cleveland_diagnosis`) | Cleveland subset of the UCI Heart Disease dataset; binary heart-disease diagnosis | 303 initially; one exact duplicate was removed, leaving 302 | 13 | `1` = heart disease; target values above 0 were mapped to 1 |
| **B: Faisalabad with follow-up time** (`B_Faisalabad_with_time`) | Heart Failure Clinical Records; death during follow-up | 299 | 12, including `time` | `1` = death (`DEATH_EVENT`) |
| **C: Faisalabad without follow-up time** (`C_Faisalabad_without_time`) | Same 299-patient heart-failure dataset and outcome as B, with `time` excluded | 299 | 11 | `1` = death (`DEATH_EVENT`) |

**Data sources**

- Cleveland Heart Disease dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
- Heart Failure Clinical Records dataset: [Kaggle dataset page](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data).

The manuscript reports 138 Cleveland records with heart disease and 164 without after duplicate removal. For the Faisalabad dataset, it reports 96 deaths and 203 survivors. No exact duplicate rows were detected in the Faisalabad dataset. The source data do not provide patient identifiers that allow patient-level uniqueness or cross-partition overlap to be independently verified.

### Why there are three configurations

- **A** assesses a diagnostic task: whether heart disease is present.
- **B** assesses a prognostic task and retains the `time` feature.
- **C** repeats the Faisalabad task after removing `time`, which could reveal information about the outcome period rather than information available at prediction time.

The Cleveland diagnosis task and Faisalabad mortality task are different prediction problems. Their performance values should not be interpreted as directly interchangeable.

## Models

Nine classifiers are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM) with a radial basis function kernel
5. K-Nearest Neighbors (KNN)
6. Gaussian Naive Bayes
7. XGBoost
8. Neural Network, implemented as a multilayer perceptron (MLP)
9. LightGBM

## Experimental variants

Each classifier is evaluated using four variants in every configuration.

| Variant | Description |
|---|---|
| `default_all` | Default model settings using all available predictors. |
| `default_k4` | Default model settings after rank-sum feature selection retains four predictors. |
| `tuned_all` | Hyperparameter-tuned model using all available predictors. |
| `tuned_selK` | Hyperparameter-tuned model with feature-selection size `K` tuned as part of the search. |

For `tuned_selK`, the recorded candidate values for `K` are `[3, 4, 5, 6, 8, 10]`. The randomized search evaluates up to 20 candidates for `tuned_all` and up to 30 for `tuned_selK`, subject to the number of combinations available. The search optimizes ROC-AUC using an inner five-fold cross-validation procedure. The search spaces are experiment-defined and should not be interpreted as recovered settings from the base study.

## Preprocessing and feature selection

The documented pipeline includes the following steps:

1. **Target preparation.** Cleveland target values are converted to a binary label: 0 for no disease and 1 for disease. In Faisalabad, `DEATH_EVENT` is used as the binary target.
2. **Missing-value handling.** The Cleveland markers `?` and `NA` are treated as missing. Predictor imputation is performed within the fitted modeling pipeline using training-fold medians.
3. **Scaling.** Min-max scaling is applied within the pipeline. The scaler is fitted only on the relevant training data.
4. **Optional rank-sum selection.** On training data only, each predictor is scored using the chi-squared statistic, the ANOVA F-statistic, and mutual information. Each score set is converted to ranks, and the ranks are added. Predictors with the smallest combined rank sums are selected.
5. **Classifier fitting and tuning.** Model fitting and hyperparameter tuning use training data only. Validation and test folds are used only for evaluation.

Preprocessing, feature selection, and tuning are refitted within each outer training fold during nested cross-validation. This reduces information leakage from those fitted steps into the outer test fold.

## Evaluation protocol

### Track 1: hold-out evaluation

- A stratified 80/20 train-test split is used, with random seed `42`.
- The Faisalabad run metadata records 239 training rows and 60 test rows.
- Hyperparameter tuning is conducted using training data only. The hold-out test set is not used to select parameters.
- The reported metrics include accuracy, precision, recall (sensitivity), F1-score, specificity, classification error, ROC-AUC, and confusion-matrix counts.
- A 95% percentile-bootstrap interval is estimated from 1,000 resamples of the hold-out rows. Shared resample indices are used for paired comparisons; bootstrap samples containing only one class are skipped for ROC-AUC.

### Track 2: repeated nested cross-validation

- The outer evaluation uses stratified five-fold cross-validation with two repeats, giving 10 outer-fold scores per model, variant, and configuration.
- Each outer training set contains its own inner stratified five-fold randomized search for tuned variants.
- All rows are used in this track; it does not use the 80/20 hold-out split as its dataset.
- The same outer splits are shared by models and variants so fold-level comparisons are paired.
- The main nested summaries are the mean and standard deviation of accuracy, F1-score, and ROC-AUC across the 10 outer folds.

Repeated nested cross-validation is the primary performance estimate in this study. Hold-out results are supplementary because one split of a small dataset can be sensitive to a few individual predictions.

## Statistical analysis

The combined results file records these statistical analyses:

- Nadeau-Bengio corrected paired t-tests for tuning and feature-selection effects on accuracy and ROC-AUC, with Holm adjustment across models within each comparison.
- Friedman tests and Nemenyi critical-difference summaries for comparing models using paired outer-fold scores.
- Wilcoxon signed-rank comparisons between the best-ranked model and other models, with Holm adjustment.
- Paired bootstrap comparisons of hold-out performance differences between variants.

**Coverage note:** the supplied combined CSV contains statistical-test records for Faisalabad configurations B and C, but no `stat_test` rows for Cleveland configuration A. This README therefore does not claim that the same formal statistical tests are available for Cleveland.

## Explainability

SHAP is applied to the `tuned_all` models using the hold-out rows. Depending on the classifier, the recorded methods include linear, tree, and kernel explainers. The kernel explainer uses 15 background rows and 300 samples per explained row. Because raw SHAP magnitudes can be on different scales across models, the combined results also record each feature's share of the total mean absolute SHAP value within its model.

The rank-sum selector and SHAP are descriptive tools. They indicate which variables the fitted pipeline uses; they do not establish causal relationships.

**Coverage note:** the supplied CSV contains SHAP-importance rows and selector-rank rows for Faisalabad configurations B and C. It does not contain SHAP or selector-rank rows for Cleveland configuration A.

## Key results in the supplied CSV

The values below are rounded to three decimals for readability. They are drawn from `heart_ml_all_results_combined.csv`; nested values are means across the 10 outer folds unless otherwise stated.

| Analysis | Result | Interpretation |
|---|---|---|
| Cleveland, `default_all` Random Forest | Hold-out accuracy **0.902** (55/61); hold-out ROC-AUC **0.954**; nested accuracy **0.836** (SD 0.053); nested ROC-AUC **0.909** (SD 0.033) | The roughly 90% hold-out result is reproduced by the default Random Forest with all predictors. The repeated nested estimate is more modest. |
| Faisalabad, `default_all` | Mean nested ROC-AUC across nine models: **0.828** with `time` and **0.710** without `time`; difference **0.118** | Across this set of models, observed follow-up duration is associated with a substantial increase in apparent discrimination. |
| Faisalabad, `tuned_selK`, with `time` | Highest nested ROC-AUC: XGBoost, **0.915** | This is the highest `tuned_selK` nested AUC recorded for configuration B. |
| Faisalabad, `tuned_selK`, without `time` | Highest nested ROC-AUC: XGBoost, **0.805** | Performance is lower when follow-up duration is excluded. This is the more relevant configuration for predictions intended to be made before follow-up ends. |
| Faisalabad Random Forest SHAP, with `time` | `time` accounts for **39.862%** of total mean absolute SHAP value | The fitted model relies heavily on follow-up duration, which is not available at the intended prediction time. |
| Faisalabad Random Forest SHAP, without `time` | `serum_creatinine` accounts for **40.226%** of total mean absolute SHAP value | After removing `time`, serum creatinine has the largest recorded SHAP share for this model. |

These are verification results, not evidence that one model is universally best. The comparison with previously published accuracy is indirect because the earlier split, seed, and placement of feature selection are not fully specified in the manuscript being examined. The experiment does not show that the earlier study is wrong.

## Combined results file

`heart_ml_all_results_combined.csv` is the consolidated output file supplied for this experiment. It contains 1,709 rows and 58 columns, with records grouped by `configuration` and `record_type`.

| `record_type` | Purpose |
|---|---|
| `run_info` | Experiment settings and selected software metadata. |
| `model_summary` | Hold-out results and aggregate nested-CV metrics for each model and variant. |
| `nestedcv_fold` | Individual outer-fold metric values. |
| `stat_test` | Paired model comparisons, Friedman/Nemenyi summaries, and bootstrap comparisons. |
| `selector_rank` | Combined rank-sum scores and selected-feature indicators. |
| `shap_importance` | SHAP importance values, percentage shares, and feature ranks. |

The CSV records 108 model summaries and 1,080 outer-fold records across the three configurations and four variants. Its statistical-test records, SHAP rows, and feature-selector ranks are not equally available for every configuration; see the coverage notes above.

## Reproducibility and software

- **Random seed:** `42`.
- **Hyperparameter search score:** ROC-AUC.
- **Inner search:** stratified five-fold cross-validation with `RandomizedSearchCV`.
- **Outer evaluation:** stratified five folds repeated twice.
- **Bootstrap resamples:** `1,000`.
- **Recorded scikit-learn version:** `1.6.1`.
- **Recorded SHAP version:** `0.52.0`.
- **XGBoost and LightGBM:** recorded as available, but their exact versions are not included in the combined CSV.

The files supplied for preparing this README include the manuscript and consolidated results CSV, but not the executable training scripts or notebooks. Therefore, an exact run command and a complete pinned dependency file cannot be provided reliably from the supplied artifacts alone. Add the original script or notebook and its tested environment file to the repository before describing it as fully reproducible from source.

## Limitations

- Both datasets are small, public cohorts, and no independent external validation was performed.
- The hold-out track uses one split and one seed.
- Scores from the 10 outer folds are correlated, and the sample size limits the power of statistical comparisons.
- Patient identifiers are unavailable in the Faisalabad data, so potential patient-level overlap cannot be independently checked.
- Configuration B intentionally includes post-outcome follow-up duration. Its scores should not be interpreted as prospective mortality-prediction performance.
- The two datasets represent different tasks, diagnosis and prognosis.
- SHAP values describe the fitted models and are not causal explanations.
- Calibration and clinical utility were not evaluated.

## Data and result consistency checks before publication

Two details in the supplied materials should be resolved before reporting the exact experimental setup as final:

1. **Cleveland split metadata:** the combined CSV's model-summary records report `train_size = 242` and `test_size = 61`, while the manuscript describes 302 usable records and a 241/61 split. The exact rows used to produce the Cleveland summaries should be checked against the preprocessing and split code.
2. **Class weighting:** the instruction prompt says that no class balancing was used, but the CSV's best-parameter records include `class_weight='balanced'` for Logistic Regression in some tuned Faisalabad variants. No SMOTE or sampling-based resampling is documented, but the use of class weighting should be clarified before making a blanket statement that no class balancing was applied.

These are listed rather than silently reconciled because the underlying code is not part of the supplied files.

## Disclaimer

This project is for research and educational purposes only. It is not a medical device and must not be used to diagnose a patient or make clinical decisions.
