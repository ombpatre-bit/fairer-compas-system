# Fairer COMPAS System — Racial Bias Audit & Mitigation

> An AI Ethics investigation into racial disparities in the COMPAS recidivism risk-scoring algorithm, using fairness-aware machine learning to audit and improve upon the original system.

---

## Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Fairness Metrics](#fairness-metrics)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Setup & Usage](#setup--usage)
- [Ethical Considerations](#ethical-considerations)
- [References](#references)

---

## Background

**COMPAS** (Correctional Offender Management Profiling for Alternative Sanctions) is a commercial risk-assessment tool used by courts across the United States to predict the likelihood of a defendant reoffending. Its scores influence critical decisions — including bail, sentencing, and parole.

In 2016, ProPublica published a landmark investigation revealing that COMPAS systematically assigned higher risk scores to Black defendants compared to White defendants with similar criminal histories. This project independently replicates and extends that analysis using modern fairness-aware ML techniques.

---

## Problem Statement

Does the COMPAS algorithm produce **racially biased predictions**, and can a fairer alternative be built without sacrificing meaningful predictive accuracy?

Specifically, this project investigates:
- Whether error rates (false positives and false negatives) differ significantly across racial groups
- Whether demographic parity and equalised odds can be achieved in recidivism prediction
- What trade-offs exist between model accuracy and algorithmic fairness

---

## Project Objectives

1. **Audit** COMPAS scores for evidence of racial disparity using statistical analysis and per-group confusion matrices
2. **Build** a baseline Random Forest recidivism classifier
3. **Evaluate** performance independently for each racial/ethnic group
4. **Mitigate** identified bias using a custom fairness-aware scoring function and hyperparameter tuning
5. **Compare** baseline vs fairness-optimised model across demographic subgroups

---

## Repository Structure

```
fairer-compas-system-main/
│
├── logistics_regression_compas_recidivism_racial_bias.ipynb   # Main analysis notebook
│
├── compas-scores-raw.csv                    # Raw COMPAS assessment scores (Broward County)
├── cox-violent-parsed.csv                   # ProPublica violent recidivism dataset (full)
├── cox-violent-parsed_filt.csv              # Filtered violent recidivism dataset
│
└── propublicaCompassRecividism_data_fairml.csv/
    └── propublica_data_for_fairml.csv       # FairML-formatted ProPublica dataset
```

---

## Datasets

| Dataset | Source | Description |
|---|---|---|
| `compas-scores-raw.csv` | Kaggle / Broward County | Raw COMPAS risk scores including decile scores, charge descriptions, and demographic attributes |
| `cox-violent-parsed.csv` | ProPublica | Two-year follow-up violent recidivism records with COMPAS scores and defendant demographics |
| `cox-violent-parsed_filt.csv` | ProPublica (filtered) | Cleaned subset of the violent recidivism dataset used for modelling |
| `propublica_data_for_fairml.csv` | ProPublica / FairML | Binary-encoded demographic features formatted for fairness-aware ML libraries |

**Key features used in modelling:** `age`, `sex`, `race`, `decile_score`, `priors_count`, `days_in_jail`, `c_days_from_compas`, `is_violent_recid`, `v_decile_score`

**Target variable:** `is_recid` — whether the defendant reoffended within two years

**Racial groups analysed:** African-American, Caucasian, Hispanic, Asian, Native American, Other

---

## Methodology

### 1. Exploratory Data Analysis
- Age distribution across the defendant population
- Risk score variance by race via box plots
- Recidivism frequency broken down by race and gender
- Feature correlation heatmap

### 2. Data Preprocessing
- Dropped personally identifiable and redundant columns
- Imputed missing values (median for numerical, mode for categorical features)
- Label-encoded categorical variables
- Applied `StandardScaler` normalisation

### 3. Baseline Modelling
- Train/test split: **70% / 30%** with stratification by race
- Model: **Random Forest Classifier** (`random_state=42`)
- Evaluation: overall accuracy, precision, recall, F1-score

### 4. Per-Group Evaluation
- Computed classification reports and confusion matrices **independently for each racial group**
- Visualised precision, recall, F1, and support per group in comparative bar charts

### 5. Fairness-Aware Optimisation
- Defined a custom **fairness metric** that penalises demographic parity violations:

  ```python
  fairness_score = accuracy - (alpha × demographic_parity_difference)
  # alpha = 0.5 (balanced trade-off)
  ```

- Used `fairlearn.metrics.demographic_parity_difference` to quantify inter-group prediction gap
- Ran `GridSearchCV` (5-fold CV) optimising for the combined fairness-accuracy score
- Hyperparameters tuned: `n_estimators` ∈ {50, 100, 200}, `max_depth` ∈ {None, 5, 10}, `min_samples_split` ∈ {2, 5, 10}

---

## Fairness Metrics

| Metric | Definition |
|---|---|
| **Demographic Parity Difference** | Gap in the rate of "high risk" predictions between the most and least favoured racial group |
| **Per-group Precision** | Of those predicted to reoffend in each group, what fraction actually did? |
| **Per-group Recall** | Of those who actually reoffended in each group, what fraction were correctly flagged? |
| **Per-group False Positive Rate** | Rate at which non-recidivists are incorrectly labelled high-risk (key equity concern) |
| **Fairness-Accuracy Score** | Composite score balancing predictive power against demographic parity violation |

---

## Key Findings

- **Disparate error rates were confirmed:** African-American defendants had higher false positive rates (labelled high-risk but did not reoffend) compared to Caucasian defendants, consistent with ProPublica's original findings.
- **Per-group accuracy varied** across all six racial categories, with smaller groups (Asian, Native American) showing higher variance due to limited support.
- **The fairness-optimised model** achieved a lower demographic parity difference compared to the baseline, demonstrating that fairness constraints can be incorporated without large accuracy trade-offs.
- **Confusion matrix analysis** showed that the primary source of bias was in the middle decile score range, where race correlated more strongly with prediction errors.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` (Random Forest, GridSearchCV, StandardScaler, LabelEncoder) |
| Fairness auditing | `fairlearn` (demographic_parity_difference) |
| Environment | Jupyter Notebook (Kaggle kernel compatible) |

---

## Setup & Usage

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn fairlearn jupyter
```

### Running the Notebook

```bash
git clone https://github.com/<your-username>/fairer-compas-system.git
cd fairer-compas-system
jupyter notebook logistics_regression_compas_recidivism_racial_bias.ipynb
```

> **Note:** The notebook was originally developed on Kaggle. If running locally, update the data paths at the top of the notebook:
> ```python
> raw_data_path = "./cox-violent-parsed.csv"   # instead of /kaggle/input/...
> ```

### Notebook Sections

1. **Data Loading & Inspection** — load datasets, check shape, nulls, and dtypes
2. **Data Cleaning** — drop redundant columns, impute missing values
3. **Exploratory Analysis** — age, gender, race distributions; risk score box plots
4. **Baseline Modelling** — train Random Forest, evaluate overall performance
5. **Per-Group Evaluation** — classification reports and confusion matrices per ethnicity
6. **Fairness-Aware Tuning** — GridSearchCV with custom fairness scorer
7. **Final Comparison** — accuracy benchmarks across demographic groups

---

## Ethical Considerations

This project engages with sensitive data about real individuals involved in the criminal justice system. Several important ethical points apply:

- **Data represents systemic inequalities.** Recidivism labels reflect re-arrest rates, which are themselves influenced by racially disparate policing practices. A model trained on this data may perpetuate upstream social biases even when "de-biased" by statistical criteria.
- **No model should be used for sentencing decisions in isolation.** Algorithmic risk scores are tools to inform, not replace, human judgement in high-stakes legal contexts.
- **Fairness definitions conflict.** Demographic parity, equalised odds, and individual fairness cannot all be satisfied simultaneously (per the impossibility theorems in algorithmic fairness). The choice of metric reflects a normative value judgement, not a purely technical one.
- **This is a research and educational project.** It is not intended for deployment in any real-world judicial setting.

---

## References

- Angwin, J. et al. (2016). *Machine Bias*. ProPublica. https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- Dressel, J. & Farid, H. (2018). *The accuracy, fairness, and limits of predicting recidivism*. Science Advances.
- Chouldechova, A. (2017). *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*. Big Data, 5(2).
- Northpointe Inc. (2015). *COMPAS Risk & Need Assessment System: User Guide*.
- Kaggle COMPAS Dataset: https://www.kaggle.com/datasets/danofer/compass
- Fairlearn Documentation: https://fairlearn.org/

---

*This project was developed as part of an AI Ethics study exploring how machine learning auditing and fairness-aware optimisation can be applied to real-world algorithmic decision systems.*
