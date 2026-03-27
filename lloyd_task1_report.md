# Customer Churn Data Report

## 1. Data Gathering and Dataset Selection

The analysis uses a multi-sheet workbook, `Customer_Churn_Data_Large.xlsx`, with five related datasets keyed on `CustomerID`.

### Selected datasets and rationale


| Dataset                 | Shape       | Role in modeling                            | Why included                                                                                                             |
| ----------------------- | ----------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `Customer_Demographics` | `(1000, 5)` | Static customer profile features            | Adds baseline segmentation (`Age`, `Gender`, `MaritalStatus`, `IncomeLevel`) that can influence churn propensity         |
| `Transaction_History`   | `(5054, 5)` | Behavioral/value features after aggregation | Provides purchase intensity and variability signals (`tx_count`, `total_spend`, `avg_spend`, `spend_std`, `active_days`) |
| `Customer_Service`      | `(1002, 5)` | Friction/support features after aggregation | Captures dissatisfaction indicators (`complaint_count`, `unresolved_count`, `unresolved_rate`)                           |
| `Online_Activity`       | `(1000, 4)` | Engagement features                         | Adds digital activity context (`LoginFrequency`, `ServiceUsage`, `LastLoginDate`)                                        |
| `Churn_Status`          | `(1000, 2)` | Target label                                | Defines the supervised learning outcome (`ChurnStatus`)                                                                  |


### Data quality and joinability checks

- All five source sheets show **no missing values** at source-level profiling.
- Key coverage by unique `CustomerID`:
  - `Customer_Demographics`: `1000`
  - `Transaction_History`: `1000`
  - `Customer_Service`: `668`
  - `Online_Activity`: `1000`
  - `Churn_Status`: `1000`
- The lower `Customer_Service` customer coverage is expected and is treated as valid "no interaction" behavior during preprocessing.

### Target distribution

From `Churn_Status`:

- `ChurnStatus = 0`: `0.796` (79.6%)
- `ChurnStatus = 1`: `0.204` (20.4%)

The dataset is moderately imbalanced and should be handled in downstream model evaluation (e.g., stratified splits and class-aware metrics).

---

## 2. Customer-Level Analytical Dataset Construction

A customer-grain analytical table was created by:

1. Aggregating `Transaction_History` to customer-level:
  - `tx_count`, `total_spend`, `avg_spend`, `spend_std`, `first_tx_date`, `last_tx_date`, `active_days`
2. Aggregating `Customer_Service` to customer-level:
  - `svc_count`, `complaint_count`, `unresolved_count`, `first_interaction_date`, `last_interaction_date`, `unresolved_rate`
3. Left-joining all feature sources onto `Churn_Status`

Resulting analytical dataset:

- `customer_df` shape: `**(1000, 22)`**
- Grain: **1 row per customer**

---

## 3. Exploratory Data Analysis (EDA)

EDA combined class balance checks, group-wise churn rates, correlation screening, and visualization-based pattern inspection.

### Statistical summaries from EDA output

**Churn class balance**

- `0`: `0.796`
- `1`: `0.204`

**Churn rate by categorical segment**

- By `Gender`:
  - `M`: `0.211`
  - `F`: `0.197`
- By `MaritalStatus`:
  - `Married`: `0.230`
  - `Single`: `0.205`
  - `Widowed`: `0.196`
  - `Divorced`: `0.185`
- By `IncomeLevel`:
  - `Low`: `0.222`
  - `Medium`: `0.199`
  - `High`: `0.192`
- By `ServiceUsage`:
  - `Mobile App`: `0.231`
  - `Online Banking`: `0.201`
  - `Website`: `0.178`

**Absolute correlation ranking vs `ChurnStatus` (numeric features)**

1. `LoginFrequency`: `-0.082`
2. `svc_count`: `-0.062`
3. `avg_spend`: `0.045`
4. `Age`: `0.029`
5. `unresolved_count`: `-0.020`
6. `spend_std`: `0.019`
7. `unresolved_rate`: `0.017`
8. `tx_count`: `-0.009`
9. `active_days`: `-0.004`
10. `complaint_count`: `0.002`
11. `total_spend`: `0.001`

### Visualizations generated in notebook output

1. **Histograms + KDE (key numeric distributions)**
  Histograms of key numerical features
2. **Scatter plots (pair relationships with churn hue)**
  Scatter relationships with churn status
3. **Box plots by churn status (separation + outlier check)**
  Box plots by churn status

### EDA interpretation highlights

- Engagement appears meaningful: lower `LoginFrequency` aligns with elevated churn tendency.
- Churn rates vary by `ServiceUsage`, suggesting channel/engagement-type effects.
- Socioeconomic segmentation is relevant (`IncomeLevel` has visible churn variation).
- Service friction features remain useful and should be retained in model inputs.

---

## 4. Data Cleaning and Preprocessing

Preprocessing was done on a copy of the analytical table (`model_df`) with explicit handling for missingness, outliers, and feature transformation.

### 4.1 Missing value strategy

For behavior-derived columns where missing often means **no recorded event** after left joins, zero-imputation was applied:

- `tx_count`, `total_spend`, `avg_spend`, `spend_std`, `active_days`
- `svc_count`, `complaint_count`, `unresolved_count`, `unresolved_rate`

For each above feature, a missingness indicator was added before imputation:

- `<feature>_was_missing` (binary flag)

Fallback imputation:

- Remaining numeric NaNs: median imputation
- Remaining categorical NaNs: mode imputation

Notebook output after this step:

- `Remaining missing values: 664`

These residual missings are associated with non-modeled date fields and are addressed by dropping raw date columns prior to model matrix construction.

### 4.2 Outlier handling

IQR capping (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) was applied to:

- `total_spend`, `avg_spend`, `spend_std`, `tx_count`, `svc_count`, `LoginFrequency`

Reported outlier rates before -> after capping:

- `total_spend`: `0.000` -> `0.000`
- `avg_spend`: `0.038` -> `0.000`
- `spend_std`: `0.008` -> `0.000`
- `tx_count`: `0.000` -> `0.000`
- `svc_count`: `0.000` -> `0.000`
- `LoginFrequency`: `0.000` -> `0.000`

### 4.3 Feature preparation for modeling

Columns dropped before feature matrix assembly:

- Identifiers / raw dates:
  - `CustomerID`
  - `first_tx_date`, `last_tx_date`
  - `first_interaction_date`, `last_interaction_date`
  - `LastLoginDate`

Transformations:

- Numeric features: standardized using `StandardScaler`
- Categorical features: one-hot encoded via `pd.get_dummies(..., drop_first=False, dtype=int)`

---

## 5. Cleaned and Preprocessed Dataset (Model-Ready Output)

Final artifacts produced in notebook:

- `X_processed` (model matrix): **shape `(1000, 32)`**
- `y` (target vector): **shape `(1000,)`**

Example processed feature groups include:

- Scaled numerics:
  - `Age`, `LoginFrequency`, `tx_count`, `total_spend`, `avg_spend`, `spend_std`, `active_days`, `svc_count`, `complaint_count`, `unresolved_count`, `unresolved_rate`
- Missingness flags:
  - `tx_count_was_missing`, `total_spend_was_missing`, `avg_spend_was_missing`, `spend_std_was_missing`, `active_days_was_missing`, etc.
- One-hot encoded categoricals:
  - Encodings from profile and usage dimensions (`Gender`, `MaritalStatus`, `IncomeLevel`, `ServiceUsage`)

This final dataset is suitable for immediate model development (train/validation/test split, model fitting, and metric evaluation).

---

## 6. Recommended Next Steps for Model Development

- Use stratified train/validation/test splitting to preserve churn class ratio.
- Benchmark a baseline model first (e.g., logistic regression), then compare tree-based models.
- Evaluate with class-aware metrics (`ROC-AUC`, `PR-AUC`, `F1`, recall for churn class).
- Inspect feature importance and calibration before production-facing decisions.

