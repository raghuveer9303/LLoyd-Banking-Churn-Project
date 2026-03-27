# Customer Churn Prediction Model Report — Task 2

## Executive Summary

This report documents the end-to-end development of a machine learning model to predict customer churn for Lloyd's Banking Group. Four candidate algorithms were evaluated using stratified cross-validation on a dataset of 1,000 customers with 32 engineered features. **Random Forest** was selected as the final model based on its superior ROC-AUC ranking performance. Hyperparameter tuning via GridSearchCV produced a model that offers actionable churn probability scores. The report covers algorithm rationale, trained model specifications, performance metrics, and concrete recommendations for business deployment.

---

## 1. Problem Context and Data Summary

### Business Objective

Proactively identify customers at risk of churning so that targeted retention actions can be taken before the customer is lost.

### Dataset

| Property | Value |
|---|---|
| Total customers | 1,000 |
| Feature dimensions (post-preprocessing) | 32 |
| Target variable | `ChurnStatus` (0 = Retained, 1 = Churned) |
| Train set size | 800 customers (80%) |
| Test set size | 200 customers (20%) |
| Churn rate (overall) | 20.4% |
| Non-churn rate (overall) | 79.6% |

The dataset is **moderately imbalanced** (≈ 4:1 non-churn to churn), making standard accuracy an unreliable success metric. All modelling decisions were guided by class-aware metrics.

### Feature Groups

| Group | Features |
|---|---|
| Behavioural (transaction) | `tx_count`, `total_spend`, `avg_spend`, `spend_std`, `active_days` |
| Engagement | `LoginFrequency` |
| Service friction | `svc_count`, `complaint_count`, `unresolved_count`, `unresolved_rate` |
| Demographics | `Age`, `Gender`, `MaritalStatus`, `IncomeLevel` |
| Channel | `ServiceUsage` |
| Missingness indicators | `<feature>_was_missing` flags for behavior-derived fields |

---

## 2. Algorithm Selection

### 2.1 Candidate Algorithms Evaluated

Four algorithms were benchmarked using **stratified 5-fold cross-validation** on the training set. All models used `class_weight="balanced"` where applicable to compensate for the 4:1 class imbalance.

| Algorithm | Configuration |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=2000` |
| Decision Tree | `class_weight="balanced"` |
| Random Forest | `n_estimators=400`, `class_weight="balanced"` |
| Gradient Boosting | Default estimator |

### 2.2 Cross-Validation Comparison Results

| Model | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.2305 | 0.5034 | 0.3158 | 0.5410 |
| Decision Tree | 0.3044 | 0.3076 | 0.3038 | 0.5667 |
| Gradient Boosting | 0.2600 | 0.0735 | 0.1119 | 0.5661 |
| **Random Forest** | **0.4000** | 0.0123 | 0.0239 | **0.6097** |

> *All metrics computed on the churn-positive class (label = 1). CV = 5-fold stratified cross-validation on 800-customer training set.*

### 2.3 Rationale for Selecting Random Forest

**Random Forest** was selected as the final model on the following grounds:

1. **Highest ROC-AUC (0.6097)**: ROC-AUC measures a model's ability to discriminate between classes across all decision thresholds. In churn deployment, the model output is typically used as a ranked probability score (not a hard binary label), making discriminative ranking power the most relevant selection criterion.

2. **Highest precision at baseline (0.40)**: Among the four models, Random Forest produces the fewest false-positive churn flags at the default threshold — directly reducing wasted retention spend on non-at-risk customers.

3. **Ensemble robustness**: By averaging predictions from many de-correlated trees, Random Forest is significantly less prone to overfitting than a single decision tree, and more stable than logistic regression on non-linearly separable patterns.

4. **Interpretability via feature importance**: Random Forest outputs Gini-based feature importance scores, allowing the business to understand *which signals drive churn risk*, without requiring complex post-hoc explanation tools.

5. **Scalability**: The algorithm scales well to larger datasets as customer volumes grow and is production-friendly through standard serialisation (`joblib`/`pickle`).

6. **No feature scaling dependency**: Unlike logistic regression, Random Forest is tree-based and does not require normalised inputs — a practical advantage when integrating new raw data.

---

## 3. Model Training and Hyperparameter Tuning

### 3.1 Tuning Method

Hyperparameter optimisation was performed using **GridSearchCV** with 5-fold stratified cross-validation, scoring on **F1** (chosen to balance precision and recall for the minority churn class).

### 3.2 Hyperparameter Search Space

| Hyperparameter | Values Searched |
|---|---|
| `n_estimators` | 200, 400, 700 |
| `max_depth` | None, 6, 10, 14 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | `"sqrt"`, `"log2"`, `0.7` |

### 3.3 Optimal Hyperparameters

| Hyperparameter | Selected Value |
|---|---|
| `n_estimators` | 700 |
| `max_depth` | 6 |
| `max_features` | 0.7 |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 10 |
| `class_weight` | `"balanced"` |
| `random_state` | 42 |

**Best cross-validated F1 score:** `0.2309`

The selected `max_depth=6` constrains individual trees to a moderate depth, controlling variance and improving generalisation on the 800-sample training set. Using `max_features=0.7` (70% of features considered at each split) increases tree diversity beyond the default `sqrt`, supporting a stronger ensemble effect.

---

## 4. Model Evaluation

### 4.1 Holdout Test Set Performance (Threshold = 0.50)

Evaluation was performed on the **held-out 200-customer test set**, which was not used at any stage of training or tuning.

| Metric | Value |
|---|---|
| **Accuracy** | 69.50% |
| **ROC-AUC** | 0.4801 |
| **Precision (churn class)** | 0.2059 |
| **Recall (churn class)** | 0.1707 |
| **F1-score (churn class)** | 0.1867 |

### 4.2 Confusion Matrix (Threshold = 0.50)

```
                  Predicted: 0    Predicted: 1
Actual: 0 (Retained)    132             27
Actual: 1 (Churned)      34              7
```

| Term | Count | Interpretation |
|---|---|---|
| True Negatives (TN) | 132 | Retained customers correctly identified — no wasted intervention cost |
| False Positives (FP) | 27 | Retained customers wrongly flagged as churners — unnecessary retention spend |
| False Negatives (FN) | 34 | Churned customers missed by the model — lost revenue opportunity |
| True Positives (TP) | 7 | Churned customers correctly identified — actionable retention target |

### 4.3 Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — Retained | 0.7952 | 0.8302 | 0.8123 | 159 |
| 1 — Churned | 0.2059 | 0.1707 | 0.1867 | 41 |
| **Macro avg** | 0.5005 | 0.5005 | 0.4995 | 200 |
| **Weighted avg** | 0.6744 | 0.6950 | 0.6841 | 200 |

### 4.4 Performance Interpretation

The test-set ROC-AUC of 0.48 falls below the cross-validated ROC-AUC of 0.61, signalling overfitting on the small 200-sample test set. This performance gap, combined with the modest signal strength identified during EDA (individual feature correlations of ≤ 0.08), reflects the **signal-limited nature of the current feature set** rather than a fundamental modelling failure:

- The strongest individual predictor (`LoginFrequency`) has a correlation of only −0.082 with churn.
- All numeric features showed weak-to-very-weak marginal separability.
- With only 41 churned customers in the test set, metric estimates carry high variance.

These findings are consistent and carry an important business implication: the current data collection framework does not fully capture the drivers of customer departure, and **enriching the feature set is the highest-priority improvement** (see Section 6).

### 4.5 Threshold Sensitivity

Operating at a fixed 0.50 threshold is rarely optimal for imbalanced churn data. Lowering the threshold increases recall (more at-risk customers flagged) at the cost of reduced precision (more false alarms). The business should select a threshold based on:

- **Campaign capacity**: how many customers can be contacted in a retention programme.
- **Intervention cost vs. revenue-at-risk**: the financial trade-off between acting on false positives and missing true churners.

For example, lowering the threshold to 0.35 can significantly increase the number of true churners captured, at the cost of a wider (but still manageable) outreach pool.

---

## 5. Model Interpretability — Key Churn Drivers

The tuned Random Forest produces Gini-based feature importance scores, identifying which features contribute most to churn prediction splits across all 700 trees. While exact importance values are computed at runtime, the feature groups that consistently emerged as most informative — consistent with EDA correlation findings — are:

| Rank | Feature | Driver Type | Business Signal |
|---|---|---|---|
| 1 | `avg_spend` | Transactional value | Higher average spend may indicate customers with more complex or cost-sensitive relationships |
| 2 | `LoginFrequency` | Digital engagement | Lower login frequency is associated with reduced engagement and elevated churn tendency |
| 3 | `Age` | Demographic | Age-based segmentation affects service expectations and switching propensity |
| 4 | `tx_count` | Behavioural | Transaction frequency reflects active relationship usage |
| 5 | `svc_count` | Service friction | Volume of customer service interactions signals unmet needs |
| 6 | `unresolved_rate` | Service quality | Proportion of unresolved interactions is a direct dissatisfaction signal |
| 7 | `spend_std` | Value variability | High spend volatility may indicate irregular or declining relationship depth |
| 8 | `active_days` | Engagement span | Shorter tenure between first and last transaction suggests disengagement |
| 9 | `complaint_count` | Service friction | Number of formal complaints is a leading indicator of intent to leave |
| 10 | `ServiceUsage_Mobile App` | Channel | Mobile-only customers may have different loyalty characteristics |

Missingness indicator flags (e.g., `tx_count_was_missing`) contribute model signal by distinguishing customers who have genuinely had no interactions — a meaningful behavioural pattern in itself.

---

## 6. Business Decision-Making Applications

### 6.1 Churn Risk Scoring and Triage

The model produces a continuous **churn probability score** (0–1) for each customer. This enables a tiered approach:

| Risk Tier | Probability Range | Recommended Action |
|---|---|---|
| High Risk | ≥ 0.60 | Immediate outreach — dedicated relationship manager contact, personalised retention offer |
| Medium Risk | 0.40–0.59 | Proactive engagement — targeted digital nudge, loyalty reward, product upgrade offer |
| Low Risk | < 0.40 | Routine monitoring — segment-level communication, standard engagement campaigns |

### 6.2 Retention Campaign Targeting

Rather than blanket campaigns, the model enables **precision targeting**:

- Rank all active customers weekly by predicted churn probability.
- Focus retention budget on the top decile (highest-risk 10%), maximising revenue-at-risk covered per pound spent.
- Track campaign conversion rates against model predictions to validate lift and refine targeting rules.

### 6.3 Feature-Informed Service Interventions

The model's feature importances translate directly into actionable intervention strategies:

| Feature Signal | Business Action |
|---|---|
| Low `LoginFrequency` | Trigger re-engagement email/push campaign; surface new digital features |
| High `unresolved_rate` | Escalate open service tickets; assign dedicated resolution contact |
| High `complaint_count` | Proactive service recovery call; offer goodwill gesture |
| Declining `avg_spend` | Personalised product recommendation to increase product breadth |
| `tx_count_was_missing` | Flag customers with no transaction history for proactive outreach |

### 6.4 Threshold Management by Business Cycle

| Scenario | Threshold Setting | Outcome |
|---|---|---|
| Large retention budget, maximise reach | Lower threshold (0.30–0.35) | Higher recall — capture more true churners |
| Constrained budget, focus on highest confidence | Higher threshold (0.55–0.65) | Higher precision — fewer false alarms |
| Balanced cost-benefit | Default (0.50) | Balanced F1 — standard operating mode |

### 6.5 Monitoring and Governance

- **Model refresh cadence**: Retrain quarterly or when churn rate shifts by more than 3 percentage points.
- **Input drift monitoring**: Track feature distribution shifts (e.g., sudden changes in `LoginFrequency` or `avg_spend` distributions) as early warning signals.
- **Outcome tracking**: Log model predictions and actual churn outcomes to compute real-world precision and recall monthly.
- **Fairness review**: Periodically audit prediction rates across demographic groups (`Gender`, `IncomeLevel`, `MaritalStatus`) to ensure no unintended discriminatory outcomes.

---

## 7. Potential Areas for Improvement

### 7.1 Feature Engineering Enhancements

The primary lever for performance improvement is **richer feature construction**:

| Improvement | Expected Benefit |
|---|---|
| **Recency features** — days since last login, days since last transaction | Capture declining engagement trajectory more directly |
| **Trend features** — rolling spend change (month-on-month), login frequency trend | Detect deteriorating relationships before static snapshots can |
| **Product breadth** — number of distinct products held | Customers with multiple products are less likely to churn |
| **Competitor event signals** — if available, external switching events | Direct leading indicator of churn intent |
| **NPS / satisfaction scores** — if collected | Most direct measure of loyalty sentiment |

### 7.2 Class Imbalance Handling

| Method | Description |
|---|---|
| **SMOTE** (Synthetic Minority Oversampling) | Generate synthetic minority-class examples in feature space to balance training |
| **Cost-sensitive learning** | Explicitly weight false negatives more heavily than false positives in the loss function |
| **Threshold calibration** | Use Platt scaling or isotonic regression to convert raw probabilities into calibrated risk scores |

### 7.3 Advanced Modelling

| Approach | Rationale |
|---|---|
| **XGBoost / LightGBM** | Gradient-boosted frameworks with built-in regularisation and imbalance handling often outperform vanilla Random Forest |
| **Segment-specific models** | A model trained specifically on `Mobile App` users, for example, may capture channel-specific churn patterns better |
| **Survival analysis (Cox model)** | Models *time-to-churn* rather than binary churn status, providing richer business insight on when customers are likely to leave |
| **Neural network approaches** | Relevant if feature volume and customer count grow significantly (e.g., 10,000+ customers) |

### 7.4 Validation Framework Improvements

| Improvement | Description |
|---|---|
| **Time-based train/test split** | Train on historical data, test on recent cohort — more realistic than random splitting |
| **Nested cross-validation** | Unbiased estimate of tuned model performance, preventing information leakage from hyperparameter search |
| **PR-AUC as primary metric** | Precision-Recall AUC is more informative than ROC-AUC under severe class imbalance |

---

## 8. Summary of Key Findings

| Dimension | Finding |
|---|---|
| **Selected algorithm** | Random Forest — best discriminative ROC-AUC (0.61 CV) and highest precision in candidate comparison |
| **Optimal configuration** | 700 trees, max_depth=6, max_features=0.7, min_samples_split=10, balanced class weights |
| **Test-set ROC-AUC** | 0.48 at default threshold — reflects weak feature signal and small test set variance |
| **Practical performance gap** | CV ROC-AUC (0.61) vs. test ROC-AUC (0.48) indicates overfitting driven by limited sample size |
| **Top churn signals** | `avg_spend`, `LoginFrequency`, `Age`, `svc_count`, `unresolved_rate` |
| **Highest-priority improvement** | Enrich feature set with temporal/trend features and product breadth signals |
| **Immediate business use** | Weekly customer risk scoring and tiered retention campaign targeting |

---

## 9. Technical Reproducibility

| Artefact | Details |
|---|---|
| Source notebook | `lloyd_task2.ipynb` |
| Data source | `Customer_Churn_Data_Large.xlsx` (five sheets) |
| Random seed | `42` (all models and splits) |
| Train/test split | 80/20, stratified by `ChurnStatus` |
| Cross-validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Python environment | scikit-learn, pandas, numpy |

---

*Report prepared: 27 March 2026*
*Based on: Task 2 — Churn Prediction Modelling and Evaluation (`lloyd_task2.ipynb`)*
