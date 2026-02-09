# Credit Risk Classification — End-to-End Machine Learning Project

## 📌 Problem Statement
The objective of this project is to predict whether a loan applicant is likely to default using historical application data.  
Given the highly imbalanced nature of credit default data, the primary goal is to **identify defaulters effectively (high recall)** while maintaining reasonable precision and stable generalization.

---

## 📊 Dataset Overview
The dataset contains applicant-level information including:
- Demographics (age, gender, education, region)
- Financial details (loan amount, annuity, goods price)
- Employment history
- Housing-related attributes
- External credit bureau scores

The target variable is binary:
- `0` → Non-defaulter  
- `1` → Defaulter

---

## 🧠 Approach & Methodology

### 1. Baseline Modeling
- Built an initial **Logistic Regression** model.
- Identified limitations in recall due to class imbalance and linear assumptions.

### 2. Feature Engineering
- **Ratio features** to encode financial burden:
  - Credit-to-income ratio
  - Annuity-to-income ratio
  - Employment-age ratio
- **Missingness indicators** for external credit scores to preserve information loss during imputation.
- Careful handling of categorical missing values (e.g., occupation).

### 3. Model Selection
- Evaluated **Decision Tree** and **Random Forest** models to capture non-linear interactions.
- Addressed class imbalance using `class_weight="balanced"`.

### 4. Validation & Tuning
- Used **Stratified K-Fold Cross-Validation** for reliable performance estimation.
- Tuned key hyperparameters (`max_depth`, `min_samples_leaf`, `n_estimators`) to improve stability rather than peak performance.

### 5. Error Analysis
- Conducted bias–variance diagnosis using learning curves.
- Performed slice-based analysis (e.g., missing external credit scores, high credit-to-income ratio).
- Identified false positives as the dominant error due to recall prioritization.

### 6. Explainability
- Used **feature importance** and **SHAP** to validate that the model relies on meaningful risk drivers.
- Confirmed that no single feature dominates predictions and that feature effects align with domain intuition.

---

## 🏆 Final Model
**Tuned Random Forest Classifier with class weighting**

Chosen because it:
- Achieves the best balance between recall and precision
- Generalizes stably across folds
- Relies on sensible financial and credit-related features
- Is explainable and defensible for real-world use

---

## 📈 Model Performance Comparison

| Model | ROC-AUC | Recall | Precision | F1-score |
|------|--------|--------|-----------|---------|
| Logistic Regression | 0.7499 | 0.0105 | 0.5778 | 0.0206 |
| Decision Tree | 0.7095 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.7342 | 0.0000 | 0.0000 | 0.0000 |
| Decision Tree (balanced) | 0.7122 | **0.7088** | 0.1373 | 0.2301 |
| Random Forest (balanced) | 0.7311 | 0.6415 | 0.1575 | 0.2529 |
| **Best Random Forest (tuned)** | **0.7348** | **0.6318** | **0.1620** | **0.2579** |

---

## 🔍 Key Insights
- External credit scores are the strongest predictors of default risk.
- Employment stability and proportional financial burden significantly influence predictions.
- Performance degrades for applicants without credit history, indicating inherent data limitations rather than modeling issues.
- The final model prioritizes recall, which is appropriate in conservative credit-risk settings.

---

## ⚠️ Limitations & Future Work
- Decision threshold tuning for business-specific tradeoffs.
- Separate handling or modeling strategy for thin-file applicants.
- Exploration of gradient boosting methods (e.g., XGBoost, LightGBM).
- Cost-sensitive evaluation aligned with real financial impact.

---

## 🛠️ Tech Stack & Requirements
This project uses only standard Python ML libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `shap`

---

## ▶️ How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn shap
2. Run notebooks in order