<p align="center">
  <img src="assets/images/project_banner.png" width="100%" alt="Ad click prediction project banner">
</p>

# Ad Click Prediction

An end-to-end machine learning case study that predicts rare ad clicks from
user, product, campaign, webpage, and temporal context. The project turns an
imbalanced digital-advertising dataset into business-ready CTR insights,
model comparisons, test predictions, and a compact submission PDF.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Weighted%20Classifier-FF6600)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Feature%20Engineering-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Case%20Study-F37626?logo=jupyter&logoColor=white)](ad_click_prediction_case_study.ipynb)
[![Report](https://img.shields.io/badge/Submission-PDF-B91C1C)](reports/ad_click_prediction_case_study_submission.pdf)
[![Last Commit](https://img.shields.io/github/last-commit/banshiAbp/ad-click-prediction)](https://github.com/banshiAbp/ad-click-prediction/commits/main)

> **Verified outcome:** The final temporal-holdout model selection favored
> **Random Forest with class balancing and a tuned threshold**. It achieved
> **ROC-AUC 0.574**, **F1 0.142**, **recall 0.560**, and **precision 0.081**
> on the latest-day validation window. The low precision is expected for a
> rare-click problem with a training click rate of only **6.76%**.

## Why This Project?

Digital advertising is a high-volume, low-click environment. Most impressions
do not lead to clicks, yet each irrelevant impression can waste budget and
degrade user experience. A useful CTR model must do more than predict the
majority class: it must identify subtle click-propensity patterns while
managing false positives and false negatives.

This project builds a reproducible classification workflow that answers both
technical and business questions: which products perform best, whether weekends
behave differently, which user profiles show stronger click intent, whether
personalized features improve signal, and whether SMOTE is worth the extra
training footprint for real-time ad serving.

## Project Highlights

- Temporal validation that mimics future ad-serving conditions
- Time features from impression timestamp: hour, weekday, weekend, night, and business-hour flags
- Personalized interaction features: user-product, campaign-product, and webpage-product context
- Smoothed historical CTR aggregates for products, campaigns, webpages, user groups, categories, and segments
- Missing-value handling and categorical encoding inside reproducible sklearn pipelines
- Comparison of balanced Logistic Regression, balanced Random Forest, weighted XGBoost, and lightweight SMOTE-style sampling
- Threshold tuning focused on F1 and recall for rare click detection
- Product CTR, profile CTR, feature-importance, inventory, and bidding-strategy insights
- Final session-level test predictions with click probabilities and predicted labels
- Submission-ready PDF under both page and file-size limits

## Project Workflow

```mermaid
flowchart LR
    A[Raw Ad Impression Data] --> B[Data Quality Review]
    B --> C[Temporal Feature Engineering]
    C --> D[CTR Aggregates and Interactions]
    D --> E[Imbalance Handling]
    E --> F[Model Comparison]
    F --> G[Threshold Tuning]
    G --> H[Business Interpretation]
    H --> I[Test Predictions and PDF]
```

## Operations Performed

| Stage | Operations |
|---|---|
| Data loading | Imported train/test CSV files and parsed `DateTime` |
| Data understanding | Checked shape, target imbalance, missingness, unique values, and train/test structure |
| Feature engineering | Added temporal flags, personalized interaction keys, and smoothed CTR aggregate features |
| Preprocessing | Imputed categorical/numeric fields, one-hot encoded categories, and scaled numeric CTR features |
| Validation design | Used older dates for development and latest date as holdout validation |
| Imbalance strategy | Compared class weighting, threshold tuning, and lightweight SMOTE-style oversampling |
| Model training | Trained Logistic Regression, Random Forest, and XGBoost classifiers |
| Evaluation | Compared ROC-AUC, PR-AUC, F1, precision, recall, TP, FP, TN, and FN |
| Interpretability | Reviewed feature-importance groups and business-level CTR tables |
| Business output | Answered product, weekend, profile, SMOTE, bidding, and inventory questions |
| Submission | Generated consolidated notebook, compact PDF, charts, artifacts, and test predictions |

## Dataset

| Split | Rows | Columns | Purpose |
|---|---:|---:|---|
| Training | 463,291 | 15 | Model development, temporal validation, feature engineering, and final training |
| Test | 128,858 | 14 | Final ad-click probability scoring |

Target: `is_click`

Observed training click rate: **6.76%**

Core fields include:

- `session_id`
- `DateTime`
- `user_id`
- `product`
- `campaign_id`
- `webpage_id`
- `product_category_1`
- `product_category_2`
- `user_group_id`
- `gender`
- `age_level`
- `user_depth`
- `city_development_index`
- `var_1`

The original dataset is referenced in the project requirement as:
[Ad click prediction data](https://drive.google.com/drive/folders/1ySbTboXX1_gzWexW8AsvFFy313uEaDnh?usp=sharing)

## Project Structure Philosophy

The repository now follows a clean, industry-style ML layout:

- `assets/data/` stores raw source datasets and project input documents.
- `assets/images/` stores README visuals and generated chart assets.
- `outputs/` stores machine-generated prediction files and metric snapshots.
- `reports/` stores final presentation/submission documents.
- The root stays focused on the executable notebook and project documentation.

## Final Results

| Model | Threshold | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest - balanced - tuned threshold** | **0.458** | **0.574** | 0.074 | **0.142** | 0.081 | 0.560 |
| XGBoost - weighted - tuned threshold | 0.476 | 0.566 | **0.076** | 0.138 | 0.080 | 0.488 |
| Logistic Regression - balanced - tuned threshold | 0.072 | 0.576 | 0.075 | 0.133 | 0.074 | **0.654** |

The final selected model favors the best F1 balance on the temporal holdout.
Logistic Regression produced higher recall, but at a larger false-positive cost.
This makes Random Forest the more balanced choice for the submitted operating
point.

## Business Questions Answered

### Weekend vs Weekday Users

Weekend and weekday CTR were compared directly. The weekend signal is useful as
a monitored bid modifier, but it should not be used as a standalone targeting
rule without live stability checks.

### Product CTR Performance

Products were ranked by historical CTR and expected clicks per 100,000
impressions. High-CTR products can receive more reserved inventory, while
low-performing products should be reviewed for creative quality, targeting fit,
or landing-page mismatch.

### Personalized Feature Impact

User-product, campaign-product, and webpage-product interaction features help
capture affinity and context that raw IDs cannot express alone. These features
are especially valuable in ad-tech because user intent often emerges from
combinations of user profile, product, campaign, and placement.

### Feature Importance

CTR aggregate features and contextual placement variables are the most useful
feature families. They can be amplified through bid multipliers, placement
quality monitoring, product-level creative refreshes, and segment-specific
campaign rules.

### SMOTE Trade-Off

SMOTE-style oversampling can improve rare-event sensitivity in offline
experiments, but it increases training data size and may complicate real-time
refresh workflows. For production ad serving, class weighting plus threshold
tuning is the preferred first-line strategy.

### Inventory Forecasting

Aggregated product CTR can forecast expected clicks for a planned block of
impressions. Products with higher expected clicks per inventory block should
receive stronger inventory reservations and faster creative iteration.

### User Profile Strategy

High-propensity user profiles by age, gender, and city development index can
inform bid adjustments. These recommendations should use minimum-volume
thresholds and should be monitored for fairness, privacy, and drift.

## Generated Artifacts

| File | Description |
|---|---|
| `ad_click_prediction_case_study.ipynb` | Executed consolidated notebook |
| `reports/ad_click_prediction_case_study_submission.pdf` | Final submission PDF |
| `outputs/ad_click_prediction_test_predictions.csv` | Test click probabilities and predicted labels |
| `outputs/ctr_artifacts.json` | Model metrics and business insight snapshot |
| `assets/images/product_ctr.png` | Product CTR chart |
| `assets/images/model_f1.png` | Model F1 comparison chart |
| `assets/images/feature_importance.png` | Feature importance chart |

Submission PDF checks:

- **11 pages**
- **0.13 MB**
- Below the required **50-page** and **20 MB** limits

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/banshiAbp/ad-click-prediction.git
cd ad-click-prediction
```

### 2. Create and activate an environment

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn pypdf nbformat nbconvert jupyter
```

### 4. Run the complete case study

```bash
jupyter notebook ad_click_prediction_case_study.ipynb
```

Run all cells from top to bottom. The notebook performs data preparation,
feature engineering, model comparison, threshold tuning, business analysis,
chart generation, and final test prediction export.

## Repository Structure

```text
.
|-- assets/
|   |-- data/
|   |   |-- Ad_click_prediction_train (1).csv
|   |   |-- Ad_Click_prediciton_test.csv
|   |   `-- Click-Through Rate (CTR) Prediction Project.pdf
|   `-- images/
|       |-- project_banner.png
|       |-- feature_importance.png
|       |-- model_f1.png
|       `-- product_ctr.png
|-- outputs/
|   |-- ad_click_prediction_test_predictions.csv
|   `-- ctr_artifacts.json
|-- reports/
|   `-- ad_click_prediction_case_study_submission.pdf
|-- ad_click_prediction_case_study.ipynb
`-- README.md
```

## Strategic Recommendations

- Rank impressions by predicted click probability and tune thresholds by campaign budget.
- Use bid multipliers for high-CTR product, webpage, campaign, and qualified profile segments.
- Reserve more inventory for products with higher expected clicks per 100,000 impressions.
- Monitor recall, precision, F1, PR-AUC, calibration, and feature drift weekly.
- Keep SMOTE for offline experimentation; prefer class weighting and threshold tuning for production refreshes.
- A/B test personalized CTR features against a non-personalized baseline before deployment.

## Submission Note

The final deliverable is the PDF file:

```text
reports/ad_click_prediction_case_study_submission.pdf
```

The notebook and generated artifacts are included for reproducibility and later
iteration.
