# Ad Click Prediction

This project builds an intelligent click-through rate (CTR) prediction system using user, campaign, product, webpage, and device-context data. The business goal is to predict which ad impressions are likely to receive genuine user clicks, helping advertisers reduce wasted spend, improve targeting quality, and balance false positives against missed click opportunities.

## Problem Context

Digital advertising datasets are highly imbalanced because most ad impressions do not receive clicks. This project treats ad-click prediction as a binary classification problem and focuses on business-relevant metrics such as ROC-AUC, PR-AUC, F1-score, precision, recall, and confusion-matrix trade-offs.

## Files

- `Ad_click_prediction_train (1).csv` - training dataset with the `is_click` target.
- `Ad_Click_prediciton_test.csv` - test dataset for final scoring.
- `Click-Through Rate (CTR) Prediction Project.pdf` - original approach/problem document.
- `ad_click_prediction_case_study.ipynb` - executed consolidated notebook with data preparation, feature engineering, modeling, evaluation, and recommendations.
- `ad_click_prediction_case_study_submission.pdf` - final compact submission PDF.
- `ad_click_prediction_test_predictions.csv` - generated test-set click probabilities and predicted labels.

## Approach

The solution follows the structure used in the earlier `Predicting-Sales-from-Campaign-Data` project:

1. Load and validate train/test datasets.
2. Parse temporal features from `DateTime`.
3. Engineer contextual and personalized features, including smoothed CTR aggregates.
4. Use temporal validation to mimic future ad-serving conditions.
5. Compare balanced Logistic Regression, balanced Random Forest, weighted XGBoost, and a lightweight SMOTE-style experiment.
6. Select the best model using F1/recall-oriented validation metrics.
7. Generate test predictions and translate findings into business recommendations.

## Key Business Questions Covered

- Weekend versus weekday click behavior.
- Highest and lowest CTR products.
- Impact of personalized user-product and campaign-product interaction features.
- Feature drivers behind clicks, including CTR aggregates and placement context.
- SMOTE trade-offs for rare click events and real-time serving.
- Product CTR usage for inventory forecasting.
- High-propensity user profiles for bid adjustment strategies.

## Outputs

The final PDF is intentionally concise and submission-ready:

- `11` pages.
- Approximately `0.13 MB`.
- Below the required `50` page and `20 MB` limits.

## Reproducibility

Run the notebook from this folder after installing the required Python ML stack used in the workspace:

```bash
jupyter notebook ad_click_prediction_case_study.ipynb
```

The notebook writes `ad_click_prediction_test_predictions.csv` after fitting the final model.
