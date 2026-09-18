# Customer Churn Analysis

EDA and churn prediction on a telecom dataset, built as a Jupyter/Colab notebook.

## Dataset

- Source: [Kaggle — telecom-churn](https://www.kaggle.com/datasets/barun2104/telecom-churn) (auto-downloaded via `kagglehub`)
- 3,333 customer records, 11 features, no missing values
- Target: `Churn` (1 = left, ~14% of customers — imbalanced)

## Notebook Sections

1. Inspecting Data
2. Handling Null Values
3. Exploratory Data Analysis — distributions, outliers, correlations
4. Churn Prediction — 6 classifiers + tuned Random Forest (GridSearchCV)
5. Model Evaluation — confusion matrices, precision/recall/F1/ROC-AUC, PR curves, feature importance
6. Conclusions

## Results

- Best F1: **Random Forest** (~0.71)
- Best ROC-AUC: **RF (Tuned)** (~0.88)
- Top churn drivers: DayMins, MonthlyCharge, CustServCalls, ContractRenewal

## How to Run

Open `Customer_Churn_Analysis.ipynb` in Jupyter/Colab notebook and run all cells.
