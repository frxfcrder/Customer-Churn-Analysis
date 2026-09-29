# Customer Churn Analysis

Exploratory data analysis and classification models that predict telecom customer churn from
customer attributes. The analysis lives in a notebook; the modeling logic is extracted into an
importable `src/` package with tests.

## Dataset

- Source: [Kaggle — telecom-churn](https://www.kaggle.com/datasets/barun2104/telecom-churn) (auto-downloaded with `kagglehub`)
- Size: 3,333 customer records × 11 columns (10 features + target), no missing values
- Target: `Churn` — binary (1 = left, ~14% of customers, imbalanced)
- Features: AccountWeeks, ContractRenewal, DataPlan, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins

## Project Structure

```
customer-churn-analysis/
├── data/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

`data/` holds the downloaded CSV and `models/` the saved artifacts; both are git-ignored (`.gitkeep` keeps the folders in the repo).

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux activate with `source .venv/bin/activate`.

Notebook (EDA + results):

```bash
jupyter notebook notebooks/01_eda.ipynb
```

As a library:

```python
from src.data_loader import load_dataset, split_features_target
from src.preprocessing import split_train_test, scale_train_test
from src.train import train_all
from src.evaluate import evaluate_models, best_model
from src.predict import save_model, load_model, predict_churn

df = load_dataset()
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = split_train_test(X, y)
X_train, X_test, _ = scale_train_test(X_train, X_test)

models = train_all(X_train, y_train)
results = evaluate_models(models, X_test, y_test)
print(results)
print("Best F1:", best_model(results))

save_model(models["XGBoost"])
print(predict_churn(load_model(), X_test.head()))
```

Tests:

```bash
pytest
```

## Results

Held-out test set (80/20 split, `random_state=42`):

| Model         | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------|----------|-----------|--------|--------|---------|
| Logistic      | 0.8576   | 0.5263    | 0.2062 | 0.2963 | 0.8091  |
| Decision Tree | 0.8921   | 0.6404    | 0.5876 | 0.6129 | 0.7657  |
| Random Forest | 0.9250   | 0.8219    | 0.6186 | 0.7059 | 0.8603  |
| AdaBoost      | 0.8711   | 0.6078    | 0.3196 | 0.4189 | 0.8509  |
| HistGB        | 0.9220   | 0.7848    | 0.6392 | 0.7045 | 0.8468  |
| MLP           | 0.9070   | 0.7011    | 0.6289 | 0.6630 | 0.8672  |
| RF (Tuned)    | 0.9190   | 0.8644    | 0.5258 | 0.6538 | 0.8803  |
| XGBoost       | 0.9235   | 0.7805    | 0.6598 | 0.7151 | 0.8472  |

- Best F1: **XGBoost** (0.7151)
- Best ROC-AUC: **RF (Tuned)** (0.8803)
- Top churn drivers: DayMins, MonthlyCharge, CustServCalls, ContractRenewal
