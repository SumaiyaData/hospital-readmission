# Hospital Readmission Risk Prediction

A machine learning pipeline that predicts hospital readmission risk scores using a synthetic healthcare dataset. The project covers feature engineering, multi-model comparison with cross-validation, XGBoost hyperparameter tuning, and result visualisation.

> **Note:** The dataset is synthetic — it was procedurally generated to simulate readmission patterns. Model metrics (~0.99 R²) reflect the deterministic structure of the data generation formula, not real-world clinical performance. This project demonstrates correct ML pipeline methodology.

---

## Results

| Model | CV R² | Test R² | RMSE |
|---|---|---|---|
| XGBoost (tuned) | **0.991** | **0.991** | **0.022** |
| Random Forest | 0.983 | 0.983 | 0.030 |
| Gradient Boosting | 0.962 | 0.962 | 0.044 |
| Decision Tree | 0.956 | 0.964 | 0.042 |
| KNN | 0.921 | 0.921 | 0.063 |
| Linear Regression | 0.809 | 0.809 | 0.098 |
| Ridge | 0.809 | 0.809 | 0.098 |

![Model Results](outputs/model_results.png)

---

## Dataset

**8,000 patients | 16 raw columns | synthetic data**

| Feature | Type | Rationale |
|---|---|---|
| `season` | Categorical | Seasonal health effects (flu, allergy, heat) |
| `age` | Numeric | Biological risk factor |
| `gender` | Categorical | Biological factor |
| `region` | Categorical | Geographic healthcare access differences |
| `primary_diagnosis` | Categorical | Disease severity varies by condition |
| `length_of_stay` | Numeric | Proxy for illness severity |
| `treatment_type` | Categorical | Recovery risk varies by treatment |
| `followup_visits_last_year` | Numeric | Follow-up care reduces readmission |
| `prev_readmissions` | Numeric | Strongest historical predictor |
| `insurance_type` | Categorical | Healthcare access and affordability |
| `discharge_disposition` | Categorical | Post-discharge care setting |
| `patient_complexity_score` | Numeric (engineered) | `comorbidities_count + medications_count` |

**Dropped columns:** `patient_id`, `admission_date`, `comorbidities_count`, `medications_count`, `label` (target leakage)

---

## Project Structure

```
hospital-readmission-ml/
│
├── hospital_readmission.py   ← full pipeline script
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── README.md             ← instructions for placing dataset
│
└── outputs/
    └── model_results.png     ← generated after running the script
```

---


## Usage

### In Google Colab
Upload the dataset and run `hospital_readmission.py` cell by cell.


## Pipeline

```
Raw CSV
   │
   ├── Feature Engineering
   │     └── patient_complexity_score = comorbidities + medications
   │     └── drop: patient_id, admission_date, label
   │
   ├── Train / Test Split  (80 / 20, random_state=42)
   │
   ├── Preprocessor (inside Pipeline — no leakage)
   │     ├── OneHotEncoder   → categorical columns
   │     └── StandardScaler  → numeric columns
   │
   ├── Model Comparison (7 models, 5-fold CV)
   │     └── metrics: CV R², Train R², Test R², MAE, RMSE, MedAE
   │
   ├── XGBoost Hyperparameter Tuning
   │     └── GridSearchCV: n_estimators, max_depth, learning_rate,
   │                       subsample, colsample_bytree
   │
   ├── Model Versioning → xgb_model_vYYYYMMDD_HHMM.pkl
   │
   └── Plots → outputs/model_results.png
         ├── Test R² by Model
         ├── Predicted vs Actual
         └── Top 15 Feature Importances
```

---

## Key Design Decisions

**Why cross-validation instead of a single split for model comparison?**
A single train/test split gives results that depend on which samples happen to land in the test set. 5-fold CV evaluates every model across 5 different splits and reports the mean and standard deviation — a fairer and more stable comparison.

**Why MedAE instead of MAPE?**
MAPE (Mean Absolute Percentage Error) becomes unreliable when target values are close to zero — a small absolute error produces a huge percentage. MedAE (Median Absolute Error) is unaffected by this and is also more robust to outliers.

**Why subsample and colsample_bytree in the grid?**
These two XGBoost parameters introduce randomness into tree building (row sampling and column sampling). They often reduce overfitting on real-world data and are worth tuning alongside `n_estimators`, `max_depth`, and `learning_rate`.

**Why Pipeline?**
Wrapping the preprocessor and model in a Pipeline guarantees that the scaler and encoder are fitted only on training data, never on the test set. This prevents data leakage.

---

## Limitations

- **Synthetic data**: metrics reflect the data generation formula, not real clinical generalisability
- **No temporal split**: a production model should use a future time window as the test set
- **No calibration**: predicted scores are not probability-calibrated
- **No fairness audit**: subgroup performance (by gender, region, insurance type) not evaluated

---

## License

MIT — for educational and portfolio use only. Not intended for clinical deployment.
