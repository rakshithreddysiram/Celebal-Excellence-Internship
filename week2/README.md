# EDA + Modeling Pipeline
This workspace contains an end-to-end exploratory data analysis and modeling workflow for the Tesla deliveries dataset. The main artifact is the Jupyter notebook `week2_rakshith.ipynb` which implements:
- Data loading and date/target auto-detection
- Exploratory data analysis and aggregated visualizations
- Preprocessing & feature engineering (time features, lags, rolling means)
- Regression modeling (baseline pipelines) and hyperparameter tuning
- Monthly time-series forecasting (SARIMAX, Prophet fallback)

**How to run**
- Open the notebook `week2_rakshith.ipynb` in Jupyter (or JupyterLab).
- Run cells in order. The first code cell installs required packages (useful in fresh environments).
- If you prefer to install requirements manually, run:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels plotly joblib scikit-optimize prophet
```

**Notebook structure (important cells)**
- Cell 1: Package install
- Cell 2: Imports and environment flags (`HAS_XGB`, `HAS_PROPHET`)
- Cells 3–5: Data load, date parsing, EDA and aggregated plots
- Cells 6–10: Preprocessing, feature engineering (lags), pipelines, train/test split
- Cells 11–12: Baseline training and hyperparameter tuning (RandomizedSearchCV)
- Cell 13: Time-series forecasting (SARIMAX + Prophet fallback)

**Key pipeline decisions**
- Numeric imputation: median (`SimpleImputer(strategy='median')`).
- Categorical imputation: constant `'Unknown'` + `OneHotEncoder(handle_unknown='ignore')`.
- Lag features: default lags = `[1, 3, 6, 12]` (months) and matching rolling means.
- Preprocessing built with `ColumnTransformer` and scikit-learn `Pipeline`.

**Compatibility & Troubleshooting notes**
- RMSE: computed as `math.sqrt(mean_squared_error(y_true, y_pred))` to avoid `squared` parameter compatibility issues across scikit-learn versions.
- Plotting: the notebook tries Plotly first and falls back to Matplotlib (guaranteed static charts without Plotly/Kaleido).
- Resampling: uses `resample('ME')` (month-end) to avoid deprecated aliases.
- If you hit memory or compute issues during hyperparameter tuning, reduce `n_iter` in the `RandomizedSearchCV` cells.

**Next steps / suggestions**
- Add cross-validated time-series-aware CV (e.g., `TimeSeriesSplit`) for more robust model evaluation.
- Add model explainability (SHAP) for the chosen model.
- Produce a short report summarizing best model metrics and recommendations.

If you want, I can run the notebook cells here to validate execution and upload the saved `best_pipeline.joblib` artifact. Tell me whether to run the full notebook or just the preprocessing + baseline cells first.
# End-to-End EDA & Modeling Plan — Tesla Deliveries Dataset

Project: Build an end-to-end machine learning pipeline for sales/price/deliveries data using the provided CSV (`tesla_deliveries_dataset_2015_2025.csv`). The pipeline covers data ingestion, exploratory data analysis (EDA), preprocessing, feature engineering, regression modeling with hyperparameter tuning, and time-series forecasting.

## Objectives
- Explore dataset and understand temporal patterns and trends.
- Engineer features suitable for regression and time-series forecasting.
- Train and tune regression models to predict a sales/price/deliveries target.
- Produce forecasts using time-series models (Prophet / SARIMAX).
- Evaluate models, produce visualizations, and save model artifacts.

## High-level Plan (Phases)

Phase 1 — Project Setup
- Create a reproducible notebook environment with requirements installed.
- Add README with plan and steps.

Phase 2 — Data Loading & EDA
- Load CSV robustly and detect probable date and target columns.
- Print dataset summary, types, missing values, and quick plots.

Phase 3 — Preprocessing & Feature Engineering
- Parse dates and create time features (year, month, day, weekday).
- Handle missing values; numeric imputation, categorical fill.
- Create lag features and rolling means for time-aware regression.

Phase 4 — Regression Modeling & Hyperparameter Tuning
- Select candidate models: Linear Regression, Random Forest, XGBoost.
- Build scikit-learn pipeline for preprocessing and modeling.
- Use RandomizedSearchCV for hyperparameter tuning.
- Evaluate models (MAE, RMSE, R2) and compare.

Phase 5 — Time-Series Forecasting
- Aggregate series to daily/weekly/monthly as appropriate.
- Fit Prophet (if available) and SARIMAX; compare forecasts.

Phase 6 — Evaluation, Visualization & Export
- Visualize true vs predicted and forecast plots.
- Save best model(s) using `joblib` and document next steps.

Phase 7 — Wrap Up
- Summarize findings and recommendations for production.

## Notebook Structure (implemented in `week2_rakshith.ipynb`)
- Cell 1: Install requirements (`pip install` commands)
- Cell 2: Imports
- Cell 3: Configuration and helper functions (detect date/target)
- Cell 4: Load dataset and initial summary
- Cell 5-7: EDA visualizations and missing/value analysis
- Cell 8-10: Preprocessing & feature engineering
- Cell 11-14: Regression modeling and hyperparameter tuning
- Cell 15-17: Time-series forecasting (Prophet & SARIMAX)
- Cell 18: Evaluation, save models and next steps

## Notes & Assumptions
- The notebook will try to auto-detect a date column and a numeric target column (e.g., `deliveries`, `sales`, `price`). If auto-detection fails, set `TARGET_COL` manually in the notebook.
- Forecasting uses monthly aggregation by default; adjust aggregation level if needed.
- `prophet` or `fbprophet` may require extra system dependencies on some systems; notebook falls back to `statsmodels` SARIMAX if Prophet is not available.

## Plotting choices and guidance
- Histograms (frequency plots): show the distribution of values across individual records. The y-axis is the number of records (frequency) that fall into each bin. Use histograms when you want to inspect the distribution or detect outliers. Example columns: `Estimated_Deliveries`, `Avg_Price_USD`, `Range_km`.
- Aggregated totals (bar charts): show sums or averages aggregated along a grouping (e.g., total deliveries per year, total charging stations per region). Use aggregated charts when the meaningful metric is the sum/total rather than per-record frequency. The notebook includes examples: `Total Estimated_Deliveries by Year` and `Total Estimated_Deliveries by Region`.
- Time-series line charts: use when the data has a date/time dimension. The notebook aggregates to month-end by default and produces both Plotly interactive charts and Matplotlib static fallbacks so charts render in environments without Plotly/Kaleido.

Examples:
- Frequency histogram: `sns.histplot(df['Estimated_Deliveries'])` — y-axis shows how many rows have deliveries in that bin.
- Aggregated total: `df.groupby('Year')['Estimated_Deliveries'].sum().plot.bar()` — y-axis shows total deliveries for the year (sum across records).

When to use which:
- Use histograms to understand the distribution across records (e.g., are most records small deliveries or large?).
- Use aggregated bar charts to report totals or compare totals across groups (year, region, model).


## Next actions
1. Run the notebook cells in order to reproduce the analysis and models.
2. If errors appear for prophet/xgboost installs, run installation steps from the first cell manually in the environment (see notes inside notebook).

---
Created as part of an end-to-end demonstration. See `week2_rakshith.ipynb` for the implementation.
