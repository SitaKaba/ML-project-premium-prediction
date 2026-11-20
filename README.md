# Health Insurance Premium Prediction

A Streamlit app that predicts a customer's **health insurance premium** from their profile, using two regression models segmented by age.

**Live demo:** https://sitakaba-ml-project-premium-prediction.streamlit.app

## What it does

The user fills in 12 fields: age, number of dependants, income (in lakhs), genetical risk (0–5), insurance plan (Bronze / Silver / Gold), employment status, gender, marital status, BMI category, smoking status, region and medical history. The app returns the predicted premium.

## Approach

- **Two models, segmented by age:**
  - **age ≤ 25:** `LinearRegression`;
  - **age > 25:** `XGBRegressor`.
  Each segment has its own fitted scaler (`artifacts/scaler_young.joblib`, `artifacts/scaler_rest.joblib`).
- **Feature engineering:**
  - **Medical history → normalized risk score.** Each condition has a weight (heart disease 8, diabetes 6, high blood pressure 6, thyroid 5, none 0). Up to two conditions are summed, then divided by 14, the maximum possible score.
  - **Insurance plan → ordinal encoding** (Bronze 1, Silver 2, Gold 3).
  - **Other categorical fields → one-hot encoding.**
- The training notebooks are not included in this repository; the trained models are in `artifacts/`.

## Run locally

```bash
git clone https://github.com/SitaKaba/ML-project-premium-prediction.git
cd ML-project-premium-prediction
pip install -r requirements.txt
streamlit run main.py
```

A Dev Container configuration (`.devcontainer/`) is also provided.

## Structure

```
main.py                 # Streamlit interface
prediction_helper.py    # risk score, encoding, scaling, model selection by age
artifacts/              # two models and two scalers (joblib)
requirements.txt
```

## Credits

Project built while following the [Codebasics](https://codebasics.io) machine learning course. Deployment on Streamlit Community Cloud by Sita KABA.
