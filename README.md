# Changi Airport: Fuel Price Stress Test

A machine learning dashboard that stress-tests all 371 of Singapore Changi Airport's international routes against jet fuel price scenarios. Set a fuel price, choose a scenario type, and see which routes are most vulnerable to demand loss — ranked in real time.

**Live dashboard:** https://changi-fuel-risk.streamlit.app

---

## What it does

Jet fuel typically accounts for 20 to 30% of an airline's operating costs. When prices spike, demand on price-sensitive routes drops — but not uniformly. A fuel shock hits a thin low-GDP corridor much harder than a high-frequency route to a wealthy market. This tool makes those differences visible.

The model scores every active Changi route for predicted year-over-year demand change under a user-defined fuel price. Routes are coloured by risk tier (Critical / High / Moderate / Low) and ranked from most to least vulnerable.

---

## Model

**Algorithm:** Random Forest Regressor (200 trees, max depth 8, min samples per leaf 5)

**Target variable:** Year-over-year log-difference in passenger arrivals

**Features:** Jet fuel price, 1/2/3-month lagged fuel prices, 3-month and 6-month fuel changes, 6-month fuel volatility, GDP per capita, GDP YoY growth, LCC share, month, quarter, peak season flag

**Training data:** Monthly Singapore air passenger arrivals from 8 source countries, 2006 to 2018

**Test data:** 2019 to 2024

**Performance:** R² = 0.39 on the full test set (vs R² = 0.09 for a linear baseline)

The Random Forest was chosen because the relationship between fuel prices and arrivals is non-linear. The same fuel spike hurts less in December (peak season) than in February, and hits low-GDP corridors harder than high-GDP ones. A linear model misses these interactions entirely.

**Key limitation:** Random forests cannot extrapolate. For fuel prices above roughly $4/gal (outside the training range), the model underestimates risk. The dashboard flags this with a warning.

---

## Scenario types

**Shock** — the price has just jumped today. Lag features stay at recent historical prices. Models the immediate impact of a sudden spike.

**Sustained** — the price has been elevated for 3 or more months. All lag features match the scenario price. Models what happens if high prices persist.

---

## Route scoring

The model was trained on 8 countries. To score all 371 Changi routes across 48 countries, each route is assigned:
- Destination-country GDP per capita and YoY growth (World Bank)
- Route distance from Changi via the Haversine formula
- Binary LCC/FSC carrier type (OurAirports + airline classifications)

Fuel elasticity is calibrated from the trained model and applied route by route.

---

## Project structure

```
changi-fuel-risk/
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   └── requirements.txt    # Python dependencies
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_integration.ipynb
│   ├── 04_EDA.ipynb
│   ├── 05_features.ipynb
│   ├── 06_modelling.ipynb
│   └── 07_route_scoring.ipynb
├── data/
│   └── processed/          # Cleaned CSVs used by the dashboard
├── models/
│   └── rf_model.pkl        # Trained Random Forest model
├── runtime.txt             # Pins Python 3.12 for Streamlit Cloud
└── README.md
```

---

## Data sources

| Data | Source |
|---|---|
| Singapore passenger arrivals | Singapore Civil Aviation Authority via data.gov.sg |
| Jet fuel prices | US Energy Information Administration (EIA) |
| GDP per capita | World Bank Open Data |
| Route network and airport data | OurAirports |

Coverage: 2005 to 2026

---

## Running locally

```bash
git clone https://github.com/prishasureka/changi-fuel-risk.git
cd changi-fuel-risk
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Then open http://localhost:8501 in your browser.

---

## Tech stack

Python, scikit-learn, pandas, NumPy, Plotly, Streamlit
