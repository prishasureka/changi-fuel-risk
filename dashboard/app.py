# basic structure and imports

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import sys
import os
import datetime
import math

# Add parent directory to path so we can import from models/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration — always comes first in Streamlit
st.set_page_config(
    page_title="Changi Route Resilience",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# load data and model

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'rf_model.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    route_features = pd.read_csv(os.path.join(base, 'route_features.csv'))
    fuel_monthly   = pd.read_csv(os.path.join(base, 'fuel_monthly.csv'))
    return route_features, fuel_monthly


model = load_model()
route_features, fuel_monthly = load_data()

# header and context

st.title("Changi Airport: Fuel Price Stress Test")
st.markdown("""
This tool lets you simulate what happens to passenger demand across Changi Airport's
**371 international routes** when jet fuel prices change. Set a fuel price, pick a
scenario type, and see which routes take the biggest hit — ranked from most to least
vulnerable.
""")

with st.expander("How to use this dashboard", expanded=True):
    st.markdown("""
    **Step 1 — Set a fuel price.**
    Use the slider on the left. It starts at the current market price. Drag it up to
    simulate a price spike, or down to model a price drop.

    **Step 2 — Choose a scenario type.**
    - *Shock*: the price has just jumped today. Airlines haven't had time to adjust,
      and recent months were at normal prices. This tests the immediate impact.
    - *Sustained*: the price has been elevated for 3 or more months. Lag features
      match the scenario price. This tests what happens if high prices persist.

    **Step 3 — Filter if needed.**
    Use the carrier filter (LCC vs FSC) and minimum distance slider to narrow the
    view. Low-cost carriers operate on thinner margins and tend to be more
    fuel-sensitive than full-service carriers.

    **Step 4 — Read the chart.**
    Routes are ranked from most to least vulnerable. The colour shows the risk tier:
    - **Red (Critical)**: predicted demand drop of more than 20%
    - **Orange (High)**: 10 to 20% drop
    - **Yellow (Moderate)**: 3 to 10% drop
    - **Green (Low)**: less than 3% drop, or positive

    **Step 5 — Use the full table.**
    Scroll down past the chart for the complete 371-route table. You can sort any
    column. The "Fuel cost delta/seat" column shows how much more (or less) it costs
    the airline to fly one seat on that route under your scenario vs. the current price.

    **One important distinction:** these scores predict demand change, not whether an
    airline will suspend a route. A major carrier running London 14 times a week has
    plenty of slack to absorb a -15% demand hit through fare changes and load factor
    adjustments. A thin 3x-weekly route to a smaller market might get cut at -5%.
    Use distance and carrier type alongside the risk score to judge suspension
    likelihood yourself.
    """)

with st.expander("What is the model actually doing?"):
    st.markdown("""
    **The model: Random Forest Regressor**

    A random forest builds hundreds of individual decision trees. Each tree is a long
    chain of if/then rules: "if fuel price is above $3.50 AND GDP per capita is below
    $5,000 AND the month is July, then predict X." Each tree is trained on a slightly
    different random slice of the data and a random subset of features. The forest's
    final prediction is the average across all 200 trees.

    **Why not just use a simpler model?**

    The relationship between fuel prices and passenger arrivals is not linear. A basic
    linear regression assumes that every $1 fuel increase cuts arrivals by the same
    fixed amount regardless of context. That is wrong. A fuel spike in December hurts
    less because seasonal demand is strong. The same spike on a low-GDP corridor hurts
    more than on a high-GDP one. The Random Forest captures these interactions. In this
    project, a linear model achieved R² = 0.09 on the test set. The Random Forest
    achieved R² = 0.39, more than four times better.

    **What it was trained to predict**

    Given a set of features for a country-month (fuel price, 3-month and 6-month fuel
    changes, lagged fuel prices, GDP per capita, GDP growth rate, LCC share, month,
    quarter, peak season flag), it predicts the year-over-year log-difference in
    passenger arrivals. The log-difference roughly maps to a percentage:
    - -0.030 is about -3% arrivals
    - -0.105 is about -10% arrivals
    - -0.223 is about -20% arrivals

    **Training details**

    - 200 trees, each with a maximum depth of 8 levels
    - Each leaf required at least 5 training samples (prevents overfitting to outliers)
    - Trained on 2006 to 2018 monthly data from 8 countries, tested on 2019 to 2024
    - Calibrated elasticity parameters are then applied to all 371 Changi routes using
      each route's destination-country GDP and carrier type

    **The key limitation**

    Random forests cannot extrapolate beyond what they were trained on. For fuel prices
    above roughly $4/gal (which never appeared in training), the model returns a
    compressed, underestimated prediction. That is why the dashboard shows a warning
    above $3.96/gal. Treat those scores as lower bounds on the actual risk.
    """)

st.caption(
    "Data: Singapore arrivals (data.gov.sg) · Fuel prices (EIA) · "
    "GDP (World Bank) · Route network (OurAirports) · 2005 to 2026"
)

st.divider()

# sidebar with controls

with st.sidebar:
    st.header("Scenario settings")

    # Derive current and historical fuel prices from data
    fuel_monthly['year_month_dt'] = pd.to_datetime(fuel_monthly['year_month'])
    fuel_sorted = fuel_monthly.sort_values('year_month_dt').reset_index(drop=True)
    current_fuel = float(fuel_sorted['jet_fuel_usd_per_gallon'].iloc[-1])
    fuel_6m_ago  = float(fuel_sorted['jet_fuel_usd_per_gallon'].iloc[-7]) if len(fuel_sorted) >= 7 else current_fuel

    fuel_scenario = st.slider(
        "Jet fuel price (USD/gallon)",
        min_value=1.0,
        max_value=7.5,
        value=current_fuel,
        step=0.05,
        help=f"Current price: ${current_fuel:.2f}. Drag to simulate a shock."
    )

    # Fix 3: warn when scenario is outside the model's training range
    TRAINING_MAX_FUEL = 3.96
    if fuel_scenario > TRAINING_MAX_FUEL:
        st.warning(
            f"**Outside training range.** The model was trained on fuel prices up to "
            f"${TRAINING_MAX_FUEL:.2f}/gal. At ${fuel_scenario:.2f}/gal the Random Forest "
            f"is extrapolating — it will systematically **underestimate** risk at these levels."
        )

    st.caption(f"""
    **Reference prices:**
    - 2005 average: ~$1.70
    - 2008 peak: ~$4.00
    - 2020 COVID low: ~$0.75
    - 2022 Ukraine spike: ~$4.30
    - Current: ~${current_fuel:.2f}
    """)

    st.divider()

    # Fix 2: shock vs sustained scenario type
    scenario_type = st.radio(
        "Scenario type",
        options=["Shock (price spikes today)", "Sustained (elevated for months)"],
        index=0,
        help=(
            "**Shock**: fuel price jumps today; lag features stay at recent historical prices. "
            "**Sustained**: price has been at this level for 3+ months; lag features match the scenario price."
        )
    )

    st.divider()

    carrier_filter = st.radio(
        "Carrier type",
        options=["All", "LCC only", "FSC only"],
        index=0,
        help="LCC = Low-Cost Carrier. LCC routes operate on thinner margins and are more fuel-sensitive."
    )

    min_distance = st.slider(
        "Minimum route distance (km)",
        min_value=0, max_value=10000, value=0, step=500,
        help="Filter out short-haul routes below this distance."
    )

    show_top_n = st.select_slider(
        "Show top N routes in chart",
        options=[15, 20, 25, 30, 40, 50],
        value=25
    )

# Feature columns — must match training
feature_cols = [
    'jet_fuel_usd_per_gallon', 'fuel_lag1', 'fuel_lag2', 'fuel_lag3',
    'fuel_change_3m', 'fuel_change_6m', 'fuel_volatility_6m',
    'gdp_per_capita', 'gdp_yoy_change', 'lcc_share',
    'month', 'quarter', 'is_peak_season'
]

# Apply carrier and distance filters
scenario_df = route_features.copy()
if carrier_filter == "LCC only":
    scenario_df = scenario_df[scenario_df['carrier_type'] == 'LCC']
elif carrier_filter == "FSC only":
    scenario_df = scenario_df[scenario_df['carrier_type'] == 'FSC']
scenario_df = scenario_df[scenario_df['distance_km'] >= min_distance]

# Derive actual historical lag prices and rolling volatility from data (Fix 2, Fix 7)
fuel_lag1_actual = float(fuel_sorted['jet_fuel_usd_per_gallon'].iloc[-2]) if len(fuel_sorted) >= 2 else current_fuel
fuel_lag2_actual = float(fuel_sorted['jet_fuel_usd_per_gallon'].iloc[-3]) if len(fuel_sorted) >= 3 else current_fuel
fuel_lag3_actual = float(fuel_sorted['jet_fuel_usd_per_gallon'].iloc[-4]) if len(fuel_sorted) >= 4 else current_fuel
fuel_volatility_actual = float(fuel_sorted['jet_fuel_usd_per_gallon'].tail(6).std())

# For a shock: lags stay at recent historical values (price only just changed).
# For sustained: lags match the scenario price (it's been elevated for months).
if scenario_type == "Shock (price spikes today)":
    lag1 = fuel_lag1_actual
    lag2 = fuel_lag2_actual
    lag3 = fuel_lag3_actual
else:
    lag1 = lag2 = lag3 = fuel_scenario

# Set fuel scenario features
_now = datetime.datetime.now()
scenario_df['jet_fuel_usd_per_gallon'] = fuel_scenario
scenario_df['fuel_lag1']               = lag1
scenario_df['fuel_lag2']               = lag2
scenario_df['fuel_lag3']               = lag3
scenario_df['fuel_change_3m']          = (fuel_scenario - current_fuel) / current_fuel * 100
scenario_df['fuel_change_6m']          = (fuel_scenario - fuel_6m_ago)  / fuel_6m_ago  * 100
scenario_df['fuel_volatility_6m']      = fuel_volatility_actual
scenario_df['month']                   = _now.month
scenario_df['quarter']                 = math.ceil(_now.month / 3)
scenario_df['is_peak_season']          = int(_now.month in [6, 7, 8, 12])

# Impute any missing GDP values with column medians before prediction
X_scenario = scenario_df[feature_cols].fillna(scenario_df[feature_cols].median(numeric_only=True))
scenario_df['predicted_demand_change'] = model.predict(X_scenario)

# Per-seat fuel cost delta vs current price
FUEL_BURN_GAL_PER_SEAT_KM = 0.05
scenario_df['fuel_cost_delta_per_seat'] = (
    scenario_df['distance_km'] * FUEL_BURN_GAL_PER_SEAT_KM * (fuel_scenario - current_fuel)
).round(2)

# Risk classification
scenario_df['risk_level'] = pd.cut(
    scenario_df['predicted_demand_change'],
    bins=[-np.inf, -0.223, -0.105, -0.030, np.inf],
    labels=['Critical', 'High', 'Moderate', 'Low']
)

# Route label for display
scenario_df['route_label'] = (
    scenario_df['airline_iata'] + ' → ' +
    scenario_df['destination_city'] + ' (' +
    scenario_df['destination_airport_iata'] + '), ' +
    scenario_df['destination_country']
)

scenario_df = scenario_df.sort_values('predicted_demand_change')

# Summary metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fuel scenario", f"${fuel_scenario:.2f}/gal",
            delta=f"{fuel_scenario - current_fuel:+.2f} vs current",
            delta_color="inverse")
col2.metric("Critical-risk routes",
            int((scenario_df['predicted_demand_change'] < -0.223).sum()))
col3.metric("High-risk routes",
            int(((scenario_df['predicted_demand_change'] >= -0.223) &
                 (scenario_df['predicted_demand_change'] < -0.105)).sum()))
col4.metric("Routes analysed", len(scenario_df))


# main bar chart

st.subheader(f"Top {show_top_n} highest-risk routes under ${fuel_scenario:.2f}/gal scenario")

top_risk = scenario_df.head(show_top_n).copy()
top_risk['gdp_per_capita'] = top_risk['gdp_per_capita'].round(0)
top_risk['distance_km']    = top_risk['distance_km'].round(0)

color_map = {
    'Critical': '#d62728',
    'High':     '#ff7f0e',
    'Moderate': '#ffd700',
    'Low':      '#2ca02c'
}

fig_bar = px.bar(
    top_risk,
    x='predicted_demand_change',
    y='route_label',
    orientation='h',
    color='risk_level',
    color_discrete_map=color_map,
    title=f'Predicted YoY demand change at ${fuel_scenario:.2f}/gal  (log-diff — e.g. −0.105 ≈ −10% arrivals)',
    labels={
        'predicted_demand_change': 'Predicted demand change (log-diff)',
        'route_label':  'Route',
        'risk_level':   'Risk level'
    },
    hover_data={
        'carrier_type':             True,
        'distance_km':              True,
        'gdp_per_capita':           True,
        'fuel_cost_delta_per_seat': True,
        'route_label':              False
    }
)

fig_bar.add_vline(x=0, line_dash='dash', line_color='gray')
fig_bar.update_layout(height=650, yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_bar, use_container_width=True)


# historical fuel price context chart

st.subheader("Historical context: where does your scenario sit?")

fig_fuel = go.Figure()
fig_fuel.add_trace(go.Scatter(
    x=fuel_sorted['year_month_dt'],
    y=fuel_sorted['jet_fuel_usd_per_gallon'],
    mode='lines',
    name='Historical fuel price',
    line=dict(color='steelblue', width=1.5)
))
fig_fuel.add_hline(
    y=fuel_scenario,
    line_dash='dash',
    line_color='red',
    annotation_text=f'Your scenario: ${fuel_scenario:.2f}',
    annotation_position='bottom right'
)
fig_fuel.update_layout(
    title='Jet fuel price history 2005–2026',
    xaxis_title='',
    yaxis_title='USD per gallon',
    height=300
)
st.plotly_chart(fig_fuel, use_container_width=True)


# detailed route risk table

st.subheader("Full route risk table")
st.info(
    "**Demand change ≠ route suspension.** These scores predict how much passenger demand "
    "is expected to fall on each corridor, and not whether the airline will suspend the route. "
    "A high-frequency SQ route to London can absorb a −15% demand shock through fare adjustment or "
    "load-factor changes. A thin 3×-weekly route to Ulaanbaatar may be suspended at −5%. "
    "Use the distance and carrier type columns alongside the risk score to judge actual suspension likelihood."
)

display_df = scenario_df[[
    'airline_iata', 'destination_city', 'destination_country',
    'carrier_type', 'distance_km',
    'predicted_demand_change', 'fuel_cost_delta_per_seat',
    'risk_level', 'gdp_per_capita'
]].copy()

display_df.columns = [
    'Airline', 'City', 'Country',
    'Type', 'Distance (km)',
    'Predicted demand change', 'Fuel cost delta/seat ($)',
    'Risk level', 'GDP per capita (USD)'
]
display_df['Predicted demand change'] = display_df['Predicted demand change'].round(4)
display_df['Distance (km)']           = display_df['Distance (km)'].round(0).astype('Int64')
display_df['GDP per capita (USD)']    = display_df['GDP per capita (USD)'].round(0).astype('Int64')

st.dataframe(display_df, use_container_width=True, hide_index=True)


# methodology note

with st.expander("Data sources and limitations"):
    st.markdown(f"""
    **Data sources**
    - Passenger arrivals: Singapore Civil Aviation Authority via data.gov.sg
    - Jet fuel prices: US Energy Information Administration (EIA)
    - GDP per capita: World Bank Open Data
    - Route network: OurAirports
    - Coverage: 2005 to 2026, {len(route_features):,} routes across {route_features['destination_country'].nunique()} countries

    **Fuel cost delta/seat** = distance x 0.05 gal/seat-km x (scenario price minus current price).
    A positive number means the route becomes more expensive to operate per seat under your scenario.

    **Limitations**
    - Training data covers 8 countries. Predictions for the other 40+ are calibrated extrapolations,
      with higher uncertainty for countries far from the training distribution.
    - The model was trained on fuel prices up to $3.96/gal. Above this level it will underestimate risk.
    - Route network is a current snapshot, not historical.
    - The model does not capture one-off events such as pandemics, airline bankruptcies, or geopolitical shocks.
    - Scores reflect predicted demand change only. Route suspension depends on how marginal the
      route was before the shock, which is not modelled here.
    """)
