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
This dashboard models how jet fuel price shocks affect **each specific Changi route**.
All 371 active Changi corridors are scored for demand impact under the selected fuel
scenario - ranked from most vulnerable to most resilient. Drag the slider to simulate
a shock and see which routes are first in line to lose passengers.

**Data:** Singapore arrivals (data.gov.sg) · Fuel prices (EIA) · GDP (World Bank) ·
OurAirports route network · 2005–2026
**Model:** Random Forest calibrated on historical arrival changes across 8 source countries.
""")

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

with st.expander("Methodology and limitations"):
    st.markdown(f"""
    **How the model works:**
    A Random Forest Regressor was trained on monthly Singapore air passenger arrival data
    (2005–2019, 8 source countries) to learn how jet fuel prices, GDP conditions, corridor
    LCC share, and seasonality relate to year-over-year arrival changes. The trained model
    is then applied to all **{len(route_features):,} Changi routes across {route_features['destination_country'].nunique()} countries**,
    using each route's destination-country GDP, LCC composition, and the fuel scenario you set.

    **Predicted demand change** is in log-difference units. Approximate interpretation:
    - −0.030 ≈ −3% arrivals
    - −0.105 ≈ −10% arrivals
    - −0.223 ≈ −20% arrivals

    **Fuel cost delta/seat** = distance × 0.05 gal/seat-km × (scenario − current price).
    A positive number means the route becomes more expensive to operate per seat.

    **Data sources:**
    - Passenger arrivals: Singapore Civil Aviation Authority via data.gov.sg
    - Jet fuel prices: US Energy Information Administration (EIA)
    - GDP per capita: World Bank Open Data
    - Route network: OurAirports

    **Important distinctions:**
    - Scores reflect **predicted demand change**, not route suspension probability. Suspension
      depends on how marginal the route was before the shock, which is not modelled.
    - Training data covers only 8 countries; predictions for the other 40 are calibrated
      extrapolations — higher uncertainty for countries far from the training distribution.
    - The model was trained on fuel prices up to **$3.96/gal**. Scenarios above this level
      are outside the training range; the model will underestimate risk there.
    - **Shock** mode sets fuel lags to recent historical prices (price just spiked today).
      **Sustained** mode sets lags equal to the scenario price (elevated for 3+ months).
      Airlines respond differently to these two situations.
    - Route network is a current snapshot, not historical.
    - Model does not capture one-off events (pandemics, airline bankruptcies, geopolitics).
    """)
