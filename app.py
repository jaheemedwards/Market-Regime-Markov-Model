import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Regime Markov Model", layout="wide")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Model Configuration")
tickers = ["^GSPC", "AAPL", "TSLA", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]
ticker = st.sidebar.selectbox("Select Stock or Crypto Symbol", options=tickers)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# -----------------------------
# Data Download
# -----------------------------
st.title("📈 Market Regime Markov Model")
with st.spinner(f"Fetching data for {ticker}..."):
    data = yf.download(ticker, start=start_date, end=end_date)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

if data.empty:
    st.error("No data found. Please check the symbol or date range.")
    st.stop()

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(subset=['Close'], inplace=True)

# -----------------------------
# Define Market States (Drawdown-Based)
# -----------------------------
rolling_window = 60  # ~3 months
data['RollingMax'] = data['Close'].rolling(rolling_window, min_periods=1).max()
data['Drawdown'] = (data['Close'] - data['RollingMax']) / data['RollingMax']

def get_market_state(drawdown):
    if drawdown <= -0.2:
        return "Bear"
    elif -0.2 < drawdown <= -0.1:
        return "Neutral"  # Correction / sideways
    else:
        return "Bull"

data['State'] = data['Drawdown'].apply(get_market_state)
color_map = {"Bull": "green", "Neutral": "gray", "Bear": "red"}
data['Color'] = data['State'].map(color_map)

st.subheader(f"Price & Regime Classification for {ticker}")

# -----------------------------
# Classic Overlay Plot
# -----------------------------
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines',
                               line=dict(color='lightgray'), name='Close Price'))

for state, color in color_map.items():
    subset = data[data['State'] == state]
    fig_price.add_trace(go.Scatter(
        x=subset.index,
        y=subset['Close'],
        mode='lines',
        name=state,
        line=dict(width=2, color=color),
        hoverinfo='text',
        text=[f"{state} - {float(p):.2f}" if not pd.isna(p) else f"{state}" for p in subset['Close']]
    ))

fig_price.update_layout(title=f"{ticker} Closing Prices with Market Regimes",
                        xaxis_title="Date", yaxis_title="Price",
                        legend_title="Regime")
st.plotly_chart(fig_price, width='stretch')

# -----------------------------
# Smoothed Background Shading Plot with prominent price line
# -----------------------------
st.subheader("Price with Smoothed Shaded Regimes")

fig_shade = go.Figure()

# Add shaded regions for market regimes first
data['State_shift'] = data['State'].shift(1)
change_idx = data[data['State'] != data['State_shift']].index.tolist()
change_idx = [data.index[0]] + change_idx + [data.index[-1]]

for start, end in zip(change_idx[:-1], change_idx[1:]):
    state = data.loc[start, 'State']
    fig_shade.add_vrect(x0=start, x1=end, fillcolor=color_map[state],
                        opacity=0.2, line_width=0)

# Add a prominent price line on top of shading
fig_shade.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    line=dict(color='blue', width=1),  # thicker, easy to see
    name='Close Price'
))

fig_shade.update_layout(
    title=f"Smoothed Shaded Regimes: {ticker}",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig_shade, width='stretch')


# -----------------------------
# Two-Panel Layout: Step Line Colored by Regime
# -----------------------------
st.subheader("Two-Panel View: Price & Regimes Separately")

# Map State to numeric for step line
data['State_num'] = data['State'].map({'Bear': -1, 'Neutral': 0, 'Bull': 1})

fig_sub = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.1)

# Top panel: Price
fig_sub.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'), row=1, col=1)

# Bottom panel: Regime Level with colored step lines
# Split data into consecutive segments with same state
data['State_shift'] = data['State'].shift(1)
change_idx = data[data['State'] != data['State_shift']].index.tolist()
change_idx = [data.index[0]] + change_idx + [data.index[-1]]

for start, end in zip(change_idx[:-1], change_idx[1:]):
    segment = data.loc[start:end]
    state = segment['State'].iloc[0]
    fig_sub.add_trace(go.Scatter(
        x=segment.index,
        y=segment['State_num'],
        mode='lines',
        name=state,
        line=dict(color=color_map[state], width=3),
        showlegend=False,
        line_shape='hv'  # step-style
    ), row=2, col=1)

fig_sub.update_layout(title=f"Two-Panel View: {ticker}", xaxis_title="Date",
                      yaxis_title="Price / Regime Level", height=700)
st.plotly_chart(fig_sub, width='stretch')

# -----------------------------
# Heatmap of Regimes
# -----------------------------
st.subheader("Regime Heatmap Over Time")
data['State_cat'] = data['State'].map({'Bear':0, 'Neutral':1, 'Bull':2})
fig_heat = go.Figure(data=go.Heatmap(z=data['State_cat'].values.reshape(1,-1),
                                     x=data.index, y=[ticker],
                                     colorscale=[[0,'red'], [0.5,'gray'], [1,'green']]))
fig_heat.update_layout(title=f"Regime Heatmap: {ticker}", xaxis_title="Date",
                       yaxis_title="", height=200)
st.plotly_chart(fig_heat, width='stretch')

# -----------------------------
# Transition Matrix
# -----------------------------
states = ["Bull", "Neutral", "Bear"]
transition_matrix = pd.DataFrame(0, index=states, columns=states)
for (prev, curr) in zip(data['State'][:-1], data['State'][1:]):
    transition_matrix.loc[prev, curr] += 1
transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

st.subheader("Transition Probability Matrix")
st.dataframe(transition_matrix.round(2))

fig_matrix = go.Figure(data=go.Heatmap(
    z=transition_matrix.values,
    x=states,
    y=states,
    text=transition_matrix.round(2).values,
    texttemplate="%{text}",
    colorscale="Blues"
))
fig_matrix.update_layout(title="Market Regime Transition Matrix", xaxis_title="To State", yaxis_title="From State")
st.plotly_chart(fig_matrix, width='stretch')

# -----------------------------
# Steady-State Probabilities
# -----------------------------
P = transition_matrix.values
eigvals, eigvecs = np.linalg.eig(P.T)
steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])
steady_state = steady_state / steady_state.sum()
steady_state = steady_state.flatten()
steady_df = pd.Series(steady_state, index=states)

st.subheader("Steady-State (Long-Term) Probabilities")
fig_steady = px.bar(steady_df, x=steady_df.index, y=steady_df.values,
                    text=steady_df.round(3).values,
                    color=steady_df.index, color_discrete_map=color_map,
                    labels={'x': 'State', 'y': 'Probability'},
                    title='Steady-State Probabilities')
fig_steady.update_traces(textposition='outside')
st.plotly_chart(fig_steady, width='stretch')

# -----------------------------
# Simulate Future States
# -----------------------------
n_days = st.sidebar.slider("Days to Simulate", 30, 364, 60)
current_state = data['State'].iloc[-1]
simulated = [current_state]

for _ in range(n_days):
    current_idx = states.index(current_state)
    next_state = np.random.choice(states, p=P[current_idx])
    simulated.append(next_state)
    current_state = next_state

sim_df = pd.Series(simulated).value_counts(normalize=True)

st.subheader(f"Simulated {n_days}-Day Regime Distribution")
fig_sim = px.bar(sim_df, x=sim_df.index, y=sim_df.values, text=sim_df.round(3).values,
                 color=sim_df.index, color_discrete_map=color_map,
                 labels={'x': 'State', 'y': 'Probability'}, title=f'Simulated {n_days}-Day Regime Distribution')
fig_sim.update_traces(textposition='outside')
st.plotly_chart(fig_sim, width='stretch')

# -----------------------------
# Summary Insights
# -----------------------------
most_likely_state = sim_df.idxmax()
most_likely_prob = sim_df.max()

st.markdown(f"""
### 📊 Insights

* The **price chart** shows historical closing prices with highlighted regimes:

  * **Bull (green):** market trending upward
  * **Bear (red):** market trending downward
  * **Neutral (gray):** correction or sideways movement
* The **transition matrix** heatmap displays the probabilities of the market switching from one regime to another.
* **Steady-state probabilities** indicate the long-term likelihood of being in each market regime.
* The **simulation** provides a forecast of the distribution of market regimes over a user-specified number of future days.

**Predicted most likely regime in {n_days} days:** **{most_likely_state}** with probability **{most_likely_prob:.2f}**.

> This prediction is based on the current market state and the historical transition probabilities captured in the Markov Chain.

💡 *Tip:* Use this app to explore and compare the behavior of different assets like S&P500 (^GSPC), BTC-USD, AAPL, or TSLA, and understand how volatility and regime patterns differ.
""")
