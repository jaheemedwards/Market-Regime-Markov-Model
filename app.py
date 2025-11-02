import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Market Regime Markov Model", layout="wide")

# -----------------------------
# 1️⃣ Sidebar Inputs
# -----------------------------
st.sidebar.header("Model Configuration")

# Predefined list of tickers (stocks and crypto)
tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "DOGE-USD"]

ticker = st.sidebar.selectbox("Select Stock or Crypto Symbol", options=tickers)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# -----------------------------
# 2️⃣ Data Download
# -----------------------------
st.title("📈 Market Regime Markov Model")

with st.spinner(f"Fetching data for {ticker}..."):
    data = yf.download(ticker, start=start_date, end=end_date)

# Fix MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

if data.empty:
    st.error("No data found. Please check the symbol or date range.")
    st.stop()

# Ensure 'Close' is numeric
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(subset=['Close'], inplace=True)

# Compute returns
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# -----------------------------
# 3️⃣ Define Market States
# -----------------------------
def get_state(r):
    if r > 0.005:
        return "Bull"
    elif r < -0.005:
        return "Bear"
    else:
        return "Neutral"

data['State'] = data['Return'].apply(get_state)

# Assign colors for regimes
color_map = {"Bull": "green", "Neutral": "gray", "Bear": "red"}
data['Color'] = data['State'].map(color_map)

st.subheader(f"Price & Regime Classification for {ticker}")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', line=dict(color='lightgray'), name='Close Price'))

# Add colored segments for each regime
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
# 4️⃣ Build Transition Matrix
# -----------------------------
states = ["Bull", "Neutral", "Bear"]
transition_matrix = pd.DataFrame(0, index=states, columns=states)

for (prev, curr) in zip(data['State'][:-1], data['State'][1:]):
    transition_matrix.loc[prev, curr] += 1

transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

st.subheader("Transition Probability Matrix")
st.dataframe(transition_matrix.round(2))

# -----------------------------
# 5️⃣ Visualize Transition Matrix with Plotly
# -----------------------------
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
# 6️⃣ Compute Steady-State Probabilities
# -----------------------------
P = transition_matrix.values
eigvals, eigvecs = np.linalg.eig(P.T)
steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])
steady_state = steady_state / steady_state.sum()
steady_state = steady_state.flatten()

steady_df = pd.Series(steady_state, index=states)

st.subheader("Steady-State (Long-Term) Probabilities")
fig_steady = px.bar(steady_df, x=steady_df.index, y=steady_df.values, text=steady_df.round(3).values,
                    color=steady_df.index, color_discrete_map=color_map,
                    labels={'x': 'State', 'y': 'Probability'}, title='Steady-State Probabilities')
fig_steady.update_traces(textposition='outside')
st.plotly_chart(fig_steady, width='stretch')

# -----------------------------
# 7️⃣ Simulate Future States
# -----------------------------
n_days = st.sidebar.slider("Days to Simulate", 10, 100, 30)
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
# 8️⃣ Summary Insights
# -----------------------------

# Determine most likely regime after simulation

most_likely_state = sim_df.idxmax()
most_likely_prob = sim_df.max()

st.markdown(f"""

### 📊 Insights

* The **price chart** shows historical closing prices with highlighted regimes:

  * **Bull (green):** market trending upward
  * **Bear (red):** market trending downward
  * **Neutral (gray):** market relatively stable
* The **transition matrix** heatmap displays the probabilities of the market switching from one regime to another.
* **Steady-state probabilities** indicate the long-term likelihood of being in each market regime.
* The **simulation** provides a forecast of the distribution of market regimes over a user-specified number of future days.

**Predicted most likely regime in {n_days} days:** **{most_likely_state}** with probability **{most_likely_prob:.2f}**.

> This prediction is based on the current market state and the historical transition probabilities captured in the Markov Chain.

💡 *Tip:* Use this app to explore and compare the behavior of different assets like BTC-USD, AAPL, or TSLA, and understand how volatility and regime patterns differ.
""")
