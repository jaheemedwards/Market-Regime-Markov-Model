# Market Regime Markov Model

[Live App Link](https://your-live-app-link.streamlit.app/)

---

## 📌 Overview

The **Market Regime Markov Model** is a Streamlit web app that analyzes historical stock and cryptocurrency prices to identify market regimes and model their transitions using a Markov Chain. The app allows users to visualize price movements, regime classification, transition probabilities, steady-state probabilities, and simulated future market states.

---

## 🧠 Theory & Concepts

### Market Regimes

Financial markets can switch between different **regimes**:

* **Bull Market:** Prices trending upward
* **Bear Market:** Prices trending downward
* **Neutral Market:** Prices relatively stable

Understanding regimes helps investors adjust strategies, forecast likely transitions, and assess long-term tendencies.

### Markov Chains in Finance

A **Markov Chain** is a stochastic process where the **next state depends only on the current state**:

* **States:** Bull, Bear, Neutral
* **Transition Matrix:** Probability of moving from one state to another
* **Steady-State Probabilities:** Long-term probabilities of being in each state

The app models daily market regimes as a Markov Chain to predict future states and understand long-term market tendencies.

### Key Components

| Concept             | Theory                             | Implementation                           | Purpose                                        |
| ------------------- | ---------------------------------- | ---------------------------------------- | ---------------------------------------------- |
| Market Regimes      | Bull/Bear/Neutral states           | `get_state()` function                   | Discretize returns for Markov modeling         |
| Markov Chain        | Next state depends only on current | Transition matrix computation            | Model probabilities of regime changes          |
| Steady-State        | Long-run probabilities             | Eigenvector method                       | Identify long-term tendencies                  |
| Simulation          | Forecast future regimes            | `np.random.choice` with transition probs | Scenario analysis                              |
| Interactive Visuals | Communicate insights               | Plotly charts                            | Help users interpret regimes and probabilities |

---

## ⚙️ Implementation

1. **Data Selection:** Users choose from a predefined list of stocks and cryptocurrencies via a dropdown menu.
2. **Data Download:** Historical price data is fetched from Yahoo Finance.
3. **Return Calculation:** Daily returns are calculated to normalize prices.
4. **Market Regime Classification:** Each day's return is categorized into Bull, Bear, or Neutral based on thresholds.
5. **Transition Matrix:** Counts and normalizes transitions between regimes to form a probability matrix.
6. **Steady-State Probabilities:** Eigenvector method identifies long-term likelihood of being in each regime.
7. **Simulation:** Generates possible future regimes using the transition probabilities.
8. **Interactive Visualization:** Plotly charts show price trends, regimes, transition matrix heatmap, steady-state probabilities, and simulated future distribution.

---

## 🖥️ How the App Looks

* **Price Chart:** Closing prices with colored segments for Bull (green), Bear (red), and Neutral (gray) regimes.
* **Transition Matrix:** Heatmap showing probabilities of switching from one regime to another.
* **Steady-State Probabilities:** Bar chart showing long-term regime probabilities.
* **Simulation:** Bar chart showing distribution of simulated future market regimes.
* **Sidebar:** Controls to select ticker, date range, and number of days for simulation.

![App Screenshot](assets/project_portfolio_projectTab_screenshot.png)

---

## 🔗 Live Website

Explore the app live: [Market Regime Markov Model](https://your-live-app-link.streamlit.app/)

---

## 📝 Notes

* The app currently uses predefined thresholds for regime classification (0.5% up/down per day).
* Users can compare volatility and regime behavior across different assets like AAPL, TSLA, BTC-USD, and ETH-USD.
* All visualizations are interactive and rendered using Plotly.
