# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:14:03 2025

@author: kushp
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import norm, ttest_ind
from statsmodels.tsa.stattools import adfuller, acf
import io

# ------------------------------
# Data Acquisition Module
# ------------------------------

class DataFetcher:
    """
    Fetch historical stock price data using yfinance.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self) -> pd.DataFrame:
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError("No data fetched. Please check ticker and date range.")
        data.dropna(inplace=True)
        return data

# ------------------------------
# Strategy Builder Module (Stock Strategies)
# ------------------------------

class Strategy:
    """
    Base Strategy class. Subclasses should implement generate_signals.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate_signals()")

class SMACrossoverStrategy(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff().fillna(0.0)
        return signals

class RSITradingStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_rsi(self, data: pd.DataFrame) -> pd.Series:
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = self.compute_rsi(data)
        sig = [1 if r < self.oversold else 0 for r in signals['rsi']]
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        return signals

class BollingerBandsStrategy(Strategy):
    def __init__(self, window: int = 20, std_multiplier: float = 2.0):
        self.window = window
        self.std_multiplier = std_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        rolling_mean = data['Close'].rolling(window=self.window, min_periods=1).mean()
        rolling_std = data['Close'].rolling(window=self.window, min_periods=1).std()
        upper_band = rolling_mean + self.std_multiplier * rolling_std
        lower_band = rolling_mean - self.std_multiplier * rolling_std
        sig = []
        current = 0
        for price, lb, ub in zip(data['Close'], lower_band, upper_band):
            if price < lb:
                current = 1
            elif price > ub:
                current = 0
            sig.append(current)
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        signals['rolling_mean'] = rolling_mean
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        return signals

class SecondDerivativeMAStrategy(Strategy):
    def __init__(self, ma_window: int = 50, threshold: float = 0.1):
        self.ma_window = ma_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['ma'] = data['Close'].rolling(window=self.ma_window, min_periods=1).mean()
        signals['second_deriv'] = signals['ma'].diff().diff()
        sig = []
        prev = 0
        for val in signals['second_deriv']:
            if pd.isna(val):
                sig.append(prev)
            elif val > self.threshold:
                prev = 1
                sig.append(1)
            elif val < -self.threshold:
                prev = 0
                sig.append(0)
            else:
                sig.append(prev)
        signals['signal'] = sig
        signals['positions'] = pd.Series(sig, index=signals.index).diff().fillna(0.0)
        return signals

# ------------------------------
# Options Strategies (Simplified Simulation with Actual Options Data)
# ------------------------------

class WheelStrategyOptions(Strategy):
    def __init__(self, put_offset=0.05, call_offset=0.05, holding_period=5, shares=100):
        self.put_offset = put_offset
        self.call_offset = call_offset
        self.holding_period = holding_period
        self.shares = shares

    def simulate(self, data: pd.DataFrame, ticker: str, initial_capital: float) -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        option_dates = tkr.options
        if not option_dates:
            raise ValueError("No options data available for ticker.")
        exp_date = option_dates[0]
        chain = tkr.option_chain(exp_date)
        puts = chain.puts
        calls = chain.calls
        cash = initial_capital
        state = "no_stock"
        entry_price = None
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = float(data['Close'].iloc[i])
            if state == "no_stock":
                desired_strike = price * (1 - self.put_offset)
                idx = (puts['strike'] - desired_strike).abs().idxmin()
                chosen_put = puts.loc[idx]
                premium = float(chosen_put['lastPrice'])
                cash += premium * self.shares
                if i + self.holding_period < len(dates):
                    exp_price = float(data['Close'].iloc[i + self.holding_period])
                else:
                    exp_price = price
                if exp_price < chosen_put['strike']:
                    cost = chosen_put['strike'] * self.shares
                    if cash >= cost:
                        cash -= cost
                        state = "stock_owned"
                        entry_price = chosen_put['strike']
            elif state == "stock_owned":
                desired_strike = entry_price * (1 + self.call_offset)
                idx = (calls['strike'] - desired_strike).abs().idxmin()
                chosen_call = calls.loc[idx]
                premium_call = float(chosen_call['lastPrice'])
                if i + self.holding_period < len(dates):
                    exp_price = float(data['Close'].iloc[i + self.holding_period])
                else:
                    exp_price = price
                cash += premium_call * self.shares
                if exp_price > chosen_call['strike']:
                    cash += chosen_call['strike'] * self.shares
                    state = "no_stock"
                    entry_price = None
            portfolio_value = cash if state == "no_stock" else cash + float(data['Close'].iloc[i]) * self.shares
            portfolio_values.append(portfolio_value)
            i += 1
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

class CreditSpreadsStrategyOptions(Strategy):
    def __init__(self, spread_width=5.0, spread_offset=0.05, holding_period=5):
        self.spread_width = spread_width
        self.spread_offset = spread_offset
        self.holding_period = holding_period

    def simulate(self, data: pd.DataFrame, ticker: str, initial_capital: float) -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        option_dates = tkr.options
        if not option_dates:
            raise ValueError("No options data available for ticker.")
        exp_date = option_dates[0]
        chain = tkr.option_chain(exp_date)
        puts = chain.puts
        cash = initial_capital
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = float(data['Close'].iloc[i])
            desired_strike = price * (1 - self.spread_offset)
            idx = (puts['strike'] - desired_strike).abs().idxmin()
            chosen_put = puts.loc[idx]
            premium = float(chosen_put['lastPrice'])
            if i + self.holding_period < len(dates):
                exp_price = float(data['Close'].iloc[i + self.holding_period])
            else:
                exp_price = price
            if exp_price > chosen_put['strike']:
                profit = premium
            else:
                profit = - (self.spread_width - premium)
            cash += profit
            for j in range(self.holding_period):
                portfolio_values.append(cash)
            i += self.holding_period
        portfolio_values = portfolio_values[:len(dates)]
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

class IronCondorsStrategyOptions(Strategy):
    def __init__(self, lower_put_offset=0.05, upper_call_offset=0.05,
                 inner_put_offset=0.02, inner_call_offset=0.02, holding_period=5):
        self.lower_put_offset = lower_put_offset
        self.upper_call_offset = upper_call_offset
        self.inner_put_offset = inner_put_offset
        self.inner_call_offset = inner_call_offset
        self.holding_period = holding_period

    def simulate(self, data: pd.DataFrame, ticker: str, initial_capital: float) -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        option_dates = tkr.options
        if not option_dates:
            raise ValueError("No options data available for ticker.")
        exp_date = option_dates[0]
        chain = tkr.option_chain(exp_date)
        puts = chain.puts
        calls = chain.calls
        cash = initial_capital
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = float(data['Close'].iloc[i])
            inner_put_target = price * (1 - self.inner_put_offset)
            inner_call_target = price * (1 + self.inner_call_offset)
            outer_put_target = price * (1 - self.lower_put_offset)
            outer_call_target = price * (1 + self.upper_call_offset)
            idx_put = (puts['strike'] - inner_put_target).abs().idxmin()
            chosen_put = puts.loc[idx_put]
            idx_call = (calls['strike'] - inner_call_target).abs().idxmin()
            chosen_call = calls.loc[idx_call]
            premium_put = float(chosen_put['lastPrice'])
            premium_call = float(chosen_call['lastPrice'])
            total_premium = premium_put + premium_call
            if i + self.holding_period < len(dates):
                exp_price = float(data['Close'].iloc[i + self.holding_period])
            else:
                exp_price = price
            if (exp_price >= chosen_put['strike']) and (exp_price <= chosen_call['strike']):
                profit = total_premium
            else:
                loss_put = (chosen_put['strike'] - outer_put_target) if exp_price < chosen_put['strike'] else 0
                loss_call = (outer_call_target - chosen_call['strike']) if exp_price > chosen_call['strike'] else 0
                profit = total_premium - (loss_put + loss_call)
            cash += profit
            for j in range(self.holding_period):
                portfolio_values.append(cash)
            i += self.holding_period
        portfolio_values = portfolio_values[:len(dates)]
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

# ------------------------------
# Stock Backtesting Engine
# ------------------------------

class Backtester:
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 100000.0, shares: int = 100):
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.shares = shares

    def run_backtest(self) -> pd.DataFrame:
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions['positions'] = self.shares * self.signals['signal']
        close_prices = self.data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()
        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['holdings'] = positions['positions'] * close_prices
        pos_diff = positions['positions'].diff().fillna(0.0)
        portfolio['cash'] = self.initial_capital - (pos_diff * close_prices).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

    def run_backtest_custom(self, profit_target: float, stop_loss: float) -> pd.DataFrame:
        position = 0
        entry_price = None
        cash = self.initial_capital
        total_values = []
        positions_series = []
        for date, price in self.data['Close'].items():
            signal = self.signals['signal'].loc[date]
            if position == 0:
                if signal == 1:
                    position = self.shares
                    entry_price = price
                    cash -= price * self.shares
            else:
                if price >= entry_price * (1 + profit_target) or price <= entry_price * (1 - stop_loss):
                    cash += price * self.shares
                    position = 0
                    entry_price = None
                elif signal == 0:
                    cash += price * self.shares
                    position = 0
                    entry_price = None
            total = cash + position * price
            total_values.append(total)
            positions_series.append(position)
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['total'] = total_values
        portfolio['positions'] = positions_series
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

# ------------------------------
# Advanced Analytics Functions
# ------------------------------

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-9)

def compute_sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate/252
    negative_returns = excess_returns[excess_returns < 0]
    downside_std = negative_returns.std()
    return np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-9)

def compute_calmar_ratio(annual_return, max_drawdown):
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

def compute_drawdown_metrics(portfolio):
    total = portfolio['total']
    peak = total.cummax()
    drawdown = (total - peak) / peak
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean()
    recovery_times = []
    trough = total.iloc[0]
    trough_date = total.index[0]
    for date, value in total.items():
        if value < trough:
            trough = value
            trough_date = date
        if value >= peak.loc[date] and trough < value:
            recovery_times.append((date - trough_date).days)
            trough = value
    avg_recovery = np.mean(recovery_times) if recovery_times else np.nan
    return max_drawdown, avg_drawdown, avg_recovery

def monte_carlo_simulation(returns, initial_value, num_simulations, horizon):
    simulated_values = []
    daily_returns = np.ravel(returns.dropna().values)
    for _ in range(num_simulations):
        sim_return = np.random.choice(daily_returns, size=int(horizon), replace=True)
        sim_growth = np.prod(1 + sim_return)
        simulated_values.append(initial_value * sim_growth)
    return np.array(simulated_values)

def compute_VaR_CVaR(simulated_values, confidence_level=0.95):
    VaR = np.percentile(simulated_values, (1 - confidence_level) * 100)
    CVaR = simulated_values[simulated_values <= VaR].mean()
    return VaR, CVaR

def compute_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

def get_market_data(ticker="SPY", start_date="2020-01-01", end_date="2021-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def perform_statistical_tests(strategy_returns, market_returns):
    a = np.ravel(strategy_returns.dropna().values)
    b = np.ravel(market_returns.dropna().values)
    t_stat, p_value = ttest_ind(a, b, equal_var=False)
    adf_result = adfuller(strategy_returns.dropna())
    autocorr = acf(strategy_returns.dropna(), nlags=20)
    return t_stat, p_value, adf_result, autocorr

def compute_beta(strategy_returns, market_returns):
    common_index = strategy_returns.index.intersection(market_returns.index)
    if len(common_index) < 2:
        return np.nan
    a = np.ravel(strategy_returns.loc[common_index].dropna().values)
    b = np.ravel(market_returns.loc[common_index].dropna().values)
    if len(b) < 2:
        return np.nan
    covariance = np.cov(a, b)[0, 1]
    variance = np.var(b)
    return covariance / variance if variance != 0 else np.nan

# ------------------------------
# Visualization Functions
# ------------------------------

def plot_results(portfolio: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio")
    ax.set_title("Portfolio Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Value")
    ax.legend()
    ax.grid(True)
    return fig

def plot_buy_hold_comparison(portfolio: pd.DataFrame, data: pd.DataFrame, initial_capital: float):
    bh_shares = initial_capital / data['Close'].iloc[0]
    buy_hold = bh_shares * data['Close']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio")
    ax.plot(data.index, buy_hold, label="Buy & Hold", linestyle='--')
    ax.set_title("Strategy vs. Buy & Hold Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)
    return fig

def plot_monte_carlo(simulated_values):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(simulated_values, bins=50, alpha=0.7, color='orange')
    ax.set_title("Monte Carlo Simulation: Final Portfolio Values")
    ax.set_xlabel("Final Portfolio Value")
    ax.set_ylabel("Frequency")
    return fig

# ------------------------------
# New Interactive Visualizations for Advanced Analytics
# ------------------------------

def plot_beta_comparison(strategy_returns, market_returns):
    # Squeeze to ensure inputs are 1-dimensional
    if hasattr(strategy_returns, "squeeze"):
        strategy_returns = strategy_returns.squeeze()
    if hasattr(market_returns, "squeeze"):
        market_returns = market_returns.squeeze()
    df = pd.DataFrame({
        'Strategy Returns': strategy_returns,
        'Market Returns': market_returns
    }).dropna()
    fig = px.scatter(
        df, 
        x='Market Returns', 
        y='Strategy Returns',
        trendline='ols',
        title='Strategy vs. Market Returns (Interactive Beta Analysis)',
        labels={'Market Returns': 'Market Returns', 'Strategy Returns': 'Strategy Returns'}
    )
    return fig

def plot_qq(returns):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot(returns, line='s', ax=ax, alpha=0.5)
    ax.set_title('QQ-Plot of Strategy Returns')
    fig.tight_layout()
    return fig

def generate_report(portfolio, market_data, annual_return, max_dd, avg_dd, rec_time, sharpe, sortino, calmar, beta):
    report_dict = {
        "Total Return (%)": [(portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0] * 100],
        "Annualized Return (%)": [annual_return * 100],
        "Sharpe Ratio": [sharpe],
        "Sortino Ratio": [sortino],
        "Calmar Ratio": [calmar],
        "Max Drawdown (%)": [max_dd * 100],
        "Average Drawdown (%)": [avg_dd * 100],
        "Average Recovery Time (days)": [rec_time],
        "Portfolio Beta": [beta]
    }
    report_df = pd.DataFrame(report_dict)
    return report_df

# ------------------------------
# Streamlit Web App
# ------------------------------

def main():
    st.title("QuantBacktest Web App")
    st.markdown("A quantitative backtesting platform for stock and options trading strategies with advanced analytics.")

    # Sidebar – Backtest settings
    st.sidebar.header("Backtest Settings")
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2021, 1, 1))
    initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0)
    shares = st.sidebar.number_input("Number of Shares", value=100, step=1)

    strategy_options = ["SMA Crossover", "RSI Trading", "Bollinger Bands",
                        "Custom Profit/Stop", "Second Derivative MA",
                        "Wheel Strategy", "Credit Spreads", "Iron Condors"]
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_options)

    portfolio = None

    try:
        fetcher = DataFetcher(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        data = fetcher.fetch()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # ------------------------------
    # Run Selected Strategy
    # ------------------------------
    if selected_strategy in ["SMA Crossover", "RSI Trading", "Bollinger Bands", "Custom Profit/Stop", "Second Derivative MA"]:
        if selected_strategy in ["SMA Crossover", "Custom Profit/Stop"]:
            st.sidebar.subheader("SMA Parameters")
            sma_short = st.sidebar.slider("Short Window", min_value=5, max_value=100, value=50)
            sma_long = st.sidebar.slider("Long Window", min_value=20, max_value=300, value=200)
            strategy = SMACrossoverStrategy(short_window=sma_short, long_window=sma_long)
        elif selected_strategy == "RSI Trading":
            st.sidebar.subheader("RSI Parameters")
            rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14)
            oversold = st.sidebar.slider("Oversold Threshold", min_value=10, max_value=50, value=30)
            overbought = st.sidebar.slider("Overbought Threshold", min_value=50, max_value=90, value=70)
            strategy = RSITradingStrategy(period=rsi_period, oversold=oversold, overbought=overbought)
        elif selected_strategy == "Bollinger Bands":
            st.sidebar.subheader("Bollinger Bands Parameters")
            bb_window = st.sidebar.slider("Window", min_value=10, max_value=100, value=20)
            bb_std_multiplier = st.sidebar.slider("Std Dev Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            strategy = BollingerBandsStrategy(window=bb_window, std_multiplier=bb_std_multiplier)
        elif selected_strategy == "Second Derivative MA":
            st.sidebar.subheader("Second Derivative MA Parameters")
            sd_ma_window = st.sidebar.slider("MA Window", min_value=5, max_value=100, value=50)
            sd_threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            strategy = SecondDerivativeMAStrategy(ma_window=sd_ma_window, threshold=sd_threshold)
        signals = strategy.generate_signals(data)
        use_custom = False
        if selected_strategy == "Custom Profit/Stop":
            use_custom = True
            st.sidebar.subheader("Profit-taking / Stop-loss Settings")
            profit_target = st.sidebar.slider("Profit Target (%)", min_value=1, max_value=50, value=10) / 100.0
            stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=1, max_value=50, value=5) / 100.0
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                backtester = Backtester(data, signals, initial_capital, shares)
                portfolio = backtester.run_backtest_custom(profit_target, stop_loss) if use_custom else backtester.run_backtest()
    elif selected_strategy == "Wheel Strategy":
        st.sidebar.subheader("Wheel Strategy Parameters")
        put_offset = st.sidebar.slider("Put Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        call_offset = st.sidebar.slider("Call Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = WheelStrategyOptions(put_offset, call_offset, holding_period=holding_period, shares=shares)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Wheel Strategy simulation..."):
                portfolio = strategy.simulate(data, ticker, initial_capital)
    elif selected_strategy == "Credit Spreads":
        st.sidebar.subheader("Credit Spreads Parameters")
        spread_width = st.sidebar.number_input("Spread Width", value=5.0)
        spread_offset = st.sidebar.slider("Spread Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = CreditSpreadsStrategyOptions(spread_width, spread_offset, holding_period=holding_period)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Credit Spreads simulation..."):
                portfolio = strategy.simulate(data, ticker, initial_capital)
    elif selected_strategy == "Iron Condors":
        st.sidebar.subheader("Iron Condors Parameters")
        lower_put_offset = st.sidebar.slider("Lower Put Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        upper_call_offset = st.sidebar.slider("Upper Call Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        inner_put_offset = st.sidebar.slider("Inner Put Offset (%)", min_value=1, max_value=10, value=2) / 100.0
        inner_call_offset = st.sidebar.slider("Inner Call Offset (%)", min_value=1, max_value=10, value=2) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = IronCondorsStrategyOptions(lower_put_offset, upper_call_offset, inner_put_offset, inner_call_offset, holding_period=holding_period)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Iron Condors simulation..."):
                portfolio = strategy.simulate(data, ticker, initial_capital)

    # ------------------------------
    # Results & Visualization
    # ------------------------------
    if portfolio is not None:
        st.subheader("Performance Summary")
        total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
        days = (portfolio.index[-1] - portfolio.index[0]).days
        annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
        sharpe = compute_sharpe_ratio(portfolio['returns'])
        st.write({
            "Total Return (%)": f"{total_return * 100:.2f}%",
            "Annualized Return (%)": f"{annual_return * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}"
        })
        max_dd, avg_dd, rec_time = compute_drawdown_metrics(portfolio)
        sortino = compute_sortino_ratio(portfolio['returns'])
        calmar = compute_calmar_ratio(annual_return, max_dd)
        st.write({
            "Max Drawdown (%)": f"{max_dd * 100:.2f}%",
            "Average Drawdown (%)": f"{avg_dd * 100:.2f}%",
            "Avg Recovery Time (days)": f"{rec_time:.1f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Calmar Ratio": f"{calmar:.2f}"
        })

        st.subheader("Portfolio Performance")
        fig1 = plot_results(portfolio)
        st.pyplot(fig1)
        st.subheader("Strategy vs. Buy & Hold Comparison")
        fig2 = plot_buy_hold_comparison(portfolio, data, initial_capital)
        st.pyplot(fig2)

        # ------------------------------
        # Advanced Analytics Dashboard (Always Run All Analytics)
        # ------------------------------
        st.markdown("### Advanced Analytics Dashboard")
        st.markdown("Below are all the advanced analytics for your strategy.")

        # Retrieve market data for statistical tests and beta analysis
        market_data = get_market_data("SPY", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        market_returns = market_data['Close'].pct_change().fillna(0.0)
        beta = compute_beta(portfolio['returns'], market_returns)

        tabs = st.tabs([
            "Performance Overview", 
            "Beta Analysis", 
            "QQ Plot", 
            "Monte Carlo Simulation", 
            "Statistical Edge", 
            "Hedge Optimization", 
            "Options Greeks", 
            "Export Report"
        ])

        with tabs[0]:
            st.markdown("#### Performance Overview")
            st.write(f"**Annualized Return (%):** {annual_return * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
            st.write(f"**Sortino Ratio:** {sortino:.2f}")
            st.write(f"**Calmar Ratio:** {calmar:.2f}")
            st.write(f"**Max Drawdown (%):** {max_dd * 100:.2f}%")
            st.write(f"**Average Drawdown (%):** {avg_dd * 100:.2f}%")
            st.write(f"**Average Recovery Time (days):** {rec_time:.1f}")
            st.write(f"**Portfolio Beta (vs. SPY):** {beta:.2f}")

        with tabs[1]:
            st.markdown("#### Beta Analysis")
            beta_fig = plot_beta_comparison(portfolio['returns'], market_returns)
            st.plotly_chart(beta_fig, use_container_width=True)

        with tabs[2]:
            st.markdown("#### QQ Plot")
            qq_fig = plot_qq(portfolio['returns'].dropna())
            st.pyplot(qq_fig)

        with tabs[3]:
            st.markdown("#### Monte Carlo Simulation & Risk Metrics")
            num_simulations = 1000
            horizon = 252
            simulated_vals = monte_carlo_simulation(portfolio['returns'], portfolio['total'].iloc[-1], num_simulations, horizon)
            fig_mc = plot_monte_carlo(simulated_vals)
            st.pyplot(fig_mc)
            VaR, CVaR = compute_VaR_CVaR(simulated_vals, confidence_level=0.95)
            st.write(f"Value at Risk (95%): {VaR:.2f}")
            st.write(f"Conditional VaR (95%): {CVaR:.2f}")
            blowup_prob = np.mean(simulated_vals < initial_capital) * 100
            st.write(f"Blowup Probability (% final < initial): {blowup_prob:.2f}%")

        with tabs[4]:
            st.markdown("#### Statistical Edge & Market Comparison")
            t_stat, p_value, adf_result, autocorr = perform_statistical_tests(portfolio['returns'], market_returns)
            st.write(f"t-test Statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
            st.write("ADF Test Result:")
            st.write(adf_result)
            st.write("Autocorrelation (first 10 lags):")
            st.write(autocorr[:10])

        with tabs[5]:
            st.markdown("#### Hedge Optimization")
            st.write(f"Portfolio Beta (vs. SPY): {beta:.2f}")
            if beta > 1:
                st.write("Suggestion: The portfolio is more volatile than SPY. Consider hedging with SPY put options or other risk reduction strategies.")
            elif beta < 1:
                st.write("Suggestion: The portfolio is less volatile than SPY.")
            else:
                st.write("Suggestion: The portfolio beta is around 1, similar to the market.")

        with tabs[6]:
            st.markdown("#### Options Greeks Tracking")
            st.markdown("Enter parameters below to compute options Greeks:")
            S = st.number_input("Underlying Price (S)", value=float(data['Close'].iloc[-1]))
            K = st.number_input("Strike Price (K)", value=float(data['Close'].iloc[-1]))
            T = st.number_input("Time to Expiration (years)", value=0.25, step=0.01)
            r = st.number_input("Risk-Free Rate (annual)", value=0.02, step=0.001)
            sigma = st.number_input("Volatility (annual)", value=0.2, step=0.01)
            option_type = st.selectbox("Option Type", ["call", "put"])
            greeks = compute_greeks(S, K, T, r, sigma, option_type)
            st.write("Computed Options Greeks:")
            st.write(greeks)

        with tabs[7]:
            st.markdown("#### Export Summary Report")
            report_df = generate_report(portfolio, market_data, annual_return, max_dd, avg_dd, rec_time, sharpe, sortino, calmar, beta)
            st.dataframe(report_df)
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name='quantbacktest_report.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
