import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- Internationalization ---
TRANSLATIONS = {
    "en": {
        "page_title": "Stock Strategy Simulator",
        "main_title": "📈 Stock Strategy Simulator & Optimizer",
        "settings": "Settings",
        "select_stock": "Select Stock",
        "enter_ticker": "Enter Stock Ticker",
        "initial_capital": "Initial Capital ($)",
        "target_capital": "Target Capital ($)",
        "duration": "Duration (Months)",
        "prediction_settings": "Prediction Settings",
        "target_mode": "Target Price Mode",
        "manual_input": "Manual Input",
        "auto_prediction": "Auto Prediction (Linear Trend)",
        "target_price": "Target Price ($)",
        "pred_method": "Prediction Method",
        "method_linear": "Linear Trend (Regression)",
        "method_cagr": "Historical Growth (CAGR)",
        "method_analyst": "Analyst Consensus (Wall St)",
        "method_agent": "AI Technical Agent",
        "agent_reason": "Agent Reasoning: {}",
        "auto_predicted_msg": "Auto-predicted price: ${:.2f}",
        "sim_params": "Simulation Params",
        "volatility": "Annual Volatility (Sigma)",
        "volatility_help": "Higher means more oscillation",
        "num_sims": "Number of Simulations",
        "analysis_title": "### Analysis: {} ({})",
        "current_price": "Current Price",
        "required_return": "Required Return",
        "sim_section": "1. Price Path Simulation",
        "sim_spinner": "Simulating market scenarios...",
        "chart_title": "Simulated Price Paths for {}",
        "chart_xaxis": "Trading Days",
        "chart_yaxis": "Price",
        "strat_section": "2. Strategy Comparison",
        "select_strats": "Select Strategies to Compare",
        "strat_buy_hold": "Buy & Hold",
        "strat_grid": "Grid Trading (Oscillation)",
        "strat_rebalance": "50/50 Rebalancing",
        "col_strategy": "Strategy",
        "col_avg_val": "Avg Final Value",
        "col_min_val": "Min Value",
        "col_max_val": "Max Value",
        "col_success": "Success Rate (>Target)",
        "best_strat_msg": "🏆 Best Performing Strategy (on average): **{}** with Avg Value **${:,.2f}**",
        "warning_msg": "⚠️ Even the best strategy averages below your target of ${:,.2f}. Consider increasing risk (leverage), extending duration, or adjusting expectations.",
        "error_ticker": "Please enter a valid ticker.",
        "error_fetch": "Error fetching data for {}: {}",
        "custom": "Custom",
        "custom_strat_settings": "Custom Strategy Settings",
        "custom_strat_config": "Configure Thresholds",
        "buy_threshold": "Buy Drop Threshold (%)",
        "buy_pct": "Buy Amount (% of Cash)",
        "sell_threshold": "Sell Rise Threshold (%)",
        "sell_pct": "Sell Amount (% of Shares)",
        "strat_custom": "Custom Threshold Strategy (Dynamic)"
    },
    "zh": {
        "page_title": "股票策略模拟器",
        "main_title": "📈 股票策略模拟与优化器",
        "settings": "设置",
        "select_stock": "选择股票",
        "enter_ticker": "输入股票代码",
        "initial_capital": "初始本金 ($)",
        "target_capital": "目标金额 ($)",
        "duration": "时长 (月)",
        "prediction_settings": "预测设置",
        "target_mode": "目标价模式",
        "manual_input": "手动输入",
        "auto_prediction": "自动预测 (线性趋势)",
        "target_price": "目标价格 ($)",
        "pred_method": "预测机制",
        "method_linear": "线性趋势 (回归分析)",
        "method_cagr": "历史增长率 (CAGR)",
        "method_analyst": "分析师一致预期 (华尔街)",
        "method_agent": "AI 技术分析代理",
        "agent_reason": "代理分析逻辑: {}",
        "auto_predicted_msg": "自动预测价格: ${:.2f}",
        "sim_params": "模拟参数",
        "volatility": "年化波动率 (Sigma)",
        "volatility_help": "数值越高代表震荡越剧烈",
        "num_sims": "模拟次数",
        "analysis_title": "### 分析: {} ({})",
        "current_price": "当前价格",
        "required_return": "所需回报率",
        "sim_section": "1. 价格路径模拟",
        "sim_spinner": "正在模拟市场情景...",
        "chart_title": "{} 的模拟价格路径",
        "chart_xaxis": "交易日",
        "chart_yaxis": "价格",
        "strat_section": "2. 策略对比分析",
        "select_strats": "选择要对比的策略",
        "strat_buy_hold": "买入持有 (Buy & Hold)",
        "strat_grid": "网格交易 (震荡策略)",
        "strat_rebalance": "50/50 动态平衡",
        "col_strategy": "策略",
        "col_avg_val": "平均最终价值",
        "col_min_val": "最低价值",
        "col_max_val": "最高价值",
        "col_success": "成功率 (>目标)",
        "best_strat_msg": "🏆 表现最佳策略 (平均): **{}**，平均价值 **${:,.2f}**",
        "warning_msg": "⚠️ 即使是最佳策略，平均结果也低于您的目标 ${:,.2f}。请考虑增加风险 (杠杆)、延长时长或调整预期。",
        "error_ticker": "请输入有效的股票代码。",
        "error_fetch": "获取 {} 数据时出错: {}",
        "custom": "自定义",
        "custom_strat_settings": "自定义策略设置",
        "custom_strat_config": "配置买卖阈值",
        "buy_threshold": "下跌买入阈值 (%)",
        "buy_pct": "买入资金比例 (%)",
        "sell_threshold": "上涨卖出阈值 (%)",
        "sell_pct": "卖出持仓比例 (%)",
        "strat_custom": "自定义阈值策略 (动态)"
    }
}

# --- Configuration & Helper Functions ---

def fetch_stock_data(ticker):
    """Fetches historical data and current info."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y") # Fetch 2 years for trend analysis
        info = stock.info
        current_price = hist['Close'].iloc[-1] if not hist.empty else 0
        return stock, hist, info, current_price
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, pd.DataFrame(), {}, 0

def predict_target_price(hist, months):
    """
    Simple prediction based on linear regression of the last year's data.
    """
    if hist.empty:
        return 0
    
    # Use last 1 year for trend
    df = hist.reset_index()
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    
    X = df['Date_Ordinal'].values.reshape(-1, 1)
    y = df['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_date = datetime.now() + timedelta(days=months*30)
    future_ordinal = np.array([[future_date.toordinal()]])
    prediction = model.predict(future_ordinal)[0]
    
    return max(0, prediction) # Ensure positive

def calculate_technical_indicators(hist):
    # Simple RSI and MACD calculation
    df = hist.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def predict_price_agent(hist, months, current_price):
    # "AI Agent" Logic
    if hist.empty: return current_price, "No Data"
    
    df = calculate_technical_indicators(hist)
    last_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
    last_macd = df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0
    last_signal = df['Signal_Line'].iloc[-1] if not pd.isna(df['Signal_Line'].iloc[-1]) else 0
    
    # Base drift from last year (CAGR)
    lookback = min(len(hist), 252)
    start_price_period = hist['Close'].iloc[-lookback]
    annual_return = (current_price / start_price_period) - 1
    
    # Adjust based on technicals
    sentiment_score = 0
    reason_parts = [f"Base Annual Return: {annual_return:.1%}"]
    
    # RSI Logic
    if last_rsi < 30: 
        sentiment_score += 0.15 
        reason_parts.append("RSI Oversold (+15%)")
    elif last_rsi > 70: 
        sentiment_score -= 0.15
        reason_parts.append("RSI Overbought (-15%)")
    else:
        reason_parts.append(f"RSI Neutral ({last_rsi:.0f})")
    
    # MACD Logic
    if last_macd > last_signal: 
        sentiment_score += 0.05
        reason_parts.append("MACD Bullish (+5%)")
    else: 
        sentiment_score -= 0.05
        reason_parts.append("MACD Bearish (-5%)")
    
    # Combine
    adjusted_return = annual_return + sentiment_score
    
    # Project
    future_price = current_price * (1 + adjusted_return * (months/12))
    return max(0, future_price), ", ".join(reason_parts)

def simulate_price_paths(start_price, target_price, months, volatility, num_simulations=50):
    """
    Monte Carlo simulation of price paths.
    Returns a DataFrame where each column is a simulation path.
    """
    days = int(months * 21)
    dt = 1/252
    
    # Drift to hit target price at end of period (on average)
    # E[S_T] = S_0 * exp(mu * T)
    T = days / 252
    if start_price > 0:
        mu = np.log(target_price / start_price) / T
    else:
        mu = 0

    paths = []
    for _ in range(num_simulations):
        prices = [start_price]
        for _ in range(days):
            shock = np.random.normal(0, 1)
            price = prices[-1] * np.exp((mu - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shock)
            prices.append(price)
        paths.append(prices)
    
    return np.array(paths).T # Shape: (days+1, num_simulations)

# --- Strategy Implementations ---

def strategy_buy_and_hold(prices, initial_cash):
    """Simple Buy and Hold."""
    if len(prices) == 0: return []
    shares = initial_cash / prices[0]
    values = prices * shares
    return values

def strategy_grid_trading(prices, initial_cash, grid_size_pct=0.03, trade_share_pct=0.1):
    """
    Simple Grid Trading:
    - Buy when price drops by grid_size_pct
    - Sell when price rises by grid_size_pct
    """
    cash = initial_cash
    shares = 0
    values = []
    
    # Initial entry: buy 50% position to have inventory for selling
    initial_entry_shares = (initial_cash * 0.5) / prices[0]
    cash -= initial_entry_shares * prices[0]
    shares += initial_entry_shares
    
    last_trade_price = prices[0]
    
    for price in prices:
        # Calculate portfolio value
        values.append(cash + shares * price)
        
        # Grid Logic
        price_change = (price - last_trade_price) / last_trade_price
        
        if price_change <= -grid_size_pct:
            # Buy
            cost = cash * trade_share_pct
            if cost > 0:
                num_buy = cost / price
                shares += num_buy
                cash -= cost
                last_trade_price = price
                
        elif price_change >= grid_size_pct:
            # Sell
            num_sell = shares * trade_share_pct
            if num_sell > 0:
                shares -= num_sell
                cash += num_sell * price
                last_trade_price = price
                
    return np.array(values)

def strategy_rebalance(prices, initial_cash, target_allocation=0.5, rebalance_threshold=0.05):
    """
    Maintains a fixed ratio of Cash vs Stock (e.g., 50/50).
    """
    cash = initial_cash
    shares = 0
    values = []
    
    # Initial Buy
    shares = (initial_cash * target_allocation) / prices[0]
    cash -= shares * prices[0]
    
    for price in prices:
        total_value = cash + shares * price
        values.append(total_value)
        
        current_allocation = (shares * price) / total_value
        
        if abs(current_allocation - target_allocation) > rebalance_threshold:
            target_stock_value = total_value * target_allocation
            current_stock_value = shares * price
            diff = target_stock_value - current_stock_value
            
            if diff > 0: # Buy
                shares_to_buy = diff / price
                if cash >= diff:
                    shares += shares_to_buy
                    cash -= diff
            else: # Sell
                shares_to_sell = abs(diff) / price
                if shares >= shares_to_sell:
                    shares -= shares_to_sell
                    cash += abs(diff)
                    
    return np.array(values)

def strategy_custom_threshold(prices, initial_cash, buy_threshold=0.05, buy_pct=0.3, sell_threshold=0.05, sell_pct=0.2):
    """
    Custom Threshold Strategy:
    - Buy buy_pct of cash when price drops by buy_threshold from last trade.
    - Sell sell_pct of shares when price rises by sell_threshold from last trade.
    """
    cash = initial_cash
    shares = 0
    values = []
    
    # Initial entry: 50% position
    initial_entry_shares = (initial_cash * 0.5) / prices[0]
    cash -= initial_entry_shares * prices[0]
    shares += initial_entry_shares
    
    last_trade_price = prices[0]
    
    for price in prices:
        # Calculate portfolio value
        values.append(cash + shares * price)
        
        # Calculate change from last trade
        price_change = (price - last_trade_price) / last_trade_price
        
        if price_change <= -buy_threshold:
            # Buy Dip
            amount_to_spend = cash * buy_pct
            if amount_to_spend > 0:
                num_buy = amount_to_spend / price
                shares += num_buy
                cash -= amount_to_spend
                last_trade_price = price
                
        elif price_change >= sell_threshold:
            # Sell Rally
            shares_to_sell = shares * sell_pct
            if shares_to_sell > 0:
                shares -= shares_to_sell
                cash += shares_to_sell * price
                last_trade_price = price
                
    return np.array(values)

# --- Main App UI ---

st.set_page_config(page_title="Stock Strategy Simulator", layout="wide")

# Language Selector
lang_code = st.sidebar.selectbox("Language / 语言", ["English", "中文"])
lang = "en" if lang_code == "English" else "zh"
t = TRANSLATIONS[lang]

st.title(t["main_title"])

# Sidebar Controls
with st.sidebar:
    st.header(t["settings"])
    
    # Quick Select for popular stocks
    popular_stocks = {
        "Tesla (TSLA)": "TSLA",
        "Nvidia (NVDA)": "NVDA",
        "Alphabet (GOOGL)": "GOOGL",
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Amazon (AMZN)": "AMZN",
        "Meta (META)": "META",
        t["custom"]: "CUSTOM"
    }
    
    stock_choice = st.selectbox(t["select_stock"], options=list(popular_stocks.keys()))
    
    if stock_choice == t["custom"] or stock_choice == "CUSTOM":
        ticker = st.text_input(t["enter_ticker"], value="TSLA").upper()
    else:
        ticker = popular_stocks[stock_choice]

    initial_capital = st.number_input(t["initial_capital"], value=1_000_000, step=100_000)
    target_capital = st.number_input(t["target_capital"], value=2_000_000, step=100_000)
    duration_months = st.slider(t["duration"], 1, 24, 11)
    
    st.subheader(t["prediction_settings"])
    prediction_mode = st.radio(t["target_mode"], [t["manual_input"], t["auto_prediction"]])
    
    stock, hist, info, current_price = fetch_stock_data(ticker)
    
    if prediction_mode == t["manual_input"]:
        target_price = st.number_input(t["target_price"], value=480.0)
    else:
        # Prediction Method Selection
        pred_method = st.selectbox(t["pred_method"], [
            t["method_linear"], 
            t["method_cagr"], 
            t["method_analyst"],
            t["method_agent"]
        ])
        
        predicted = 0
        reason = ""
        
        if pred_method == t["method_linear"]:
            predicted = predict_target_price(hist, duration_months)
        elif pred_method == t["method_cagr"]:
            # CAGR Logic
            if not hist.empty:
                lookback = min(len(hist), 252)
                start_p = hist['Close'].iloc[-lookback]
                cagr = (current_price / start_p) - 1
                predicted = current_price * (1 + cagr * (duration_months/12))
        elif pred_method == t["method_analyst"]:
            # Analyst Target
            predicted = info.get('targetMeanPrice', current_price)
            if predicted is None: predicted = current_price
        elif pred_method == t["method_agent"]:
            # AI Agent
            predicted, reason = predict_price_agent(hist, duration_months, current_price)
            st.caption(t["agent_reason"].format(reason))

        target_price = st.number_input(t["target_price"], value=float(f"{predicted:.2f}"))
        st.info(t["auto_predicted_msg"].format(predicted))

    st.subheader(t["custom_strat_settings"])
    with st.expander(t["custom_strat_config"], expanded=True):
        c_buy_thresh = st.slider(t["buy_threshold"], 1, 20, 5) / 100.0
        c_buy_pct = st.slider(t["buy_pct"], 10, 100, 30) / 100.0
        c_sell_thresh = st.slider(t["sell_threshold"], 1, 20, 5) / 100.0
        c_sell_pct = st.slider(t["sell_pct"], 10, 100, 20) / 100.0

    st.subheader(t["sim_params"])
    volatility = st.slider(t["volatility"], 0.1, 1.0, 0.45, help=t["volatility_help"])
    num_sims = st.slider(t["num_sims"], 1, 100, 20)

# Main Content
if stock:
    # Display Company Name
    company_name = info.get('longName', ticker)
    st.markdown(t["analysis_title"].format(company_name, ticker))

    col1, col2, col3 = st.columns(3)
    col1.metric(t["current_price"], f"${current_price:.2f}")
    col2.metric(t["target_price"], f"${target_price:.2f}")
    required_return = (target_capital - initial_capital) / initial_capital
    col3.metric(t["required_return"], f"{required_return:.1%}")

    # 1. Run Simulation
    st.subheader(t["sim_section"])
    with st.spinner(t["sim_spinner"]):
        sim_paths = simulate_price_paths(current_price, target_price, duration_months, volatility, num_sims)
    
    # Plot Simulations
    fig_sim = go.Figure()
    # Plot first 10 paths to avoid clutter
    for i in range(min(10, num_sims)):
        fig_sim.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', name=f'Sim {i+1}', opacity=0.3, line=dict(width=1)))
    
    fig_sim.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text=t["target_price"])
    fig_sim.add_hline(y=current_price, line_dash="dash", line_color="gray", annotation_text=t["current_price"])
    fig_sim.update_layout(title=t["chart_title"].format(ticker), xaxis_title=t["chart_xaxis"], yaxis_title=t["chart_yaxis"])
    st.plotly_chart(fig_sim, use_container_width=True)

    # 2. Strategy Comparison
    st.subheader(t["strat_section"])
    
    # Define custom strategy with current parameters
    from functools import partial
    custom_strat_func = partial(strategy_custom_threshold, 
                                buy_threshold=c_buy_thresh, 
                                buy_pct=c_buy_pct, 
                                sell_threshold=c_sell_thresh, 
                                sell_pct=c_sell_pct)

    strategies = {
        t["strat_buy_hold"]: strategy_buy_and_hold,
        t["strat_grid"]: strategy_grid_trading,
        t["strat_rebalance"]: strategy_rebalance,
        t["strat_custom"]: custom_strat_func
    }
    
    selected_strategies = st.multiselect(t["select_strats"], list(strategies.keys()), default=list(strategies.keys()))
    
    results = []
    
    # We will run strategies on ALL simulation paths and average the results
    for strat_name in selected_strategies:
        strat_func = strategies[strat_name]
        final_values = []
        success_count = 0
        
        for i in range(num_sims):
            path = sim_paths[:, i]
            portfolio_curve = strat_func(path, initial_capital)
            final_val = portfolio_curve[-1]
            final_values.append(final_val)
            if final_val >= target_capital:
                success_count += 1
        
        avg_final = np.mean(final_values)
        min_final = np.min(final_values)
        max_final = np.max(final_values)
        prob_success = (success_count / num_sims) * 100
        
        results.append({
            t["col_strategy"]: strat_name,
            t["col_avg_val"]: avg_final,
            t["col_min_val"]: min_final,
            t["col_max_val"]: max_final,
            t["col_success"]: prob_success
        })

    results_df = pd.DataFrame(results)
    
    # Display Results Table
    st.table(results_df.style.format({
        t["col_avg_val"]: "${:,.2f}",
        t["col_min_val"]: "${:,.2f}",
        t["col_max_val"]: "${:,.2f}",
        t["col_success"]: "{:.1f}%"
    }))
    
    # Best Strategy Highlight
    if not results_df.empty:
        best_strat = results_df.loc[results_df[t["col_avg_val"]].idxmax()]
        st.success(t["best_strat_msg"].format(best_strat[t["col_strategy"]], best_strat[t["col_avg_val"]]))

        if best_strat[t["col_avg_val"]] < target_capital:
            st.warning(t["warning_msg"].format(target_capital))
        else:
            st.balloons()

else:
    st.warning(t["error_ticker"])
