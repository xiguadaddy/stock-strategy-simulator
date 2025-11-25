import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fetch_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Get the last available price
        history = stock.history(period="1d")
        if not history.empty:
            return history['Close'].iloc[-1]
        else:
            return 350.0 # Fallback assumption if fetch fails
    except Exception as e:
        print(f"Error fetching price: {e}")
        return 350.0

def simulate_price_path(start_price, end_price, months, volatility=0.4):
    """
    Simulates a geometric brownian motion path with a drift towards end_price
    """
    days = months * 21 # Trading days
    dt = 1/252
    
    # Calculate required drift to hit end_price
    # E[S_T] = S_0 * exp(mu * T)
    # end_price = start_price * exp(mu * (days/252))
    T = days / 252
    mu = np.log(end_price / start_price) / T
    
    prices = [start_price]
    for _ in range(days):
        shock = np.random.normal(0, 1)
        # dS = S * (mu * dt + sigma * dW)
        price = prices[-1] * np.exp((mu - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shock)
        prices.append(price)
        
    return np.array(prices)

def grid_trading_strategy(prices, initial_cash, grid_size=0.05):
    """
    Simple grid trading simulation
    """
    cash = initial_cash
    shares = 0
    portfolio_values = []
    
    # Setup grid
    # This is a simplified backtest
    
    for price in prices:
        # Logic for grid trading would go here
        # For now, let's just track buy and hold to compare
        portfolio_values.append(cash + shares * price)
        
    return portfolio_values

def main():
    ticker = "TSLA"
    current_price = fetch_current_price(ticker)
    target_price = 480
    months = 11
    principal = 1_000_000
    
    print(f"Current {ticker} Price: ${current_price:.2f}")
    print(f"Target Price: ${target_price}")
    print(f"Goal: ${principal} -> ${2_000_000}")
    
    # Check simple buy and hold return
    simple_return = (target_price - current_price) / current_price
    print(f"Buy and Hold Return: {simple_return:.2%}")
    
    if simple_return < 1.0:
        print("Simple Buy and Hold will NOT reach the target (requires 100% return).")
        print("Need to utilize volatility (Grid Trading / Swing Trading) or Leverage.")
    
    # Simulate
    np.random.seed(42)
    prices = simulate_price_path(current_price, target_price, months)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(prices, label='Simulated Price Path')
    plt.axhline(y=target_price, color='r', linestyle='--', label='Target Price')
    plt.title(f"Simulated {ticker} Price Path (11 Months)")
    plt.xlabel("Trading Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig('simulation.png')
    print("Simulation plot saved to simulation.png")

if __name__ == "__main__":
    main()
