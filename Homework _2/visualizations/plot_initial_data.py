import matplotlib.pyplot as plt
import pandas as pd

def plot_stock_prices(data: pd.DataFrame) -> None:
    """
    Plots the historical stock prices for the given data.
    
    Parameters:
    data (DataFrame): Historical stock data.
    """
    plt.figure(figsize=(12, 6))
    for ticker in data.columns:
        plt.plot(data.index, data[ticker], label=ticker)
    plt.title('Historical Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_stock_returns(returns: pd.DataFrame) -> None:
    """
    Plots the historical stock returns for the given data.
    
    Parameters:
    returns (DataFrame): Historical stock returns.
    """
    plt.figure(figsize=(12, 6))
    for ticker in returns.columns:
        plt.plot(returns.index, returns[ticker], label=f'{ticker} Returns')
    plt.title('Historical Stock Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_histogram_of_returns(returns: pd.DataFrame) -> None:
    """
    Plots histograms of the stock returns to visualize their distribution.
    
    Parameters:
    returns (DataFrame): Historical stock returns.
    """
    plt.figure(figsize=(12, 6))
    for ticker in returns.columns:
        plt.hist(returns[ticker], bins=50, alpha=0.5, label=f'{ticker} Returns')
    plt.title('Histogram of Stock Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()