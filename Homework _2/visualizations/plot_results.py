import matplotlib.pyplot as plt
import pandas as pd

def plot_series(series: pd.Series, title: str, ylabel: str) -> None:
    """
    Plots a financial time series over time.
    
    Parameters:
    series (Series): Time series data to plot.
    title (str): Title of the plot.
    ylabel (str): Y-axis label.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series.values, marker='o', linestyle='-', markersize=4)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_net_worth(net_worth_series: pd.Series) -> None:
    """
    Plots the net worth trajectory over time.
    
    Parameters:
    net_worth_series (Series): Time series of net worth values over time.
    """
    plot_series(net_worth_series, 'Net Worth Over Time', 'Net Worth ($)')

def plot_cash(cash_series: pd.Series) -> None:
    """
    Plots the cash trajectory over time.
    
    Parameters:
    cash_series (Series): Time series of cash values over time.
    """
    plot_series(cash_series, 'Cash Over Time', 'Cash ($)')

def plot_portfolio_value(portfolio_value_series: pd.Series) -> None:
    """
    Plots the portfolio value trajectory over time.
    
    Parameters:
    portfolio_value_series (Series): Time series of portfolio values over time.
    """
    plot_series(portfolio_value_series, 'Portfolio Value Over Time', 'Portfolio Value ($)')

def plot_rewards(reward_series: pd.Series) -> None:
    """
    Plots the reward trajectory over time.
    
    Parameters:
    reward_series (Series): Time series of reward values over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(reward_series.index, reward_series.values, marker='o', linestyle='-', markersize=4, color='tab:orange')
    plt.title('Rewards Over Time')
    plt.xlabel('Date')
    plt.ylabel('Reward ($)')
    plt.grid(True)
    plt.show()