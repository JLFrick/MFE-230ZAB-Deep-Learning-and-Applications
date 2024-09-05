import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Dict

from strategy.portfolio import initialize_portfolio, get_portfolio_value
from strategy.actions import apply_action
from data.fetch_data import fetch_data
from data.process_data import calculate_returns
from strategy.rewards import calculate_reward

def run_simulation(
    tickers: List[str], 
    start_date: str,
    end_date: str,
    initial_cash: float, 
    action_function: Callable[[pd.Series, Dict[str, int]], np.ndarray],
    discount_factor: float = 1.0,
    seed: int = 123
) -> pd.Series:
    """
    Runs the trading simulation with a given action function.
    
    Parameters:
    tickers (list): List of stock tickers.
    start_date (str): Start date for fetching stock data.
    end_date (str): End date for fetching stock data.
    initial_cash (float): Initial cash balance.
    action_function (function): Function to decide actions based on state.
    seed (int): Random seed for reproducibility.
    
    Returns:
    Series: Time series of net worth over time.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Fetch data
    data = fetch_data(tickers, start_date, end_date, '1wk')
    
    # Process data to calculate returns
    returns = calculate_returns(data, 'fill_zero')

    # Initialize portfolio
    cash, portfolio = initialize_portfolio(initial_cash)
    net_worth_dict = {}
    cash_dict = {}
    portfolio_value_dict = {}
    reward_dict = {}

    dates = data.index

    # Run the simulation for each data entry
    for i in range(len(dates)):
        current_prices = data.iloc[i]
        current_returns = returns.iloc[i]

        # Current portfolio value
        portfolio_value = get_portfolio_value(portfolio, current_prices)
        net_worth = cash + portfolio_value
        
        # Store values in dictionaries
        net_worth_dict[dates[i]] = net_worth
        cash_dict[dates[i]] = cash
        portfolio_value_dict[dates[i]] = portfolio_value

        # Check if the simulation should terminate
        if net_worth <= 0:
            print(f"Simulation terminated on {dates[i]} due to portfolio value <= 0.")
            break

        # Use the action function to determine the next action
        actions = action_function(current_returns, portfolio)
        cash, portfolio = apply_action(cash, portfolio, current_prices, actions)
        reward = calculate_reward(cash, portfolio, current_prices, actions, discount_factor)
        reward_dict[dates[i]] = reward
        
    # Final portfolio value and net worth (if the loop ended naturally or was broken)
    final_portfolio_value = get_portfolio_value(portfolio, current_prices)
    net_worth = cash + final_portfolio_value
    print(net_worth)
    
    # Add the final portfolio value as an additional reward at the end
    final_reward = portfolio_value_dict[dates[i]] * discount_factor
    reward_dict[dates[i]] = final_reward
    
    # Convert dictionaries to pandas Series
    net_worth_series = pd.Series(net_worth_dict)
    cash_series = pd.Series(cash_dict)
    portfolio_value_series = pd.Series(portfolio_value_dict)
    reward_series = pd.Series(reward_dict)
    
    return net_worth_series, cash_series, portfolio_value_series, reward_series
