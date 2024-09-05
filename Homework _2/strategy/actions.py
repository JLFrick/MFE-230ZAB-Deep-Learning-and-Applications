import numpy as np
import pandas as pd
from typing import Dict, Tuple

def random_action(current_returns: pd.Series, portfolio: Dict[str, int]) -> np.ndarray:
    """
    Generates a random action for the trading strategy.
    
    Parameters:
    current_returns (Series): The returns of the stocks for the current period.
    portfolio (dict): The current holdings of the portfolio.
    
    Returns:
    ndarray: Array with random actions for each stock (1 = long, -1 = short, 0 = neutral).
    """
    # np.random.seed(123)  # Seed for reproducibility
    actions = np.random.choice([-1, 0, 1], size=len(current_returns))
    return actions

def apply_action(
    cash: float, 
    portfolio: Dict[str, int], 
    prices: pd.Series, 
    actions: np.ndarray
) -> Tuple[float, Dict[str, int]]:
    """
    Applies the chosen action to the portfolio and updates the cash balance.
    
    Parameters:
    cash (float): Current cash balance.
    portfolio (dict): Current portfolio holdings.
    prices (Series): Current prices of the stocks.
    actions (ndarray): Array of actions to take (1 = long, -1 = short, 0 = neutral).
    
    Returns:
    tuple: Updated cash balance and portfolio.
    """
    for i, ticker in enumerate(prices.index):
        action = actions[i]
        if action == 1:  # Long 1 share
            cash -= prices[ticker]
            portfolio[ticker] = portfolio.get(ticker, 0) + 1
        elif action == -1:  # Short 1 share
            cash += prices[ticker]
            portfolio[ticker] = portfolio.get(ticker, 0) - 1
    return cash, portfolio
