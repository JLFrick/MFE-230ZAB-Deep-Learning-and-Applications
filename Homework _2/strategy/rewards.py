from typing import Dict
import pandas as pd
import numpy as np

def calculate_reward(
    cash: float, 
    portfolio: Dict[str, int], 
    current_prices: pd.Series, 
    actions: np.ndarray,
    discount_factor: float = 1.0
) -> float:
    """
    Calculates the immediate reward based on the actions taken.
    
    Parameters:
    cash (float): Current cash balance before actions.
    portfolio (dict): Current portfolio holdings before actions.
    current_prices (Series): Current prices of the stocks.
    actions (ndarray): Array of actions taken (1 = buy, -1 = sell, 0 = hold).
    discount_factor (float): Discount factor for the reward. Default is 1 (no discount).
    
    Returns:
    float: The immediate reward based on the actions taken.
    """
    immediate_reward = 0.0
    
    for i, ticker in enumerate(current_prices.index):
        action = actions[i]
        if action == 1:  # Buy 1 share
            immediate_reward -= current_prices[ticker]  # Cost of buying
        elif action == -1:  # Sell 1 share
            immediate_reward += current_prices[ticker]  # Gain from selling
    
    return immediate_reward

