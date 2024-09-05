from typing import Tuple, Dict
import pandas as pd

def initialize_portfolio(initial_cash: float) -> Tuple[float, Dict[str, int]]:
    """
    Initializes the portfolio with a given amount of cash and zero stock holdings.
    
    Parameters:
    initial_cash (float): Initial cash in the portfolio.
    
    Returns:
    tuple: Initial cash and an empty portfolio dictionary.
    """
    return initial_cash, {}

def get_portfolio_value(portfolio: Dict[str, int], prices: pd.Series) -> float:
    """
    Calculates the total value of the portfolio based on current prices.
    
    Parameters:
    portfolio (dict): Dictionary with stock holdings.
    prices (Series): Current prices of the stocks.
    
    Returns:
    float: The total value of the portfolio.
    """
    return sum(portfolio[ticker] * prices[ticker] for ticker in portfolio)
