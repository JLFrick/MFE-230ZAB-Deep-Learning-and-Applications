import pandas as pd
from typing import Optional

def calculate_returns(data: pd.DataFrame, fill_method: Optional[str] = 'drop') -> pd.DataFrame:
    """
    Calculates returns for the given stock data.
    
    Parameters:
    data (DataFrame): Historical stock data.
    fill_method (str, optional): Method to handle NaN values at the start of the returns series.
                                 Options are 'drop' (default), 'fill_zero', or 'none'.
    
    Returns:
    DataFrame: Returns for each stock with NaN handling as specified.
    """
    returns = data.pct_change()
    
    if fill_method == 'drop':
        returns = returns.dropna()
    elif fill_method == 'fill_zero':
        returns = returns.fillna(0)
    # If fill_method is 'none', do not alter the NaN values (keep them as NaN)
    
    return returns
