import yfinance as yf
import pandas as pd
from typing import List

def fetch_data(tickers: List[str], start_date: str, end_date: str, interval: str = "1wk") -> pd.DataFrame:
    """
    Fetches historical stock data for the given tickers from Yahoo Finance.
    
    Parameters:
    tickers (list): List of stock tickers.
    start_date (str): Start date for the data.
    end_date (str): End date for the data.
    
    Returns:
    DataFrame: Adjusted close prices for the tickers.
    """
    return yf.download(tickers, start=start_date, end=end_date, interval=interval)['Adj Close']
