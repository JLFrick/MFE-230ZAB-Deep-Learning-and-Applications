import numpy as np
import pandas as pd
from typing import List, Dict, Callable

from strategy.portfolio import initialize_portfolio, get_portfolio_value
from strategy.actions import apply_action
from data.fetch_data import fetch_data
from data.process_data import calculate_returns
from strategy.rewards import calculate_reward


class TradingSimulation:
    def __init__(
        self, 
        tickers: List[str], 
        start_date: str,
        end_date: str,
        initial_cash: float, 
        action_function: Callable[[pd.Series, Dict[str, int]], np.ndarray], 
        discount_factor: float = 1.0,
        seed: int = 123
    ):
        """
        Initializes the trading simulation.

        Parameters:
        tickers (list): List of stock tickers.
        start_date (str): Start date for fetching stock data.
        end_date (str): End date for fetching stock data.
        initial_cash (float): Initial cash balance.
        action_function (function): Function to decide actions based on state.
        discount_factor (float): Discount factor for the reward.
        seed (int): Random seed for reproducibility.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.action_function = action_function
        self.discount_factor = discount_factor
        self.seed = seed
        
        # Initialize state variables
        self.data = None
        self.returns = None
        self.cash = None
        self.portfolio = None
        self.net_worth_series = pd.Series()
        self.cash_series = pd.Series()
        self.portfolio_value_series = pd.Series()
        self.reward_series = pd.Series()
        
        # Set random seed
        np.random.seed(self.seed)

    def fetch_data(self):
        """Fetches historical stock data and calculates returns."""
        self.data = fetch_data(self.tickers, self.start_date, self.end_date, '1wk')
        self.returns = calculate_returns(self.data, 'fill_zero')

    def initialize_simulation(self):
        """Initializes the portfolio and cash for the simulation."""
        self.cash, self.portfolio = initialize_portfolio(self.initial_cash)

    def run(self):
        """Runs the trading simulation and computes net worth, cash, portfolio value, and rewards."""
        self.initialize_simulation()
        self.fetch_data()
        
        net_worth_dict = {}
        cash_dict = {}
        portfolio_value_dict = {}
        reward_dict = {}

        dates = self.data.index

        # Run the simulation for each data entry
        for i in range(len(dates)):
            current_prices = self.data.iloc[i]
            current_returns = self.returns.iloc[i]

            # Calculate current portfolio value and net worth
            portfolio_value = get_portfolio_value(self.portfolio, current_prices)
            net_worth = self.cash + portfolio_value
            
            # Store values in dictionaries
            net_worth_dict[dates[i]] = net_worth
            cash_dict[dates[i]] = self.cash
            portfolio_value_dict[dates[i]] = portfolio_value

            # Check if the simulation should terminate
            if net_worth <= 0:
                print(f"Simulation terminated on {dates[i]} due to portfolio value <= 0.")
                break

            # Use the action function to determine the next action
            actions = self.action_function(current_returns, self.portfolio)
            
            # Calculate the immediate and potentially discounted reward
            reward = calculate_reward(self.cash, self.portfolio, current_prices, actions, self.discount_factor)
            reward_dict[dates[i]] = reward
            
            # Update cash and portfolio with actions
            self.cash, self.portfolio = apply_action(self.cash, self.portfolio, current_prices, actions)

        # Final portfolio value and net worth (if the loop ended naturally or was broken)
        final_portfolio_value = get_portfolio_value(self.portfolio, self.data.iloc[-1] if net_worth > 0 else current_prices)
        net_worth = self.cash + final_portfolio_value
        
        # Store final values in dictionaries
        if net_worth > 0:  # If ended naturally
            net_worth_dict[dates[-1]] = net_worth
            cash_dict[dates[-1]] = self.cash
            portfolio_value_dict[dates[-1]] = final_portfolio_value
        else:  # If terminated early
            net_worth_dict[dates[i]] = net_worth
            cash_dict[dates[i]] = self.cash
            portfolio_value_dict[dates[i]] = final_portfolio_value
        
        # Add the final portfolio value as an additional reward at the end
        final_reward = final_portfolio_value * self.discount_factor
        reward_dict[dates[i] if net_worth <= 0 else dates[-1]] = final_reward

        # Convert dictionaries to pandas Series
        self.net_worth_series = pd.Series(net_worth_dict)
        self.cash_series = pd.Series(cash_dict)
        self.portfolio_value_series = pd.Series(portfolio_value_dict)
        self.reward_series = pd.Series(reward_dict)

    def plot_results(self):
        """Plots the results of the simulation."""
        from visualizations.plot_results import plot_net_worth, plot_cash, plot_portfolio_value, plot_rewards

        plot_net_worth(self.net_worth_series)
        plot_cash(self.cash_series)
        plot_portfolio_value(self.portfolio_value_series)
        plot_rewards(self.reward_series)
