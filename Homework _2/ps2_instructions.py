# -*- coding: utf-8 -*-
"""PS2-TradingSimulationRL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rc12R07Mo5Ra1ZgteRnl3X1_ftwOgF5M

Building the following trading simulation.

You can trade three stocks: Apple, Coca-Cola, and IBM.

Your portfolio is rebalanced every week and held for a week.

At the beginning of each week, you make a decision to adjust your portfolio holding based on the following state variables: last week's return on these 3 stocks and your current holding on these three stocks.

Your simulation (episode) starts in the beginning of 2010 with an initial cash holding of 100

Your simulation (episode) terminates if the following conditions are met: your current cash holding + your current portfolio value <= 0 dollar. OR you have reached the end of the year 2017.

Your actions are to long 1 share, short 1 share, stay neutral for each stock -- your actions are three dimensional vectors.

At the end of each period, determine whether your simulation has ended by checking the date and the sum of your cash holding and portfolio value. The portofolio value is the sum of the value of the three stocks in your portfolio based on how many shares you hold.

When you commit an action, the immediate reward is the gain of the shares sold (shorted) minus the cost of the number of shares bought. We ignore the transaction cost here. When you reach the end, you receive an additional reward that is equal to the value of your portfolio at the end.

First build this experiment, and then plot the trajectory of your networth: cash+portfolio value over time under a random policy (randomly choosing actions at each state). Use a random state of 123. Assume no discounting. Use the data from Yahoo Finance.
"""



"""Implement a Q-learning algorithm. Start with a random policy and update the policy every 2 weeks. Tune the hyperparameters of the model you choose to estimate Q -- the objective is to maximize the total discounted reward. Do not use any data in 2018 (using any data in 2018 in this stage would disqualify this assignment). Report the total reward and also report the cash holding and portfolio value separately."""



"""Freeze this policy. Assume you start with no stock holdings and 100 dollars in the beginning of 2018. Deploy this policy for the next month and next year. Report the total reward the policy achieves in the next month and next year, separately.

The two rewards will be ranked separately, and the ranking of the average ranking will be your overall ranking for this assignment. The tiebreaker is the total reward in the next month.
"""

