This file gives an overview of each file and how to run them.

To run the pset, all that is needed is an empty folder `data` in the directory to store the pulled data as well as the need to run each file sequentially: `1_get_data.ipynb` $\to$ `2_construct_fine_tune_dataset.ipynb` $\to$ `3_gics_strategy.ipynb`

### Notebook 1: `1_get_data.ipynb`: 
1. Gets necessary data from CRSP for S&P constituents
2. Transforms PDFs of FOMC statements to URLs (so that all FOMCs are now in a url)
3. Scrapes necessary text from all URLs
4. Removes noise from FOMC statements


### Notebook 2: `2_construct_fine_tune_dataset.ipynb`: 
1. Creates a labeled dataset according to the format needed to fine tune a model
2. Checks efficacy of strategy (i.e. what the performance is given the correct label was forecasted)

### Notebook 3: `3_gics_strategy.ipynb`: 
1. Fine tunes the model
2. Construc portfolio
3. Assess Performance

### `annualized_returns_test.csv`: 
- contains all annualized returns in the testing set (each statement is one row).