# Loan Data Analysis

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)  
- [File Structure](#file-structure)
- [Charts](#charts)

## Description

This project performs exploratory data analysis on loan data to uncover insights that can improve loan approval decisions, pricing, and risk management. Statistical analysis and data visualization techniques are used. 

The goal is to gain a deeper understanding of the risks and returns associated with the loan portfolio to enhance overall performance and profitability.

## Installation

The following packages need to be installed:
- pandas
- numpy
- matplotlib
- missingno
- scikit learn
- seaborn
- sqlalchemy
- psycopg2
- pyyaml

Run `pip install -r requirements.txt` to install all required packages.

## Usage

The main scripts are:

- `db_utils.py` - connects to the AWS RDS database and queries the database
- `cleaning.py` - cleans and transforms the raw data
- `analysis.py` - performs exploratory data analysis and visualization 

Run `python analysis.py` to generate analysis and plots from the data.

## File Structure
    
    ├── data/ 
    │   └── loans.csv  
    ├── plots/
    │   ├── Chart.png
    │   └── Another Chart.png ...
    ├── src/
    │   ├── db_utils.py
    │   ├── cleaning.py 
    │   └── analysis.py
    ├── credentials.yaml
    ├── requirements.txt
    ├── LICENSE.txt
    └── README.md

## Charts
### Null Values Before and After Handling
![Null Values](https://github.com/mathias-wb/exploratory-analysis/blob/master/plots/Null%20Values%20(Before%20and%20After%20Handling).png)
### Correlation Heatmap
![Correlation](https://github.com/mathias-wb/exploratory-analysis/blob/master/plots/Correlation.png)
### Percentage of Loans Recovered
![Percentage Loans Recovered](https://github.com/mathias-wb/exploratory-analysis/blob/master/plots/Percentage%20of%20Loans%20Recovered.png)
