# Loan Data Analysis

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)  
- [File Structure](#file-structure)
- [License](#license)

## Description

This project performs exploratory data analysis on loan data to uncover insights that can improve loan approval decisions, pricing, and risk management. Statistical analysis and data visualization techniques are used. 

The goal is to gain a deeper understanding of the risks and returns associated with the loan portfolio to enhance overall performance and profitability.

## Installation

The following packages need to be installed:
- pandas
- numpy
- matplotlib
- seaborn
- sqlalchemy
- psycopg2
- yaml

Run `pip install -r requirements.txt` to install all required packages.

## Usage

The main scripts are:

- `db_utils.py` - connects to the AWS RDS database and executes SQL queries
- `data_processing.py` - cleans and transforms the raw data
- `main.py` - performs exploratory data analysis and visualization 

Run `python eda.py` to generate analysis and plots from the data.

## File Structure
    
    ├── data/ 
    │   └── loans.csv  
    ├── images/
    │   └── plots/
    ├── src/
    │   ├── db_utils.py
    │   ├── data_processing.py 
    │   └── main.py
    ├── credentials.yaml
    └── README.md

## License

[MIT](https://choosealicense.com/licenses/mit/)