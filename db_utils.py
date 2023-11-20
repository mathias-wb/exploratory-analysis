import os
import yaml
from sqlalchemy import create_engine
import pandas as pd

"""
Here I'm going to perform exploratory data analysis on a loan 
portfolio, using various statistical and data visualisation 
techniques to uncover patterns, relationships, and anomalies in 
the loan data.

This information will enable the business to make more informed 
decisions about loan approvals, pricing, and risk management.

By conducting exploratory data analysis on the loan data, 
I'll aim to gain a deeper understanding of the risk and return 
associated with the business' loans.

Ultimately, my goal is to improve the performance and 
profitability of the loan portfolio.
"""

class RDSDatabaseConnector:
    def __init__(self):
        credentials: dict = get_credentials()

        self.database_type: str = "postgresql"
        self.database_api: str = "psycopg2"

        self.user: str = credentials["RDB_USER"]  # Attributes set to 
        self.password: str = credentials["RDB_PASSWORD"]
        self.host: str = credentials["RDB_HOST"]
        self.port : str = credentials["RDB_PORT"]
        self.database: str = credentials["RDB_DATABASE"]
        

    def create_engine(self):
        return create_engine(f"{self.database_type}+{self.database_api}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")



def get_credentials(filename) -> dict:
    """Creates a dictionary from `credentials.yaml` which can be used to access a database.

    Returns
    -------
    `dict`
        A dictionary of credentails that can be used to access a database.
    """
    filepath: str = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "r") as file:
        credentials_dict: dict = yaml.safe_load(file)
    return credentials_dict