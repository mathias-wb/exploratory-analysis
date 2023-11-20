import os
import yaml
import psycopg2
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
    """
    A connection to an AWS database that can be queried.

    Attributes
    ----------
    db_type : str
    > The type of database.
    db_api : str
    > The API that is being used to connect to the database.
    engine : psycopg2.Engine
    > An Engine created from data in `credentials.yaml` to connect to the AWS database.
    """
    def __init__(self, db_type, db_api="psycopg2"):
        """
        Constructor method for the class.

        Parameters
        ----------
        db_type : str
        > The type of database.
        db_api : str
        > The API that is being used to connect to the database.
        """
        credentials: dict = self._get_yaml_credentials("credentials.yaml")

        self.db_type: str = db_type
        self.db_api: str = db_api

        self.__user: str = credentials["RDS_USER"]  # Attributes set to data extracted from credentials dict, 
        # ... no need to access them outside of this method.
        self.__password: str = credentials["RDS_PASSWORD"]
        self.__host: str = credentials["RDS_HOST"]
        self.__port: str = credentials["RDS_PORT"]
        self.__db: str = credentials["RDS_DATABASE"]

        self.engine: Engine = create_engine(f"{self.db_type}+{self.db_api}://{self.__user}:{self.__password}@{self.__host}:{self.__port}/{self.__db}")


    def _get_yaml_credentials(self, filename: str) -> dict:
        """
        Gets credentials from a YAML file and converts to a Python readable dict.

        Parameters
        ----------
        filename : str
        > The name of the file which contains all of the credentials to access an AWS database.

        Returns
        -------
        credentials_dict : dict
        > A dictionary containing the credentials found in the YAML file.
        """
        filepath: str = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, "r") as file:
            credentials_dict: dict = yaml.safe_load(file)
        return credentials_dict


    def query(self, query: str) -> pd.DataFrame:
        """
        Runs a query on connected database.

        Parameters
        ----------
        query : str
        > The SQL query you wish to run.

        Returns
        -------
        df : DataFrame
        > A Pandas DataFrame which can be manipulated in Python.
        """
        with self.engine.connect() as con:
            df: DataFrame = pd.read_sql(query, con)
        return df


    def create_csv(self, df: pd.DataFrame):
        """
        Creates a `data.csv` file from a DataFrame. 
        
        If there is already a `data.csv` file present, it will iterate through variations 
        of the name until it finds one that doesn't exist yet.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame you wish to convert to a .csv file.
        """
        filename = "data.csv"
        counter = 1
        while os.path.exists(filename):
            filename = "data" + "(" + str(counter) + ").csv"
            counter += 1
        df.to_csv(filename, index=False)
        

db = RDSDatabaseConnector("postgresql")
db.create_csv(db.query("SELECT * FROM loan_payments;"))