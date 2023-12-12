from sqlalchemy import create_engine
import os
import pandas as pd
import psycopg2
import yaml


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
        The type of database.
    db_api : str
        The API that is being used to connect to the database.
    engine : psycopg2.Engine
        An Engine created from data in `credentials.yaml` to connect to the AWS database.
    """
    def __init__(self, database_type: str, database_api: str = "psycopg2"):
        """
        Constructor method for the class.

        Parameters
        ----------
        db_type : str
            The type of database.
        db_api : str
            The API that is being used to connect to the database.
        """
        credentials = self._get_yaml_credentials()

        self.database_type: str = database_type
        self.database_api: str = database_api

        # Attributes set to data extracted from credentials dict.
        # There's no need to access them outside of this method, so they're mangled.
        self.__user: str = credentials["RDS_USER"]  
        self.__password: str = credentials["RDS_PASSWORD"]
        self.__host: str = credentials["RDS_HOST"]
        self.__port: str = credentials["RDS_PORT"]
        self.__database: str = credentials["RDS_DATABASE"]

        self.engine: Engine = create_engine(f"{self.database_type}+{self.database_api}://{self.__user}:{self.__password}@{self.__host}:{self.__port}/{self.__database}")

    def _get_yaml_credentials(self) -> dict:
        """
        Gets credentials from a YAML file and converts to a Python readable dict.

        Parameters
        ----------
        filename : str
            The name of the file which contains all of the credentials to access an AWS database.

        Returns
        -------
        credentials_dict : dict
            A dictionary containing the credentials found in the YAML file.
        """
        filepath = os.path.join("..", "credentials.yaml")
        with open(filepath, "r") as file:
            credentials_dict = yaml.safe_load(file)
        return credentials_dict


    def query(self, SQL_query: str) -> pd.DataFrame:
        """
        Runs a query on connected database.

        Parameters
        ----------
        query : str
            The SQL query you wish to run.

        Returns
        -------
        df : pd.DataFrame
            A Pandas DataFrame which can be manipulated in Python.
        """
        with self.engine.connect() as con:
            df = pd.read_sql(SQL_query, con)
        return df

    def create_csv(self, SQL_query: str) -> None:
        """
        Creates a .csv file from the result of an SQL query.
        
        Parameters
        ----------
        query : str
            The SQL query you wish to generate a .csv file from.
        """
        data_dir = os.path.join("..", "data")
        filename = "loans.csv"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filepath = os.path.join(data_dir, filename)

        counter = 0
        while os.path.exists(filepath): # Iterates until there is a new filename.
            counter += 1
            filename = "loans" + "(" + str(counter) + ").csv"
            filepath = os.path.join(data_dir, filename)
        self.query(SQL_query).to_csv(filepath, index=False)


def df_from_csv(filename="loans.csv") -> pd.DataFrame:
    """
    Converts data from a .csv file into a Pandas DataFrame.

    Parameters
    ----------
    filename : str
        The name of the .csv file you wish to convert to a DataFrame.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the converted data from the .csv file.    
    """
    data_dir = os.path.join("..", "data")
    filepath = os.path.join(data_dir, filename)

    with open(filepath, "r") as file:
        df = pd.read_csv(file)
    return df
        

if __name__ == "__main__":
    # The commented code below gets the data from the AWS database and converts it into a .csv file.

    # db = RDSDatabaseConnector("postgresql") # BUG make sure this works
    # db.query("SELECT * FROM loan_payments;")
    # db.create_csv("SELECT * FROM loan_payments;")

    df = df_from_csv()
    display(df)