import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import db_utils

pd.set_option("display.max_columns", None)
df = db_utils.df_from_csv("data.csv")

# === Step 1. Transforming Data ===

class DataTransform:
    """
    This class is used to transform the data into a format that is more suitable for analysis.
    
    Attributes
    ----------
    df : pandas.DataFrame
        The dataframe to be transformed.
    """
    def __init__(self, dataframe):
        """
        Initializes the DataTransform class.        

        Paramaters
        ----------
        dataframe : pandas.DataFrame
            The dataframe to be transformed.
        """
        self.df = dataframe
        self.remove_excess()
        self.convert_to_int()
        self.convert_to_datetime()
        self.convert_to_bool()
        self.convert_to_category()

    def remove_excess(self):
        """
        Cleans data within columns that require specific cleaning.
        """
        self.df["sub_grade"] = self.df["sub_grade"].apply(lambda x: x[1:])
        self.df["term"] = self.df["term"].str.replace(" months", "") #if " months" in str(x) else x
        #self.df["term"] = self.df["term"].apply(lambda x: str(x).replace(" months", "") if " months" in str(x) else x)

    def convert_to_int(self):
        """
        Converts columns that had unnecessary float values to integers.
        """
        for col in ["term", "sub_grade", "loan_amount", "funded_amount", "mths_since_last_delinq", "mths_since_last_record", "collections_12_mths_ex_med", "mths_since_last_major_derog"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")

    def convert_to_category(self):
        """
        Converts columns that have <20 values to categories.
        """
        for col in self.df:
            if self.df[col].nunique() < 20 and self.df[col].dtype not in ["bool", "int64", "Int64", "float64", "datetime64[ns]"]:
                self.df[col] = self.df[col].astype("category")

    def convert_to_datetime(self):
        for col in ["issue_date","last_payment_date", "next_payment_date", "last_credit_pull_date", "earliest_credit_line"]:
            self.df[col] = pd.to_datetime(self.df[col], format="%b-%Y", errors="coerce")

    def convert_to_bool(self):
        for col in ["payment_plan"]:
            self.df[col] = self.df[col] == "y"


# === Step 2. Get Information ===

class DataFrameInfo:
    def __init__(self, dataframe):
        self.df = dataframe

    def get_datatypes(self):
        return self.df.dtypes

    def get_median(self, column=None):
        if column is None:
            output = f"=== Median ===\n"
            for col in self.df:
                output += f"{col}: {self.get_median(col)}\n"
            return output[:-1]
        elif self.df[column].dtype not in ["category"]:
            return self.df[column].median()
        else:
            return None

    def get_mean(self, column=None, sig_figure=2):
        if column is None:
            output = f"=== Mean ===\n"
            for col in self.df:
                output += f"{col}: {self.get_mean(col)}\n"
            return output[:-1]
        elif self.df[column].dtype not in ["category", "datetime64[ns]", "bool"]:
            return round(self.df[column].mean(), sig_figure)
        elif self.df[column].dtype != "category":
            self.df[column].mean()
        else:
            return None

    def get_mode(self, column=None):
        if column is None:
            output = f"=== Mode ===\n"
            for col in self.df:
                output += f"{col}: {self.get_mode(col)}\n"
            return output[:-1]
        else:
            return list(self.df[column].mode().head(5))

    def get_standard_deviation(self, column=None, sig_figure=2):
        if column is None:
            output = f"=== Standard Deviation ===\n"
            for col in self.df:
                if self.df[column].dtype == "category":
                    output += f"{col}: {self.get_standard_deviation(col)}\n"
            return output[:-1]
        elif self.df[column].dtype not in ["category", "datetime64[ns]", "bool"]:
            return round(self.df[column].std(), sig_figure)
        elif self.df[column].dtype != "category":
            self.df[column].std()
        else:
            return None

    def get_distinct(self, column=None):
        if column is None:
            output = f"=== Unique Values ===\n"
            for col in self.df:
                if self.df[col].dtype == "category":
                    output += f"{col}: {self.df[col].nunique()} {list(self.df[col].unique())}\n"
            return output[:-1]
        elif self.df[column].dtype == "category":
            return self.df[column].nunique()
        else:
            return None

    def get_shape(self):
        return self.df.shape

    def get_nulls(self, column=None, sig_figures=2):
        if column is None:
            output = f"=== Null Values ===\n"
            for col in self.df:
                output += f"{col}: {sum(self.df[col].isna())} ({round(sum(self.df[col].isna()) * 100 / self.get_shape()[0], sig_figures)}%)\n"
            return output[:-1]
        else:
            return f"{sum(self.df[column].isna())} ({round(sum(self.df[column].isna()) * 100 / self.get_shape()[0], sig_figures)}%)"

    def get_description(self):
        return self.df.describe()
    
    def get_info(self):
        return self.df.info()


# === Step 3. Removing and Imputing Missing Values ===

class DataFrameTransform:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def drop_missing(self, column=None):
        if column is None:
            for col in self.df:
                self.drop_missing(col)
        else:
            self.df = self.df.dropna(subset=[column])

    def impute_missing(self, column=None, method="mean"):
        if column is None:
            for col in self.df:
                self.impute_missing(col)
        else:
            if method == "mean":
                self.df[column] = self.df[column].fillna(self.df[column].mean())
            elif method == "median":
                self.df[column] = self.df[column].fillna(self.df[column].median())
            elif method == "mode":
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
            elif method == "drop":
                self.df = self.df.dropna(subset=[column])
            else:
                self.df[column] = self.df[column].fillna(method)


class Plotter:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_line(self, x, y):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df[x], self.df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()


# === A. Transforming Data ===
df = DataTransform(df).df

# === B. Getting Information ===
info = DataFrameInfo(df)

# === C. Removing and Imputing Missing Values ===