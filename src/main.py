from scipy.stats import skew, boxcox, yeojohnson

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import math
import db_utils

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

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
        self.convert_to_datetime()
        self.convert_to_bool()
        self.convert_to_category()
        self.convert_to_int()

    def remove_excess(self):
        """
        Cleans data within columns that require specific cleaning.
        """
        self.df["sub_grade"] = self.df["sub_grade"].apply(lambda x: x[1:])
        self.df["term"] = self.df["term"].str.replace(" months", "").astype("Int64")
        self.df["verification_status"] = self.df["verification_status"] != "Not Verified"

    def convert_to_int(self, dataframe=None):
        """
        Converts columns that had unnecessary float values to integers.
        """
        if dataframe is None:
            for col in ["sub_grade", "loan_amount", "funded_amount", "mths_since_last_delinq", "mths_since_last_record", "collections_12_mths_ex_med", "mths_since_last_major_derog"]:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")
                return self.df
        else:
            for col in ["sub_grade", "loan_amount", "funded_amount", "mths_since_last_delinq", "mths_since_last_record", "collections_12_mths_ex_med", "mths_since_last_major_derog"]:
                dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce").astype("Int64")
                return dataframe

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
        self.df["payment_plan"] = self.df["payment_plan"] == "y"
        self.df["policy_code"] = self.df["policy_code"] == 1


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

    def get_mean(self, column=None, sig_figures=2):
        if column is None:
            output = f"=== Mean ===\n"
            for col in self.df:
                output += f"{col}: {self.get_mean(col)}\n"
            return output[:-1]
        elif self.df[column].dtype not in ["category", "datetime64[ns]", "bool"]:
            return round(self.df[column].mean(), sig_figures)
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

    def get_standard_deviation(self, column=None, sig_figures=2):
        if column is None:
            output = f"=== Standard Deviation ===\n"
            for col in self.df:
                if self.df[column].dtype == "category":
                    output += f"{col}: {self.get_standard_deviation(col)}\n"
            return output[:-1]
        elif self.df[column].dtype not in ["category", "datetime64[ns]", "bool"]:
            return round(self.df[column].std(), sig_figures)
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
                if sum(self.df[col].isna()) != 0:
                    output += f"{col}: {sum(self.df[col].isna())} ({round(sum(self.df[col].isna()) * 100 / self.get_shape()[0], sig_figures)}%)\n"
            return output[:-1]
        else:
            return f"{sum(self.df[column].isna())} ({round(sum(self.df[column].isna()) * 100 / self.get_shape()[0], sig_figures)}%)"

    def get_skew(self, column=None, sig_figures=2):
        if column is None:
            output = f"=== Skew ===\n"
            for col in self.df:
                if self.df[col].dtype not in ["category", "datetime64[ns]", "bool"]:
                    output += f"[{col}] : {round(skew(self.df[col]), sig_figures)}\n"
            return output[:-1]
        else:
            return f"[{column}] : {round(skew(self.df[column]), sig_figures)}"

    def get_description(self):
        return self.df.describe()
    
    def get_info(self):
        return self.df.info()


# === Step 3. Removing and Imputing Missing Values ===

class DataFrameTransform:
    """Initializes DataFrameTransform with a dataframe.
    
    Attributes
    ----------
    dataframe : (pandas.DataFrame)
        The dataframe to transform.
    """
    def __init__(self, dataframe):
        self.df = dataframe
        self.original_df = dataframe.copy()

        # Columns with more than 50% null values get dropped.
        self.drop_columns(["mths_since_last_delinq", "mths_since_last_record", "next_payment_date", "mths_since_last_major_derog"])

        # Columns that have null values that are likely should be zeros are filled with zeros.
        self.fill_zero_columns(["funded_amount"])
        
        self.impute_median_columns(["last_payment_date", "last_credit_pull_date"])
        self.impute_mean_columns(["collections_12_mths_ex_med"])

        self.drop_rows(["term", "int_rate", "employment_length"])

        self.fix_skews()

    def drop_columns(self, columns:list):
        """Drops specified columns from the dataframe.

        Parameters
        ----------
        columns : (list)
            The column names to drop.
        """
        for col in columns:
            self.df = self.df.drop(col, axis=1)

    def fill_zero_columns(self, columns:list):
        for col in columns:
            self.df[col] = self.df[col].fillna(0)

    def impute_median_columns(self, columns:list):
        for col in columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())

    def impute_mean_columns(self, columns:list, sig_figures=2):
        for col in columns:
            self.df[col] = self.df[col].fillna(round(self.df[col].mean(), 2))

    def drop_rows(self, columns:list):
        for col in columns:
            self.df = self.df[~self.df[col].isna()]

    def fix_skews(self):
        """Fixes skewed columns by applyin Log, Box-Cox or Yeo-Johnson transformation.
        """
        for col in self.df:
            if self.df[col].dtype not in ["category", "datetime64[ns]", "bool"]:
                if -0.5 < skew(self.df[col].astype("float64")) > 0.5:  # a common skew threshold to filter in the skewed columns
                    log = self.df[col].map(lambda x: np.log(x) if x > 0 else 0)
                    if self.df[col].all() > 0:
                        box = pd.Series(boxcox(self.df[col].astype("float64"))[0])
                    else:
                        box = pd.Series(yeojohnson(self.df[col].astype("float64"))[0])
                    self.df[col] = log if abs(skew(log) < abs(skew(box))) else box  # chooses the closest to 0

    def remove_outliers(self, columns:list):  # TODO add outlier removal
        pass

class Plotter:
    def __init__(self, dataframe):
        self.df = dataframe

    def null(self, comparison_df=None):
        if comparison_df is None:
            fig = plt.figure(figsize=(30, 30))
            msno.bar(self.df, color="green")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(50, 50))
            msno.bar(comparison_df, ax=axes[0], color="darkred")
            msno.bar(self.df, ax=axes[1], color="green")
        plt.show()
        
        
    def distribution(self):  # TODO add comparison to original data
        fig = plt.figure(figsize=(30, 30))
        for col in self.df:
            if self.df[col].dtype not in ["category", "datetime64[ns]", "bool"]:
                sns.displot(self.df[col], color="green")
                plt.show()

        

df = db_utils.df_from_csv()
dt = DataTransform(df)
dft = DataFrameTransform(dt.df)

dfi = DataFrameInfo(df)
dfti = DataFrameInfo(dft.df)

plot_df = Plotter(dft.df)
plot_df.distribution()

# ===
