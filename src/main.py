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
    """This class is used to convert data into formats that are more 
    suitable for analysis.
    """
    def remove_excess(self, df:pd.DataFrame) -> pd.DataFrame:
        """Cleans data within columns that require specific cleaning.

        Parameters
        ----------
        df `pd.DataFrame` - 
            The dataframe to be cleaned.

        Returns
        -------
        `pd.DataFrame` -
            The cleaned dataframe.
        """
        df["sub_grade"] = df["sub_grade"].apply(lambda x: x[1:]).astype("Int64")  # Removes repeated data from "grade" in "sub_grade".
        df["term"] = df["term"].str.replace(" months", "").astype("Int64")  # Removes characters aside from numbers in "term".
        df["verification_status"] = df["verification_status"] != "Not Verified"  # Simplifies "verification_status" to boolean.
        return df

    def convert_column_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts columns to the appropriate data type by calling 
        other functions.

        Parameters
        ----------
        df `pd.DataFrame` - 
            The dataframe with columns to be converted.
        """
        df = self.convert_to_bool(df)
        df = self.convert_to_category(df)
        df = self.convert_to_datetime(df)
        df = self.convert_to_int(df)
        return df

    def convert_to_int(self, df:pd.DataFrame) -> pd.DataFrame:
        """ Converts columns that had whole float values to integers.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe with columns to be converted.

        Returns
        -------
        `pd.DataFrame` -
            The dataframe with columns converted to integers.
        """
        for col in df:
            if df[col].dtype in ["float64", "int64"]:
                if (df[col].fillna(0) % 1 == 0).all():
                    df[col] = df[col].astype("Int64")
        return df

    def convert_to_category(self, df:pd.DataFrame) -> pd.DataFrame:
        """ Converts columns that have <20 values to categories.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe with columns to be converted.

        Returns
        -------
        `pd.DataFrame` -
            The dataframe with columns converted to categories.
        """
        for col in df:
            if df[col].nunique() < 20 and df[col].dtype not in ["bool", "Int64","float64", "datetime64[ns]"]:
                df[col] = df[col].astype("category")
        return df

    def convert_to_datetime(self, df:pd.DataFrame) -> pd.DataFrame:
        """ Converts columns that have dates to datetime64[ns].

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe with columns to be converted.

        Returns
        -------
        `pd.DataFrame` -
            The dataframe with columns converted to datetime64[ns].
        """
        for col in df:
            try:
                df[col] = pd.to_datetime(df[col], format="%b-%Y")
            except TypeError:
                pass
            except ValueError:
                pass
        return df

    def convert_to_bool(self, df:pd.DataFrame) -> pd.DataFrame:  # NOTE Could change this in the future, but not necessary right now.
        """ Converts columns that have up to 2 values to boolean.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe with columns to be converted.
        
        Returns
        -------
        `pd.DataFrame` -
            The dataframe with columns converted to boolean.
        """
        df["payment_plan"] = df["payment_plan"] == "y"
        df["policy_code"] = df["policy_code"] == 1
        return df

# === Step 2. Get Information ===

class DataFrameInfo:
    """This class is used to get information about the dataframe.
    """
    def get_shape(self, df:pd.DataFrame) -> tuple:
        """Returns the shape of the dataframe.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe to get the shape of.

        Returns
        -------
        `tuple` -
            The shape of the dataframe.
        """
        return df.shape

    def get_datatype(self, df:pd.DataFrame, column:str) -> str:
        """Returns the data type of the column in the dataframe.

        Parameters
        ----------
        df `pd.DataFrame` -
             The dataframe to get the data type of the column from.
        
        column `str` -
            The column to get the data type of.

        Returns
        -------
        `str` -
            The data type of the column in the dataframe.
        """
        return df[column].dtype

    def get_datatypes(self, df:pd.DataFrame) -> pd.Series:
        """Returns the data types of the columns in the dataframe.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe to get the data types of the columns from.

        Returns
        -------
        `pd.Series` -
            The data types of the columns in the dataframe.
        """
        return df.dtypes

    def get_median(self, df:pd.DataFrame, column:str) -> int | float | datetime64[ns] | None:
        """Returns the median of the column in the dataframe.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe to get the median of the column from.

        column `str` -
            The column to get the median of.
        
        Returns
        -------
        `int` | `float` | `datetime64[ns]` | `None` - 
            The median value of the column in the dataframe.
        """
        if df[column].dtype not in ["category"]:
            return df[column].median()
        else:
            return None

    def get_medians(self, df:pd.DataFrame):
        """Prints the medians of all columns in the dataframe.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe to get the medians of the columns from.
        """
        output = f"=== Medians ===\n"
        for col in df:
            output += f"{col}: {get_median(df, col)}\n"
        print(output[:-1])

    def get_mean(self, df:pd.DataFrame, column:str, sig_figures:int=2) -> int | float | datetime64[ns] | bool | None:
        """Returns the mean of the column in the dataframe.

        Parameters
        ----------
        df `pd.DataFrame` -
            The dataframe to get the mean of the column from.

        column `str` -
            The column to get the mean of.
        
        sig_figures `int` -
            The number of significant figures to round to.
        
        Returns
        -------
        `int` | `float` | `datetime64[ns]` | `bool` | `None` -
            The mean value of the column in the dataframe.
        """
        if df[column].dtype not in ["category", "datetime64[ns]", "bool"]:
            return round(df[column].mean(), sig_figures)
        elif df[column].dtype != "category":
            df[column].mean()
        else:
            return None

    def get_means(self, df:pd.DataFrame):
        output = f"=== Means ===\n"
        for col in df:
            output += f"{col}: {self.get_mean(df, col)}\n"
        print(output[:-1])
    
    def get_mode(self, df:pd.DataFrame, column:str, max_values:int=5):
            return list(df[column].mode().head(max_values))

    def get_modes(self, df:pd.DataFrame):
        output = f"=== Modes ===\n"
        for col in df:
            output += f"{col}: {self.get_mode(df, col)}\n"
        print(output[:-1])

    def get_standard_deviation(self, df:pd.DataFrame, column:str, sig_figures:int=2):
        if df[column].dtype not in ["category", "datetime64[ns]", "bool"]:
            return round(df[column].std(), sig_figures)
        elif df[column].dtype != "category":
            df[column].std()
        else:
            return None

    def get_standard_deviations(self, df:pd.DataFrame):
        output = f"=== Standard Deviations ===\n"
        for col in df:
            if df[column].dtype == "category":
                output += f"{col}: {self.get_standard_deviation(df, col)}\n"
        return output[:-1]

    def get_distinct(self, df:pd.DataFrame, column:str):
        return df[column].nunique()

    def get_distincts(self, df:pd.DataFrame):
        output = f"=== Unique Values ===\n"
        for col in df:
            output += f"{col}: {self.get_distinct(df, col)} {list(df[col].unique())}\n"
        print(output[:-1])

    

    def get_null(self, df:pd.DataFrame, column:str, sig_figures:int=2):
        return (sum(self.df[column].isna()), 
            round(sum(self.df[column].isna()) * 100 / self.get_shape()[0], sig_figures))

    def get_nulls(self, column=None, sig_figures=2):
        output = f"=== Null Values ===\n"
        for col in self.df:
            if sum(self.df[col].isna()) != 0:
                output += f"{col}: {sum(self.df[col].isna())} ({round(sum(self.df[col].isna()) * 100 / self.get_shape()[0], sig_figures)}%)\n"
        return output[:-1]
        else:
            

    def get_skew(self, column=None, sig_figures=2):
        if column is None:
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
    """
    def __init__(self):
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

# 1. Convert column format.
df = DataTransform().remove_excess(df)
df = DataTransform().convert_column_formats(df)

# ===
