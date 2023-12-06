from scipy.stats import skew, boxcox, yeojohnson
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import math
import db_utils

pd.set_option("display.max_columns", None)

class DataTransform:
    """This class is used to convert data into formats that are more 
    suitable for analysis.
    """
    def remove_excess(self, df:pd.DataFrame) -> pd.DataFrame:  # NOTE this is hardcoded, might be a better way.
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
        df = self.convert_to_boolean(df)
        df = self.convert_to_datetime(df)
        df = self.convert_to_integer(df)
        df = self.convert_to_category(df)
        return df

    def convert_to_integer(self, df:pd.DataFrame) -> pd.DataFrame:
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

    def convert_to_boolean(self, df:pd.DataFrame) -> pd.DataFrame:  # NOTE Could change this in the future, but not necessary right now.
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


class DataFrameInfo:
    """This class is used to get information about the dataframe.
    """
    def get_description(self, df:pd.DataFrame, sig_figures:int=2):
        return round(df.describe(), sig_figures)
    
    def get_info(self, df:pd.DataFrame):
        return df.info()

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
        return str(df[column].dtype)

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

    def get_median(self, df:pd.DataFrame, column:str) -> int | float | datetime | None:
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
            output += f"{col}: {self.get_median(df, col)}\n"
        print(output[:-1])

    def get_mean(self, df:pd.DataFrame, column:str, sig_figures:int=2) -> int | float | datetime | bool | None:
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
            if df[col].dtype == "category":
                output += f"{col}: {self.get_standard_deviation(df, col)}\n"
        return output[:-1]

    def get_distinct(self, df:pd.DataFrame, column:str):
        return df[column].nunique()

    def get_distincts(self, df:pd.DataFrame):
        output = f"=== Unique Values ===\n"
        for col in df:
            if df[col].dtype == "category":
                output += f"{col}: {self.get_distinct(df, col)} {list(df[col].unique())}\n"
        print(output[:-1])

    def get_null(self, df:pd.DataFrame, column:str, sig_figures:int=2) -> tuple[int, int|float]:
        return (sum(df[column].isna()), 
            round(sum(df[column].isna()) * 100 / self.get_shape(df)[0], sig_figures))

    def get_nulls(self, df:pd.DataFrame, sig_figures:int=2):
        output = f"=== Null Values ===\n" 
        for col in df:
            if sum(df[col].isna()) != 0:
                null_count, null_percent = self.get_null(df, col)
                output += f"{col}: {null_count} ({null_percent}%)\n"
        print(output[:-1])

    def get_skew(self, df:pd.DataFrame, column:str, sig_figures:int=2) -> int | float:
        try:
            return round(skew(df[column].fillna(0)), sig_figures)  # FIXME Needs to convert column to int64 to work and then back to Int64
        except ValueError:
            return None

    def get_skews(self, df:pd.DataFrame, sig_figures:int=2):
        output = f"=== Skews ===\n"
        for col in df:
            if df[col].dtype in ["Int64", "float64"]:
                output += f"[{col}] : {self.get_skew(df, col)}\n"
        print(output[:-1])


class DataFrameTransform:
    """Initializes DataFrameTransform with a dataframe.
    """
    def drop_columns(self, df:pd.DataFrame, threshold:float=0.5):
        """Drops columns from the dataframe with more nulls than the threshold percentage.
        """
        for col in df:
            if DataFrameInfo().get_null(df, col)[1] >= threshold:
                df = df.drop(col, axis=1)
        return df

    def fill_zero_columns(self, df:pd.DataFrame, columns:list):
        for col in columns:
            df[col] = df[col].fillna(0)
        return df

    def impute_median_columns(self, df:pd.DataFrame, columns:list):
        for col in columns:
            df[col] = df[col].fillna(df[col].median())
        return df

    def impute_mean_columns(self, df:pd.DataFrame, columns:list, sig_figures=2):
        for col in columns:
            df[col] = df[col].fillna(round(df[col].mean(), 2))
        return df

    def drop_rows(self, df:pd.DataFrame, columns:list):
        for col in columns:
            df = df[~df[col].isna()]
        return df

    def fix_skews(self, df:pd.DataFrame):
        """Fixes skewed columns by applyin Log, Box-Cox or Yeo-Johnson transformation.
        """
        for col in df:
            if df[col].dtype not in ["category", "datetime64[ns]", "bool"]:
                if -0.5 < skew(df[col].astype("float64")) > 0.5:  # a common skew threshold to filter in the skewed columns
                    log = df[col].map(lambda x: np.log(x) if x > 0 else 0)
                    if df[col].all() > 0:
                        box = pd.Series(boxcox(df[col].astype("float64"))[0])
                    else:
                        box = pd.Series(yeojohnson(df[col].astype("float64"))[0])
                    df[col] = log if abs(skew(log) < abs(skew(box))) else box  # chooses the closest to 0
        return df

    def remove_outliers_zscore(self, df:pd.DataFrame):
        #Z-Score Method
        new_df = df.copy()
        for col in df:
            if df[col].dtype not in ["category", "bool"]:
                upper_limit = df[col].mean() + 3*df[col].std()
                lower_limit = df[col].mean() - 3*df[col].std()
                new_df = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit)]
                
        print(f"{len(df)} -> {len(new_df)}\nOutliers: {len(df)-len(new_df)}")
        return new_df

    def remove_outliers_iqr(self, df:pd.DataFrame):
        #Interquartile Range Method
        new_df = df.copy
        for col in df:
            if df[col].dtype not in ["category", "bool"]:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3-q1
                upper_limit = q3 + (1.5*iqr)
                lower_limit = q1 - (1.5*iqr)
                print(f"{col}: ({lower_limit} - {upper_limit})")
                new_df = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit)]

        print(f"{len(df)} -> {len(new_df)}\nOutliers: {len(df)-len(new_df)}")
        return new_df

    def remove_outliers_percentile(self, df:pd.DataFrame, low:float=0.01, high:float=0.99):
        new_df = df.copy()
        for col in df:
            if df[col].dtype not in ["category", "bool"]:
                lower_limit = df[col].quantile(low)
                upper_limit = df[col].quantile(high)
                print(f"{col}: ({lower_limit} - {upper_limit})")
                new_df = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit)]

        print(f"{len(df)} -> {len(new_df)}\nOutliers: {len(df)-len(new_df)}")
        return new_df



class Plotter:
    def null(self, df:pd.DataFrame, comparison_df:pd.DataFrame=None):
        if comparison_df is None:
            fig = plt.figure(figsize=(30, 30))
            msno.bar(df, color="green")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(50, 50))
            msno.bar(comparison_df, ax=axes[0], color="darkred")
            msno.bar(df, ax=axes[1], color="green")
        plt.show()
        
    def distribution(self, df:pd.DataFrame):  # TODO add comparison to original data
        fig = plt.figure(figsize=(30, 30))
        for col in df:
            if df[col].dtype not in ["category", "datetime64[ns]", "bool"]:
                sns.displot(df[col].astype(int), color="green", kde=True)
                sns.despine()
                plt.title(col)
                plt.show()

    def box(self, df:pd.DataFrame):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=column)
        plt.title(column)
        plt.show()
    
if __name__ == "__main__":
    df = db_utils.df_from_csv()

    # 1. Convert column format
    df = DataTransform().remove_excess(df)
    df = DataTransform().convert_column_formats(df)

    # 2. Data information
    df_info = DataFrameInfo()

    # df_info.get_info(df)
    # df_info.get_description(df)  
    # df_info.get_datatypes(df)
    # df_info.get_shape(df)

    # df_info.get_distincts(df)
    # df_info.get_means(df)
    # df_info.get_medians(df) 
    # df_info.get_modes(df)
    # df_info.get_nulls(df)
    # df_info.get_standard_deviations(df)

    # 3. Remove / impute data
    df_transform = DataFrameTransform()
    
    df = df_transform.fill_zero_columns(df, ["funded_amount"])
    df = df_transform.impute_mean_columns(df, ["collections_12_mths_ex_med"])
    df = df_transform.impute_median_columns(df, ["last_payment_date", "last_credit_pull_date"])
    df = df_transform.drop_rows(df, ["term", "int_rate", "employment_length"])
    df = df_transform.drop_columns(df)
    
    # 4. Transform skewed data 
    # TODO actually fix skews
    plot = Plotter()
    original_df = df.copy()

    # plot.null(df)
    

    # df = df_transform.fix_skews(df)
    # plot.null(df)
    

    # 5. Dealing with outliers
    # plot.distribution(df)
    df = df_transform.remove_outliers_iqr(df)

    

