import pandas as pd
import numpy as np
import db_utils

# DONE : Task 1 - Create a DataTransfom class with methods that can be applied to DataFrame columns.
class DataTransform:
    """
    A class that transforms the data from a DataFrame object into data that is more easily readable.

    Arguments
    ---------
    dataframe : pd.DataFrame
        The dataframe you wish to transform.
    """ 
    def __init__(self, dataframe: pd.DataFrame):
        """
        Constructor method which transforms the dataframe.

        Parameters:
            dataframe -- dataframe
        """
        self.df = dataframe

        for col in self.df.columns:
            # Converts columns with dates in them into datetime64[ns] datatypes.
            if col in ["issue_date", "earliest_credit_line", "last_payment_date", "next_payment_date", "last_credit_pull_date"]:
                self.df[col] = pd.to_datetime(self.df[col], format="mixed")

            # Converts payment_plan's datatype into a boolean value since entries in column are either "y" or "n". 
            if col == "payment_plan":
                self.df[col] = self.df[col] == "y"

            # Removes redundant data from sub_grade.
            if col == "sub_grade":
                self.df[col] = self.df[col].map(lambda x: x[1:]).astype("Int64")

            # Turns columns with unnecessary float64 type into Int64 (accepts NaN values).
            if col in ["funded_amount", "mths_since_last_delinq", "mths_since_last_record", "collections_12_mths_ex_med", "mths_since_last_major_derog"]:
                self.df[col] = self.df[col].astype("Int64")

            # Converts all previously int64 columns into Int64 (notice capital "I") to remain uniform with the other columns.
            if self.df[col].dtype == "int64":
                self.df[col] = self.df[col].astype("Int64")

            # Converts all columns with text and fewer than 25 unique options into categories.       
            if self.df[col].nunique() < 25 and self.df[col].dtypes not in ["Int64", "float64", "bool"]:
                self.df[col] = self.df[col].astype("category")


class DataFrameInfo:
    def __init__(self, dataframe) -> None:
        self.df = dataframe

    def __str__(self) -> str:
        return f"shape:  {self.df.shape}\n\
cols:   {', '.join(np.array(pd.Categorical(self.df.columns)))}"

    def get_datatype(self, column:str) -> str:
        if self.df[column].dtype == "category":
            return f"{self.df[column].dtype} : {', '.join(np.array(pd.Categorical(self.df[column])))}"
        else:
            return f"{self.df[column].dtype}"
    
    def get_mean(self, column:str, significant_figures:int=2) -> float | None:
        if self.df[column].dtype in ["Int64", "float64"]:
            return round(self.df[column].mean(), significant_figures)
        elif self.df[column].dtype == "datetime64[ns]":
            return self.df[column].mean()
        else: return None

    def get_standard_deviation(self, column:str, significant_figures:int=2) -> float | None:
        if self.df[column].dtype in ["Int64", "float64"]:
            return round(self.df[column].std(), significant_figures)
        elif self.df[column].dtype == "datetime64[ns]":
            return self.df[column].std()
        else: return None

    def get_median(self, column:str) -> str | None:
        if self.df[column].dtype in ["Int64", "float64", "datetime64[ns]"]:
            return str(self.df[column].median())
        else: return None

    def get_mode(self, column:str) -> str:
        return str(self.df[column].mode().values[0])
    
    def get_min(self, column:str) -> str:
        return self.df[column].min()

    def get_max(self, column:str) -> str:
        return self.df[column].max()

    def get_unique(self, column:str) -> int:
        return self.df[column].nunique()

    def get_nulls(self, column:str) -> int:
        nulls = self.df[column].isna().sum()
        return nulls, (round(nulls * 100 / len(self.df), 2))


    def get_details(self, column:str) -> str:
        return f"name:   {column}\n\
dtype:  {self.get_datatype(column)}\n\
st.dev: {self.get_standard_deviation(column)}\n\
mean:   {self.get_mean(column)}\n\
median: {self.get_median(column)}\n\
mode:   {self.get_mode(column)}\n\
min:    {self.get_min(column)}\n\
max:    {self.get_max(column)}\n\
unique: {self.get_unique(column)}\n\
nulls:  {self.get_nulls(column)[0]} ({self.get_nulls(column)[1]}%)"
    


if __name__ == "__main__":  
    pd.set_option("display.max_columns", None)
    transformed = DataTransform(db_utils.df_from_csv("data.csv"))
    df_info = DataFrameInfo(transformed.df)

    column = "mths_since_last_major_derog"

    print(df_info, "\n===")
    print(df_info.get_details(column))