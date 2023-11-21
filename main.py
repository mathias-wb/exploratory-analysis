import pandas as pd
import numpy as np
import db_utils

# FIXME : M3. Task 1 : Create a DataTransfom class with methods that can be applied to DataFrame columns.
class DataTransform:  
    def __init__(self, dataframe: pd.DataFrame):
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
            if self.df[col].nunique() < 25 and self.df[col].dtypes not in []:
                self.df[col] = self.df[col].astype("category")
            
       
        
pd.set_option("display.max_columns", None)

transformed = DataTransform(db_utils.df_from_csv("data.csv"))
transformed.df.info()