import pandas as pd
import numpy as np
import db_utils

# TODO : Create a DataTransfom class with methods that can be applied to DataFrame columns.
class DataTransform:  
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # loan_amount (amount of loan the applicant received) 
        self.df.loan_amount = self.df.loan_amount.astype(np.float64)

        # term (number of monthly payments for the loan)
        self.df.term = pd.to_numeric(self.df.term.map(lambda x: str(x).rstrip(" months")), errors="coerce")
        self.df.term = self.df.term.fillna(0).astype(np.int64)

        # sub_grade (LC assigned loan sub grade)
        self.df.sub_grade = self.df.sub_grade.map(lambda x: list(x)[1])

        # mths_since_last_delinq
        self.df.mths_since_last_delinq.fillna(0)

        # turn what's left that isn't numbers into categorical data
        for col in df.columns:
            if df[col].nunique() < 25 and df[col].dtypes not in ["int64", "float64"]:
                df[col] = df[col].astype('category')
        

    def drop_NaNs(self):
        self.df = self.df.dropna(how="any")
        

        
        
pd.set_option("display.max_columns", None)

transformed = DataTransform(db_utils.df_from_csv("data.csv"))

transformed.df.info()
display(transformed.df)