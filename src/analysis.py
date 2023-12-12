from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import db_utils, cleaning

# 1. Summarise what percentage of the loans are recovered against the 
# investor funding and the total amount funded. 
# Visualise your results on an appropriate graph.
def percentage_of_loans_recovered(df: pd.DataFrame):
        total_funded = df["funded_amount"].sum()
        total_recovered = df["total_payment"].sum()
        total_funded_inv = df["funded_amount_inv"].sum()
        total_recovered_inv = df["total_payment_inv"].sum()

        percent_recovered_total = (total_recovered / total_funded) * 100
        percent_recovered_inv = (total_recovered_inv / total_funded_inv) * 100
        
        data = {
            "Category": ["Against\nTotal Funding", "Against\nInvestor Funding"],
            "Percentage Recovered": [percent_recovered_total, percent_recovered_inv]
        }
        df_loan_summary = pd.DataFrame(data)

        sns.set_theme(style="whitegrid", font="JetBrains Mono")
        plt.figure(figsize=(5,6))
        ax = sns.barplot(data=df_loan_summary, x="Category", y="Percentage Recovered")
        plt.title("Percentage of Loans Recovered")
        plt.ylabel("Percentage (%)")
        plt.ylim([0,100])
        plt.xlabel(None)
        
        for bar in ax.patches: # placing percentage text on bars
            ax.annotate(f'{round(bar.get_height(), 2)}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va='center', xytext=(0, -20), textcoords="offset points", color="white")
        plt.show()


#1. (cont.) Additionally visualise what percentage of the total amount 
# would be recovered up to 6 months' in the future.
def percentage_of_loans_recovered_in_6_months(df: pd.DataFrame):
    pass


df = cleaning.clean_data(db_utils.df_from_csv("loans.csv"))

percentage_of_loans_recovered(df)
display(df)
