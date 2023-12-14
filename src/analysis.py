from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import math, db_utils, cleaning

# 1.
def percentage_of_loan_recovery(df: pd.DataFrame) -> None:
    """Calculates and visualizes percentage of loans recovered against total funding and investor funding.

    This function takes a pandas DataFrame containing loan data, calculates recovery percentages against total and investor funding, 
    projects recovery percentage in 6 months, and visualizes the results as a bar chart.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing loan data 
    """
    
    total_funded = df["funded_amount"].sum()
    total_recovered = df["total_payment"].sum()
    total_funded_inv = df["funded_amount_inv"].sum()
    total_recovered_inv = df["total_payment_inv"].sum()

    percent_recovered_total = (total_recovered / total_funded) * 100
    percent_recovered_inv_total = (total_recovered_inv / total_funded_inv) * 100
    
    df["months_left"] = df["out_prncp"] / df["instalment"]
    df["months_left_inv"] = df["out_prncp_inv"] / df["instalment"]
    df["amount_recovered_in_6_mths"] = np.where(df["months_left"] < 6, df["months_left"] * df["instalment"], 6 * df["instalment"])
    df["amount_recovered_in_6_mths_inv"] = np.where(df["months_left_inv"] < 6, df["months_left_inv"] * df["instalment"], 6 * df["instalment"])
    
    total_recovered_in_6_mths = df["amount_recovered_in_6_mths"].sum()
    total_recovered_in_6_mths_inv = df["amount_recovered_in_6_mths_inv"].sum()
    
    percent_total_recovered_in_6_mths = (total_recovered_in_6_mths / total_funded) * 100
    percent_total_recovered_in_6_mths_inv = (total_recovered_in_6_mths_inv / total_funded_inv) * 100

    data = {
        "Category": ["x", "y", "Against\nTotal Funding", "Against\nInvestor Funding"],
        "Percentage Recovered": [percent_recovered_total, percent_recovered_inv_total, percent_total_recovered_in_6_mths, percent_total_recovered_in_6_mths_inv]
    }
    df_loan_summary = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", font="JetBrains Mono")
    plt.figure(figsize=(9,16))

    # Plot the bars for recovered funding
    ax = sns.barplot(data=df_loan_summary.iloc[:2], x="Category", y="Percentage Recovered", label=["Recovered Total Funding", "Recovered Investor Funding"])
    # Plot the stacked bars for projected recoveries
    ax = sns.barplot(data=df_loan_summary.iloc[2:], x="Category", y="Percentage Recovered", bottom=df_loan_summary.iloc[:2]["Percentage Recovered"], color="#76ae55", label="Projected Recovery in 6 Months")
    
    plt.title("Percentage of Loans Recovered")
    plt.ylabel("Percentage (%)")
    plt.xlabel(None)
    
    # Write the percentage of the portion in the centre of the each stacked bar.
    for bar, label in zip(ax.patches, df_loan_summary["Percentage Recovered"]):
        height = bar.get_height() + bar.get_y()
        ax.annotate(f'{round(label, 2)}%', (bar.get_x() + bar.get_width() / 2, (bar.get_y() + height) / 2),
                    ha="center", va='center', xytext=(0, 0), textcoords="offset points", color="white")
    
    plt.legend()
    plt.show()


# 2.
def percentage_loss(df: pd.DataFrame) -> None:
    df["loss"] = df["loan_status"] == "Charged Off"
    percent_charged_off = round(((sum(df["loss"] == True) / df.shape[0]) * 100), 2)
    amount_lost = sum(np.where(df["loss"] == True, df["loan_amount"], 0))
    print(f"{percent_charged_off}% of loans were charged off.\nTotal amount lost: £{amount_lost}")


# 3. 
def projected_loss(df: pd.DataFrame) -> None:
    charged_off_loans = df[df["loan_status"] == "Charged Off"]
    display(charged_off_loans)

    projected_loss = charged_off_loans["funded_amount"].sum() - charged_off_loans["total_payment"].sum()
    projected_loss_inv = charged_off_loans["funded_amount_inv"].sum() - charged_off_loans["total_payment_inv"].sum()
    projected_loss_extra = charged_off_loans["collection_recovery_fee"].sum() - charged_off_loans["recoveries"].sum()
    
    data = {
        "Category": ["Projected\nFunding Loss", "Projected Investor\nFunding Loss", "Other Loss"],
        "Amount Lost": [projected_loss, projected_loss_inv, projected_loss_extra]
    }

    data = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", font="JetBrains Mono")
    
    plt.figure(figsize=(6, 6))
    sns.barplot(data=data, x="Category", y="Amount Lost")
    sns.barplot(data=data, x="Category", y="Amount Lost")
    sns.barplot(data=data, x="Category", y="Amount Lost")

    plt.axhline(y=0, color="red", linestyle="-", linewidth=1, label="Zero Line")

    plt.xlabel(None)
    plt.ylabel('Projected Loss (Currency)')
    plt.title('Projected Loss for Charged Off Loans')

    plt.legend()
    plt.show()


df = cleaning.clean_data(db_utils.df_from_csv("loans.csv"))

# percentage_of_loan_recovery(df)
# percentage_loss(df)
projected_loss(df)
