import pandas as pd
import numpy as np
import re
from sklearn import preprocessing

#######
# Utils
#######

def encode_category(df, col):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[col].unique())
    df[col] = encoder.transform(df[col])
    return df

def format_date(df, col, format='%y%m%d'):
    df[col] = pd.to_datetime(df[col], format=format)
    return df

def split_birth(birth_number):
    year = 1900 + (birth_number // 10000)
    month = (birth_number % 10000) // 100
    day = birth_number % 100

    # Male
    if month < 50:
       gender = 1
    # Female
    else:
        gender = 0
        month = month - 50
    
    birth_date = year*10000 + month*100 +day
    birth_date = pd.to_datetime(birth_date, format='%Y%m%d')

    return gender, birth_date

def parse_date(date):
    year = int(str(date)[0:2])
    month = int(str(date)[2:4])
    day = int(str(date)[4:6])
    return {"year": year, "month": month, "day": day}


def parse_gender(row, birth_date):
    female = birth_date["month"] >= 50

    if female:
        row["gender"] = "female"
        birth_date["month"] -= 50
    else:
        row["gender"] = "male"


def calculate_birth_loan_difference(birth_date, loan_date):
    frame = { 'birth': birth_date, 'granted': loan_date }
    dates = pd.DataFrame(frame)

    dates['birth'] = pd.to_datetime(dates['birth'], format='%Y-%m-%d')
    dates['granted'] = pd.to_datetime(dates['granted'], format='%Y-%m-%d')
    dates['difference'] = (dates['granted'] - dates['birth']).dt.days // 365

    return dates['difference']

def calculate_age_loan(row):
    date_loan = row["date"]
    birth_number = row["birth_number"]

    birth_date = parse_date(birth_number)

    parse_gender(row, birth_date)

    if date_loan is not None:
        date_loan = parse_date(row["date"])
        date_loan = (
            date_loan["year"]
            - birth_date["year"]
            - (
                (date_loan["month"], date_loan["day"])
                < (birth_date["month"], birth_date["day"])
            )
        )

    row["age_loan"] = date_loan

    return row


def calculate_average_commited_crimes(df):
    def nan_commited_crimes(year):
        return df["no._of_commited_crimes_'" + str(year)].isna()

    # convert '?' to NaN

    df["no._of_commited_crimes_'95"] = pd.to_numeric(
        df["no._of_commited_crimes_'95"], errors="coerce"
    )
    df["no._of_commited_crimes_'96"] = pd.to_numeric(
        df["no._of_commited_crimes_'96"], errors="coerce"
    )

    # NaN values will be equaled to the value of the other column

    df.loc[nan_commited_crimes(95), "no._of_commited_crimes_'95"] = df[
        "no._of_commited_crimes_'96"
    ]
    df.loc[nan_commited_crimes(96), "no._of_commited_crimes_'96"] = df[
        "no._of_commited_crimes_'95"
    ]

    # create column with mean from both years

    df["avg_commited_crimes"] = df[
        ["no._of_commited_crimes_'95", "no._of_commited_crimes_'96"]
    ].mean(axis=1) / df["no._of_inhabitants"]

    return df


def calculate_average_unemployment_rate(df):
    def nan_unemployment_rate(year):
        return df["unemploymant_rate_'" + str(year)].isna()

    # convert '?' to NaN

    df["unemploymant_rate_'95"] = pd.to_numeric(
        df["unemploymant_rate_'95"], errors="coerce"
    )
    df["unemploymant_rate_'96"] = pd.to_numeric(
        df["unemploymant_rate_'96"], errors="coerce"
    )

    # NaN values will be equaled to the value of the other column

    df.loc[nan_unemployment_rate(95), "unemploymant_rate_'95"] = df[
        "unemploymant_rate_'96"
    ]
    df.loc[nan_unemployment_rate(96), "unemploymant_rate_'96"] = df[
        "unemploymant_rate_'95"
    ]

    # create column with mean from both years and drop previous and now useless columns

    df["unemployment_rate"] = df[
        ["unemploymant_rate_'95", "unemploymant_rate_'96"]
    ].mean(axis=1)

    return df


def calculate_number_of_disponents(df):
    disp_count = df.groupby(["account_id"])["disp_id"].nunique()
    return df.merge(disp_count, on="account_id", suffixes=("", "_count"), how="left")


def calculate_diff_salary_loan(df):
    df["diff_salary_loan"] = df["average salary"] - df["payments"]
    return df


def calculate_transaction_count(df):
    transaction_count = df.groupby(["account_id"])["trans_id"].nunique()
    df = df.merge(
        transaction_count, on="account_id", suffixes=["_", "_count"], how="left"
    )
    return df


def calculate_credit_debit_ratio(df):
    count_transactions_per_type = (
        df.groupby(["account_id", "type"]).size().unstack(fill_value=0)
    )
    count_transactions_per_type["debit"] = (
        count_transactions_per_type["withdrawal"]
        + count_transactions_per_type["withdrawal in cash"]
    )
    count_transactions_per_type.drop(
        labels=["withdrawal", "withdrawal in cash"], axis=1, inplace=True
    )
    count_transactions_per_type["credit_debit_ratio"] = (
        count_transactions_per_type["credit"] / count_transactions_per_type["debit"]
    )

    df = df.merge(count_transactions_per_type, on="account_id", how="left")
    df = df.replace([np.inf, -np.inf], 0)
    return df


def drop_duplicated_accounts(df):
    df.drop_duplicates(subset="account_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def drop_irrelevant_columns_from_df(df):
    to_drop = [
        "account_id",
        "district_id",
        "date",
        "date_loan",
        "client_id",
        "disp_id",
        "birth_number",
        "district_id_client",
    ]
    df = df.drop(to_drop, axis=1).reset_index(drop=True)
    return df

def drop_demographic_columns_from_df(df):
    to_drop = [
        "region",
        "code",
        "name",
        "no. of inhabitants",
        "no. of municipalities with inhabitants < 499",
        "no. of municipalities with inhabitants 500-1999",
        "no. of municipalities with inhabitants 2000-9999",
        "no. of municipalities with inhabitants >10000",
        "no. of cities",
        "ratio of urban inhabitants",
        "average salary",
        "no. of enterpreneurs per 1000 inhabitants",
        "unemploymant rate '95",
        "unemploymant rate '96",
        "no. of commited crimes '95",
        "no. of commited crimes '96"
    ]
    df = df.drop(to_drop, axis=1).reset_index(drop=True)
    return df

def drop_irrelevant_columns_from_main_df(df):
    to_drop = [
        "account_id",
        "client_id",
        "disp_id",
        "card_id",
        "name ",
        "code",
        "region",
        "gender",
        "type_card",
        "issued",
        "unemploymant rate '95",
        "unemploymant rate '96",
        "no. of commited crimes '95",
        "no. of commited crimes '96",
    ]
    df = df.drop(to_drop, axis=1).reset_index(drop=True)
    return df


def drop_irrelevant_columns_from_transactions_df(df):
    to_drop = [
        "k_symbol",
        "account_id",
        "trans_id_",
        "date_transaction",
    ]
    df = df.drop(to_drop, axis=1).reset_index(drop=True)
    return df


def drop_demographic_columns_from_transactions_df(df):
    df = df.drop(
        [
            "no. of inhabitants",
            "no. of municipalities with inhabitants < 499",
            "no. of municipalities with inhabitants 500-1999",
            "no. of municipalities with inhabitants 2000-9999 ",
            "no. of municipalities with inhabitants >10000",
            "no. of cities",
            "ratio of urban inhabitants",
            "average salary",
            "no. of enterpreneurs per 1000 inhabitants",
            "unemployment_rate",
            "commited_crimes",
        ],
        axis=1,
    ).reset_index(drop=True)
    return df


def rename_main_df_columns(df):
    to_rename = {}
    df = df.rename(to_rename, axis=1)
    return df


def drop_irrelevant_columns(df, to_drop):
    df = df.drop(to_drop, axis=1).reset_index(drop=True)
    return df


def rename_transactions_df_columns(df):
    to_rename = {
        "trans_id_count": "transactions_count",
        "credit": "credits_count",
        "debit": "debits_count",
    }
    df = df.rename(to_rename, axis=1)
    return df


def drop_outliers(df, col_name):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df_out


def convert_n_numerical_to_numerical(df, col_name):
    mapping = {k: v for v, k in enumerate(df[col_name].unique())}
    df[col_name] = df[col_name].map(mapping)
    return df

def clean_district_columns(column):
    column = column.strip()
    column = column.split()
    return '_'.join(column)


#######
# Clean
#######

def clean_loans(df):
    # Format loan date
    df = format_date(df, 'date')
    df.rename(columns={'date': 'loan_date'}, inplace=True)

    return df

def clean_accounts(df):
    # Format creation date
    df = format_date(df, 'date')

    # Encode frequency
    df = encode_category(df, 'frequency')

    df.rename(columns={'district_id': 'account_district_id','date': 'creation_date'}, inplace=True)

    return df

def clean_disp(df):
    # Keep only the owner of the account, because only the owner can ask for a loan
    owners_df = df[df['type']=='OWNER']

    disp_count = df.groupby(["account_id"]).agg({'type': 'count'}).reset_index()
    disp_count.columns = ['account_id', 'disp_count']
    df = owners_df.merge(disp_count, on="account_id", how="left")
    df.fillna(0, inplace=True)

    df['has_disponent'] = df['disp_count'] > 1
    df.drop(columns=['type', 'disp_count'], inplace=True)
    df = encode_category(df, 'has_disponent')

    return df

def clean_clients(df):
    # Gender and Date of birth
    df['gender'], df['birth_date'] = zip(*df['birth_number'].map(split_birth))

    df.rename(columns={'district_id': 'client_district_id'}, inplace=True)
    df.drop(columns=["birth_number"], inplace=True)

    return df

def clean_districts(df):
    # Rename wrongly named columns
    df = df.rename(clean_district_columns, axis='columns')

    # Feature extraction
    df = calculate_average_unemployment_rate(df)
    df = calculate_average_commited_crimes(df)

    # Entrepreneurs ratio
    df['ratio_entrepreneurs'] = df['no._of_enterpreneurs_per_1000_inhabitants'] / 1000

    # From percentage to ratio
    df['ratio_of_urban_inhabitants'] = df['ratio_of_urban_inhabitants'] / 100

    # Criminality growth
    df['criminality_growth'] = (df["no._of_commited_crimes_'96"] - df["no._of_commited_crimes_'95"]) / df['no._of_inhabitants']

    # Unemployment growth
    df['unemployment_growth'] = df["unemploymant_rate_'96"] - df["unemploymant_rate_'95"]

    # Drop
    df.drop(columns=[
        "no._of_commited_crimes_'95",
        "no._of_commited_crimes_'96",
        "unemploymant_rate_'95",
        "unemploymant_rate_'96",
        'no._of_enterpreneurs_per_1000_inhabitants',
        'name'
    ], inplace=True)

    # Encode Region
    df = encode_category(df, 'region')

    return df

def clean_transactions(df):

    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # Drop bank column - only 75% not null values
    df.drop(columns=['bank'], inplace=True)

    # Fix Operation values
    df["operation"].fillna("interest credited", inplace=True)

    # Rename values
    df.loc[df["operation"]=="credit in cash", "operation"] = "CashC"
    df.loc[df["operation"]=="collection from another bank", "operation"] = "Coll"
    df.loc[df["operation"]=="interest credited", "operation"] = "Interest"
    df.loc[df["operation"]=="withdrawal in cash", "operation"] = "CashW"
    df.loc[df["operation"]=="remittance to another bank", "operation"] = "Rem"
    df.loc[df["operation"]=="credit card withdrawal", "operation"] = "CardW"

    # Fix K_symbol values
    df["k_symbol"].fillna("None", inplace=True)

    # Rename values
    df.loc[df["k_symbol"]=="insurrance payment", "k_symbol"] = "Insurance"
    df.loc[df["k_symbol"]=="interest credited", "k_symbol"] = "Interest"
    df.loc[df["k_symbol"]=="household", "k_symbol"] = "Household"
    df.loc[df["k_symbol"]=="payment for statement", "k_symbol"] = "Statement"
    df.loc[df["k_symbol"]=="sanction interest if negative balance", "k_symbol"] = "Sanction"
    df.loc[df["k_symbol"]=="old-age pension", "k_symbol"] = "Pension"

    # Type & Amount
    # Rename withdrawal in cash - wrong label
    df.loc[df['type'] == 'withdrawal in cash','type'] = 'withdrawal'

    # Make withdrawal amount negative
    df.loc[df["type"]=="withdrawal", "amount"] *= -1

    # Format date
    df = format_date(df, 'date')

    # Feature Extraction

    # Average Amount by type
    avg_amounts = df.groupby(['account_id', 'type'], as_index=False)['amount'].mean()
    
    credit_amount_mean = avg_amounts[avg_amounts['type'] == 'credit']
    credit_amount_mean.columns = ['account_id', 'type', 'avg_amount_credit']

    withdrawal_amount_mean = avg_amounts[avg_amounts['type'] == 'withdrawal']
    withdrawal_amount_mean.columns = ['account_id', 'type', 'avg_amount_withdrawal']

    credit_amount_mean = credit_amount_mean.drop(columns=["type"])
    withdrawal_amount_mean = withdrawal_amount_mean.drop(columns=["type"])

    avg_amount_df = pd.merge(credit_amount_mean, withdrawal_amount_mean, on="account_id", how="outer")
    avg_amount_df.fillna(0, inplace=True)

    avg_amount_total = df.groupby(['account_id']).agg({'amount':['mean', 'min', 'max']}).reset_index()
    avg_amount_total.columns = ['account_id', 'avg_amount_total', 'min_amount', 'max_amount']
    new_df = pd.merge(avg_amount_df, avg_amount_total, on="account_id", how="outer")
    new_df.fillna(0, inplace=True)

    # Number of withdrawals and credits
    type_counts = df.groupby(['account_id', 'type']).size().reset_index(name='counts')

    credit_counts = type_counts[type_counts['type'] == 'credit']
    credit_counts.columns = ['account_id', 'type', 'num_credits']
    credit_counts = credit_counts.drop(columns=["type"])

    withdrawal_counts = type_counts[type_counts['type'] == 'withdrawal']
    withdrawal_counts.columns = ['account_id', 'type', 'num_withdrawals']
    withdrawal_counts = withdrawal_counts.drop(columns=["type"])

    trans_type_count_df = pd.merge(credit_counts, withdrawal_counts, on="account_id", how="outer")
    trans_type_count_df.fillna(0, inplace=True)
    trans_type_count_df['credit_ratio'] = trans_type_count_df['num_credits'] / (trans_type_count_df['num_credits'] + trans_type_count_df['num_withdrawals'])
    # trans_type_count_df['withdrawal_ratio'] = trans_type_count_df['num_withdrawals'] / (trans_type_count_df['num_credits'] + trans_type_count_df['num_withdrawals'])

    trans_type_count_df.drop(columns=['num_credits', 'num_withdrawals'], inplace=True)
    new_df = pd.merge(new_df, trans_type_count_df, on="account_id", how="outer")

    # Average, Min, Max Balance & Num Transactions
    balance_count_df = df.groupby(['account_id'])["balance"]

    balance_count_df = df.groupby(['account_id']).agg({'balance':['count', 'mean', 'min', 'max', 'std']}).reset_index()
    balance_count_df.columns = ['account_id', 'num_trans', 'avg_balance', 'min_balance', 'max_balance', 'std_balance']

    balance_count_df['negative_balance'] = balance_count_df['min_balance'] < 0
    # balance_count_df.drop(columns=['min_balance'], inplace=True)
    balance_count_df = encode_category(balance_count_df, 'negative_balance')

    # Last Transaction
    last_balance_df = df.sort_values('date', ascending=False).groupby(['account_id']).head(1).reset_index()
    last_balance_df['last_balance_negative'] = last_balance_df['balance'] < 0
    last_balance_df = encode_category(last_balance_df, 'last_balance_negative')
    last_balance_df = last_balance_df[["account_id", "last_balance_negative"]]

    new_df = pd.merge(new_df, balance_count_df, on="account_id", how="outer")
    new_df = pd.merge(new_df, last_balance_df, on="account_id", how="outer")

    return new_df

def clean_cards(df_card, df_disp):

    df = df_disp.merge(df_card, on="disp_id", how="left")

    df = df.groupby(['account_id']).agg({'card_id':['count']}).reset_index()
    df.columns = ['account_id', 'num_cards']

    # Has card
    df['has_card'] = df['num_cards'] > 0
    df.drop(columns=['num_cards'], inplace=True)
    df = encode_category(df, 'has_card')

    return df

def clean_columns(df):
    return df.drop(columns=["account_id", "disp_id", "client_id", "code",
        "account_district_id", "client_district_id",
        "loan_date", "creation_date", "birth_date",
        "region", "no._of_inhabitants",
        "no._of_municipalities_with_inhabitants_<_499",
        "no._of_municipalities_with_inhabitants_500-1999",
        "no._of_municipalities_with_inhabitants_2000-9999",
        "no._of_municipalities_with_inhabitants_>10000", "no._of_cities"])

#######
# Merge
#######

def merge_dfs(dfs):
    [ account, disp, cards, client, district, loan, transaction ] = dfs

    df = pd.merge(loan, account, on='account_id', how="left")
    df = pd.merge(df, disp,  on='account_id', how="left")
    df = pd.merge(df, client,  on='client_id', how="left")
    df = pd.merge(df, district, left_on='client_district_id', right_on='code')
    df = pd.merge(df, transaction, how="left", on="account_id")
    df = pd.merge(df, cards, how="left", on="account_id")

    return df


##########
# Features
##########

def extract_other_features(df):

    # Age when the loan was requested
    df['age_at_loan'] = calculate_birth_loan_difference(df['birth_date'], df['loan_date'])

    # Days between loan and account creation
    df['days_between'] = (df['loan_date'] - df['creation_date']).dt.days

    # Boolean value telling if the account was created on the owner district
    df['same_district'] = df['account_district_id'] == df['client_district_id']

    df = df.set_index('loan_id')

    return df
