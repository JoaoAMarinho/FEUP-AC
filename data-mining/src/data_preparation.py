import pandas as pd
import numpy as np

def parse_date(date):
  year = int(str(date)[0:2])
  month = int(str(date)[2:4])
  day = int(str(date)[4:6])
  return { 'year': year, 'month': month, 'day': day }


def parse_gender(row, birth_date):
  female = birth_date['month'] >= 50

  if female:
    row['gender'] = 'female' 
    birth_date['month'] -= 50
  else:
    row['gender'] = 'male' 


def calculate_age_loan(row):
  date_loan = row['date_loan']
  birth_number = row['birth_number']

  birth_date = parse_date(birth_number)

  parse_gender(row, birth_date)
  
  if (date_loan is None):
    date_loan = parse_date(row['date_loan'])
    date_loan = date_loan['year'] - birth_date['year'] - ((date_loan['month'], date_loan['day']) < (birth_date['month'], birth_date['day']))
  
  row['age_loan'] = date_loan
    
  return row


def calculate_average_commited_crimes(df):
    def nan_commited_crimes(year): return df["no. of commited crimes '" + str(year) + " "].isna()

    # convert '?' to NaN

    df['no. of commited crimes \'95 '] = pd.to_numeric(df['no. of commited crimes \'95 '], errors='coerce')
    df['no. of commited crimes \'96 '] = pd.to_numeric(df['no. of commited crimes \'96 '], errors='coerce')

    # NaN values will be equaled to the value of the other column

    df.loc[nan_commited_crimes(95), 'no. of commited crimes \'95 '] = df['no. of commited crimes \'96 ']
    df.loc[nan_commited_crimes(96), 'no. of commited crimes \'96 '] = df['no. of commited crimes \'95 ']

    # create column with mean from both years and drop previous and now useless columns

    df['commited_crimes'] = df[['no. of commited crimes \'95 ', 'no. of commited crimes \'96 ']].mean(axis=1)

    return df


def calculate_average_unemployment_rate(df):
    def nan_unemployment_rate(year): return df["unemploymant rate '" + str(year) + " "].isna()

    # convert '?' to NaN

    df['unemploymant rate \'95 '] = pd.to_numeric(df['unemploymant rate \'95 '], errors='coerce')
    df['unemploymant rate \'96 '] = pd.to_numeric(df['unemploymant rate \'96 '], errors='coerce')

    # NaN values will be equaled to the value of the other column

    df.loc[nan_unemployment_rate(95), 'unemploymant rate \'95 '] = df['unemploymant rate \'96 ']
    df.loc[nan_unemployment_rate(96), 'unemploymant rate \'96 '] = df['unemploymant rate \'95 ']

    # create column with mean from both years and drop previous and now useless columns

    df['unemployment_rate'] = df[['unemploymant rate \'95 ', 'unemploymant rate \'96 ']].mean(axis=1)

    return df


def calculate_number_of_disponents(df):
    disp_count = df.groupby(['account_id'])['disp_id'].nunique() 
    return df.merge(disp_count, on='account_id', suffixes=('', '_count'), how='left')


def calculate_diff_salary_loan(df):
    df['diff_salary_loan'] = df['average salary '] - df['payments']
    return df


def calculate_transaction_count(df):
    transaction_count = df.groupby(['account_id'])['trans_id'].nunique()
    df = df.merge(transaction_count, on='account_id', suffixes=['_', '_count'], how='left')
    return df


def calculate_credit_debit_ratio(df):
    count_transactions_per_type = df.groupby(['account_id', 'type']).size().unstack(fill_value=0)
    count_transactions_per_type['debit'] = count_transactions_per_type['withdrawal'] + count_transactions_per_type['withdrawal in cash']
    count_transactions_per_type.drop(labels=['withdrawal', 'withdrawal in cash'], axis=1, inplace=True)
    count_transactions_per_type['credit_debit_ratio'] = count_transactions_per_type['credit'] / count_transactions_per_type['debit']

    df = df.merge(count_transactions_per_type, on='account_id', how='left')
    df = df.replace([np.inf, -np.inf], 0)
    return df


def drop_duplicated_accounts(df):
    df.drop_duplicates(subset='account_id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def drop_irrelevant_columns_from_main_df(df):
    to_drop = [
        'account_id',
        'loan_id',
        'client_id',
        'disp_id',
        'card_id',
        'name ',
        'code',
        'region',
        'gender',
        'type_card',
        'issued',
        'unemploymant rate \'95 ', 
        'unemploymant rate \'96 ',
        'no. of commited crimes \'95 ', 
        'no. of commited crimes \'96 '
    ]
    df = df.drop(to_drop, axis=1).reset_index(drop=True)
    return df


def drop_irrelevant_columns_from_transactions_df(df):
    to_drop = [
        'k_symbol',
        'account_id',
        'trans_id_',
        'date_transaction',
        ]
    df = df.drop(to_drop, axis=1).reset_index(drop=True)
    return df

def drop_demographic_columns_from_transactions_df(df):
    df = df.drop(['no. of inhabitants', 'no. of municipalities with inhabitants < 499 ',
       'no. of municipalities with inhabitants 500-1999',
       'no. of municipalities with inhabitants 2000-9999 ',
       'no. of municipalities with inhabitants >10000 ', 'no. of cities ',
       'ratio of urban inhabitants ', 'average salary ',
       'no. of enterpreneurs per 1000 inhabitants ', 'unemployment_rate',
       'commited_crimes'], axis=1).reset_index(drop=True)
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
        'trans_id_count': 'transactions_count', 
        'credit': 'credits_count', 
        'debit': 'debits_count' 
    }
    df = df.rename(to_rename, axis=1)
    return df

def drop_outliers(df, col_name):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df_out

def convert_n_numerical_to_numerical(df, col_name):
    mapping = {k: v for v, k in enumerate(df[col_name].unique())}
    df[col_name] = df[col_name].map(mapping)
    return df
