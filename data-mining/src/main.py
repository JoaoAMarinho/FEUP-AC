from tkinter import TRUE
from pandas import read_csv, DataFrame

account = read_csv('../data/account.csv',';')
card_dev = read_csv('../data/card_dev.csv',';')
client = read_csv('../data/client.csv',';')
disp = read_csv('../data/disp.csv',';')
district = read_csv('../data/district.csv',';')
district = district.rename(columns={'code':'district_id'})
loan_dev = read_csv('../data/loan_dev.csv',';')
trans_dev = read_csv('../data/trans_dev.csv',';')


def main():
  print(district)
  account_loan = loan_dev.join(account.set_index('account_id'), on='account_id', lsuffix='_loan')
  account_loan_crimes = account_loan.join(district.set_index('district_id'), on='district_id', rsuffix='_district')
  print(account_loan_crimes)

if __name__ == "__main__":
  main()