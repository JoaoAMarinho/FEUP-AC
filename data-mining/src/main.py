from pandas import read_csv

account = read_csv('../data/account.csv',';')
card_dev = read_csv('../data/card_dev.csv',';')
client = read_csv('../data/client.csv',';')
disp = read_csv('../data/disp.csv',';')
district = read_csv('../data/district.csv',';')
loan_dev = read_csv('../data/loan_dev.csv',';')
trans_dev = read_csv('../data/trans_dev.csv',';')


def main():
  print(account)

if __name__ == "__main__":
  main()