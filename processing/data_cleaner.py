
#This cleans data.


import pandas as pd

def clean_data():

    df = pd.read_csv("data/raw/XAUUSD_H1_raw.csv")

    df = df.drop_duplicates()

    df = df.dropna()

    df['returns'] = df['close'].pct_change()

    df.to_csv("data/clean/XAUUSD_H1_clean.csv", index=False)

    print("Clean data saved")


if __name__ == "__main__":
    clean_data()
