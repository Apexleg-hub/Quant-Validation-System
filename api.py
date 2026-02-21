

from fastapi import FastAPI
import pandas as pd

app = FastAPI()


@app.get("/trades")

def get_trades():

    df = pd.read_csv("logs/trades_log.csv")

    return df.to_dict(orient="records")



@app.get("/equity")

def get_equity():

    df = pd.read_csv("logs/trades_log.csv")

    df['equity'] = df['profit'].cumsum()

    return df[['equity']].to_dict(orient="records")
