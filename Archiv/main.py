import requests
import pandas as pd

API_KEY = "YOUR_ALPHA_VANTAGE_KEY"
TICKERS = ["NVDA", "AMD", "INTC", "QCOM", "AVGO"]

def get_stock_df(ticker, api_key=API_KEY):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }

    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adjusted_close",
        "6. volume": "volume",
        "7. dividend amount": "dividend_amount",
        "8. split coefficient": "split_coefficient",
    })

    df = df.astype(float)
    df["ticker"] = ticker
    return df

# one dataframe per company
nvda_df = get_stock_df("NVDA")
amd_df = get_stock_df("AMD")

# dictionary of dataframes
dfs = {ticker: get_stock_df(ticker) for ticker in TICKERS}

# one combined dataframe
combined_df = pd.concat(dfs.values()).reset_index().rename(columns={"index": "date"})