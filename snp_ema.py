import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# list all nasdaq 100 stocks
def get_nasdaq100_tickers():
    # get nasdaq 100 data from wikipedia
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    nasdaq100 = tables[4]
    symbols = nasdaq100["Symbol"].tolist()
    return symbols


def get_snp500_tickers():
    # get nasdaq 100 data from wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    nasdaq100 = tables[0]
    symbols = nasdaq100["Symbol"].tolist()
    return symbols


def get_current_diffs(tickers):
    all_diffs = {}
    for ticker in tqdm(tickers):
        try:
            data = yf.download(ticker, start="2020-01-01")
            ema_1m = data["Close"].ewm(span=21, adjust=False)
            ema_1m = ema_1m.mean().values.tolist()
            diff = data["Close"].values[-1] - ema_1m[-1]
            all_diffs[ticker] = np.concatenate([diff, diff / data["Close"].values[-1]])
        except Exception as e:
            print(f"Failed to get data for {ticker}")
            print(e)
    return all_diffs


# get nasdaq 100 data from yahoo finance
# plot the data
def plot_nasdaq100(ticker):
    data = yf.download(ticker, start="2020-01-01")

    # Calculate the EMAs for different periods
    data["1Y_EMA"] = (
        data["Close"].ewm(span=252, adjust=False).mean()
    )  # 1 year EMA (252 trading days)
    data["3M_EMA"] = (
        data["Close"].ewm(span=63, adjust=False).mean()
    )  # 3 months EMA (63 trading days)
    data["1M_EMA"] = (
        data["Close"].ewm(span=21, adjust=False).mean()
    )  # 1 month EMA (21 trading days)
    data["1W_EMA"] = (
        data["Close"].ewm(span=5, adjust=False).mean()
    )  # 1 week EMA (5 trading days)

    # Plot the data and the EMAs
    plt.figure(figsize=(12, 8))
    plt.plot(data["Close"], label=f"{ticker} Closing Price", color="blue", alpha=0.7)
    plt.plot(data["1Y_EMA"], label="1 Year EMA", color="red")
    plt.plot(data["3M_EMA"], label="3 Months EMA", color="green")
    plt.plot(data["1M_EMA"], label="1 Month EMA", color="orange")
    plt.plot(data["1W_EMA"], label="1 Week EMA", color="purple")

    # Add labels and title
    plt.title(
        f"{ticker} Stock Price and Exponential Moving Averages (EMAs)", fontsize=16
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    tickers = get_nasdaq100_tickers() + get_snp500_tickers()
    tickers = list(set(tickers))
    all_diffs = get_current_diffs(tickers)
    all_diffs = {
        k: v for k, v in sorted(all_diffs.items(), key=lambda item: item[1][1])
    }
    print(all_diffs)
