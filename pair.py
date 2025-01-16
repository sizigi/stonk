import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
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


def get_tickers():
    tickers = get_nasdaq100_tickers() + get_snp500_tickers()
    tickers = list(set(tickers))
    return tickers


def get_ticker_prices(tickers, cache_path="./cache"):
    all_diffs = {}
    for ticker in tqdm(tickers):
        try:
            data = yf.download(ticker, start="2020-01-01")
            if os.path.exists(cache_path):
                data.to_csv(os.path.join(cache_path, f"{ticker}.csv"))
            ema_1m = data["Close"].ewm(span=21, adjust=False)
            ema_1m = ema_1m.mean().values.tolist()
            diff = data["Close"].values[-1] - ema_1m[-1]
            all_diffs[ticker] = np.concatenate([diff, diff / data["Close"].values[-1]])
        except Exception as e:
            print(f"Failed to get data for {ticker}")
            print(e)
    return all_diffs


def norm_price_corr(cache_path, norm=True, span=90):
    files = os.listdir(cache_path)
    equity_prices = []
    ticker_names = []
    for file in files:
        data = pd.read_csv(os.path.join(cache_path, file))
        equity_price = data["Close"].values[-span:]
        if equity_price.shape[0] < span:
            continue
        equity_prices.append(equity_price.astype(np.float32))
        ticker_names.append(file.split(".")[0])
    equity_prices = np.array(equity_prices)

    if norm:
        equity_prices = equity_prices / equity_prices[:, 0:1]
    
    dates = np.arange(equity_prices.shape[1]).reshape(1, -1).repeat(equity_prices.shape[0], axis=0)
    ones = np.ones_like(dates)
    equity_prices_homo = np.stack([dates, equity_prices, ones], axis=-1)
    equity_prices_points = np.stack([dates, equity_prices], axis=-1)
    
    all_diffs = []
    for equity_price_homo in equity_prices_homo:
        diffs = []
        for equity_price_points in equity_prices_points:
            A, res, rank, s = np.linalg.lstsq(equity_price_homo, equity_price_points)
            diff = np.linalg.norm(equity_price_homo @ A - equity_price_points)
            diffs.append(diff)
        all_diffs.append(np.array(diffs))
    all_diffs = np.array(all_diffs)

    rng = np.arange(len(ticker_names))
    x, y = np.meshgrid(rng, rng)
    all_diff_tups = np.stack((all_diffs, y, x), -1).reshape(-1, 3)
    sorted_diff_tups = all_diff_tups[all_diff_tups[:, 0].argsort()]
    sorted_diff_tups = sorted_diff_tups[len(ticker_names):]

    # diff_map = all_diffs[:20, :20]
    # ticker_names = ticker_names[:20]
    # fig, ax = plt.subplots(figsize=(10, 10))
    # cax = ax.matshow(diff_map, cmap='coolwarm')
    # fig.colorbar(cax)
    # ax.set_xticks(np.arange(len(ticker_names)))
    # ax.set_yticks(np.arange(len(ticker_names)))
    # ax.set_xticklabels(ticker_names, rotation=90)
    # ax.set_yticklabels(ticker_names)
    # plt.title("Difference Heatmap")
    # plt.savefig("corr.png")

    return sorted_diff_tups, ticker_names
    

if __name__ == "__main__":
    # ema sort
    tickers = get_tickers()
    all_diffs = get_ticker_prices(tickers)
    all_diffs = {
        k: v for k, v in sorted(all_diffs.items(), key=lambda item: item[1][1])
    }
    print(all_diffs)

    sorted_diff_tups, ticker_names = norm_price_corr("./cache")
    cnt = 0
    for diff_tup in sorted_diff_tups:
        diff, i, j = diff_tup
        name_i, name_j = ticker_names[int(i)], ticker_names[int(j)]
        if name_i == "K" or name_j == "K":
            continue
        print(f"{ticker_names[int(i)]} - {ticker_names[int(j)]}: {diff}")
        cnt += 1
        if cnt == 400:
            break
