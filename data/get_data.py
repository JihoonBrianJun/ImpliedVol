import json
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.stats import norm
from scipy.optimize import newton

# Black-Scholes formula for call option price
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Vega of the call option, derivative of price w.r.t. volatility
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# Objective function to find implied volatility
def objective_function(sigma, S, K, T, r, C_market):
    return black_scholes_call_price(S, K, T, r, sigma) - C_market

# Function to calculate implied volatility using Newton-Raphson method
def implied_volatility(S, K, T, r, C_market, initial_guess=0.2):
    # Use the Newton-Raphson method and provide the Vega function as the derivative
    implied_vol = newton(
        lambda sigma: objective_function(sigma, S, K, T, r, C_market),
        initial_guess,
        fprime=lambda sigma: vega(S, K, T, r, sigma),
    )
    return implied_vol


def main(args):
    if args.download_data:
        interest_rates = yf.download(args.treasury_ticker, interval="1d", start=args.start, end=args.end)['Adj Close']
        underlying_price_df = yf.download(args.underlying_ticker, interval="1d", start=args.start, end=args.end)['Adj Close']
        dates = underlying_price_df.index
        remain_dates = (pd.to_datetime('2024-06-21') - dates).days / 366
        underlying_prices = underlying_price_df.to_numpy()
        
        all_implied_vol_list = []
        for idx, date in tqdm(enumerate(dates)):
            interest_rate = interest_rates.loc[date]
            underlying_price = underlying_prices[idx]
            remain_date = remain_dates[idx]
            implied_vol_list = []
            for ratio in (np.arange(args.bins) - 5):
                strike = int(np.round((underlying_price * (1 + ratio / 100)) // 50) * 50)
                try:
                    option_price_df = yf.download(f"{args.base_ticker}0{strike}000", interval="1d", progress=False)['Adj Close']
                    if date in option_price_df.index:
                        try:
                            implied_vol = implied_volatility(underlying_price, strike, remain_date, interest_rate / 100, option_price_df.loc[date])
                        except:
                            implied_vol = np.nan
                    else:
                        implied_vol = np.nan
                except:
                    implied_vol = np.nan
                implied_vol_list.append(implied_vol)
            all_implied_vol_list.append(implied_vol_list)

        with open(args.save_dir, 'w') as f:
            json.dump(all_implied_vol_list, f)
    
    with open(args.save_dir, 'r') as f:
        all_implied_vol_list = json.load(f)
        
    for idx in range(len(all_implied_vol_list)):
        mean_vol = np.nanmean(all_implied_vol_list[idx])
        for sub_idx in range(args.bins):
            if np.isnan(all_implied_vol_list[idx][sub_idx]):
                all_implied_vol_list[idx][sub_idx] = mean_vol
    
    print(np.mean(all_implied_vol_list[-1]))
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='data/all.json')
    parser.add_argument('--bins', type=int, default=11)
    parser.add_argument('--download_data', type=bool, default=False)
    parser.add_argument('--treasury_ticker', type=str, default='^IRX')
    parser.add_argument('--underlying_ticker', type=str, default='ESM24.CME')
    parser.add_argument('--base_ticker', type=str, default='SPX240621C')
    parser.add_argument('--start', type=str, default='2023-06-21')
    parser.add_argument('--end', type=str, default='2024-06-10')
    args = parser.parse_args()
    main(args)