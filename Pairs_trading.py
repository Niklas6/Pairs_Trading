
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(slots=True)
class pairmodel:
    left: str #= 'LRCX'
    right: str #= 'AMAT'
    gamma: float =1/2
    threshold: float =1/50
    #max_exposure: float =200


@dataclass(slots=True)
class position:
    left: float = 0.0
    right: float = 0.0




def run_strategy(prices: pd.DataFrame,model: pairmodel) -> tuple[ float, int, pairmodel]:
    pos = position()
    left = model.left
    right = model.right
    revenue=0
    trades=0
    for t in prices.index:
        npos = build_target_position( prices.loc[t,left], prices.loc[t,right], model.gamma,model.threshold,pos)
        if npos != pos:
            revenue += prices.loc[t,left] * (pos.left - npos.left) + prices.loc[t,right] * (pos.right - npos.right)
            pos = npos
            trades += 1
        model.gamma=49/50*model.gamma+ 1/50 * prices.loc[t,left]/prices.loc[t,right]
    revenue += prices.loc[t, left] * pos.left  + prices.loc[t, right] * pos.right

    return revenue, trades,model




def build_target_position(y1: float, y2: float, gamma: float, thresh: float, pos: position) -> position:
    npos=position()
    if pos.left > 0:
        if y1 - gamma * y2 < 0:
            return pos
    elif pos.left < 0:
        if y1 - gamma * y2 > 0:
            return pos
    if y1 - gamma * y2 > thresh * (y1 + gamma * y2):  # y1 is overvalued
        if pos.left >= 0:
            npos.left = -float(1 / (y1 + y2)/(1+gamma) * (100))
            npos.right = float(1 / (y1 + y2)/(1+gamma) * gamma * (100))
    elif y1 - gamma * y2 < - thresh * (y1 + gamma * y2):  # y2 is overvalued
        if pos.left <= 0:
            npos.left = float(1 / (y1 + y2)/(1+gamma) * (100))
            npos.right= -float(1 / (y1 + y2/(1+gamma) * gamma * (100)))
    return npos

def build_model(prices_analysis: pd.DataFrame, left: str, right: str):
    gamma = (prices_analysis.loc[:,left] / prices_analysis.loc[:,right]).mean()
    return pairmodel(left,right, gamma)


def run_analysis_trade_model(prices_analysis: pd.DataFrame,prices_trading: pd.DataFrame, left: str, right: str):


    model=build_model(prices_analysis,left,right)

    revenue_analysis, trades_analysis,model_analysis=run_strategy(prices_analysis,model)
    revenue_trading, trades_trading,model_trade=run_strategy(prices_trading,model_analysis)

    return(revenue_analysis, trades_analysis,revenue_trading, trades_trading)




def main(left: str = 'LRCX',right: str = 'AMAT') -> None:

    prices_analysis = pd.read_csv("semiconductor_close_analysis.csv", index_col="date", parse_dates=True).sort_index().loc[:,[left, right ]]
    prices_trading = pd.read_csv("semiconductor_close_trade.csv", index_col="date", parse_dates=True).sort_index().loc[:,[left, right ]]
    #categories = pd.read_csv("categories.csv", index_col="ticker")
    revenue_analysis, trades_analysis,revenue_trading, trades_trading=run_analysis_trade_model(prices_analysis,prices_trading,left, right )




    return(revenue_analysis, trades_analysis,revenue_trading, trades_trading)
    #model, revenue = run_train_test_strategy(
    #    train_prices=prices_analysis,
    #    test_prices=prices_trading,
    #    left=args.left,
    #    right=args.right,
    #)







if __name__ == "__main__":
    main()