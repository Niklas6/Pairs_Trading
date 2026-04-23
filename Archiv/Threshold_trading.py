import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



prices = pd.read_csv("semiconductor_close_analysis.csv", index_col="date", parse_dates=True).sort_index()
categories = pd.read_csv("categories.csv", index_col="ticker")


name1='LRCX'
name2='AMAT'



stock1=prices.loc[:,name1]
stock2=prices.loc[:,name2]

gamma=(stock1/stock2).mean()
#print(stock1.iloc[0],stock2.iloc[0],stock2.iloc[0]/stock1.iloc[0])
#(stock1.mean(),stock2.mean(),gamma)

for t in stock1.index:
    print()

stock1=prices.loc[:,name1]
stock2=prices.loc[:,name2]
#print(stock1.head())
#print(stock2.head())

def signal(pos,y1,y2,gamma):
    nbuy=1/40
    npos = [0, 0]
    if pos[0]>0:
        if y1-gamma*y2>0:
            npos = [0,0]
        else:
            return pos

    elif pos[0]<0:
        if y1-gamma*y2<0:
            npos = [0,0]
        else:
            return pos

    if y1-gamma*y2>nbuy*(y1+gamma*y2):#y1 is overvalued
        if pos[0]>=0:
            npos[0]=-float(1/(y1+y2)*(100))
            npos[1]=float(1/(y1+y2)*gamma*(100))
    elif y1 - gamma * y2 < - nbuy*(y1+gamma*y2):#y2 is overvalued
        if pos[0]<=0:
            npos[0]=float(1/(y1+y2)*(100))
            npos[1]=-float(1/(y1+y2)*gamma *(100))
    return npos


#We test with historical data
pos=[0,0]
revenue=0
trades=0
for t in stock1.index:
    y1=stock1[t]
    y2=stock2[t]
    npos=signal(pos, y1, y2, gamma)
    if npos!=pos:
        revenue+=y1*(pos[0]-npos[0])+y2*(pos[1]-npos[1])
        #print(t, revenue,pos,npos)
        pos=npos

        trades +=1
revenue +=y1*pos[0]+y2*pos[1]
print(revenue,trades)




prices_trading = pd.read_csv("semiconductor_close_trade.csv", index_col="date", parse_dates=True).sort_index()


stock1_trading=prices_trading.loc[:,name1]
stock2_trading=prices_trading.loc[:,name2]



print((stock1_trading/stock2_trading).mean(),gamma)
pos=[0,0]
revenue=0
trades=0
for t in stock1_trading.index:
    #print(t)
    y1=stock1_trading[t]
    y2=stock2_trading[t]
    npos=signal(pos, y1, y2, gamma)
    if npos!=pos:
        revenue+=y1*(pos[0]-npos[0])+y2*(pos[1]-npos[1])
        print(t, revenue,pos,npos)
        pos=npos

        trades +=1
revenue +=y1*pos[0]+y2*pos[1]
print(revenue,trades)


