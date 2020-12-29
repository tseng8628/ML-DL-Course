import matplotlib.pyplot as plt
import pandas as pd
import csv
from itertools import zip_longest

Original = pd.read_csv('^GSPC.csv')
adj_close = list(Original['Adj Close'])
volume = list(Original['Volume'])
High = list(Original['High'])
Low = list(Original['Low'])
Date = list(Original['Date'])

target = pd.read_csv('target150.csv')
tIndex = list(target['Index'])

rsi_nDays = 15
# 交易日往回推15天
# https://kknews.cc/zh-tw/news/p6e4nj2.html
# https://stockdog.blog/2019/04/07/%E7%9B%B8%E5%B0%8D%E5%BC%B7%E5%BC%B1%E6%8C%87%E6%A8%99rsi/

# def RSI(adj_close):
rsiList = []
fakeData = [0] * (rsi_nDays-1)

for x in range(rsi_nDays-1, len(adj_close)):

    U = []
    D = []

    tempClose = adj_close[x-(rsi_nDays-1):x+1].copy()
    for c in range(len(tempClose)-1):
        temp = tempClose[c+1] - tempClose[c]
        if temp > 0:
            U.append(temp)
        else:
            D.append(abs(temp))

    RS = (sum(U)/rsi_nDays) / (sum(D)/rsi_nDays)
    rsiValue = round(100 * (RS / 1+RS), 4)
    rsiList.append(rsiValue)

rsiList = fakeData + rsiList
rsiList = rsiList[tIndex[0]:tIndex[-1]+1]

# https://www.ezchart.com.tw/inds.php?IND=WMR
nDays = 15
# def WR(adj_close, volume):
wrList = []
for x in range(nDays-1, len(adj_close)-1):
    tempClose = adj_close[x-14:x+1].copy()
    Hn = max(tempClose)     # n日內最高價
    Ln = min(tempClose)     # n日內最低價
    C = adj_close[x]
    wrValue = round(100 - (Hn-C / Hn-Ln)*100, 4)
    wrList.append(wrValue)

wrList = fakeData + wrList
wrList = wrList[tIndex[0]:tIndex[-1]+1]

# WMA加權移動平均線
# def WMA(adj_close, volume):
wmaList = []


def EMA(adj_close, volume):
    RSI_list = []
    a = 0

# SMA簡單移動平均線；設定天數：10天
# def SMA(adj_close, volume):
smaList = []
smaDays = 10
sma_fakeData = [0] * (smaDays-1)
for i in range(smaDays-1, len(adj_close)):
    temp = adj_close[i-smaDays+1:i+1].copy()
    smaValue = round(sum(temp) / smaDays, 4)
    smaList.append(smaValue)

smaList = sma_fakeData + smaList
smaList = smaList[tIndex[0]:tIndex[-1]+1]

def HMA(adj_close, volume):
    RSI_list = []
    a = 0

def TRIPLe(adj_close, volume):
    RSI_list = []
    a = 0

# CCI商品通道指標
# https://www.moneydj.com/KMDJ/wiki/wikiViewer.aspx?keyid=0951f779-00d3-4f70-878f-23b5a1ce316b
# def CCI(adj_close, volume):
cciList = []
apList = []
mapList = []
mdList = []
for i in range(len(adj_close)):
    apValue = (High[i]+Low[i]+adj_close[i]) / 3
    apList.append(apValue)

for i in range(len(adj_close)):
    temp_mapList = apList[i-i:i+1]
    mapValue = sum(temp_mapList) / len(temp_mapList)
    mapList.append(mapValue)

for i in range(len(adj_close)):
    a = mapList[i-i:i+1]
    b = apList[i-i:i+1]
    c = [a[i] - b[i] for i in range(len(a))]
    res = list(map(abs, c))
    mdValue = sum(res) / len(res)
    mdList.append(mdValue)

for i in range(len(adj_close)):
    try:
        cciValue = round((apList[i]-mdList[i]) / (0.015*mdList[i]), 4)
        cciList.append(cciValue)
    except ZeroDivisionError:
        cciList.append(0.0000)

cciList = cciList[tIndex[0]:tIndex[-1]+1]

# CMO
# def CMO(adj_close, volume):
cmoList = []
for x in range(nDays-1, len(adj_close)-1):
    U = []          # 上漲總幅度
    D = []          # 下跌總幅度

    tempClose = adj_close[x-14:x+1].copy()
    for c in range(len(tempClose)-1):
        temp = tempClose[c+1] - tempClose[c]
        if temp > 0:
            U.append(temp)
        else:
            D.append(temp)

    cmoValue = round((sum(U)-sum(D)) / (sum(U)+sum(D)), 4)
    cmoList.append(cmoValue)

cmoList = fakeData + cmoList
cmoList = cmoList[tIndex[0]:tIndex[-1]+1]

def MACD(adj_close, volume):
    RSI_list = []
    a = 0

def PPO(adj_close, volume):
    RSI_list = []
    a = 0

# ROC變動率指標
# def ROC(adj_close, volume):
rocList = []
for i in range(nDays-1, len(adj_close)-1):
    n_close = i-14
    rocValue = ((adj_close[i]-adj_close[n_close])/adj_close[n_close])*100
    rocValue = round(rocValue, 4)
    rocList.append(rocValue)

rocList = fakeData + rocList
rocList = rocList[tIndex[0]:tIndex[-1]+1]

# CMFI家慶資金流量指標
# bug
# def CMFI(adj_close, volume):
Period = 21
cmfi_fakeData = [0] * (Period-1)
cmfiList = []
for x in range(Period-1, len(adj_close)-1):

    tempCmfi = adj_close[x-Period+1:x+1].copy()

    multList = []
    mfvList = []
    for i in range(len(tempCmfi)-1):
        multiplier = (adj_close[i]-Low[i])-(High[i]-adj_close[i]) / (High[i]-Low[i])
        mfv = volume[i] * multiplier
        multList.append(multiplier)
        mfvList.append(mfv)

    cmfiValue = sum(mfvList) / sum(multList)
    cmfiValue = round(cmfiValue, 4)
    cmfiList.append(cmfiValue)

cmfiList = cmfi_fakeData + cmfiList
cmfiList = cmfiList[tIndex[0]:tIndex[-1]+1]

def DMI(adj_close, volume):
    RSI_list = []
    a = 0

def SAR(adj_close, volume):
    RSI_list = []
    a = 0

# saveToCSV
saveIndex = list(range(tIndex[0], tIndex[-1]+1))
saveDate = Date[tIndex[0]:tIndex[-1]+1]
saveData = list(zip_longest(saveIndex, saveDate, rsiList, wrList, smaList, cciList, cmoList, rocList))

with open('Input-Data.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'Date', 'RSI', 'W%R', 'SMA', 'CCI', 'CMO', 'ROC'])

    for items in saveData:
        writer.writerow(items)



