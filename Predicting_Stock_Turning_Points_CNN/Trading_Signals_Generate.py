import matplotlib.pyplot as plt
import pandas as pd
import csv
from itertools import zip_longest
import numpy as np

df = pd.read_csv('^GSPC.csv')
adj_close = list(df['Adj Close'])
Date = list(df['Date'])

threshold = 150
local_min = []
local_max = []
local_max_idx = []
local_min_idx = []
TP = []
ntp_idx = []

# Generate local maximun / local minimun points.
for i in range(1, len(adj_close) - 1):

    if adj_close[i] > adj_close[i-1] and adj_close[i] > adj_close[i+1]:
        TP.append(adj_close[i])
        local_max_idx.append(i)
        local_max.append(adj_close[i])

    elif adj_close[i] < adj_close[i-1] and adj_close[i] < adj_close[i+1]:
        TP.append(adj_close[i])
        local_min_idx.append(i)
        local_min.append(adj_close[i])

new_tp = TP.copy()

while True:

    attitude = []
    for i in range(len(new_tp) - 1):
        bc = float(new_tp[i]) - float(new_tp[i+1])
        attitude.append(abs(bc))

    index = attitude.index(min(attitude))       # 最小值索引值 等於 b點
    min_bc = min(attitude)                      # 最小段bc長度

    if min_bc < threshold:
        del new_tp[index]
        del new_tp[index]                       # 因為刪除TP整個index 往前一格 所以不用index+1

    if min_bc >= threshold:
        new_tp = new_tp.copy()
        break

new_tp_org = new_tp.copy()

# 處理連續上升或下降

t = []
index_t = []
for i in range(len(new_tp) - 1):
    bc = float(new_tp[i+1]) - float(new_tp[i])

    if bc < 0:
        t.append(-1)
        index_t.append(i)
    else:
        t.append(1)
        index_t.append(i)

removeIndex = []
for i in range(len(t)-1):
    if t[i] == t[i+1]:
        removeIndex.append(i+1)

new_tp = [i for j, i in enumerate(new_tp) if j not in removeIndex]

# Trading Signal trans
new_tp1 = new_tp.copy()
new_adj_close = adj_close.copy()
TS = []                                     # TradingSingals list
ii = []                                     # index in adj_close

# 找出 local_min 跟 local_max 在 close 索引
for i in new_tp1:
    if i in new_adj_close:
        index = new_adj_close.index(i)
        ii.append(index)

for x in range(len(ii)-1):

    temp = new_adj_close[ii[x+1]] - new_adj_close[ii[x]]

    if temp > 0:
        max_Xt = new_adj_close[ii[x+1]]
        min_Xp = new_adj_close[ii[x]]

        for i in range(ii[x], ii[x+1]):
            tradingSingals = (2 * (new_adj_close[i]) - min_Xp - max_Xt) / (min_Xp - max_Xt)
            TS.append(tradingSingals)

    else:
        min_Xp = new_adj_close[ii[x+1]]
        max_Xt = new_adj_close[ii[x]]

        for i in range(ii[x], ii[x+1]):
            tradingSingals = (2 * (new_adj_close[i]) - min_Xp - max_Xt) / (min_Xp - max_Xt)
            TS.append(tradingSingals)

print('min TS： ' + str(min(TS)))
print('max TS： ' + str(max(TS)))

# plt.figure()
# plt.plot(TS)
# plt.legend(loc='upper left')
# plt.show()

# 中文版交易訊號
# 買入B：1、賣出S：-1、持有H：0
r = 0.2
targetList = []
for x in range(len(ii)-1):

    temp = new_adj_close[ii[x+1]] - new_adj_close[ii[x]]

    if temp > 0:
        Xh = new_adj_close[ii[x+1]]
        Xl = new_adj_close[ii[x]]

        for i in range(ii[x], ii[x+1]):
            if Xl <= new_adj_close[i] <= Xl + r*(Xh - Xl):
                targetList.append(1)
            elif Xh - r*(Xh-Xl) <= new_adj_close[i] <= Xh:
                targetList.append(-1)
            else:
                targetList.append(0)
    else:
        Xl = new_adj_close[ii[x+1]]
        Xh = new_adj_close[ii[x]]

        for i in range(ii[x], ii[x+1]):
            if Xl <= new_adj_close[i] <= Xl + r*(Xh - Xl):
                targetList.append(1)
            elif Xh - r*(Xh-Xl) <= new_adj_close[i] <= Xh:
                targetList.append(-1)
            else:
                targetList.append(0)

# plt.figure()
# plt.plot(targetList)
# plt.legend(loc='upper left')
# plt.show()

# save_to_csv
saveIndex = list(range(ii[0], ii[-1]))
saveDate = Date[ii[0]:ii[-1]]
saveClosePrice = adj_close[ii[0]:ii[-1]]

saveTarget = list(zip_longest(saveIndex, saveDate, saveClosePrice, targetList, TS))

title = 'target' + str(threshold)

with open(str(title) + '.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'Date', 'Close', 'Strategy', 'TradingSingals'])

    for items in saveTarget:
        writer.writerow(items)
#
# for num in new_tp:
#     if num in adj_close:
#         ntp_idx.append(adj_close.index(num))
#
# new_tp2 = dict(zip(ntp_idx, new_tp))
#
# # plot figure1
# plt.figure()
# plt.plot(adj_close, label='Original')
# plt.scatter(local_min_idx, local_min, label='local_min', marker='^')
# plt.scatter(local_max_idx, local_max, label='local_max', marker='x')
# plt.legend(loc='upper left')
# plt.show()
#
# # plot figure2
# plt.figure()
# plt.plot(adj_close, label='Original')
# new_tp_y = []
# new_tp_idx = []
# for key in sorted(new_tp2.keys()):
#     new_tp_y.append(new_tp2[key])
#     new_tp_idx.append(key)
# plt.plot(new_tp_idx, new_tp_y, label='Segmentation')
# plt.legend(loc='upper left')
# plt.show()