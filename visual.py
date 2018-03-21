# coding: utf-8
import dateutil, random
import pylab as pl
from datetime import datetime, timedelta
import matplotlib as mpl
from matplotlib import font_manager

# 交易策略结果的正式可视化
fig = pl.figure()
ax = pl.gca()
zh_font = font_manager.FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=14)
dates = [datetime(2015, 3, 11)] + [datetime.strptime(i, '%Y-%m-%d') for i in d[1::2]]
values_pred = money_nona
plot1 = pl.plot_date(pl.date2num(dates), values_pred, linestyle='-', label=u'策略交易')
date_format = mpl.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
values_real = [1] + [i / price_test[0] for i in trade_price]
plot2 = pl.plot_date(pl.date2num(dates), values_real, linestyle='-', color='red', label=u'买入并持有')
pl.title(u'累计总资产变动情况(2015-03-11至2017-05-24)', fontsize=20, fontproperties=zh_font)
xtext = pl.xlabel(u'交易时间', fontsize=15, fontproperties=zh_font)
ytext = pl.ylabel(u'累计总资产', fontsize=15, fontproperties=zh_font)
handles, labels = ax.get_legend_handles_labels()
pl.legend(handles[::-1], labels[::-1], loc=3, prop=zh_font)
pl.savefig('E:/Data Mining And Machine Learning/dataset/stocks/1503-1705cum_profit_comparision.png')

####################  将15-03-11至15-08-30这段时间去掉 ####################### 2015-08-31 至 2017-05-24
l_test2 = final_test['2']
l_test2 = pd.Series([i for i in l_test2[120:]])
price_test2 = final_test[(final_test['0'] >= '2015-08-31') & (final_test['0'] <= '2017-05-24')]['1']
price_test2 = pd.Series([i for i in price_test2])
date_test2 = final_test[(final_test['0'] >= '2015-08-31') & (final_test['0'] <= '2017-05-24')]['0']
date_test2 = pd.Series([i for i in date_test2])
print 'cum_profit:', cum_profit(l_test2, price_test2)
money2 = total_money(l_test2, price_test2)
money_nona2 = pd.Series([i for i in money2.dropna()])
print 'max_retracement:', max_retracement(money_nona2)
print 'trade time:', trade_date(l_test2, date_test2)
print 'max_count_down:', max_count_down(money_nona2)

d2 = trade_date(l_test2, date_test2)
trade_price2 = []
trade_index2 = []
for i in d2[1::2]:
    trade_index2.append(date_test2.tolist().index(i))
    trade_price2.append(price_test2[date_test2.tolist().index(i)])
print trade_index2
print trade_price2

# 可视化
fig = pl.figure()
ax = pl.gca()
zh_font = font_manager.FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=14) # 设定中文字体
dates2 = [datetime(2015, 8, 31)] + [datetime.strptime(i, '%Y-%m-%d') for i in d2[1::2]]
values_pred2 = money_nona2
plot3 = pl.plot_date(pl.date2num(dates2), values_pred2, linestyle='-', label=u'策略交易')
date_format = mpl.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
values_real2 = [1] + [i / price_test2[0] for i in trade_price2]
plot4 = pl.plot_date(pl.date2num(dates2), values_real2, linestyle='-', color='red', label=u'买入并持有')
pl.title(u'累计总资产变动情况(2015-08-31至2017-05-24)', fontsize=20, fontproperties=zh_font)
xtext2 = pl.xlabel(u'交易时间', fontsize=15, fontproperties=zh_font)
ytext2 = pl.ylabel(u'累计总资产', fontsize=15, fontproperties=zh_font)
handles2, labels2 = ax.get_legend_handles_labels()
pl.legend(handles2[::-1], labels2[::-1], loc=2, prop=zh_font)
pl.savefig('E:/Data Mining And Machine Learning/dataset/stocks/add_index_1509-1705cum_profit_comparision.png')

# 2015-03-11 至 2017-05-24最大回撤及最大连续亏损次数
sw = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv')
price1 = sw[(sw.date >= '2015-03-11') & (sw.date <= '2017-05-24')]['close']
price1 = pd.Series([i for i in price1])
print max_retracement(price1)
print max_count_down(price1)

# 2015-08-31 至 2017-05-24最大回撤及最大连续亏损次数
price2 = sw[(sw.date >= '2015-08-31') & (sw.date <= '2017-05-24')]['close']
price2 = pd.Series([i for i in price2])
print max_retracement(price2)
print max_count_down(price2)

# 训练集数据可视化
fig3 = pl.figure()
zh_font = font_manager.FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=14)
ax = pl.gca()
dates = sw[(sw.date >= '2011-03-24') & (sw.date <= '2015-03-10')]['date']
dates = [datetime.strptime(i, '%Y-%m-%d') for i in dates]
dates = np.array(dates)
pl.plot_date(dates, sw[(sw.date >= '2011-03-24') & (sw.date <= '2015-03-10')]['close'], 'r-')
date_format = mpl.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)
fig3.autofmt_xdate()
pl.title(u'2011-03-24至2015-03-10期间 上证指数', fontsize=22, fontproperties=zh_font)
pl.xlabel(u'时间', fontsize=17, fontproperties=zh_font)
pl.ylabel(u'指数值ֵ', fontsize=17, fontproperties=zh_font)
pl.savefig('E:/Data Mining And Machine Learning/dataset/stocks/1103-1503_index.png')

# 测试集数据可视化
fig2 = pl.figure()
zh_font = font_manager.FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=14)
ax = pl.gca()
dates = sw[(sw.date >= '2015-03-11') & (sw.date <= '2017-05-24')]['date']
dates = [datetime.strptime(i, '%Y-%m-%d') for i in dates]
dates = np.array(dates)
pl.plot_date(dates, sw[(sw.date >= '2015-03-11') & (sw.date <= '2017-05-24')]['close'], 'r-')
date_format = mpl.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)
fig2.autofmt_xdate()
pl.title(u'2015-03-11至2017-05-24期间 上证指数', fontsize=22, fontproperties=zh_font)
pl.xlabel(u'时间', fontsize=17, fontproperties=zh_font)
pl.ylabel(u'ָ指数值ֵ', fontsize=17, fontproperties=zh_font)
pl.savefig('E:/Data Mining And Machine Learning/dataset/stocks/1503-1705_index.png')

# Moreover, ......
