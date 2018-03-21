# coding: utf-8
import numpy as np
import pandas as pd
import tushare as ts

sw = ts.get_k_data(code='000001', index=True, start='2008-05-12', end='2017-06-10', ktype='D', retry_count=3)
del sw['code']
sw.to_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv', index=None)
sw = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv')

# 移动平均（价格）
def ma(s, n):
    m = s.shape[0]
    mavalue = pd.Series([0.0]*m)
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        mavalue[i+n-1] = np.mean(s_tem)
    return mavalue

sw.loc[:,'ma5'] = ma(sw['close'],5)
sw.loc[:,'ma10'] = ma(sw['close'],10)
sw.loc[:,'ma20'] = ma(sw['close'],20)

# 移动平均（成交量）
def v_ma(v, n):
    m = v.shape[0]
    v_mavalue = pd.Series([0.0]*m)
    for i in range(m-n+1):
        v_tem = v[i:i+n]
        v_mavalue[i+n-1] = np.mean(v_tem)
    return v_mavalue

sw.loc[:,'v_ma5'] = v_ma(sw['volume'],5)
sw.loc[:,'v_ma10'] = v_ma(sw['volume'],10)
sw.loc[:,'v_ma20'] = v_ma(sw['volume'],20)

# 价格变动
def price_change(s):
    m = s.shape[0]
    pc = pd.Series([0.0]*m)
    for i in range(m-1):
        pc[i+1] = s[i+1] - s[i]
    return pc

sw.loc[:,'price_change'] = price_change(sw['close'])

# 涨跌幅
def price_change_rate(s):
    m = s.shape[0]
    pcr = pd.Series([0.0]*m)
    for i in range(m-1):
        pcr[i+1] = float(s[i+1]-s[i])/s[i] *100
    return pcr

sw.loc[:,'p_change'] = price_change_rate(sw['close'])

# 相对于过去10天的收益率
temp = sw[['date', 'close']]
for i in np.arange(1,11,1):
    column_name = 'close_%s' %i
    temp_temp = temp['close'].copy()[:-i]
    temp_temp = pd.concat([pd.Series([0]*i),temp_temp], axis=0)
    lt = [i for i in temp_temp]
    temp.loc[:,column_name] = pd.Series(lt)

for i in np.arange(1,11,1):
    column_name = 'return_rate_%s' %i
    temp.loc[:,column_name] = temp.apply(lambda s: (s[1]/s[i+1] - 1)*100 if s[i+1]>0 else np.nan, axis=1)

sw = pd.merge(sw,temp, on=['date','close'], how='left')

for i in np.arange(1,11,1):
    column_name = 'close_%s' %i
    del sw[column_name]

# 平均价格
t = sw[['high','close','low']]
t.loc[:,'ave_price'] = t.apply(lambda s: (s[0]+s[1]+s[2])/3, axis=1)

# 未来十天的平均价格
for j in np.arange(1,11,1):
    column_name = 'ave_price_%s' %j
    future = t['ave_price'].copy()[j:]
    future = pd.concat([ future, pd.Series([0]*j) ], axis=0)
    lt = [i for i in future]
    t.loc[:,column_name] = pd.Series(lt)


# 未来十天的收益率
for j in np.arange(1,11,1):
    column_name = 'future_rate_%s' %j
    t.loc[:,column_name] = t.apply(lambda s: (s[j+3] / s[1] - 1) * 100 if s[j+3] > 0 else np.nan, axis=1)

for j in np.arange(1,11,1):
    column_name = 'ave_price_%s' %j
    del t[column_name]

sw = pd.merge(sw,t, on=['high','close','low'], how='left')


######### 技术指标 ##########

# MACD  平滑异动移动平均线
def ema(s, n):  # s is the price of close ( type: series )
    m = s.shape[0]
    e = np.zeros((1,m))
    e[0,0] = s[0]
    for i in range(m-1):
        e[0,i+1] = float(2)/(n+1)*s[i+1] + float(n-1)/(n+1)*e[0,i]
    return pd.Series( [ i for i in e[0] ] )

def dif(s, start, end):
    return ema(s,start) - ema(s,end)

def dea(s, start, end):
    m = s.shape[0]
    d = np.zeros((1,m))
    d[0,0] = dif(s,start,end)[0]
    for i in range(m-1):
        d[0,i+1] = float(2)/10*dif(s,start,end)[i+1] + float(8)/10*d[0,i]
    return pd.Series( [ i for i in d[0] ] )

def macd(s, start, end):
    return dif(s,start,end)-dea(s,start,end)

sw.loc[:,'macd'] = macd(sw['close'],12,26)

# KDJ  随机指标
def rsv(s,n):
    m = s.shape[0]
    r = pd.Series( [0]*m )
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        r[i+n-1] = 100 * float(s_tem[i+n-1]-min(s_tem)) / (max(s_tem)-min(s_tem))
    return r

r = rsv(sw['close'], 10)

def k(s, n, r):
    m = s.shape[0]
    kvalue = pd.Series( [0.0]*m )
    kvalue[0] = 50
    rseries = r
    for i in range(m-1):
        kvalue[i+1] = float(1)/3*rseries[i+1] + float(2)/3*kvalue[i]
    return kvalue

k = k(sw['close'], 10, r)

def d(s, n, k):
    m = s.shape[0]
    dvalue = pd.Series( [0.0]*m )
    dvalue[0] = 50.0
    kseries = k
    for i in range(m-1):
        dvalue[i+1] = float(1)/3*kseries[i+1] + float(2)/3*dvalue[i]
    return dvalue

d = d(sw['close'],10,k)

j = 3*k - 2*d

sw.loc[:,'k_value'] = k
sw.loc[:,'d_value'] = d
sw.loc[:,'j_value'] = j


#  CCI 顺势指标
def cci(s, n, h, l):
    m = s.shape[0]
    tp = (h + s + l) / 3
    ma = np.zeros((1,m))
    md = np.zeros((1,m))
    c = np.zeros((1,m))
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        ma[0,i+n-1] = np.mean((s_tem))
        md[0,i+n-1] = np.sum( np.abs(ma[0,i+n-1]-s_tem) )
        c[0,i+n-1] = 1/0.015*( tp[i+n-1]-ma[0,i+n-1] ) / md[0,i+n-1]
    return pd.Series( [ i for i in c[0] ] )

sw.loc[:,'cci'] = cci(sw['close'], 12, sw['high'],sw['low'])


# RSI 相对强弱指标  n取9或14
def rsi(piord, n):
    m = piord.shape[0]
    a = pd.Series([0.0]*m)
    b = pd.Series([0.0]*m)
    rs = pd.Series([0.0]*m)
    for i in range(m-n+1):
        p_tem = piord[i:i+n]
        a[i+n-1] = p_tem.apply(lambda x: x if x>=0 else 0).sum()
        b[i+n-1] = p_tem.apply(lambda x: np.abs(x)
        if x<0 else 0).sum()
           rs[i+n-1] = a[i+n-1]/( a[i+n-1]+b[i+n-1] ) * 100
    return rs

sw.loc[:,'rsi'] = rsi(sw['p_change'],14)

#  ROC 变动速率
def roc(s,n):  # n天前
    m = s.shape[0]
    ro = pd.Series([0.0]*m)
    for i in range(m-n):
        ro[i+n] = s[i+n] / s[i]
    return ro

sw.loc[:,'roc'] = roc(sw['close'],10)

# OBV
def obv(h, s, l, v):
    fz = 2*s-h-l
    return (fz/(h-l) * v).replace(np.NAN,0)

sw.loc[:,'obv'] = obv(sw['high'],sw['close'],sw['low'],sw['volume'])

# BULL
def bullsd(s, n):
    m = s.shape[0]
    bs = pd.Series([0.0]*m)
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        bs[i+n-1] = np.sum( (np.mean(s_tem) -s_tem)**2 )/n
        bs[i+n-1] = np.sqrt( bs[i+n-1] )
    return bs

bsdvalue = bullsd(sw['close'],10)

def bullmean(s, n):
    m = s.shape[0]
    bm = pd.Series([0.0]*m)
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        bm[i+n-1] = np.mean(s_tem)
    return bm
bmean = bullmean(sw['close'],10)
bupper = bmean + 2*bsdvalue
blowwer = bmean - 2*bsdvalue

sw.loc[:,'bull_mid'] = bmean
sw.loc[:,'bull_upper'] = bupper
sw.loc[:,'bull_lowwer'] = blowwer


# label
def future(v):
    s = 0
    for i in range(10):
        if (v[i]>1) | (v[i]<-1):
            s = s+v[i]
    return s

column_name =[]
for i in np.arange(1,11,1):
    temp_name = 'future_rate_%s' %i
    column_name.append(temp_name)

t2 = sw[column_name]
t2.loc[:,'future_cond_sum'] = t2.apply(future, axis=1)

def trade_or_not(x):
    if x>15:
        return 1
    elif x<-15:
        return -1
    else:
        return 0
t2.loc[:,'label'] = t2['future_cond_sum'].apply(trade_or_not)

sw = pd.merge(sw,t2, on=column_name, how='left')

sw.to_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv', index=None)
sw = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv')


from sklearn import preprocessing
from sklearn.svm import SVC
sw = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv')

# n日K线值
data = sw[['open','close','high','low']]
def nk_line(data,n):
    temp = data[['close']]
    column_name = ['open_%s' %n, 'high_%s' %n, 'low_%s' %n]
    temp.loc[:,column_name[0]] = data['open'].shift(n)
    temp.loc[:,column_name[1]] = data['high'].shift(n)
    temp.loc[:,column_name[2]] = data['low'].shift(n)
    return temp.apply( lambda x: (x[0]-x[1])/(x[2]-x[3]) ,axis=1)

sw.loc[:,'0kline'] = nk_line(data,0)
sw.loc[:,'3kline'] = nk_line(data,3)
sw.loc[:,'6kline'] = nk_line(data,6)

# n日乖离线率（BIAS）
def bias(close,n):
    temp = close
    temp.loc[:,'close_ave'] = 0
    for i in np.arange(1,n,1):
        temp['close_ave'] += temp['close'].shift(i)
    temp.loc[:,'close_ave'] = (temp['close_ave']+temp['close']) / n
    return temp.apply(lambda x: (x[0]-x[1])/x[1]*100 ,axis=1)

sw.loc[:,'bias_6'] = bias(sw[['close']],6)
sw.loc[:,'bias_10'] = bias(sw[['close']],10)


# 总资产，每进行一次卖出交易便清算一次现有资产
def total_money(l,price):
    m = l.shape[0]
    teml = l.tolist()
    start = teml.index(max(teml))  # 寻找首次交易对应的索引
    money = pd.Series([np.NAN] * m)  # 资金的积累
    money_start = 1
    money[start] = money_start
    s = start
    money_exchange = money_start / price[start]
    for i in np.arange(start, m - 1, 1):
        while l[i + 1] * l[s] < 0:
            money_exchange = price[i + 1] ** -l[i + 1] * money_exchange
            if l[i + 1] < 0:
                money[i + 1] = money_exchange
            s = i + 1
    return money

# 添加手续费后总资产
def total_money2(l,price):
    m = l.shape[0]
    teml = l.tolist()
    start = teml.index(max(teml))
    money = pd.Series([np.NAN] * m)
    money_start = 1
    money[start] = money_start
    s = start
    money_exchange = money_start / price[start] * (1-3.0/1000)
    for i in np.arange(start, m - 1, 1):
        while l[i + 1] * l[s] < 0:
            money_exchange = price[i + 1] ** -l[i + 1] * money_exchange * (1-3.0/1000)
            if l[i + 1] < 0:
                money[i + 1] = money_exchange
            s = i + 1
    return money


# 累计收益率
def cum_profit(l,price):
    m = l.shape[0]
    teml = l.tolist()
    start = teml.index(max(teml))
    s = start
    money_exchange = 1 / price[start]
    for i in np.arange(start, m - 1, 1):
        while l[i + 1] * l[s] < 0:
            money_exchange = price[i + 1] ** -l[i + 1] * money_exchange
            s = i + 1
    if l[s]>0:
        return money_exchange*price[s] - 1
    else:
        return money_exchange - 1

# 最大回撤
def max_retracement(money_nona):
    n = money_nona.shape[0]
    if n==1:
        return 0
    else:
        retrace = pd.Series( [0.0]*(n-1) )
        for t in range(n-1):
            money_tem = money_nona[:t+2]
            number = money_tem.shape[0]
            loss = pd.Series( [0.0]*(number-1) )
            for i in range(number-1):
                loss[i] = float(money_tem[number-1]) / money_tem[i] - 1
            retrace[t] = min(loss)
        return min(retrace)

# 每次交易的日期
def trade_date(l, date):
    m = l.shape[0]
    tem_l = l.tolist()
    start = tem_l.index(max(tem_l))  # 寻找首次交易对应的索引
    trade_d = [date[start]]
    s = start
    for i in np.arange(start, m - 1, 1):
        while l[i + 1] * l[s] < 0:
            trade_d.append(date[i + 1])
            s = i + 1
    if l[s] > 0:
        return trade_d[:-1]
    else:
        return trade_d

# 最大连续下跌次数
def max_count_down(money_nona):
    n = money_nona.shape[0]
    sign = pd.Series([0.0] * (n - 1))
    for i in np.arange(0, n - 1, 1):
        sign[i] = money_nona[i + 1] - money_nona[i]
    sign[n - 1] = -sign[n - 2]
    s, count = 0, 1
    c_list = []
    for i in np.arange(0, n - 1, 1):
        if sign[i + 1] * sign[s] > 0:
            count += 1
        else:
            s = i + 1
            c_list.append(count)
            count = 1
    if sign[0] > 0:
        return max(c_list[1::2])
    else:
        return max(c_list[0::2])

# 最优参数的对应位置
def location(id):
    g = id/11
    c = np.mod(id,11)
    return g,c

# 年化收益率
def year_profit(cp, t):
    return (1+cp)**(250.0/t)-1
y_pt = []


data_try = sw[11:-10] # 2188
data_try = data_try[data_try.date>='2011-03-24'] # 1500
data_try.to_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10_data_try2.csv', index=None)