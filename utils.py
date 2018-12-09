# coding: utf-8
import numpy as np
import pandas as pd
import tushare as ts

sk = ts.get_k_data(code='000001', index=True, start='2008-05-12', end='2017-06-10', ktype='D', retry_count=3)
del sk['code']
sk.to_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv', index=None)
sk = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv')

# 移动平均(价格)
def p_ma(p, n):
    m = p.shape[0]
    p_mavalue = pd.Series([0.0]*m)
    for i in range(m-n+1):
        p_tem = p[i:i+n]
        p_mavalue[i+n-1] = np.mean(p_tem)
    return p_mavalue

sk['ma5'] = p_ma(sk['close'], 5)
sk['ma10'] = p_ma(sk['close'], 10)
sk['ma20'] = p_ma(sk['close'], 20)

# 移动平均(成交量)
def v_ma(v, n):
    m = v.shape[0]
    v_mavalue = pd.Series([0.0]*m)
    for i in range(m-n+1):
        v_tem = v[i:i+n]
        v_mavalue[i+n-1] = np.mean(v_tem)
    return v_mavalue

sk['v_ma5'] = v_ma(sk['volume'], 5)
sk['v_ma10'] = v_ma(sk['volume'], 10)
sk['v_ma20'] = v_ma(sk['volume'], 20)

# 价格变动
def price_change(s):
    m = s.shape[0]
    pc = pd.Series([0.0]*m)
    for i in range(m-1):
        pc[i+1] = s[i+1] - s[i]
    return pc

sk['price_change'] = price_change(sk['close'])

# 涨跌幅
def price_change_rate(s):
    m = s.shape[0]
    pcr = pd.Series([0.0]*m)
    for i in range(m-1):
        pcr[i+1] = float(s[i+1]-s[i])/s[i] *100
    return pcr

sk['p_change'] = price_change_rate(sk['close'])

# 相对于过去10天的收益率
new = sk[['date', 'close']]
for i in range(1, 11, 1):
    column_name = 'close_%s' % i
    # 按列拼接后得到滞后i个交易日后的close数据(Series)
    temp = pd.concat([pd.Series([0] * i), new['close'][:-i]], axis=0)
    new.insert(new.shape[1], column_name, temp.values)

for i in range(1, 11, 1):
    column_name = 'return_rate_%s' %i
    temp = new.apply(lambda s: (s[1]/s[i+1] - 1)*100 if s[i+1]>0 else np.nan, axis=1) # 按行进行lambda函数运算
    new.insert(new.shape[1], column_name, temp.values)

sk = pd.merge(sk, new, on=['date','close'], how='left')

for i in range(1, 11, 1):
    column_name = 'close_%s' %i
    del sk[column_name]

# 平均价格
new = sk[['high','close','low']]
temp = new.apply(lambda s: (s[0]+s[1]+s[2])/3, axis=1)
new.insert(new.shape[1], 'ave_price', temp.values)

# 未来十天相对于当天的收益率
for j in range(1, 11, 1):
    column_name = 'ave_price_%s' % j
    # 按列拼接后得到未来i个交易日的平均价格数据
    future = pd.concat([new['ave_price'][j:], pd.Series([0]*j)], axis=0)
    new.insert(new.shape[1], column_name, future.values)

for j in range(1, 11, 1):
    column_name = 'future_rate_%s' % j
    temp = new.apply(lambda s: (s[j+3] / s[1] - 1) * 100 if s[j+3] > 0 else np.nan, axis=1)
    new.insert(new.shape[1], column_name, temp.values)

for j in range(1, 11, 1):
    column_name = 'ave_price_%s' % j
    del new[column_name]

sk = pd.merge(sk, new, on=['high','close','low'], how='left')


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
    return dif(s, start, end) - dea(s, start, end)

sk['macd'] = macd(sk['close'], 12, 26)

# KDJ  随机指标
def rsv(s,n):
    m = s.shape[0]
    r = pd.Series( [0]*m )
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        r[i+n-1] = 100 * float(s_tem[i+n-1]-min(s_tem)) / (max(s_tem)-min(s_tem))
    return r

r = rsv(sk['close'], 10)

def k(s, n, r):
    m = s.shape[0]
    kvalue = pd.Series( [0.0]*m )
    kvalue[0] = 50
    rseries = r
    for i in range(m-1):
        kvalue[i+1] = float(1)/3*rseries[i+1] + float(2)/3*kvalue[i]
    return kvalue

k = k(sk['close'], 10, r)

def d(s, n, k):
    m = s.shape[0]
    dvalue = pd.Series( [0.0]*m )
    dvalue[0] = 50.0
    kseries = k
    for i in range(m-1):
        dvalue[i+1] = float(1)/3*kseries[i+1] + float(2)/3*dvalue[i]
    return dvalue

d = d(sk['close'], 10, k)

j = 3*k - 2*d

sk['k_value'] = k
sk['d_value'] = d
sk['j_value'] = j


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

sk['cci'] = cci(sk['close'], 12, sk['high'], sk['low'])


# RSI 相对强弱指标  n取9或14
def rsi(piord, n):
    m = piord.shape[0]
    a = pd.Series([0.0]*m)
    b = pd.Series([0.0]*m)
    rs = pd.Series([0.0]*m)
    for i in range(m-n+1):
        p_tem = piord[i:i+n]
        a[i+n-1] = p_tem.apply(lambda x: x if x>=0 else 0).sum()
        b[i+n-1] = p_tem.apply(lambda x: np.abs(x) if x<0 else 0).sum()
        rs[i+n-1] = a[i+n-1]/( a[i+n-1]+b[i+n-1] ) * 100
    return rs

sk['rsi_14'] = rsi(sk['p_change'], 14)


#  ROC 变动速率
def roc(s, n):  # n天前
    m = s.shape[0]
    ro = pd.Series([0.0]*m)
    for i in range(m-n):
        ro[i+n] = s[i+n] / s[i]
    return ro

sk['roc'] = roc(sk['close'], 10)

# OBV
def obv(h, s, l, v):
    fz = 2*s-h-l
    return (fz/(h-l) * v).replace(np.NAN,0)

sk['obv'] = obv(sk['high'], sk['close'], sk['low'], sk['volume'])

# BULL
def bullsd(s, n):
    m = s.shape[0]
    bs = pd.Series([0.0]*m)
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        bs[i+n-1] = np.sum( (np.mean(s_tem) -s_tem)**2 )/n
        bs[i+n-1] = np.sqrt( bs[i+n-1] )
    return bs

bsdvalue = bullsd(sk['close'], 10)

def bullmean(s, n):
    m = s.shape[0]
    bm = pd.Series([0.0]*m)
    for i in range(m-n+1):
        s_tem = s[i:i+n]
        bm[i+n-1] = np.mean(s_tem)
    return bm
bmean = bullmean(sk['close'], 10)
bupper = bmean + 2*bsdvalue
blowwer = bmean - 2*bsdvalue

sk['bull_mid'] = bmean
sk['bull_upper'] = bupper
sk['bull_lowwer'] = blowwer


# n日K线值
data = sk[['open', 'close', 'high', 'low']]
def nk_line(data, n):
    temp = data[['close']]
    column_name = ['open_%s' %n, 'high_%s' %n, 'low_%s' %n]
    temp.loc[:,column_name[0]] = data['open'].shift(n)
    temp.loc[:,column_name[1]] = data['high'].shift(n)
    temp.loc[:,column_name[2]] = data['low'].shift(n)
    return temp.apply( lambda x: (x[0]-x[1])/(x[2]-x[3]), axis=1)

sk['0kline'] = nk_line(data, 0)
sk['3kline'] = nk_line(data, 3)
sk['6kline'] = nk_line(data, 6)

# n日乖离线率(BIAS)
def bias(close, n):
    for i in range(1, n, 1):
        column_name = 'close_%s' % i
        temp = close['close'].shift(i)
        close.insert(close.shape[1], column_name, temp.values)
    ave = close.apply(lambda x: np.mean(x), axis=1)
    posi = close.shape[1]
    close.insert(posi, 'close_ave', ave.values)
    return close.apply(lambda x: (x[0]-x[posi])/x[posi]*100, axis=1)

sk['bias_6'] = bias(sk[['close']], 6)
sk['bias_10'] = bias(sk[['close']], 10)


# label
# 计算未来10个交易日相对于当前交易日的总条件收益率
def future(v):
    s = 0
    for i in range(10):
        if (v[i]>1) | (v[i]<-1):
            s += v[i]
    return s

column_name =[]
for i in range(1, 11, 1):
    temp_name = 'future_rate_%s' %i
    column_name.append(temp_name)

new = sk[column_name]
temp = new.apply(future, axis=1)
new.insert(new.shape[1], 'future_cond_sum', temp.values)

# 判断label的类型
def trade_or_not(x):
    if x>15:
        return 1
    elif x<-15:
        return -1
    else:
        return 0

temp = new['future_cond_sum'].apply(trade_or_not)
new.insert(new.shape[1], 'label', temp.values)

sk = pd.merge(sk, new, on=column_name, how='left')

# 将特征工程后的数据存入csv文件
sk.to_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv', index=None)


sw = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10.csv')

# 总资产的变动
def total_money(l, price, fee):
    """
    Args:
        l: 预测交易信号
        price: 标的价格
        fee: 交易费率
    """
    m = l.shape[0]
    teml = l.tolist()
    start = teml.index(max(teml))  # 寻找首次交易对应的索引
    money = pd.Series([np.NAN] * m)  # 资金的积累
    money_start = 100
    for i in range(start+1):
        money[i] = money_start
    s = start
    money_exchange = money_start / price[start] * (1-fee)
    for i in np.arange(start, m - 1, 1):
        if l[i + 1] * l[s] < 0:
            if l[i + 1] < 0:
                # 卖出
                money_exchange *= price[i + 1]
                money[i + 1] = money_exchange * (1-fee)
            else:
                # 买入
                money_exchange /= price[i + 1]
                money[i+1] = money[i] * (1-fee)
            s = i + 1
        else:
            if l[s] < 0:
                money[i+1] = money[i]
            else:
                money[i+1] = money[i] / price[i] * price[i+1]
    cum_profit = money[m-1]/money[0] - 1.0
    year_profit = (1+cum_profit)**(250.0/m) - 1.0
    return (money, cum_profit, year_profit)


# 最大回撤
def max_retracement(money_nona):
    """
    Args: money_nona: 累计总资产
    """
    n = money_nona.shape[0]
    if n==1:
        return 0
    else:
        retrace = pd.Series([0.0]*(n-1))
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
