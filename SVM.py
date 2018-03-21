# coding: utf-8
from utils import *
import numpy as np
import pandas as pd

data_try = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_new_day10_data_try2.csv')

del data_try['future_cond_sum']
del data_try['date']
for i in np.arange(1, 11, 1):
    column_name = 'future_rate_%s' % i
    del data_try[column_name]
## (1500-240-60)/60.0 = 20.0
time = 12
vali_index = []
for i in np.arange(0, time, 1):
    vali_index.append(60 * i)
print vali_index

tuned_parameters = {'gamma': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2, 2 ** 3],
                    'C': [2 ** -5, 2 ** -3, 2 ** -1, 2, 2 ** 3, 2 ** 5, 2 ** 7, 2 ** 9, 2 ** 11, 2 ** 13, 2 ** 15]}
pt = np.zeros((tuned_parameters['gamma'].__len__(), tuned_parameters['C'].__len__()))  # 9*11
retrace = np.zeros((tuned_parameters['gamma'].__len__(), tuned_parameters['C'].__len__()))  # 9*11
for j in vali_index:
    cv_train = data_try.loc[j:j + 239, :]
    cv_vali = data_try.loc[j + 240:j + 299, :]
    cv_train_y = cv_train['label']
    cv_train_y = np.array(cv_train_y)
    cv_train_y = np.array([int(s) for s in cv_train_y])
    cv_train_x = cv_train.copy()
    del cv_train_x['label']
    cv_train_x = np.array(cv_train_x)
    cv_train_x_scaled = preprocessing.scale(cv_train_x)

    cv_vali_x = cv_vali.copy()
    del cv_vali_x['label']
    cv_vali_x = np.array(cv_vali_x)
    cv_vali_x_scaled = preprocessing.scale(cv_vali_x)

    for g in range(tuned_parameters['gamma'].__len__()):
        for c in range(tuned_parameters['C'].__len__()):
            clf = SVC(decision_function_shape='ovo', C=tuned_parameters['C'][c], gamma=tuned_parameters['gamma'][g],
                      kernel='rbf')
            clf.fit(cv_train_x_scaled, cv_train_y)
            l_tem = clf.predict(cv_vali_x_scaled)
            l = pd.Series([i for i in l_tem])
            price = pd.Series([s for s in cv_vali['close']])
            pt[g, c] += cum_profit(l, price)
            money = total_money(l, price)
            money_nona = pd.Series([i for i in money.dropna()])
            if max_retracement(money_nona) <= 0:
                retrace[g, c] += max_retracement(money_nona)
            else:
                retrace[g, c] += 0

number = vali_index.index(vali_index[-1])
pt_ave = pt / (number + 1.0)
retrace_ave = retrace / (number + 1.0)

pt_ave_list = [pt_ave[i][j] for i in range(9) for j in range(11)]
pt_ave_list_sorted = sorted(pt_ave_list, reverse=True)[:5]
print pt_ave_list_sorted

index = []
for i in pt_ave_list_sorted:
    index.append(pt_ave_list.index(i))
print index

retrace_ave_list = []
index_trans = []
for k in index:
    g = location(k)[0]
    c = location(k)[1]
    index_trans.append((g, c))
    retrace_ave_list.append(retrace_ave[g, c])
print index_trans
print retrace_ave_list

train = data_try.loc[time * 60:time * 60 + 239, :]
test = data_try.loc[time * 60 + 240:time * 60 + 299, :]

train_y = train['label']
train_y = np.array(train_y)
train_y = np.array([int(s) for s in train_y])
train_x = train.copy()
del train_x['label']
train_x = np.array(train_x)
train_x_scaled = preprocessing.scale(train_x)

test_x = test.copy()
del test_x['label']
test_x = np.array(test_x)
test_x_scaled = preprocessing.scale(test_x)

clf = SVC(decision_function_shape='ovo', gamma=tuned_parameters['gamma'][2], C=tuned_parameters['C'][9], kernel='rbf')
clf.fit(train_x_scaled, train_y)
l_tem = clf.predict(test_x_scaled)
l = pd.Series([i for i in l_tem])
price = pd.Series([s for s in test['close']])
print cum_profit(l, price)
money = total_money(l, price)
money_nona = pd.Series([i for i in money.dropna()])
print 'max_retracement:', max_retracement(money_nona)

############# 将 2015-03-11 至 2017-05-24 预测结果整合在一起，一次性回测结果：
l_test = pd.concat([l1, l2, l3, l4, l5, l6, l7, l8, l9], axis=0)
l_test = pd.Series([i for i in l_test])
price_test = sw[(sw.date >= '2015-03-11') & (sw.date <= '2017-05-24')]['close']
price_test = pd.Series([i for i in price_test])
date_test = sw[(sw.date >= '2015-03-11') & (sw.date <= '2017-05-24')]['date']
date_test = pd.Series([i for i in date_test])
final_test = pd.concat((date_test, price_test, l_test), axis=1)
final_test = pd.DataFrame(final_test)
final_test.to_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_final_test2.csv', index=None)

final_test = pd.read_csv('E:\Data Mining And Machine Learning\dataset\stocks\index_final_test2.csv')
l_test = final_test['2']
price_test = final_test['1']
date_test = final_test['0']
print 'cum_profit:', cum_profit(l_test, price_test)
money = total_money(l_test, price_test)
money_nona = pd.Series([i for i in money.dropna()])
print 'max_retracement:', max_retracement(money_nona)
print 'trade time:', trade_date(l_test, date_test)
print 'max_count_down:', max_count_down(money_nona)

d = trade_date(l_test, date_test)
trade_price = []
trade_index = []
for i in d[1::2]:
    trade_index.append(final_test['0'].tolist().index(i))
    trade_price.append(final_test.loc[final_test['0'].tolist().index(i), '1'])
print trade_index, trade_price