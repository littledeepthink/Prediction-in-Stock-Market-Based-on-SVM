# coding: utf-8
from sklearn.linear_model import ElasticNet,LinearRegression
import numpy as np
import pandas as pd
bd_index = pd.read_csv(u'E:/Data Mining And Machine Learning/量化投资/统计建模比赛/文本挖掘/index_add_final.csv')

################### 最优参数的选择 ####################
def bic(X,coef,y,N,dof):
    return ( np.linalg.norm(np.mat(X)*np.mat(coef).T - np.mat(y), ord=2) )**2 + np.log(N)*dof

number_remove = [] # 移除变量的个数
bd_index.columns = np.arange(0,71,1)
# bd_index[6] = 0
# bd_index[13] = 0
y = bd_index[1]
y = np.array([i for i in y])
X_temp = bd_index.copy()
del X_temp[0]; del X_temp[1]
X = np.array(X_temp)
alpha_para = range(200,3000,100)
l_para = range(1,9,1)
l_para = [i/10.0 for i in l_para]
bic_value = np.zeros( (len(l_para),len(alpha_para)) )
for l in range( len(l_para) ):
    for a in range( len(alpha_para) ):
        clf = ElasticNet(alpha=alpha_para[a], l1_ratio=l_para[l])
        clf.fit(X, y)
        coef = [i for i in clf.coef_]
        zero_loc = []
        for i in range(len(coef)):
            if np.abs(coef[i]) < (10 ** -6):
                zero_loc.append(i+2)
                coef[i] += 1
        ols_index = range(2,71)
        for i in zero_loc:
            ols_index.remove(i)
        number_remove.append( len(zero_loc) )
        print number_remove
        number_remove = []
        X_m = np.array(bd_index.loc[:,ols_index])
        y_m = np.array( [i for i in bd_index.loc[:,1]] )
        model = LinearRegression().fit(X_m, y_m)
        ols_coef = model.coef_
        N = X.shape[0]
        dof = X.shape[1] - len(zero_loc)
        bic_value[l,a] = bic(X_m, ols_coef, y_m, N, dof)

l_num = len(l_para)
a_num = len(alpha_para)
bic_list = [bic_value[i][j] for i in range(l_num) for j in range(a_num)]
id = bic_list.index(np.min(bic_list))

def location2(id,l_para,alpha_para,m):
    g = id/m
    c = np.mod(id,m)
    return l_para[g],alpha_para[c]

print location2(id,l_para,alpha_para,28)

clf = ElasticNet(alpha=2500,l1_ratio=0.2)
clf.fit(X, y)
coef = [i for i in clf.coef_]
zero_loc = []
for i in range(len(coef)):
    if np.abs(coef[i]) < (10**-6):
        zero_loc.append( i+1 )
        coef[i] += 1
print zero_loc