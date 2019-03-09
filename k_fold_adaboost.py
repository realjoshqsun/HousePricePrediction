import xgboost as xgb
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


now = time.time()

dataset = pd.read_csv('./train.csv')
tests = pd.read_csv('./test.csv')

def pre(train):
    tt = train['房屋朝向'].values
    train.drop(['房屋朝向'], axis=1, inplace=True)
    N = []
    S = []
    E = []
    W = []
    SP = []
    for x in tt:
        e = x.count('东')
        s = x.count('南')
        n = x.count('北')
        w = x.count('西')
        sp = x.count(' ')
        N.append(1.0 * n / (n + s + e + w))
        E.append(1.0 * e / (n + s + e + w))
        W.append(1.0 * w / (n + s + e + w))
        S.append(1.0 * s / (n + s + e + w))
        SP.append(sp + 1)
    train['N'] = N
    train['E'] = E
    train['S'] = S
    train['W'] = W
    train['SP'] = SP
    return train
dataset=pre(dataset)
tests=pre(tests)

dataset = dataset.fillna(value=-999)
tests = tests.fillna(value=-999)

x_data = dataset.drop(['月租金'], axis = 1).values
y_data = dataset['月租金'].values

test = tests.drop(['id'],axis = 1).values
testsid = tests['id']

n_splits = 10
kf = KFold(n_splits= n_splits, shuffle=True,random_state=0)
prediction_sum = np.array(0)
S_train_adaboost = np.zeros((dataset.shape[0]))


for train_index, test_index in kf.split(x_data):
    train_x, train_y = x_data[train_index], y_data[train_index]
    test_x, test_y = x_data[test_index], y_data[test_index]

    ada = AdaBoostRegressor(
        DecisionTreeRegressor(min_samples_split=4, max_depth=30)
        ,n_estimators=70,learning_rate=0.25)
    model = ada.fit(train_x,train_y)
    S_train_adaboost[test_index] = model.predict(test_x)

    print('训练集均方误差为:', mean_squared_error(model.predict(train_x), train_y))
    print('验证集均方误差为:',mean_squared_error(model.predict(test_x),test_y))

    preds = model.predict(test)
    prediction_sum = prediction_sum + preds
prediction_mean = prediction_sum / 10
print(prediction_mean)

pediction = {'id':testsid,'price':prediction_mean}
columns = ['id','price']
pediction = pd.DataFrame(pediction,columns=columns)
pediction.to_csv('adaboost_kfold.csv',encoding = 'utf-8',columns=columns,index=None)

S_train_adaboost = pd.DataFrame(S_train_adaboost)
S_train_adaboost.to_csv('ada_train.csv',encoding = 'utf-8')

cost_time = time.time() - now
print("end ......", '\n', "cost time:", cost_time, "(s)......")
