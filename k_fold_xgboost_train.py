from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import KFold
import time
import xgboost as xgb
import pandas as pd
import numpy as np

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

kf = KFold(n_splits=10, shuffle=True,random_state=0)
prediction_sum = np.array(0)
S_train_xgb = np.zeros((dataset.shape[0]))

params = {
    'max_depth': 15,  # 构建树的深度 [1:]
    'min_child_weight':3, # 节点的最少特征数 用于防止过拟合问题：较大的值能防止过拟合，过大的值会导致欠拟合问题
    'subsample':0.9, # 采样数据率
    #'colsample_bytree':0.9,
    'silent': 1,
    'eta': 0.12,  # 如同学习率
}
plst = list(params.items())
num_rounds = 5000  # 迭代你次数
mean_1 = 0
mean_2 = 0
for train_index, test_index in kf.split(x_data):
    train_x, train_y = x_data[train_index], y_data[train_index]
    test_x, test_y = x_data[test_index], y_data[test_index]

    xgtrain = xgb.DMatrix(train_x, label=train_y)
    xgval = xgb.DMatrix(test_x, label=test_y)
    xgtest = xgb.DMatrix(test)

    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30)
    S_train_xgb[test_index] = model.predict(xgval,ntree_limit=model.best_iteration)

    mean_1 += mean_squared_error(model.predict(xgtrain,
                                               ntree_limit=model.best_iteration), train_y)
    mean_2 += mean_squared_error(model.predict(xgval,
                                                        ntree_limit=model.best_iteration), test_y)
    print("训练集均方误差为:", mean_squared_error(model.predict(xgtrain,
                                                        ntree_limit=model.best_iteration), train_y))
    print("验证集均方误差为:", mean_squared_error(model.predict(xgval,
                                                        ntree_limit=model.best_iteration), test_y))

    preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    prediction_sum = prediction_sum + preds
prediction_mean = prediction_sum / 10

print(mean_1/10,mean_2/10)
print(prediction_mean)


pediction = {'id':testsid,'price':prediction_mean}
columns = ['id','price']
pediction = pd.DataFrame(pediction,columns=columns)
pediction.to_csv('xgb_train_kfold.csv',encoding = 'utf-8',columns=columns,index=None)

S_train_xgb = pd.DataFrame(S_train_xgb)
S_train_xgb.to_csv('xgb_train_train.csv',encoding = 'utf-8')

cost_time = time.time() - now
print("end ......", '\n', "cost time:", cost_time, "(s)......")
