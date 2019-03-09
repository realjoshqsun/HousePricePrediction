import pandas as pd
import time
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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

ss_x = StandardScaler()
x_data = ss_x.fit_transform(x_data)
test = ss_x.transform(test)

ss_y = StandardScaler()
y_data = ss_y.fit_transform(y_data[:, None])[:,0]

kf = KFold(n_splits=10, shuffle=True,random_state=0)
prediction_sum_lasso = np.array(0)
prediction_sum_enet = np.array(0)
S_train_lasso = np.zeros((dataset.shape[0]))
S_train_enet = np.zeros((dataset.shape[0]))
for train_index, test_index in kf.split(x_data):
    train_x, train_y = x_data[train_index], y_data[train_index]
    test_x, test_y = x_data[test_index], y_data[test_index]

    # 线性模型 lasso
    lasso = LassoCV(0.0005).fit(train_x, train_y) # 0.0005是调出来的最好alpha
    y_pred_lasso = lasso.predict(test_x)
    S_train_lasso[test_index] = ss_y.inverse_transform(y_pred_lasso)
    mean_squared_error_lasso = mean_squared_error(ss_y.inverse_transform(y_pred_lasso)
                                                  ,ss_y.inverse_transform(test_y))
    print("lasso",mean_squared_error_lasso)

    # ElasticNet模型
    enet = linear_model.ElasticNetCV(alphas=[0.01],l1_ratio=[0.1],max_iter=1000).fit(train_x, train_y)
    #print(enet.alpha_,enet.l1_ratio_,enet.max_iter)
    y_pred_enet = enet.predict(test_x)
    S_train_enet[test_index] = ss_y.inverse_transform(y_pred_enet)
    mean_squared_error_enet = mean_squared_error(ss_y.inverse_transform(y_pred_lasso)
                                                  ,ss_y.inverse_transform(test_y))
    print("Elastic",mean_squared_error_enet)

    preds_lasoo = ss_y.inverse_transform(lasso.predict(test))
    preds_enet = ss_y.inverse_transform(enet.predict(test))
    prediction_sum_lasso = prediction_sum_lasso + preds_lasoo
    prediction_sum_enet = prediction_sum_enet + preds_enet

prediction_mean_lasso = prediction_sum_lasso / 10
prediction_mean_enet = prediction_sum_enet / 10
print(prediction_mean_lasso,prediction_mean_enet)

pediction_lasso = {'id':testsid,'price':prediction_mean_lasso}
pediction_enet = {'id':testsid,'price':prediction_mean_enet}
columns = ['id','price']
pediction_lasso = pd.DataFrame(pediction_lasso,columns=columns)
pediction_enet = pd.DataFrame(pediction_enet,columns=columns)
pediction_lasso.to_csv('lasso_kfold.csv',encoding = 'utf-8',columns=columns,index=None)
pediction_enet.to_csv('enet_kfold.csv',encoding = 'utf-8',columns=columns,index=None)

S_train_lasso = pd.DataFrame(S_train_lasso)
S_train_enet = pd.DataFrame(S_train_enet)
S_train_lasso.to_csv('lasso_train.csv',encoding = 'utf-8')
S_train_enet.to_csv('enet_train.csv',encoding = 'utf-8')

cost_time = time.time() - now
print("end ......", '\n', "cost time:", cost_time, "(s)......")

