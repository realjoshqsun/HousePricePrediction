import pandas as pd
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

now = time.time()

randlist = np.random.randint(196539,size=50000)

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

'''
grid = GridSearchCV(SVR(), param_grid={"kernel":['rbf','linear'],"C":[5,10,15],
                                       "gamma": [0.005,0.01,0.015]}, cv=10)
grid.fit(x_data, y_data)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
'''

n_splits = 10
kf = KFold(n_splits= n_splits, shuffle=True,random_state=0)
prediction_sum = np.array(0)
S_train_svr = np.zeros((dataset.shape[0]))

for train_index, test_index in kf.split(x_data):
    train_x, train_y = x_data[train_index], y_data[train_index]
    test_x, test_y = x_data[test_index], y_data[test_index]

    svr = SVR(kernel='rbf', C=10, gamma=0.02)
    svr.fit(train_x, train_y)
    S_train_svr[test_index] = ss_y.inverse_transform(svr.predict(test_x))

    print("训练集均方误差为:", mean_squared_error(ss_y.inverse_transform(svr.predict(train_x)),
                                          ss_y.inverse_transform(train_y)))
    print("验证集均方误差为:", mean_squared_error(ss_y.inverse_transform(svr.predict(test_x)),
                                       ss_y.inverse_transform(test_y)))

    preds = ss_y.inverse_transform(svr.predict(test))
    prediction_sum = prediction_sum + preds
prediction_mean = prediction_sum / 10
print(prediction_mean)

pediction = {'id':testsid,'price':prediction_mean}
columns = ['id','price']
pediction = pd.DataFrame(pediction,columns=columns)
pediction.to_csv('svr_kfold.csv',encoding = 'utf-8',columns=columns,index=None)

S_train_svr = pd.DataFrame(S_train_svr)
S_train_svr.to_csv('svr_train.csv',encoding = 'utf-8')

cost_time = time.time() - now
print("end ......", '\n', "cost time:", cost_time, "(s)......")
