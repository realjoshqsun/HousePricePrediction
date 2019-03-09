import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

price_rf = pd.read_csv('./rf_train.csv').ix[:,1]
price_gbdt = pd.read_csv('./gbdt_train.csv').ix[:,1]
price_ada = pd.read_csv('./ada_train.csv').ix[:,1]
price_xgb = pd.read_csv('./xgb_train_train.csv').ix[:,1]
price_label = pd.read_csv('./train_preprocessing_Try.csv')['月租金'].values

rf_test = pd.read_csv('./rf_kfold.csv')['price'].values
gbdt_test = pd.read_csv('./gbdt_kfold.csv')['price'].values
ada_test = pd.read_csv('./adaboost_kfold.csv')['price'].values
xgb_test = pd.read_csv('./xgb_train_kfold.csv')['price'].values

train = pd.DataFrame({
            'ada':price_ada,'xgb':price_xgb,'label':price_label
                },index=None)
test = pd.DataFrame({
            'ada':ada_test,'xgb':xgb_test,
                },index=None)
'''
plt.figure()
rf = plt.scatter(train.rf.values,train.label.values,c='red',s=3)
ada = plt.scatter(train.ada.values,train.label.values,c='green',s=3)
xgb = plt.scatter(train.xgb.values,train.label.values,c='yellow',s=3)
plt.legend([rf,ada,xgb],['rf','ada','xgb'],prop={'size':17})
plt.show()
'''

x_train = train.drop(['label'],axis = 1).values
y_train = train['label'].values

x_test = test.values

ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train[:, None])[:,0]

kf = KFold(n_splits=10, shuffle=True,random_state=0)
prediction_sum = np.array(0)

for train_index, test_index in kf.split(x_train):
    train_x, train_y = x_train[train_index], y_train[train_index]
    test_x, test_y = x_train[test_index], y_train[test_index]

    model = xgb.XGBRegressor(max_depth=3,min_child_weight=3)
    model.fit(train_x,train_y)

    print("训练集均方误差为:", mean_squared_error(ss_y.inverse_transform(model.predict(train_x)),
                                          ss_y.inverse_transform(train_y)))
    print("验证集均方误差为:", mean_squared_error(ss_y.inverse_transform(model.predict(test_x)),
                                          ss_y.inverse_transform(test_y)))

    preds = ss_y.inverse_transform(model.predict(x_test))
    prediction_sum = prediction_sum + preds
prediction_mean = prediction_sum / 10
print(prediction_mean)

tests = pd.read_csv('./test.csv')
testsid = tests['id']

columns = ['id','price']
pediction = pd.DataFrame({'id':testsid,'price':prediction_mean},columns=columns)
pediction.to_csv('XGB(xgb,ada)_3,3_Stacking.csv',encoding = 'utf-8',columns=columns,index=None)

