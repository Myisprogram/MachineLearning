import pandas as pd
import  numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def cancer_predict():
    """
    对肿瘤进行预测
    :return:
    """
    # 获取数据
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    colums =  ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei',
               'Bland Chromatin','Normal Nucleoli','Mitoses','Class']

    data = pd.read_csv(link,names=colums)
    # print(data)
    # 缺失值处理
    data.replace(to_replace='?',value=np.nan,inplace=True)
    imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
    data = imp.fit_transform(data)
    # print(data)
    # 分割数据
    x_train,x_test,y_train,y_test = train_test_split(data[:,1:10],data[:, 10],test_size=0.25)
    # 标准化
    stand = StandardScaler()
    x_train = stand.fit_transform(x_train)
    x_test = stand.transform(x_test)
    print(x_test)
    # 逻辑回归分类问题预测
    lr = LogisticRegression(penalty='l2',C=1.0)
    lr.fit(x_train,y_train)
    print('回归系数：',lr.coef_)
    y_pred = lr.predict(x_test)
    print("准确率:",np.mean(y_pred == y_test))
    # 打印精确率和召回率
    print('精确率和召回率：\n',classification_report(y_test,y_pred,target_names=['良性','恶性']))

if __name__ == '__main__':
    cancer_predict()