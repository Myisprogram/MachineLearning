import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np


def titiannic():
    """
    对乘客的生死进行预测
    :return:
    """
    # 读取数据
    data = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    # print(data.head(20))

    # 目标值
    y = data['survived']
    # 特征值
    x = data[['pclass','age','sex']].copy()
    # print(x.head(20))

    # 补全缺失值，用平均值填充
    x['age'].fillna(x['age'].mean(),inplace=True)
    # print(x.head(20))

    # 分割数据集为训练数据和测试数据
    x_train,x_test,y_train,y_test = train_test_split(x,y)

    # 实例化字典特征抽取的类
    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(x_train.to_dict(orient='records'))
    x_test = dv.fit_transform(x_test.to_dict(orient='records'))
    # 打印特征名称和特征值
    # print(dv.get_feature_names())
    # print(x_train[:20])
    # 决策树估计器流程
    # dec = DecisionTreeClassifier(criterion='entropy',max_depth=None)
    # 用训练数据训练决策树模型
    # dec.fit(x_train,y_train)

    #对测试集进行预测
    # y_pred = dec.predict(x_test)
    # print(y_pred.head())
    # print(y_pred)
    # print('手工预测准确率:',np.mean(y_pred == y_test))
    # print('预测准确率:',dec.score(x_test,y_test))
    # 生成决策树结构图
    # dotstring = export_graphviz(dec,out_file=None,feature_names=['年龄','一等舱','二等舱','三等舱','女性','男性'])
    # print(dotstring)
    # 实例化随机深林
    rfc = RandomForestClassifier()
    params = {'n_estimators':[2,5,8,10,50,80,100,150],
              'max_depth':[5,8,15,25,30]}

    # 实例化网格搜索
    gs = GridSearchCV(rfc,param_grid=params,cv=5)
    gs.fit(x_train,y_train)

    print('最好的参数:',gs.best_params_)
    print('最好的准确率:',gs.best_score_)
    print('测试集最缺率:',gs.score(x_test,y_test))


if __name__ == '__main__':
    titiannic()