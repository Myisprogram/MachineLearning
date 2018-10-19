from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

def boston_price():
    """
    建立关系模型回归法预测房价
    :return:
    """
    # 1.获取数据
    bn_data = load_boston()
    # print(bn_data.data)
    # print(bn_data.target)

    # 2.分割数据
    x_train,x_test,y_train,y_test = train_test_split(bn_data.data,bn_data.target)

    # 3.标准化数据
    stand = StandardScaler()
    # 对训练数据标准化
    x_train = stand.fit_transform(x_train)
    # 对测试数据标准化
    x_test = stand.transform(x_test)
    # 4.用正规方程求解权重
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    # 查看权重系数
    # print(lr.coef_)

    print("测试集真实集",y_test)
    # 对测试集数据进行预测
    y_pred = lr.predict(x_test)
    print("预测值：",y_pred)

    # 评估预测结果
    print("预测结果的均方差：",mean_squared_error(y_test,y_pred))

if __name__ == '__main__':
    boston_price()