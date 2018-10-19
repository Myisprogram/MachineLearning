from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def news_classify():
    """
    对新闻进行分类
    :return:
    """
    # 加载数据
    news = fetch_20newsgroups('./news/',subset='all')
    # 进行分割成训练集合测试集
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)
    # 生成文章特征词
    tf = TfidfVectorizer()
    # 对训练数据进行tidif特征词抽取
    x_train = tf.fit_transform(x_train)

    # 对测试数据进行tifidf特征抽取
    x_test = tf.transform(x_test)
    # 实例化nb
    nb = MultinomialNB(alpha=1.0)
    nb.fit(x_train,y_train)
    print("计算测试数据的准确率:",nb.score(x_test,y_test))
    print("预测新的样本类别:",nb.predict(x_test.toarray()[0].reshape(1,-1)))


if __name__ == '__main__':
    news_classify()