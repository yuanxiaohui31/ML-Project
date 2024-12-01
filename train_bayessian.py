import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split

def load_data(preprocessed_file, train_part):
    """
    加载预处理的数据并划分训练集和测试集。
    """
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    labels = data['labels']

    if train_part == 1:
        X_train, y_train = features[0:2000], labels[0:2000]
        X_test, y_test = features[2001:4000], labels[2001:4000]
    else:
        X_train, y_train = features[2001:4000], labels[2001:4000]
        X_test, y_test = features[0:2000], labels[0:2000]

    return X_train, y_train, X_test, y_test


def train_bayesian_classifier(X_train, y_train):
    """
    使用朴素贝叶斯分类器训练模型。
    """
    #classifier = GaussianNB()
    #classifier.fit(X_train, y_train)
    #return classifier
    classifier = GaussianNB()
    param_grid = {
        'var_smoothing': np.logspace(0, -9, num=20)  # 以对数空间测试var_smoothing参数
    }

    # 使用GridSearchCV进行参数调优，交叉验证选择最优参数
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    best_classifier = grid_search.best_estimator_
    y_pred = best_classifier.predict(X_test)
    return best_classifier

def save_classifier(classifier, filename):
    """
    保存训练好的分类器到文件中。
    """
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"分类器已保存至 {filename}")


if __name__ == "__main__":
    # 加载数据
    preprocessed_file = 'preprocessed_data.pkl'
    train_part = 1

    X_train, y_train, X_test, y_test = load_data(preprocessed_file, train_part)

    # 训练贝叶斯分类器
    print("开始训练贝叶斯分类器...")
    bayesian_classifier = train_bayesian_classifier(X_train, y_train)

    # 保存分类器
    save_classifier(bayesian_classifier, 'trained_bayesian_classifier.pkl')

