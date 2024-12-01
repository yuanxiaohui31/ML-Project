import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV

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


def train_perceptron_classifier(X_train, y_train):
    """
    使用感知器分类器训练模型。
    """
    classifier = Perceptron(
        max_iter=1000,  # 增加迭代次数
        tol=1e-4,  # 更小的容忍度
        penalty='l2',  # 使用L2正则化
        alpha=0.001,  # 正则化强度
        eta0=0.1,  # 学习率
        random_state=42  # 固定随机种子，以便重现结果
    )
    classifier.fit(X_train, y_train)
    return classifier

def train_perceptron_with_tuning(X_train, y_train):
    """
    使用感知器进行训练并调优超参数。
    """
    param_grid = {
        'eta0': [0.001, 0.01, 0.1, 1],  # 学习率
        'max_iter': [1000, 2000],  # 最大迭代次数
        'tol': [1e-3, 1e-4],  # 容忍度
        'penalty': ['l2', 'l1', 'elasticnet'],  # 正则化类型
        'alpha': [0.0001, 0.001]  # 正则化参数
    }
    perceptron = Perceptron()
    grid_search = GridSearchCV(estimator=perceptron, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_classifier = grid_search.best_estimator_
    print("最佳模型:", best_classifier)
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
    # preprocessed_file = 'preprocessed_kpca_data.pkl'
    # train_part = 1
    train_part = 2

    X_train, y_train, X_test, y_test = load_data(preprocessed_file, train_part)

    # 训练perceptron分类器
    print("开始训练perceptron分类器...")
    # bayesian_classifier = train_perceptron_classifier(X_train, y_train)
    bayesian_classifier = train_perceptron_with_tuning(X_train, y_train)

    # 保存分类器
    # save_classifier(bayesian_classifier, 'trained_perceptron_classifier.pkl')
    # save_classifier(bayesian_classifier, 'trained_perceptron_classifier_kpca.pkl')
    save_classifier(bayesian_classifier, 'trained_perceptron_classifier_2.pkl')
