import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


# def train_logistic_regression_classifier(X_train, y_train):
#     """
#     使用逻辑回归分类器训练模型。
#     """
#     classifier = LogisticRegression(max_iter=1000)
#     classifier.fit(X_train, y_train)
#     return classifier

def train_logistic_regression_classifier(X_train, y_train):
    """
    使用逻辑回归分类器训练模型。
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    classifier = LogisticRegression(
        max_iter=1000,
        solver='saga',
        penalty='elasticnet',
        l1_ratio=0.5,
        C=0.1,
        class_weight='balanced'
    )
    classifier.fit(X_train, y_train)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("scaler已保存至 scaler.pkl")
    return classifier

def train_logistic_regression_with_tuning(X_train, y_train):
    """
    使用逻辑回归进行训练并调优超参数。
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    param_grid = [
        # l1 和 l2 正则化
        {'C': [0.01, 0.1], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'max_iter': [1000, 2000]},
        # elasticnet 正则化
        {'C': [0.01, 0.1], 'penalty': ['elasticnet'], 'l1_ratio': [0.1, 0.5, 0.9], 'solver': ['saga'],
         'max_iter': [1000, 2000]},
        # 仅 l2 正则化（适用于 lbfgs 和 newton-cg）
        {'C': [0.01, 0.1], 'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg'], 'max_iter': [1000, 2000]},
    ]
    lr = LogisticRegression(class_weight='balanced')  # 可以加入class_weight来应对不平衡数据
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_classifier = grid_search.best_estimator_
    print("最佳模型:", best_classifier)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("scaler已保存至 scaler.pkl")
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
    train_part = 1
    # train_part = 2

    X_train, y_train, X_test, y_test = load_data(preprocessed_file, train_part)

    # 训练logistic_regression分类器
    print("开始训练logistic_regression分类器...")
    # bayesian_classifier = train_logistic_regression_classifier(X_train, y_train)
    bayesian_classifier = train_logistic_regression_with_tuning(X_train, y_train)

    # 保存分类器
    # save_classifier(bayesian_classifier, 'trained_logistic_regression_classifier_kpca.pkl')
    save_classifier(bayesian_classifier, 'trained_logistic_regression_classifier.pkl')
    # save_classifier(bayesian_classifier, 'trained_logistic_regression_classifier_2.pkl')