import numpy as np
import pickle
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_data(preprocessed_file, train_part):
    """
    加载预处理的数据并划分训练集和测试集。
    """
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    labels = data['labels']

    if train_part == 1:
        X_train = features[0:2000]
        y_train = labels[0:2000]
    if train_part == 2:
        X_train = features[2001:4000]
        y_train = labels[2001:4000]

    print("训练集特征形状:", X_train.shape)
    print("训练集标签形状:", y_train.shape)

    return X_train, y_train


def train_svm_with_tuning(X_train, y_train):
    """
    使用 SVM 进行训练。
    """
    param_grid = [
        {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 0.1, 1]},
        {'C': [0.1, 1, 10, 100], 'kernel': ['poly'], 'gamma': [1e-3, 1e-2, 0.1, 1]}
    ]
    svc = SVC(decision_function_shape='ovr')
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_classifier = grid_search.best_estimator_
    return best_classifier


def train_random_forest(X_train, y_train):
    """
    使用随机森林进行训练。
    """
    #param_grid = {
      ##  'n_estimators': [100, 200, 300],
       # 'max_depth': [None, 10, 20, 30],
       # 'min_samples_split': [2, 5, 10],
    #}
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2', None],  # 自动选择特征数量
        'bootstrap': [True, False]  # 是否采用自助法采样
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_classifier = grid_search.best_estimator_
    return best_classifier


def save_classifier(classifier, filename):
    """
    保存训练好的分类器到文件中。
    """
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"分类器已保存至 {filename}")

# 主程序
if __name__ == "__main__":
    preprocessed_file = 'preprocessed_data.pkl'
    train_part = 1

    X_train, y_train = load_data(preprocessed_file, train_part)

    # 训练 SVM 模型
    print("开始训练 SVM 模型...")
    best_svm_classifier = train_svm_with_tuning(X_train, y_train)
    save_classifier(best_svm_classifier, 'trained_svm_classifier.pkl')

    # 训练随机森林模型
    print("开始训练随机森林模型...")
    best_rf_classifier = train_random_forest(X_train, y_train)
    save_classifier(best_rf_classifier, 'trained_rf_classifier.pkl')