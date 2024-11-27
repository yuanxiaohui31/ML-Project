import numpy as np
import pickle
from sklearn.metrics import accuracy_score

def load_data(preprocessed_file, train_part):
    """
    加载预处理的数据并划分测试集。
    """
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    labels = data['labels']

    if train_part == 1:
        X_test = features[2001:4000]
        y_test = labels[2001:4000]
    if train_part == 2:
        X_test = features[0:2000]
        y_test = labels[0:2000]

    return X_test, y_test


def load_classifier(filename):
    """
    从文件加载分类器。
    """
    with open(filename, 'rb') as f:
        best_classifier = pickle.load(f)
    print(f"分类器从 {filename} 加载成功")
    return best_classifier


def evaluate_model(best_classifier, X_test, y_test):
    """
    在测试集上评估模型的准确率。
    """
    y_pred = best_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# 主程序
if __name__ == "__main__":
    preprocessed_file = 'preprocessed_data.pkl'
    train_part = 1

    # 加载测试集
    X_test, y_test = load_data(preprocessed_file, train_part)

    # 加载 SVM 分类器并评估
    svm_classifier = load_classifier('trained_svm_classifier.pkl')
    svm_accuracy = evaluate_model(svm_classifier, X_test, y_test)
    print(f"SVM 模型的准确率: {svm_accuracy}")

    # 加载随机森林分类器并评估
    rf_classifier = load_classifier('trained_rf_classifier.pkl')
    rf_accuracy = evaluate_model(rf_classifier, X_test, y_test)
    print(f"随机森林模型的准确率: {rf_accuracy}")

    # 加载贝叶斯分类器并评估
    rf_classifier = load_classifier('trained_bayesian_classifier.pkl')
    rf_accuracy = evaluate_model(rf_classifier, X_test, y_test)
    print(f"贝叶斯分类器的准确率: {rf_accuracy}")

    # 加载fisher分类器并评估
    rf_classifier = load_classifier('trained_fisher_classifier.pkl')
    rf_accuracy = evaluate_model(rf_classifier, X_test, y_test)
    print(f"fisher分类器的准确率: {rf_accuracy}")

    # 加载logistic_regression分类器并评估
    rf_classifier = load_classifier('trained_logistic_regression_classifier.pkl')
    rf_accuracy = evaluate_model(rf_classifier, X_test, y_test)
    print(f"logistic_regression分类器的准确率: {rf_accuracy}")

    # 加载perceptron分类器并评估
    rf_classifier = load_classifier('trained_perceptron_classifier.pkl')
    rf_accuracy = evaluate_model(rf_classifier, X_test, y_test)
    print(f"perceptron分类器的准确率: {rf_accuracy}")