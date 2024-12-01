import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


# def train_fisher_classifier(X_train, y_train):
#     """
#     使用 Fisher 判别分类器训练模型。
#     """
#     classifier = LinearDiscriminantAnalysis()
#     classifier.fit(X_train, y_train)
#     return classifier

def train_fisher_classifier(X_train, y_train):
    """
    使用 Fisher 判别分类器训练模型，并调整参数优化效果。
    """
    classifier = LinearDiscriminantAnalysis(
        solver='lsqr',  # 使用正则化支持的求解器
        shrinkage='auto',  # 自动选择正则化参数
        store_covariance=True  # 存储协方差矩阵（可选）
    )
    classifier.fit(X_train, y_train)
    return classifier

def train_lda_with_tuning(X_train, y_train):
    """
    使用 LDA（Fisher 判别分析）进行训练并调优超参数。
    """
    param_grid = [
        {'solver': ['svd'], 'tol': [1e-4, 1e-3, 1e-2]},  # SVD 不支持 shrinkage
        {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.5, 1.0]}  # LSQR 和 Eigen 支持 shrinkage
    ]
    lda = LinearDiscriminantAnalysis()
    grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, cv=5, scoring='accuracy')
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
    # preprocessed_file = 'preprocessed_kpca_data.pkl'
    preprocessed_file = 'preprocessed_data.pkl'
    # train_part = 1
    train_part = 2

    X_train, y_train, X_test, y_test = load_data(preprocessed_file, train_part)

    # 训练fisher分类器
    print("开始训练fisher分类器...")
    # bayesian_classifier = train_fisher_classifier(X_train, y_train)
    bayesian_classifier = train_lda_with_tuning(X_train, y_train)

    # 保存分类器
    # save_classifier(bayesian_classifier, 'trained_fisher_classifier_kpca.pkl')
    # save_classifier(bayesian_classifier, 'trained_fisher_classifier.pkl')
    save_classifier(bayesian_classifier, 'trained_fisher_classifier_2.pkl')