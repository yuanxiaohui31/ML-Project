import numpy as np
from sklearn.decomposition import PCA
import pickle
import pandas as pd


def preprocess_data(input_file, input_file_lbs, output_file, n_components=100): ##主成分量为100
    """
    预处理数据：读取数据并使用 PCA 降维后保存。

    参数：
    - input_file: str, 原始数据文件路径
    - output_file: str, 预处理后的数据保存路径
    - n_components: int, PCA 降维的主成分数量

    返回：
    - None
    """
    # 加载数据
    # 从文本文件中读取数据
    digits_vec = np.loadtxt(input_file, delimiter='\t')  # 使用制表符分隔符读取数据

    # 假设标签数据存储在单独的文件中，例如 'digits_labels.txt'
    # 读取标签
    digits_labels = np.loadtxt(input_file_lbs, delimiter='\t')

    # 展示一组数据以确认读取正确
    # 转换为 DataFrame
    digits_df = pd.DataFrame(digits_vec)  #vec为784*4000，每一列代表一张图片的特征向量，一列代表一个样本
    # 设置 Pandas 显示选项
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 不限制行宽
    pd.set_option('display.max_colwidth', None)  # 不限制列宽

    # 展示前2行
    print("读取的特征数据（前2行）：")
    print(digits_df.head(2))  # 使用 DataFrame 的 head 方法
    print("读取的标签数据（前2个）：")
    print(digits_labels[:2])  # 显示前2个标签，digits_labels为1*4000的行向量，每一个数表示vec中一列的类别

    # 使用 PCA 对特征进行降维
    pca = PCA(n_components=n_components)
    # digits_vec_reduced = pca.fit_transform(digits_vec.T)  # 转置后再降维,不转置回来
    # digits_vec_reduced = pca.fit_transform(digits_vec.T).T  # 转置后再降维,再转置回来
    digits_vec_reduced = pca.fit_transform(digits_vec)  # 不转置

    # 将降维后的数据和标签一起保存
    with open(output_file, 'wb') as f:
        pickle.dump({'features': digits_vec_reduced, 'labels': digits_labels}, f)

    print(f"预处理完成，降维后的数据保存至 {output_file}")

# 示例调用
preprocess_data('digits4000_digits_vec.txt','digits4000_digits_labels.txt', 'preprocessed_data.pkl')  ##pkl 文件通常用于存储 Python 对象（例如字典、列表、数组等）的二进制表示
