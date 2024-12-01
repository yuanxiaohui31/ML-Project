from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

def preprocess_kpca(input_file, input_file_lbs, output_file, n_components=100):
    """
    使用 KernelPCA 进行降维，并保存结果。
    """
    # 加载数据
    digits_vec = np.loadtxt(input_file, delimiter='\t')
    digits_labels = np.loadtxt(input_file_lbs, delimiter='\t')

    # 归一化
    scaler = StandardScaler()
    digits_vec_normalized = scaler.fit_transform(digits_vec)

    # 使用 KernelPCA
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.01, max_iter=50000, eigen_solver='randomized')
    digits_vec_reduced = kpca.fit_transform(digits_vec_normalized)

    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump({'features': digits_vec_reduced, 'labels': digits_labels}, f)

    # 保存scaler
    # with open('scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)

    print(f"KernelPCA 降维完成，保存至 {output_file}")

# 示例调用
preprocess_kpca('digits4000_digits_vec.txt', 'digits4000_digits_labels.txt', 'preprocessed_kpca_data.pkl')
