"""
PCA主成分分析实验
实现PCA降维并可视化训练集和测试集的降维效果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

# Matplotlib 中文与 mathtext/负号 显示（精简版）
def _configure_fonts() -> None:
    # 以 SimHei 为主字体，保证中文不缺字
    plt.rcParams["font.family"] = ["SimHei"]
    # 英文/符号可回退到 DejaVu Sans
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    # 负号正常显示（使用 ASCII '-'）
    plt.rcParams["axes.unicode_minus"] = False
    # 让 mathtext 使用常规文本风格，避免符号乱码
    plt.rcParams["mathtext.default"] = "regular"


# 先设置 seaborn 样式，再配置字体（避免被覆盖）
sns.set_style("whitegrid")
_configure_fonts()


def load_data():
    """加载训练和测试数据"""
    print("正在加载数据...")

    X_train = loadmat('X_train_pca.mat')['X_train']
    y_train = loadmat('y_train_pca.mat')['y_train'].ravel()
    X_test = loadmat('X_test_pca.mat')['X_test']
    y_test = loadmat('y_test_pca.mat')['y_test'].ravel()

    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"类别分布: {np.unique(y_train)}")

    return X_train, y_train, X_test, y_test


def standardize_data(X_train, X_test):
    """标准化数据（可选，PCA前的常见预处理）"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # 避免除以0
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


class PCA:
    """手动实现的PCA类"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None  # 主成分（特征向量）
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.cumulative_variance_ratio = None

    def fit(self, X):
        """在训练数据上拟合PCA"""
        print(f"\n开始PCA拟合，目标维度: {self.n_components}")

        # 1. 计算均值并中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. 计算协方差矩阵
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # 3. 特征分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. 按特征值降序排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. 保存特征值和特征向量
        self.eigenvalues = eigenvalues
        self.components = eigenvectors[:, :self.n_components]

        # 6. 计算方差解释比例
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_variance
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)

        print(f"前{self.n_components}个主成分解释的方差比例: "
              f"{self.cumulative_variance_ratio[self.n_components-1]:.4f}")

        return self

    def transform(self, X):
        """将数据投影到主成分空间"""
        X_centered = X - self.mean  # 使用训练集的均值
        X_reduced = X_centered @ self.components
        return X_reduced

    def fit_transform(self, X):
        """拟合并转换数据"""
        self.fit(X)
        return self.transform(X)


def plot_variance_explained(pca, save_path='pca_variance_explained.png'):
    """绘制方差解释比例图"""
    print("\n绘制方差解释比例图...")

    n_components = len(pca.explained_variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：每个主成分的方差解释比例
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio,
            alpha=0.8, color='steelblue', edgecolor='black')
    ax1.set_xlabel('主成分编号', fontsize=12)
    ax1.set_ylabel('方差解释比例', fontsize=12)
    ax1.set_title('各主成分的方差解释比例', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 右图：累积方差解释比例
    ax2.plot(range(1, n_components + 1), pca.cumulative_variance_ratio,
             marker='o', linestyle='-', linewidth=2, markersize=6, color='crimson')
    ax2.axhline(y=0.90, color='green', linestyle='--', label='90%', alpha=0.7)
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95%', alpha=0.7)
    ax2.axhline(y=0.99, color='red', linestyle='--', label='99%', alpha=0.7)
    ax2.set_xlabel('主成分数量', fontsize=12)
    ax2.set_ylabel('累积方差解释比例', fontsize=12)
    ax2.set_title('累积方差解释比例曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")

    # 输出关键信息
    for threshold in [0.90, 0.95, 0.99]:
        n_comp = np.argmax(pca.cumulative_variance_ratio >= threshold) + 1
        print(f"保留{threshold*100:.0f}%方差需要 {n_comp} 个主成分")


def plot_2d_visualization(X_train_2d, y_train, X_test_2d, y_test,
                          save_path='pca_2d_visualization.png'):
    """可视化降维到2D后的数据分布（训练集+测试集）"""
    print("\n绘制2D降维可视化图...")

    # 获取类别和配色
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)
    colors = sns.color_palette("husl", n_classes)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制训练集（实心圆）
    for i, cls in enumerate(classes):
        mask = y_train == cls
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                  c=[colors[i]], marker='o', s=60, alpha=0.7,
                  edgecolors='black', linewidths=0.5,
                  label=f'训练集-类别{cls}')

    # 绘制测试集（叉号）
    for i, cls in enumerate(classes):
        mask = y_test == cls
        ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
                  c=[colors[i]], marker='x', s=80, alpha=0.8,
                  linewidths=2,
                  label=f'测试集-类别{cls}')

    ax.set_xlabel('第1主成分 (PC1)', fontsize=13, fontweight='bold')
    ax.set_ylabel('第2主成分 (PC2)', fontsize=13, fontweight='bold')
    ax.set_title('PCA降维到2D的数据分布（训练集 vs 测试集）',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")


def plot_original_vs_pca(X_train, y_train, X_test, y_test,
                        X_train_2d, y_train_2d, X_test_2d, y_test_2d,
                        save_path='original_vs_pca_comparison.png'):
    """对比原始数据前2维 vs PCA降维后的2维"""
    print("\n绘制原始数据 vs PCA降维对比图...")

    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)
    colors = sns.color_palette("Set2", n_classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：原始数据的前2个特征
    for i, cls in enumerate(classes):
        mask_train = y_train == cls
        mask_test = y_test == cls
        ax1.scatter(X_train[mask_train, 0], X_train[mask_train, 1],
                   c=[colors[i]], marker='o', s=50, alpha=0.6,
                   edgecolors='black', linewidths=0.5, label=f'训练-类{cls}')
        ax1.scatter(X_test[mask_test, 0], X_test[mask_test, 1],
                   c=[colors[i]], marker='x', s=70, alpha=0.8,
                   linewidths=2, label=f'测试-类{cls}')

    ax1.set_xlabel('原始特征1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('原始特征2', fontsize=12, fontweight='bold')
    ax1.set_title('原始数据（前2个特征）', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)

    # 右图：PCA降维后的2D数据
    for i, cls in enumerate(classes):
        mask_train = y_train_2d == cls
        mask_test = y_test_2d == cls
        ax2.scatter(X_train_2d[mask_train, 0], X_train_2d[mask_train, 1],
                   c=[colors[i]], marker='o', s=50, alpha=0.6,
                   edgecolors='black', linewidths=0.5, label=f'训练-类{cls}')
        ax2.scatter(X_test_2d[mask_test, 0], X_test_2d[mask_test, 1],
                   c=[colors[i]], marker='x', s=70, alpha=0.8,
                   linewidths=2, label=f'测试-类{cls}')

    ax2.set_xlabel('第1主成分 (PC1)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('第2主成分 (PC2)', fontsize=12, fontweight='bold')
    ax2.set_title('PCA降维后（2个主成分）', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle('原始数据 vs PCA降维数据对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")


def main():
    """主函数"""
    print("="*60)
    print("PCA主成分分析实验")
    print("="*60)

    # 1. 加载数据
    X_train, y_train, X_test, y_test = load_data()

    # 2. 数据标准化（可选）
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # 3. 使用原始数据拟合PCA（也可以用标准化后的数据）
    # 这里用原始数据，如果想用标准化数据，替换为 X_train_scaled
    pca_full = PCA(n_components=X_train.shape[1])  # 保留所有主成分用于分析
    pca_full.fit(X_train)

    # 4. 绘制方差解释比例图
    plot_variance_explained(pca_full)

    # 5. 降维到2D
    pca_2d = PCA(n_components=2)
    X_train_2d = pca_2d.fit_transform(X_train)
    X_test_2d = pca_2d.transform(X_test)  # 使用训练集的参数

    print(f"\n降维后形状: X_train_2d={X_train_2d.shape}, X_test_2d={X_test_2d.shape}")

    # 6. 可视化降维结果
    plot_2d_visualization(X_train_2d, y_train, X_test_2d, y_test)

    # 7. 对比原始数据前2维 vs PCA降维后的2维
    plot_original_vs_pca(X_train, y_train, X_test, y_test,
                        X_train_2d, y_train, X_test_2d, y_test)

    print("\n" + "="*60)
    print("实验完成！所有图像已保存。")
    print("="*60)


if __name__ == '__main__':
    main()
