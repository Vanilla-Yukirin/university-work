'''
环境激活conda activate ML
'''

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import seaborn as sns

# 使用自定义字体
# 获取脚本所在目录作为项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(project_root, 'fonts', 'SarasaMonoSlabSC-Regular.ttf')

# 确保输出文件保存在项目目录内
os.chdir(project_root)

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['Sarasa Mono Slab SC', 'Ubuntu', 'DejaVu Sans', 'SimHei', 'SimSun']
else:
    # 回退成系统默认
    print("无法在本地找到字体文件，将使用系统默认字体，可能会出现乱码")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Ubuntu', 'DejaVu Sans']

def load_data():
    """
    加载手写数字数据集
    Returns:
        X: 特征矩阵 (1797, 64)
        y: 标签向量 (1797,)
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"数据集形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    return X, y

def preprocess_data(X, y):
    """
    数据预处理
    Args:
        X: 原始特征数据
        y: 原始标签数据
    Returns:
        X_train, X_test, y_train, y_test: 预处理后的训练和测试数据
    """
    # 分层采样划分数据集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集大小: {X_train_scaled.shape}")
    print(f"测试集大小: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    训练多层感知机模型
    Args:
        X_train: 训练特征数据
        y_train: 训练标签数据
    Returns:
        model: 训练好的模型
    """
    # 创建MLP分类器
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    print("模型训练完成")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    Args:
        model: 训练好的模型
        X_test: 测试特征数据
        y_test: 测试标签数据
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算总体分类精度(OA)
    oa = accuracy_score(y_test, y_pred)
    print(f"总体分类精度(OA): {oa:.4f}")
    
    # 详细分类报告(包含平均分类精度AA等)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # TODO: 计算Kappa系数
    
    return y_pred

def visualize_data_pca(X, y):
    """
    使用PCA将数据降到2D并可视化
    Args:
        X: 原始特征数据
        y: 标签数据
    """
    # PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('手写数字数据集PCA可视化')
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("PCA可视化已保存为: pca_visualization.png")

def visualize_data_tsne(X, y):
    """
    使用t-SNE将数据降到2D并可视化
    Args:
        X: 原始特征数据
        y: 标签数据
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('手写数字数据集t-SNE可视化')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("t-SNE可视化已保存为: tsne_visualization.png")

def visualize_results(model, X_test, y_test, y_pred):
    """
    可视化预测结果
    Args:
        model: 训练好的模型
        X_test: 测试特征数据
        y_test: 真实标签
        y_pred: 预测标签
    """
    # 绘制混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("混淆矩阵已保存为: confusion_matrix.png")

def main():
    """
    主函数
    """
    print("=== 手写数字识别项目 ===")
    
    # 1. 加载数据
    X, y = load_data()
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # 3. 训练模型
    model = train_model(X_train, y_train)
    
    # 4. 评估模型
    y_pred = evaluate_model(model, X_test, y_test)
    
    # 5. 数据可视化
    visualize_data_pca(X, y)
    visualize_data_tsne(X, y)

    # 6. 结果可视化
    visualize_results(model, X_test, y_test, y_pred)
    
    print("项目执行完成")

if __name__ == "__main__":
    main()