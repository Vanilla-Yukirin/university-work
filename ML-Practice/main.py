'''
环境激活conda activate ML
'''

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import seaborn as sns
from typing import Tuple

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - SciPy may be unavailable
    loadmat = None

# 使用自定义字体
# 获取脚本所在目录作为项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(project_root, 'fonts', 'SarasaMonoSlabSC-Regular.ttf')

# 确保输出文件保存在项目目录内
os.chdir(project_root)
RANDOM_STATE = 42

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['Sarasa Mono Slab SC', 'Ubuntu', 'DejaVu Sans', 'SimHei', 'SimSun']
else:
    # 回退成系统默认
    print("无法在本地找到字体文件，将使用系统默认字体，可能会出现乱码")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Ubuntu', 'DejaVu Sans']

def _load_from_mat_files() -> Tuple[np.ndarray, np.ndarray]:
    """
    如果存在 .mat 文件，则尝试读取并返回其数组。
    Returns:
        X, y: 如果成功读取返回对应矩阵，否则抛出 ValueError
    """
    if loadmat is None:
        raise ValueError("SciPy 未安装，无法读取 .mat 文件。")

    x_path = os.path.join(project_root, "sklearn_dataset_X.mat")
    y_path = os.path.join(project_root, "sklearn_dataset_Y.mat")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise ValueError("未找到 .mat 数据文件。")

    raw_x = loadmat(x_path)
    raw_y = loadmat(y_path)

    def _extract_first_array(container):
        for key, value in container.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.size > 0:
                return arr
        raise ValueError("数据文件结构不包含有效数组。")

    X = _extract_first_array(raw_x)
    y = _extract_first_array(raw_y).ravel()
    if y.ndim > 1:
        y = y.flatten()

    return X.astype(np.float64), y.astype(np.int64)


def load_data(prefer_mat: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载手写数字数据集，可选地优先读取 .mat 文件。
    Args:
        prefer_mat: 若为 True 则优先尝试使用本地 .mat 数据
    Returns:
        X: 特征矩阵 (1797, 64)
        y: 标签向量 (1797,)
    """
    if prefer_mat:
        try:
            X, y = _load_from_mat_files()
            print("已从本地 .mat 文件加载数据。")
            return X, y
        except ValueError as err:
            print(f".mat 数据加载失败：{err}，改用 sklearn 内置数据。")

    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"已从 sklearn.datasets.load_digits 加载数据。")
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
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
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
        random_state=RANDOM_STATE
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
    aa = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"总体分类精度(OA): {oa:.4f}")
    print(f"平均分类精度(AA): {aa:.4f}")
    print(f"Kappa 系数: {kappa:.4f}")
    
    # 详细分类报告(包含平均分类精度AA等)
    print("\n分类报告:")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = {
        digit: accuracy_score(y_test[y_test == digit], y_pred[y_test == digit])
        for digit in np.unique(y_test)
    }
    print("各类别准确率:")
    for digit in sorted(per_class_accuracy):
        print(f"  数字 {digit}: {per_class_accuracy[digit]:.4f}")

    return {
        "y_pred": y_pred,
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "cm": cm,
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
    }


def cross_validate_model(X, y, n_splits: int = 5) -> np.ndarray:
    """
    使用分层 K 折交叉验证评估模型的稳定性。
    Args:
        X: 全量特征
        y: 全量标签
        n_splits: 折数
    Returns:
        scores: 每折的准确率
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(100, 50, 25),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
    print(
        f"{n_splits} 折交叉验证准确率: "
        f"{[f'{score:.4f}' for score in scores]} -> 平均 {scores.mean():.4f} ± {scores.std():.4f}"
    )
    return scores

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
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=RANDOM_STATE)
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

def visualize_results(conf_mat: np.ndarray):
    """
    可视化预测结果
    Args:
        conf_mat: 混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("混淆矩阵已保存为: confusion_matrix.png")


def visualize_misclassifications(
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    scaler: StandardScaler,
    max_samples: int = 16,
):
    """
    展示部分预测错误的样本，便于分析模型不足。
    """
    misclassified_idx = np.where(y_test != y_pred)[0]
    if misclassified_idx.size == 0:
        print("测试集中未发现预测错误的样本。")
        return

    sample_idx = misclassified_idx[:max_samples]
    restored_images = scaler.inverse_transform(X_test_scaled[sample_idx])

    cols = 4
    rows = int(np.ceil(len(sample_idx) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(sample_idx):
        ax = plt.subplot(rows, cols, i + 1)
        image = restored_images[i].reshape(8, 8)
        ax.imshow(np.clip(image, 0, 16), cmap="gray")
        ax.set_title(f"真: {y_test[idx]}  预测: {y_pred[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("misclassified_examples.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("误分类样本可视化已保存为: misclassified_examples.png")

def main():
    """
    主函数
    """
    print("=== 手写数字识别项目 ===")
    
    # 1. 加载数据
    X, y = load_data()

    # 交叉验证评估整体可行性
    cross_validate_model(X, y)
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # 3. 训练模型
    model = train_model(X_train, y_train)
    
    # 4. 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 5. 数据可视化
    visualize_data_pca(X, y)
    visualize_data_tsne(X, y)

    # 6. 结果可视化
    visualize_results(metrics["cm"])
    visualize_misclassifications(
        X_test, y_test, metrics["y_pred"], scaler=scaler, max_samples=12
    )
    
    print("项目执行完成")

if __name__ == "__main__":
    main()
