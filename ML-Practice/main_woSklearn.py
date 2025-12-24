'''
环境激活conda activate ML
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

#项目配置
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
RANDOM_STATE = 42

#中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Ubuntu', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def _load_from_mat_files() -> Tuple[np.ndarray, np.ndarray]:
    """从本地.mat文件加载数据"""
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


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """加载手写数字数据集（优先读取本地 .mat）"""
    X, y = _load_from_mat_files()
    print("已从本地 .mat 文件加载数据。")
    print(f"数据集形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    return X, y


def set_seed(seed: int) -> np.random.Generator:
    """设置随机种子并返回 RNG"""
    return np.random.default_rng(seed)


class StandardScalerNumpy:
    """只使用 numpy 的标准化器"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "StandardScalerNumpy":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分层采样划分训练集与测试集"""
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(np.round(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def preprocess_data(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """数据预处理：分层采样和标准化"""
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=0.2, rng=rng
    )

    scaler = StandardScalerNumpy()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"训练集大小: {X_train_scaled.shape}")
    print(f"测试集大小: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


class MLPClassifierNumpy:
    """用numpy实现多层感知机"""

    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        learning_rate: float = 0.01,
        epochs: int = 200,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_params()

    def _init_params(self):
        self.weights.clear()
        self.biases.clear()
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight = self.rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
            bias = np.zeros(out_dim)
            self.weights.append(weight)
            self.biases.append(bias)

    def _forward(self, X: np.ndarray):
        activations = [X]
        pre_activations = []
        for idx in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[idx] + self.biases[idx]
            pre_activations.append(z)
            activations.append(relu(z))
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        probs = softmax(z)
        activations.append(probs)
        return activations, pre_activations

    def _backward(
        self,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
        y_onehot: np.ndarray,
    ):
        batch_size = y_onehot.shape[0]
        delta = (activations[-1] - y_onehot) / batch_size

        for layer in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[layer]
            grad_w = a_prev.T @ delta
            grad_b = np.sum(delta, axis=0)
            self.weights[layer] -= self.learning_rate * grad_w
            self.biases[layer] -= self.learning_rate * grad_b
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * relu_grad(
                    pre_activations[layer - 1]
                )

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        n_classes = self.layer_sizes[-1]
        for epoch in range(1, self.epochs + 1):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_onehot = np.eye(n_classes)[y_batch]
                activations, pre_activations = self._forward(X_batch)
                self._backward(activations, pre_activations, y_onehot)

            if epoch % 20 == 0 or epoch == 1 or epoch == self.epochs:
                preds = self.predict(X)
                acc = np.mean(preds == y)
                print(f"Epoch {epoch:03d}: 训练集准确率 {acc:.4f}")
        print("模型训练完成")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        return activations[-1]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    classes = np.unique(y_true)
    n_classes = classes.size
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(np.mean(y_pred[mask] == cls))
    return float(np.mean(recalls)) if recalls else 0.0


def cohen_kappa(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    po = np.trace(cm) / total
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    pe = np.sum(row_sum * col_sum) / (total * total)
    if 1 - pe == 0:
        return 0.0
    return float((po - pe) / (1 - pe))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    classes = np.unique(y_true)
    lines = []
    header = "类别  精确率  召回率  F1分数  支持数"
    lines.append(header)
    for cls in classes:
        mask = y_true == cls
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = int(mask.sum())
        lines.append(
            f"{cls:>2d}  {precision:7.4f}  {recall:7.4f}  {f1:7.4f}  {support:6d}"
        )
    return "\n".join(lines)


def evaluate_model(model: MLPClassifierNumpy, X_test: np.ndarray, y_test: np.ndarray):
    """评估模型性能：计算OA、AA、Kappa系数等指标"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    oa = accuracy_score(y_test, y_pred)
    aa = balanced_accuracy(y_test, y_pred)
    kappa = cohen_kappa(cm)
    print(f"总体分类精度OA={oa:.4f}")
    print(f"平均分类精度AA={aa:.4f}")
    print(f"Kappa={kappa:.4f}")

    print("\n分类报告:")
    report = classification_report(y_test, y_pred)
    print(report)

    per_class_accuracy = {
        digit: accuracy_score(y_test[y_test == digit], y_pred[y_test == digit])
        for digit in np.unique(y_test)
    }
    print("各类别准确率:")
    for digit in sorted(per_class_accuracy):
        print(f"\t{digit}={per_class_accuracy[digit]:.4f}")

    return {
        "y_pred": y_pred,
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "cm": cm,
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
    }


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int, rng: np.random.Generator
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """生成分层 K 折的训练/测试索引"""
    fold_bins = [list() for _ in range(n_splits)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        splits = np.array_split(cls_idx, n_splits)
        for fold_id, fold_idx in enumerate(splits):
            fold_bins[fold_id].extend(fold_idx.tolist())

    all_indices = np.arange(len(y))
    folds = []
    for fold_id in range(n_splits):
        test_idx = np.array(sorted(fold_bins[fold_id]))
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_indices[train_mask]
        folds.append((train_idx, test_idx))
    return folds


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_splits: int = 5,
) -> np.ndarray:
    """使用分层K折交叉验证评估模型稳定性"""
    scores = []
    folds = stratified_kfold_indices(y, n_splits=n_splits, rng=rng)
    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        scaler = StandardScalerNumpy()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = MLPClassifierNumpy(
            layer_sizes=(X.shape[1], 100, 50, 25, len(np.unique(y))),
            learning_rate=0.01,
            epochs=150,
            batch_size=64,
            random_state=RANDOM_STATE + fold_id,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        print(f"第 {fold_id} 折准确率: {acc:.4f}")

    scores = np.array(scores)
    print(
        f"{n_splits} 折交叉验证准确率: "
        f"{[f'{score:.4f}' for score in scores]} -> 平均 {scores.mean():.4f} ± {scores.std():.4f}"
    )
    return scores


def pca_reduce_2d(X: np.ndarray) -> np.ndarray:
    """将数据PCA到2D"""
    X_centered = X - X.mean(axis=0)
    _, _, v_t = np.linalg.svd(X_centered, full_matrices=False)
    components = v_t[:2].T
    return X_centered @ components


def visualize_data_pca(X: np.ndarray, y: np.ndarray):
    """使用PCA将数据降到2D并可视化"""
    X_pca = pca_reduce_2d(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("手写数字数据集PCA可视化")
    plt.xlabel("第一主成分")
    plt.ylabel("第二主成分")
    plt.savefig("pca_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("PCA可视化完毕: pca_visualization.png")


def visualize_results(conf_mat: np.ndarray):
    """可视化混淆矩阵"""
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_mat, cmap="Blues")
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.colorbar()
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j], ha="center", va="center", color="black")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("混淆矩阵已保存为: confusion_matrix.png")


def visualize_misclassifications(
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    scaler: StandardScalerNumpy,
    max_samples: int = 16,
):
    """展示误分类样本"""
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
    plt.close()
    print("误分类样本可视化已保存为: misclassified_examples.png")

def main():
    """主函数"""
    print("=== 手写数字识别项目 ===")

    rng = set_seed(RANDOM_STATE)

    # 加载数据
    X, y = load_data()

    # 交叉验证
    cross_validate_model(X, y, rng=rng)

    # 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, rng=rng)

    # 训练模型
    model = MLPClassifierNumpy(
        layer_sizes=(X_train.shape[1], 100, 50, 25, len(np.unique(y))),
        learning_rate=0.01,
        epochs=200,
        batch_size=64,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)

    # 数据可视化
    visualize_data_pca(X, y)

    # 结果可视化
    visualize_results(metrics["cm"])
    visualize_misclassifications(
        X_test, y_test, metrics["y_pred"], scaler=scaler, max_samples=12
    )

    print("项目执行完成")

if __name__ == "__main__":
    main()
