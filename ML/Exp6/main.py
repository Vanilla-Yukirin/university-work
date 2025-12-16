
"""
综合实验六：使用奇异值分解（SVD）进行图像压缩与可视化。

1. 加载Noir_from_ZN.png
2. 执行SVD
3. 展示不同秩下的近似重建结果
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.ticker as mticker

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


_configure_fonts()


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """
    读取并转换图像为灰度矩阵。

    Args:
        image_path: 图像文件路径。

    Returns:
        灰度图像对应的二维 numpy 数组（float64，范围 0~255）。
    """
    if not image_path.exists():
        raise FileNotFoundError(f"未找到图片：{image_path}")

    # 使用 Pillow 读取图片，convert("L") 表示转换为单通道灰度图
    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image, dtype=np.float64)
    return image_array


def svd_decompose(image_matrix: np.ndarray):
    """
    对图像矩阵执行奇异值分解。

    Returns:
        U, S, VT: numpy.linalg.svd 的标准输出，S 为奇异值向量。
    """
    # full_matrices=False 可以减少计算量，只保留有效的秩部分
    return np.linalg.svd(image_matrix, full_matrices=False)


def reconstruct_image(U: np.ndarray, S: np.ndarray, VT: np.ndarray, rank: int) -> np.ndarray:
    """
    利用前 rank 个奇异值重建图像矩阵。
    """
    rank = max(1, min(rank, len(S)))
    # 等价于 U[:, :rank] @ np.diag(S[:rank]) @ VT[:rank, :]
    compressed = (U[:, :rank] * S[:rank]) @ VT[:rank, :]
    return np.clip(compressed, 0, 255)


def choose_ranks(max_rank: int) -> List[int]:
    """
    根据图像的最大秩挑选出若干个有代表性的 rank 值。
    """
    preset = [5, 20, 50, 100, 150, 200]
    ranks = [r for r in preset if r <= max_rank]

    if not ranks:
        # 针对小尺寸图像，均匀采样若干个 rank 值
        step = max(1, max_rank // 4)
        ranks = list(range(step, max_rank + 1, step))

    # 确保至少展示原图（所有奇异值）
    if ranks[-1] != max_rank:
        ranks.append(max_rank)

    return ranks


def compression_ratio(height: int, width: int, rank: int) -> float:
    """
    计算使用 rank 个奇异值时的理论压缩率。

    原矩阵需要存储 height * width 个像素；
    SVD 形式需要 rank * (height + width + 1) 个数值。
    """
    original_params = height * width
    svd_params = rank * (height + width + 1)
    return svd_params / original_params


def visualize_results(
    original: np.ndarray,
    reconstructions: Dict[int, np.ndarray],
    singular_values: np.ndarray,
    save_dir: Path,
) -> None:
    """
    绘制原图、不同秩的重建结果以及奇异值曲线。

    Args:
        original: 原始图像矩阵
        reconstructions: 不同秩的重建结果字典
        singular_values: 奇异值数组
        save_dir: 保存图片的目录
    """
    height, width = original.shape
    ranks = sorted(reconstructions.keys())

    # 第一幅图：原图 + 不同秩的对比
    total_cols = len(ranks) + 1
    fig, axes = plt.subplots(1, total_cols, figsize=(3 * total_cols, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("原始灰度图")
    axes[0].axis("off")

    for ax, rank in zip(axes[1:], ranks):
        ax.imshow(reconstructions[rank], cmap="gray")
        ratio = compression_ratio(height, width, rank)
        ax.set_title(f"rank={rank}\n压缩率≈{ratio:.2%}")
        ax.axis("off")

    fig.suptitle("SVD 不同秩下的重建效果", fontweight="bold")
    plt.tight_layout()

    # 保存第一幅图（高分辨率）
    save_path_1 = save_dir / "svd_reconstruction_comparison.png"
    fig.savefig(save_path_1, dpi=300, bbox_inches="tight")
    print(f"已保存重建对比图：{save_path_1}")

    # 第二幅图：奇异值衰减曲线（对数坐标更直观）
    fig_sv, ax_sv = plt.subplots(figsize=(8, 6))
    ax_sv.semilogy(singular_values, marker="o", linestyle="-", linewidth=2, markersize=4)
    ax_sv.set_xlabel("奇异值序号", fontsize=12)
    ax_sv.set_ylabel("奇异值大小（对数尺度）", fontsize=12)
    ax_sv.set_title("奇异值衰减曲线", fontsize=14, fontweight="bold")
    # 使用 ASCII 科学计数法，避免 Unicode 减号（U+2212）
    ax_sv.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
    ax_sv.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # 保存第二幅图（高分辨率）
    save_path_2 = save_dir / "svd_singular_values.png"
    fig_sv.savefig(save_path_2, dpi=300, bbox_inches="tight")
    print(f"已保存奇异值曲线图：{save_path_2}")

    plt.show()


def main() -> None:
    """
    脚本主入口：完成图像读取、SVD 分解、重建与可视化。
    """
    image_path = Path(__file__).with_name("Noir_from_ZN.png")
    save_dir = Path(__file__).parent  # 保存在脚本所在目录

    image_matrix = load_grayscale_image(image_path)
    height, width = image_matrix.shape

    print(f"载入图像：{image_path.name} ，尺寸：{height}x{width}")

    # 计算 SVD 分解
    U, S, VT = svd_decompose(image_matrix)
    print(f"奇异值数量：{len(S)}")

    # 选择若干个 rank 用于展示，并生成对应的重建图像
    ranks = choose_ranks(len(S))
    reconstructions = {}
    for rank in ranks:
        reconstructions[rank] = reconstruct_image(U, S, VT, rank)
        ratio = compression_ratio(height, width, rank)
        print(f"rank={rank:3d} => 压缩率≈{ratio:.2%}")

    visualize_results(image_matrix, reconstructions, S, save_dir)


if __name__ == "__main__":
    main()
