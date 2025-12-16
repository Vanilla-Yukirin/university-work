"""
SVD 矩阵裁切与重建的详细示例
演示如何提取前 k 层进行图像重建
"""

import numpy as np

# 创建一个小的示例矩阵
print("=" * 60)
print("示例：4×5 矩阵的 SVD 分解与裁切")
print("=" * 60)

A = np.array([
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10],
    [1, 1, 1, 1, 1],
    [3, 2, 1, 0, -1]
], dtype=float)

print("\n原始矩阵 A (4×5):")
print(A)

# SVD 分解
U, S, VT = np.linalg.svd(A, full_matrices=False)

print("\n" + "=" * 60)
print("完整 SVD 分解结果:")
print("=" * 60)
print(f"\nU 的形状: {U.shape}")
print("U (左奇异向量矩阵):")
print(U)

print(f"\nS 的形状: {S.shape}")
print("S (奇异值向量，按降序排列):")
print(S)

print(f"\nVT 的形状: {VT.shape}")
print("VT (右奇异向量矩阵的转置):")
print(VT)

# 验证重建
A_reconstructed = U @ np.diag(S) @ VT
print("\n验证完整重建 (U @ diag(S) @ VT):")
print(A_reconstructed)
print(f"重建误差: {np.linalg.norm(A - A_reconstructed):.2e}")

print("\n" + "=" * 60)
print("使用前 k=2 层进行低秩近似")
print("=" * 60)

k = 2

# 方法1：显式构造对角矩阵
print(f"\n方法1：U[:, :k] @ diag(S[:k]) @ VT[:k, :]")
U_k = U[:, :k]          # 取前2列
S_k = S[:k]             # 取前2个奇异值
VT_k = VT[:k, :]        # 取前2行

print(f"U_k 形状: {U_k.shape} = (4, 2)")
print(f"U_k (前2个左奇异向量):\n{U_k}")

print(f"\nS_k 形状: {S_k.shape} = (2,)")
print(f"S_k (前2个奇异值): {S_k}")

print(f"\nVT_k 形状: {VT_k.shape} = (2, 5)")
print(f"VT_k (前2个右奇异向量):\n{VT_k}")

A_k_method1 = U_k @ np.diag(S_k) @ VT_k
print(f"\n低秩重建结果 A_k (方法1):")
print(A_k_method1)

# 方法2：广播优化（你的代码使用的方法）
print(f"\n方法2（高效）：(U[:, :k] * S[:k]) @ VT[:k, :]")
A_k_method2 = (U[:, :k] * S[:k]) @ VT[:k, :]
print(f"低秩重建结果 A_k (方法2):")
print(A_k_method2)

print(f"\n两种方法结果是否相同: {np.allclose(A_k_method1, A_k_method2)}")

# 分层展示每个奇异值的贡献
print("\n" + "=" * 60)
print("每一层的贡献（每个奇异值对应的秩-1矩阵）")
print("=" * 60)

for i in range(min(3, len(S))):
    layer = S[i] * np.outer(U[:, i], VT[i, :])
    print(f"\n第 {i+1} 层 (σ_{i+1} = {S[i]:.4f}):")
    print(f"σ_{i+1} × u_{i+1} × v_{i+1}^T =")
    print(layer)
    print(f"该层的范数: {np.linalg.norm(layer):.4f}")

# 展示不同 k 值的重建效果
print("\n" + "=" * 60)
print("不同秩的重建误差对比")
print("=" * 60)

print(f"\n{'k':<5} {'重建误差':<15} {'相对误差':<15} {'存储量(相对)':<15}")
print("-" * 60)
original_size = A.shape[0] * A.shape[1]
for k in range(1, len(S) + 1):
    A_k = (U[:, :k] * S[:k]) @ VT[:k, :]
    error = np.linalg.norm(A - A_k)
    relative_error = error / np.linalg.norm(A)
    storage = k * (U.shape[0] + VT.shape[1] + 1)
    storage_ratio = storage / original_size
    print(f"{k:<5} {error:<15.6f} {relative_error:<15.6%} {storage_ratio:<15.2%}")

print("\n" + "=" * 60)
print("可视化矩阵裁切过程")
print("=" * 60)

k = 2
print(f"""
完整 SVD:
A (4×5) = U (4×4) @ Σ (4×5) @ V^T (5×5)

取前 k={k} 层:
              ┌─────┬───┐           ┌─────┬───┐         ┌─────────┐
              │     │   │           │ σ₁  │ 0 │         │ v₁^T    │
U[:, :{k}] =   │ u₁  │u₂ │   Σ[:{k}] = │  0  │σ₂ │   VT[:{k}, :] = │ ─────── │
              │     │   │           └─────┴───┘         │ v₂^T    │
              └─────┴───┘                               └─────────┘
               (4 × {k})              ({k} × {k})              ({k} × 5)

重建: A_k = U[:, :{k}] @ diag(S[:{k}]) @ VT[:{k}, :]
     形状:  (4×{k})   @    ({k}×{k})    @   ({k}×5)  = (4×5)
""")

print("\n实际数据存储量对比:")
print(f"原始矩阵 A: {A.shape[0]} × {A.shape[1]} = {A.shape[0] * A.shape[1]} 个数")
print(f"SVD k={k} 表示: U[:,:k]={U.shape[0]}×{k} + S[:k]={k} + VT[:k,:]={k}×{VT.shape[1]}")
print(f"               = {U.shape[0] * k} + {k} + {k * VT.shape[1]} = {k * (U.shape[0] + 1 + VT.shape[1])} 个数")
print(f"压缩率: {k * (U.shape[0] + 1 + VT.shape[1]) / (A.shape[0] * A.shape[1]):.2%}")
