下面按模块讲当前代码的运作原理和主要数学公式（对应 main_woSklearn.py 的实现）。

数据加载与预处理

从 sklearn_dataset_X.mat 和 sklearn_dataset_Y.mat 读取数据，得到 X ∈ R^{N×64}，y ∈ {0..9}。
分层采样 8:2：对每个类别分别打乱后按比例切分，保证训练/测试类别分布一致。
标准化（均值方差归一）：
均值 μ = mean(X_train, axis=0)
标准差 σ = std(X_train, axis=0)
变换 X' = (X - μ) / σ
MLP 多层感知机结构

网络结构（默认）：64 -> 100 -> 50 -> 25 -> 10
隐藏层激活：ReLU
ReLU(z) = max(0, z)
梯度 ReLU'(z) = 1(z>0)
前向传播

第 l 层线性变换：
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
激活：
隐藏层：a^{(l)} = ReLU(z^{(l)})
输出层：a^{(L)} = softmax(z^{(L)})
Softmax：
p_i = exp(z_i) / Σ_j exp(z_j)（代码里做了数值稳定的 z - max(z)）
损失函数与反向传播

目标是多分类交叉熵（代码隐式实现）：
L = - (1/N) Σ_n Σ_k y_{n,k} log(p_{n,k})
Softmax + 交叉熵的梯度：
δ^{(L)} = (p - y_onehot) / N
反向传播：
∂W^{(l)} = (a^{(l-1)})^T δ^{(l)}
∂b^{(l)} = Σ δ^{(l)}
δ^{(l-1)} = (δ^{(l)} W^{(l)T}) ⊙ ReLU'(z^{(l-1)})
参数更新（SGD mini-batch）：
W := W - lr * ∂W
b := b - lr * ∂b
评估指标

总体精度 OA：
OA = (1/N) Σ 1(y_i = ŷ_i)
平均精度 AA（宏平均召回）：
AA = (1/C) Σ_c Recall_c
Kappa：
po = trace(CM) / N
pe = Σ (row_sum * col_sum) / N^2
κ = (po - pe) / (1 - pe)
PCA 2D 可视化

数据中心化：X_c = X - mean(X)
SVD：X_c = U Σ V^T
取前两主成分 V[:, :2]
投影：X_2d = X_c * V[:, :2]
输出可视化

生成 PCA 散点图、混淆矩阵热力图、误分类样本拼图。
误分类展示会把标准化后的样本逆变换回原始像素值再绘图。
如果你希望我按函数逐段讲解，或者对某一段推导更细（例如 softmax+交叉熵梯度推导），告诉我即可。