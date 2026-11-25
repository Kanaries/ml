# e-means 聚类调研与 @kanaries/ml 集成方案

2026-11-25 · 面向浏览器与 Node.js 的 TypeScript 实现提案

**建议立项，先验证一个独立的 `Clusters.EMeans`。** 首个目标场景是对一批固定的稠密向量反复做最近中心分配，例如给图像或文本 embedding 建码本。首版建议以 Faiss 1.16.0 的 CPU 欧氏距离实现为数值基准。公开发布的依据应是本库在目标数据规模上达到同等误差所花的时间。下面的小实验验证了质量和已知风险，JavaScript 性能仍待测量。

**实施记录。** 上述方案现已落地。查阅[公开 API 指南](content/docs/apis/clusters/eMeans.mdx)、[复现实验说明](../scripts/e-means/README.md)和[760 次 JS 拟合的原始结果](../scripts/e-means/results.csv)。在两个运行时，GloVe 与 SIFT 都观察到了等质量耗时收益，二维样本上 Lloyd 更快。

作为版本选择的补充验证，同一组 Python 对照还运行了 11 月 25 日 Faiss 开发提交的移除代价策略。五种子中位数在手写数字上为 1.904，在稀有簇上为 6.962。实验使用该提交的 NumPy 实现、从同一提交编译的三个单线程 CPU 统计函数和 1.16.0 的其余 Faiss 原语。发货版本仍采用下文锁定的 1.16.0 配方；这组实验未测开发版的 JS 耗时。[开发提交](https://github.com/facebookresearch/faiss/commit/0c0f203669a5b7493260b268351280064f8fe561)

## 1. 用户听到的是哪一种 E-Means

Swann Bessa 等人在 2026-11-06 提交了 e-means 预印本。这里的 e 指指数移动平均（EMA）。用户仍需给定簇数 k，训练后按最近中心分配样本。更早的同名论文 *E-Means: An Evolutionary Clustering Algorithm* 用进化算法估计簇数，两者用途不同。本次集成对象是 2026 年的方法。[2026 年原论文](https://arxiv.org/abs/2611.09501) · [早期同名论文](https://citeseerx.ist.psu.edu/document?doi=2ef426d239c4bb2c52b5e57b9e0ae7822e57ef98&repid=rep1&type=pdf)

共同目标是减小每个样本到最近中心的平方欧氏距离之和（inertia）。对无权重数据，除以样本数 n 就是本文的 MSE。这个指标度量向量量化误差，不能直接解释为分类准确率。

| 方法 | 中心何时移动 | 历史样本怎样影响当前中心 |
| --- | --- | --- |
| Lloyd K-Means | 每遍历全量数据一次 | 按本轮分配重新取均值 |
| 本库 MiniBatchKMeans | 每处理一个有放回抽样的小批次 | 各簇累计计数；一个已有大量样本的簇，后续移动较小 |
| e-means | 每轮打乱行顺序，每个批次结束时更新 | 按相同因子衰减各簇的历史向量和与计数；较新的分配能更快移动中心 |

K-Means++ 是挑选初始中心的策略，可与 Lloyd 等更新规则组合使用。e-means 原论文主要用无放回的均匀随机初始化，节省顺序选点的开销。[论文 §3–4](https://arxiv.org/html/2611.09501v1) · [本库 MiniBatchKMeans](../src/clusters/miniBatchKMeans.ts)

## 2. 实现时必须区分的三件事

**一次更新。** 用批次开始时的中心完成整批分配。对每个簇，令 m 为本批样本数，s 为本批向量和，N 和 S 为历史统计量。更新为 `N ← (1−α)N + αm`、`S ← (1−α)S + αs`，中心取 `S/N`。批内没有成员的中心保持原位置。这样，即使两个批次有相同均值，含有更多该簇样本的批次也能施加更大影响。批大小 B 由 n、k、轮数 T 的经验公式求出，再由 B、n、k、T 求初始步长；训练中用余弦调度把步长降向初值的十分之一。[论文算法 1、式 5–7](https://arxiv.org/html/2611.09501v1) · [Faiss 固定版本的更新实现](https://github.com/facebookresearch/faiss/blob/v1.16.0/contrib/e_means/e_means.py)

**小簇复活。** 原方法每 5 个批次检查一次。若某簇的衰减计数低于平均值的 10%，就按计数加权选一个活跃簇，把供体拆成两个微扰中心并平分其计数。这个操作能把闲置中心移回高密度区域，也可能撤走一个负责偏远小群体的中心。独立研究既给出了同初始中心下劣于 Lloyd 的反例，也在有孤立小群体的数据上实测了加入距离条件的效果。[反例论文](https://arxiv.org/abs/2611.22065) · [距离感知复活研究](https://arxiv.org/abs/2611.18238)

**最后一轮与用户拿到的标签。** 默认收尾用最后一轮各批次实际分配的样本重算簇均值，空簇保留当前位置。各批次看到的中心会移动，所以最后一轮记下的标签可能与最终中心的最近邻不同。建议本库在收尾后额外做一次精确分配，保证 `getLabels()`、`fitPredict(X)`、`predict(X)` 和 `getInertia()` 对应同一组最终中心。预算为 T 轮时，这一设计给训练集增加一次分配扫描；基准须计入。论文还探索了面向留出误差的慢 EMA 收尾，首轮验证采用默认的最后一轮均值。[Faiss 训练与标签说明](https://github.com/facebookresearch/faiss/blob/v1.16.0/contrib/e_means/README.md) · [论文 §3.2](https://arxiv.org/html/2611.09501v1)

### 选定参考版本

Faiss 于 11 月 10 日发布 v1.16.0。该版初始计数为 2⁻¹⁶，论文为 10⁻¹⁰；工程版的批大小和步长系数也不同。工程版默认 5 轮；自动配方在 d ≤ 16 选偏质量的小批次，在 d ≤ 2 默认关闭按计数复活。应直接用固定版本生成参考数据，覆盖系数、取整、上下限和尾批。[Faiss v1.16.0](https://github.com/facebookresearch/faiss/releases/tag/v1.16.0) · [固定版本配方](https://github.com/facebookresearch/faiss/blob/v1.16.0/contrib/e_means/base.py)

11 月下旬，上游开发分支又引入了基于移除代价的复活：通过第一、第二近中心的距离差估计撤走中心会损失多少，并考虑拆分供体的预期收益。它会改变每批分配的成本和最后一轮的标签处理。建议先复现已发布版，同时把新策略放进稀有簇对照实验，取得证据后再决定是否升级。[上游移除代价改动](https://github.com/facebookresearch/faiss/commit/16a95d63f2980b90c19e5308b4baec8f050b9b5b) · [上游拆分收益改动](https://github.com/facebookresearch/faiss/commit/0c0f203669a5b7493260b268351280064f8fe561)

## 3. 性能证据能支持什么

作者在多组 embedding 上报告，e-means 用 **1.4–2.6 轮**达到或优于 Lloyd 运行 **25 轮**的损失水平，GPU 实际耗时加速 **10–45 倍**。DINO10B 实验还用到了 IVF 近似分配。其配方主要在大规模向量数据上拟合，论文给出的拟合范围为 n 从 10⁵ 到 10⁸、k 从 256 到 65,536；网页里的几千个样本、几个簇属于外推。作者也提供了可操作的浏览器演示。[论文实验与拟合范围](https://arxiv.org/html/2611.09501v1) · [作者项目与演示](https://e-means.github.io/)

精确分配下，三种方法都要比较样本与中心，e-means 每轮距离计算量仍为 O(nkd)。它可能通过减少达到目标误差所需的轮数来省时。JS 的数组访问、打乱顺序、每批归并以及收尾扫描都会影响端到端收益，二维小数据上还可能是 Lloyd 先完成。

### 本次实际运行的 CPU 对照

固定初始中心，每组运行 5 个种子。下表是独立留出集上“到最近最终中心的平方距离”的均值，再对种子取中位数，越低越好。Lloyd 使用 sklearn 单次初始化、最多 25 轮；e-means 使用 Faiss 1.16.0、5 轮。这里比较解的质量。

| 数据 | 训练 / 留出 | d / k | Lloyd | e-means 默认 | e-means 关闭复活 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sklearn 手写数字，像素缩放至 0–1 | 1,400 / 397 | 64 / 32 | 1.955 | 1.941 | 1.954 |
| 99% 主群体，1% 远端群体，初始中心覆盖远端 | 5,000 / 1,000 | 8 / 4 | 6.942 | 32.012 | 6.962 |

数字数据上默认 e-means 的中位误差略低。第二组特意把一个初始中心放在远端，以单独检验训练能否保住这个簇；默认复活的中位误差约为 Lloyd 的 4.6 倍，关闭复活后差距缩到约 0.3%。这支持公开一个能关闭复活的选项。两组结果只覆盖这里的分布和预算，尚需真实 embedding 的 JS 对照。

## 4. 如何接入现有仓库

建议新增 `src/clusters/eMeans.ts`，继承 `ClusterBase`，在 `src/clusters/index.ts` 导出类和 `EMeansProps`，由包入口已有的 `Clusters` 命名空间对外提供。采用与 MiniBatchKMeans 一致的 camelCase 配置。下面是拟议用法。[聚类模块出口](../src/clusters/index.ts) · [包入口](../src/index.ts) · [估计器契约](estimator-contract.md)

```ts
import { Clusters } from '@kanaries/ml';

const model = new Clusters.EMeans({ nClusters: 32, epochs: 5, randomState: 42 });
const labels = model.fitPredict(trainVectors);
const nearest = model.predict(newVectors);
```

| 决策 | 首版建议 |
| --- | --- |
| 配置 | `nClusters` 默认 8，与 MiniBatchKMeans 一致；`epochs` 默认 5，首版支持正整数；可选 `randomState`、`batchSize` 和 `revivalThreshold`。省略阈值使用 v1.16.0 的按维数默认值，传 0 关闭复活。步长随实际批大小重新推导。 |
| 数据与结果 | 接收有限值、等宽的稠密 `number[][]`；要求 1 ≤ k ≤ n。提供 `fit`、`fitPredict`、`predict`、`getCentroids`、`getLabels`、`getInertia`。训练失败、重复 fit、预测维数不符和 fit 前预测的行为由测试固定。首版限定无权重目标；如通过基类传入非单位样本权重应明确报错。 |
| 复用边界 | 复用现有 seeded RNG、估计器注册、序列化和 conformance 测试。MiniBatchKMeans 的分配函数目前是文件内函数，其训练环包含有放回抽样、累计计数和早停。仅当输入表示与行为一致时提取共用距离计算；e-means 单独实现训练环。 |
| CPU 与内存 | 公共接口沿用普通数组，热循环试用连续的中心、批统计与索引缓冲区；优先按行读取输入，复用工作内存。精确扫描只保留当前最近中心，总额外空间目标 O(n + kd)，含打乱索引和输出标签。大输入示例可在 Web Worker 中跑同步训练。 |
| 可复现性 | 通过 `getParams()` 返回调用方参数，在 fit 内解析依赖 n、d 的默认值。注册 `registerEstimator('EMeans', EMeans)`，验证 clone、setParams 和 JSON 模型恢复。显式 seed 保证同一 JS 实现可复现。 |
| 后续能力 | 首版聚焦静态数据与平方欧氏距离。无限流 `partialFit` 需要另定总步数和收尾语义；WebGPU、IVF、球面距离、公开平铺向量输入可分别按真实瓶颈立项。 |

## 5. 实施顺序与发布判断

| 阶段 | 交付与验收 |
| --- | --- |
| A · 数值基线 | 用固定版本 `NumpyEmeans` 生成小型 JSON fixture，记录显式初始中心、每轮行序、复活供体选择、逐批中心和最终精确标签。覆盖无复活和有复活两条路径。固定这些输入可隔离 NumPy 与本库 RNG 的差异；以 Float32 参考值和预先规定的容差比较中心与 inertia，平局另测。 |
| B · TypeScript 实现 | 按固定配方实现整数轮训练、保留尾批、最后一轮收尾与最终精确分配。加入聚类 conformance；测试固定 seed 重跑和序列化后的预测、k=1、k=n、重复点、空簇、α=1、极小数据、非整批、非法值、维数不一致。与仓库的 KMeans 和 MiniBatchKMeans 回归用例一同运行，再做全量 `npm run test` 与 `yarn build`。 |
| C · 双端基准 | 在 Node.js 与真实 Chromium 分别比较三种方法。使用固定划分和至少 5 个种子，包含上面两组、二维可视化样本，以及有明确来源的真实 embedding 子集（如 SIFT1M 的固定子集作向量检索基准，再选一组文本 embedding）。以 1 万至 10 万行、32 至 256 个中心起步，记录硬件、运行时、预热、实际轮数、峰值内存和留出 MSE。 |
| D · 发布判断 | 对每个数据集取单次初始化 Lloyd 最多 25 轮的留出 MSE 为目标，拟议容差为 +1%。在相同种子和初始中心下比较各预算首次达到目标的端到端时间；另列三个公共 API 的默认配置。要求目标 embedding 场景在目标运行时有可重复的时间收益；未达到目标的配置标为未达到。若只在一端获益，发布说明限定该环境；若两端都无收益，保留为实验。 |
| E · 用户文档 | 建议页面标题为 *E-Means Clustering in JavaScript*，介绍浏览器与 Node.js 的使用场景、快速示例、复活开关及实测范围。示例用相同预算比较三种算法，可切换有稀有簇的数据，并标明误差的计算方式。[现有 K-Means 文档](content/docs/apis/clusters/kmeans.mdx) |

基准需要两条对照线。算法对照通过内部夹具共享初始中心与单次启动，同一评估函数从最终中心重算分数；MiniBatchKMeans 的共享初始中心由测试驱动注入。实际使用对照保留各自公开默认值，因为本库 KMeans 默认有 10 次重启、MiniBatchKMeans 默认尝试 3 次初始化并允许早停。两条线都把初始化、训练、最终标签分配计入耗时，另列只算训练的时间，避免把重启开销误认成训练规则的收益。[本库 KMeans](../src/clusters/kmeans.ts) · [本库 MiniBatchKMeans](../src/clusters/miniBatchKMeans.ts)

## 附录 · 复现本次小实验

使用 Python 3.14、faiss-cpu 1.16.0、NumPy 2.5.1、scikit-learn 1.9.0。训练与测试分别构建，无外部数据下载，固定数据切分；相同初始中心只在同一数据集、同一种子下配对。运行时将 OpenMP、OpenBLAS 和 Accelerate 的线程数设为 1。下方代码输出各个种子的 MSE 与中位数。

```python
import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

faiss.omp_set_num_threads(1)
data = load_digits().data.astype("float32") / 16
order = np.random.default_rng(2026).permutation(len(data))
digits = data[order[:1400]], data[order[1400:]]
rng = np.random.default_rng(2026)
def sample_rare(n):
    x = rng.normal(0, 1, size=(n, 8)).astype("float32")
    x[-(n // 100):] += 20
    return x
rare = sample_rare(5000), sample_rare(1000)

for name, (train, test), k in [("digits", digits, 32), ("rare", rare, 4)]:
    scores = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(train) if name == "digits" else 4950,
                         k if name == "digits" else k - 1, replace=False)
        if name == "rare":
            ids = np.append(ids, 4999)
        init = train[ids].copy()
        lloyd = KMeans(k, init=init, n_init=1, max_iter=25, tol=0,
                       algorithm="lloyd").fit(train)
        centers = [lloyd.cluster_centers_]
        for threshold in (None, 0):
            model = NumpyEmeans(train.shape[1], k, nepochs=5, seed=seed,
                                unused_threshold=threshold)
            c, _ = model.train(train, init_centroids=init)
            centers.append(c)
        mse = []
        for c in centers:
            dist = ((test.astype("float64")[:, None] - c) ** 2).sum(axis=2)
            mse.append(dist.min(axis=1).mean())
        scores.append(mse)
    print(name, np.round(scores, 3), "median", np.median(scores, axis=0))
```
