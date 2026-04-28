# Swap Classifier Pretrain 需求文档与实现流程

> 版本: v0.2  
> 日期: 2026-04-25  
> 关联代码: `pretrain/swap_classifier_pretrain.py`

---

## 1. 任务目标

在 `pretrain` 阶段设计一个独立的 **三分类 classification 任务**，让模型判断输入拼图图像属于哪一种扰动类型。

输入是一张 `288 x 288` 的 `3 x 3` 拼图图像，中心块固定。  
输出是该图像所属的打乱类型：

| 类别 id | 类别名 | 含义 |
|---|---|---|
| 0 | `ordered` | 不做任何变换，图像保持正确顺序 |
| 1 | `inner_swap` | 图内部做一次 swap，且 **中心块不能参与 swap** |
| 2 | `outer_replace` | 随机将一个外部 piece 替换图内一个非中心 piece |

该任务的核心目的，是让 backbone 在预训练阶段学会区分：

- 完整正确拼图的全局一致性
- 图内局部错位带来的结构破坏
- 外来 piece 带来的风格/语义不一致

这会为后续拼图排序、归属判断、跨图交换等任务提供更强的表征能力。

---

## 2. 任务定义

### 2.1 输入

输入给模型的是一张单图：

- shape: `3 x 288 x 288`
- 来源: 原始数据集中一张正确顺序的拼图图像
- 每张图可以划分为 9 个 `96 x 96` 的 piece

拼图网格如下：

```text
(0,0) (0,1) (0,2)
(1,0) (1,1) (1,2)
(2,0) (2,1) (2,2)
```

其中：

- 中心块位置固定为 `(1,1)`，对应索引 `4`
- 可操作 piece 为其余 8 个位置

### 2.2 输出

模型输出长度为 3 的分类 logits：

```text
[logit_ordered, logit_inner_swap, logit_outer_replace]
```

训练时使用 `CrossEntropyLoss`。

预测类别定义为：

```python
pred = torch.argmax(logits, dim=-1)
```

---

## 3. 数据构造规则

### 3.1 基础样本

每次从数据集中取出一张图像及其 label。

现有数据格式与项目当前 `.npy` 数据保持一致：

- 图像: `train_x / test_x`
- 标签: `train_y / test_y`
- 图像尺寸: `288 x 288 x 3`
- `label` 为 `8 x 8` one-hot，表示外围 8 个 piece 的真实编号

需要先将现有 label 恢复为完整 9 宫格顺序：

1. 从 `8 x 8` one-hot 中恢复 8 个外围 piece 编号
2. 对编号 `>= 4` 的 piece 加 1
3. 在索引 4 处插入中心块编号 `4`

最终得到长度为 9 的 piece 顺序表。

### 3.2 三种样本生成方式

对每个基础样本，随机选择以下三种扰动之一。

#### Type 0: `ordered`

不做任何变换。

生成规则：

- 保持原图不变
- label 记为 `0`

目标：

- 让模型学习“正确拼图”的整体结构一致性

#### Type 1: `inner_swap`

在图内部进行一次 swap，但中心块不能参与。

生成规则：

1. 从 8 个非中心位置中随机选择两个不同位置
2. 交换这两个位置上的 piece
3. 中心块 `index = 4` 不允许被选中
4. label 记为 `1`

目标：

- 让模型学习图内局部错位、邻接关系破坏、全局布局错误

#### Type 2: `outer_replace`

从当前图外部随机取一个 piece，与图内一个非中心位置发生替换。

生成规则：

1. 随机选择当前图中一个非中心 slot
2. 随机选择另一张图像 `guest_idx`
3. 从 `guest_idx` 中随机选择一个非中心 piece
4. 用该外部 piece 替换当前图中该 slot 的 piece
5. 中心块 `index = 4` 不允许参与
6. label 记为 `2`

这里的“外部 piece”建议定义为：

- 来自其他图像
- 默认也只从非中心 8 个 piece 中采样

目标：

- 让模型学习“语义/风格不属于本图”的异常片段
- 为后续 ownership 或 outsider 识别打基础

### 3.3 类别采样策略

建议三类任务按均匀概率采样：

```python
task_type = random.randint(0, 2)
```

即：

- `ordered`: 1/3
- `inner_swap`: 1/3
- `outer_replace`: 1/3

如果后续发现训练不稳定，可调整为：

- `ordered`: 0.25
- `inner_swap`: 0.375
- `outer_replace`: 0.375

优先加强困难样本比例。

---

## 4. 模型设计需求

### 4.1 输入形式

本任务建议直接输入整张拼图图像，而不是 pairwise 输入。

推荐输入：

```python
image: Tensor[B, 3, 288, 288]
```

### 4.2 输出形式

模型输出三分类 logits：

```python
logits = model(image)   # [B, 3]
```

### 4.3 推荐 backbone

结合当前项目已有代码，可复用以下 backbone 之一：

- `efficientnet_b0`
- `Modulator`
- `VisionTransformer`

建议优先级：

1. `efficientnet_b0`
2. `Modulator`
3. `ViT`

理由：

- 当前项目已有较多 `ef/modulator` 相关代码
- 改动量最小
- 容易与已有预训练模块衔接

### 4.4 分类头

推荐结构：

```python
backbone -> hidden feature -> BN/ReLU/Dropout -> Linear(3)
```

例如：

```python
self.classifier = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.BatchNorm1d(hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, 3)
)
```

损失函数：

```python
loss_fn = nn.CrossEntropyLoss()
```

---

## 5. 数据集类设计

### 5.1 新建 Dataset

建议在独立文件 `pretrain/swap_classifier_pretrain.py` 中实现一个 Dataset，例如：

```python
class PuzzleTypeClassificationDataset(Dataset):
    ...
```

其职责：

1. 读取原图和原标签
2. 恢复正确 piece 排列
3. 随机采样一种扰动类型
4. 构造对应的新图像
5. 返回：

```python
return transformed_image, class_label
```

### 5.2 推荐返回格式

```python
image: torch.FloatTensor, shape [3, 288, 288]
label: torch.LongTensor, shape []
```

其中：

- `image` 建议使用 `float`
- `label` 必须为 `long`，以适配 `CrossEntropyLoss`

### 5.3 关键辅助函数

建议将 Dataset 内部逻辑拆成以下函数：

#### `decode_label(label_8x8)`

作用：

- 将当前 one-hot 标签恢复为完整 9 宫格 piece 编号

输入：

- `8 x 8` one-hot

输出：

- 长度为 9 的 piece 顺序列表

#### `split_image_to_pieces(image)`

作用：

- 将 `288 x 288` 图像切分为 9 个 `96 x 96` piece

输出：

- 长度为 9 的 piece list

#### `rebuild_image_from_pieces(pieces)`

作用：

- 将长度为 9 的 piece list 拼回 `288 x 288`

#### `apply_inner_swap(pieces)`

作用：

- 在非中心位置做一次 swap

#### `apply_outer_replace(pieces, guest_pieces)`

作用：

- 用外部 piece 替换当前图中一个非中心位置

---

## 6. 训练流程设计

### 6.1 总体流程

训练主流程如下：

1. 加载 `train_x/train_y` 与 `test_x/test_y`
2. 构建分类 Dataset 和 DataLoader
3. 初始化 backbone + classification head
4. 使用 `CrossEntropyLoss` 训练
5. 每个 epoch 在测试集上评估
6. 保存最优模型

### 6.2 单步训练流程

对每个 batch：

1. 读取 `images, labels`
2. `images.to(device)`
3. 前向传播得到 `logits`
4. 用 `CrossEntropyLoss(logits, labels)` 计算损失
5. 反向传播
6. optimizer.step()

### 6.3 评估指标

至少记录以下指标：

- Overall Accuracy
- Macro F1
- Per-class Precision / Recall / F1
- Confusion Matrix

建议三类名称固定为：

```python
CLASS_NAMES = ["ordered", "inner_swap", "outer_replace"]
```

### 6.4 推荐日志输出

每个 epoch 输出：

- train loss
- test accuracy
- macro f1
- per-class recall

示例：

```text
Epoch 12
train_loss: 0.3821
test_acc: 0.8714
macro_f1: 0.8652
ordered_recall: 0.91
inner_swap_recall: 0.84
outer_replace_recall: 0.86
```

---

## 7. 独立代码文件的具体实现流程

### 7.1 实现方式

该任务 **不在 `pretrain/pretrain_1.py` 上修改**，而是新建独立文件：

```text
pretrain/swap_classifier_pretrain.py
```

原因：

- 不干扰现有 pairwise 预训练逻辑
- 分类任务与 pairwise 邻接任务输入输出形式不同
- 独立文件更方便后续单独维护、调参与保存模型

### 7.2 建议新文件中包含的组件

建议在 `pretrain/swap_classifier_pretrain.py` 中实现：

#### Dataset

```python
class PuzzleTypeClassificationDataset(Dataset):
    ...
```

#### Model

```python
class PuzzleTypeClassifier(nn.Module):
    ...
```

#### Train function

```python
def train_classification(epoch_num=..., load=False):
    ...
```

#### Test function

```python
def test_classification(model):
    ...
```

### 7.3 推荐实现步骤

#### Step 1: 写 piece 操作工具函数

先在新文件中实现：

- `decode_label`
- `split_image_to_pieces`
- `rebuild_image_from_pieces`
- `sample_non_center_indices`

#### Step 2: 写 classification Dataset

在 `__getitem__` 中：

1. 取当前图
2. 切成 9 块
3. 随机选一种任务类型
4. 对 pieces 做变换
5. 重组为整图
6. 返回 `(image, label)`

#### Step 3: 写分类模型

推荐最简版本：

```python
efficientnet_b0(weights="DEFAULT")
```

将最后分类层改为 3 类输出。

#### Step 4: 写训练函数

包括：

- DataLoader 构造
- optimizer
- loss
- epoch 循环
- save best checkpoint

#### Step 5: 写测试函数

输出：

- accuracy
- macro f1
- confusion matrix

#### Step 6: 写 `__main__` 入口

例如：

```python
if __name__ == "__main__":
    train_classification(...)
```

---

## 8. 样本构造细节约束

### 8.1 中心块永远不能参与扰动

以下两种情况下中心块都必须固定：

- `inner_swap`
- `outer_replace`

可操作索引固定为：

```python
movable_indices = [0, 1, 2, 3, 5, 6, 7, 8]
```

### 8.2 外部 piece 的来源

建议外部 piece 来自：

- 另一张图像
- 非中心位置

即：

```python
guest_idx != current_idx
guest_piece_idx in movable_indices
```

### 8.3 替换而不是双图交换

由于本分类任务的输入只有一张图，因此 `outer_replace` 的定义建议为：

- 从外部取一块非中心块 piece
- 替换当前图中的一块非中心 piece
- 不追踪被替换出去的原 piece

这是一种单图样本生成机制，而不是环境中的双向交换过程。

### 8.4 标签必须只反映扰动类型

模型只预测：

- `ordered`
- `inner_swap`
- `outer_replace`

不需要输出：

- 哪两个块被 swap
- 哪个外部 piece 被换入
- 外部 piece 来自哪一类图

---

## 9. 推荐实验设置

### 9.1 数据划分

直接沿用当前项目已有划分：

- train: `train_img_48gap_33-001.npy`
- test: `test_img_48gap_33.npy`

如果后续需要，可加：

- valid: `valid_img_48gap_33.npy`

### 9.2 超参数建议

| 参数 | 建议值 |
|---|---|
| batch size | 64 或 128 |
| optimizer | Adam |
| learning rate | `1e-4` |
| epochs | 50 ~ 100 |
| loss | CrossEntropyLoss |
| input size | `3 x 288 x 288` |
| class num | 3 |

### 9.3 模型保存命名

建议命名为：

```text
model/swap_classifier_pretrain_ef.pth
model/swap_classifier_pretrain_modulator.pth
```

---

## 10. 预期收益

该任务相比当前 pairwise 预训练任务，多提供了以下能力：

- 学习整图级别的结构完整性
- 学习局部扰动与全局布局之间的关系
- 学习检测 outsider piece 的语义/风格不一致

对后续任务的潜在帮助：

- 作为 backbone 初始化
- 提升 ownership 判断能力
- 提升 cross-board 协调交换时对 outsider 的识别能力
- 提升全图状态编码器的判别性

---

## 11. 最终落地建议

建议按以下顺序实施：

1. 新建 `pretrain/swap_classifier_pretrain.py`
2. 用 `efficientnet_b0` 先做最小可运行版本
3. 先验证三分类任务是否可稳定收敛
4. 再尝试替换 backbone 为 `Modulator`
5. 若效果稳定，再考虑把该分类预训练权重迁移到后续拼图主任务

---

## 12. 一句话总结

这个新预训练任务本质上是在问模型：

> “这张拼图是完全正确的、只是内部换错了一次，还是混入了外来的 piece？”

这是一个比 pairwise 邻接判断更接近全局拼图理解的预训练任务，适合作为后续多智能体拼图求解模型的表征初始化。
