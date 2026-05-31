# Buffer Excavate-Fill Jigsaw Solver 需求与设计文档

> 版本: v0.3  
> 日期: 2026-05-31  
> 参考文档: `document/puzzle_solver_document.md`  
> 核心变化: 环境目标不变，agent 重新设计为 Digger + Filler 两阶段协作

---

## 修订记录

| 版本 | 日期 | 修改内容 |
|:---:|:---:|:---|
| v0.3 | 2026-05-31 | 重设计 RL 方案：Digger 与 Filler 分别采用独立训练体系，不再围绕 PPO；Filler 改为变种 Q-learning / scoring 机制；补充多种填充顺序策略 |
| v0.2 | 2026-05-31 | 统一空位表示为 learnable vacancy token；补充 env / agent / model_code / pretrain 的实现流程；新增 Digger 预训练任务与 Filler 预训练候选方案 |
| v0.1 | 2026-04-26 | 初版设计，提出“挖取 agent + 填充 agent + 公共 buffer”的双阶段拼图求解方案 |

---

## 1. 项目概述

### 1.1 背景

本方案保持原拼图环境的目标不变：给定两张 `3x3` 拼图 board，中心 anchor 固定，外围 8 个 piece 可操作，最终希望恢复正确拼图。

与现有交换式策略不同，本方案重新设计 agent 的行为范式：

1. **Digger Agent** 先观察 local board，从每张图中挖走若干非中心块。
2. 被挖走的块进入一个公共 buffer。
3. **Filler Agent** 在挖取阶段结束后，观察两张 board 和公共 buffer，选择 buffer 中的块填回空位。

该设计将“移除可疑块”和“回填合适块”拆成两个子任务，有利于显式建模：

- 哪些位置存在错位或 outsider
- 被移出的 piece 如何在公共 buffer 中重分配
- 空位与候选 piece 的匹配关系

### 1.2 核心目标

环境目标不变：

- 两张 `3x3` 拼图图像
- 每张图中心 anchor 固定
- 每张图外围 8 个非中心 slot 可操作
- 最终 board 恢复到正确 piece 归属与位置

agent 设计改变：

- Digger 只负责挖取。
- Filler 只负责从 buffer 回填。
- 两个 agent 先后运行，不同步执行。
- Digger 必须先处理完两张图，Filler 才开始填充。

---

## 2. 数据与环境保持不变的部分

以下定义沿用 `puzzle_solver_document.md` 与现有环境：

- 数据集格式不变。
- piece 切分方式不变。
- 中心 anchor 固定不动。
- 非中心 slot 编号仍为 8 个可操作位置。
- ground-truth piece id 与 slot id 的定义不变。

---

## 3. 新环境状态设计

### 3.1 Board 状态

每张 board 仍是 `3x3` 结构：

```text
s0  s1  s2
s3  A   s4
s5  s6  s7
```

其中：

- `A` 是中心 anchor，不可挖取、不可填充、不可移动。
- `s0..s7` 是 8 个非中心 slot。
- slot 状态可以是：
  - occupied: 当前有真实 piece
  - empty: 当前为空位，由 vacancy token 表示

### 3.2 空位表示

本版本统一采用 **learnable vacancy token** 表示空位，不再使用 Otsu 或传统 CV 背景色填充作为主方案。

原因：

- 本任务中的空位是逻辑状态，不一定需要伪造视觉背景。
- Otsu 在复杂艺术图像上不稳定。
- learnable token 更适合与 `fen_model` / board encoder / transformer 类结构结合。
- Digger 预训练和 Filler 预训练都需要对空位做显式 mask，learnable token + explicit empty mask 更清晰。

环境中建议同时保留两类信息：

```text
slot_visual_or_token
empty_mask
```

其中：

- `slot_visual_or_token`: occupied 时为真实 piece 图像或特征，empty 时为 learnable vacancy token。
- `empty_mask`: 标记每个 slot 是否为空。

### 3.3 公共 Buffer 状态

Digger 挖走的所有 piece 进入公共 buffer。

每个 buffer item 至少包含：

```text
(
    piece_image,
    piece_id,
    source_board,
    source_slot,
    gt_board,
    gt_slot
)
```

其中：

- `source_board/source_slot` 表示该块被挖出前的位置。
- `gt_board/gt_slot` 表示该 piece 的正确归属和位置。
- `piece_id` 用于 reward 与指标计算。

---

## 4. 两阶段交互流程

### 4.1 总体流程

每个 episode 按固定阶段执行：

```text
初始化两张 board
      ↓
Digger 挖 Board A 的非中心块
      ↓
Digger 挖 Board B 的非中心块
      ↓
公共 buffer 构建完成
      ↓
Filler 从 buffer 中选择 piece 并填入空位
      ↓
buffer 清空或达到终止条件
```

### 4.2 Digger 阶段

Digger 对每张 board 执行挖取。

默认设计：

- 每张 board 挖走 2 个非中心块。
- 两张 board 共挖走 4 个 piece。
- buffer 大小固定为 4。

可扩展设计：

- curriculum 初期每张 board 只挖 1 块。
- 后续增加到每张 board 挖 2 块。

### 4.3 Filler 阶段

Filler 在 Digger 完成两张图挖取后开始工作。

Filler 不直接输出离散放置动作，而是输出对候选成品图像的评分。

外部填充策略再根据评分选择动作组合。

---

## 5. Digger Agent 设计

### 5.1 观察

Digger 每次只看到一张 local board：

```text
o_d^i = (
    board_i_slots[0..7],
    anchor_i,
    empty_mask_i
)
```

其中：

- `i ∈ {A, B}`
- board 中可能已有空位
- 空位由 learnable vacancy token 表示
- 空位在动作选择时必须 mask 掉，防止 Digger 重复挖空位

### 5.2 动作

Digger 输出一个非中心 slot：

```text
a_d^i ∈ {0..7}
```

该动作表示：

> 从当前 local board 中挖走 slot `a_d^i` 上的 piece。

若每张 board 需要挖 2 块，则 Digger 顺序执行两次单输出动作。

### 5.3 动作 mask

Digger 不能选择：

- 中心 anchor
- 已经为空的 slot

因此 Digger 动作 logits 需要使用 mask：

```text
valid_digger_slot = not empty_mask[slot]
```

非法位置对应 logits 应设置为极小值，例如 `-1e9`。

### 5.4 模型建议

Digger 模型结构：

```text
fen_model backbone
      ↓
board feature
      ↓
slot scoring head
      ↓
8-way logits
      ↓
mask empty slots
```

backbone 要使用 `model_code/fen_model.py` 中已经封装好的模型，并通过 `model_name` 参数切换：

```python
fen_model(
    hidden_size1=...,
    hidden_size2=...,
    model_name=model_name
)
```

其中 `model_name` 可选：

- `ef`
- `modulator`
- 后续可扩展其他已封装 backbone

---

## 6. Filler Agent 设计

### 6.1 观察

Filler 观察全局填充状态：

```text
o_f = (
    board_A_slots,
    board_B_slots,
    anchor_A,
    anchor_B,
    empty_mask_A,
    empty_mask_B,
    buffer_pieces
)
```

### 6.2 动作语义

Filler 的最终决策不是直接输出一个离散动作，而是：

1. 枚举所有候选填充组合。
2. 将某块 buffer piece 填入某个空缺后，构造成品图像。
3. 对该成品图像打分。
4. 选择评分最高的方案。

也就是说，Filler 的核心是一个 **scoring network**，它类似 `sd2rl_torch.py` 中的变种 Q-learning 模式：

- 列举所有动作
- 对每个动作构造候选结果
- 给候选结果打分
- 选择总分最大的动作组合

### 6.3 动作空间

Filler 的动作空间由填充策略决定，支持以下超参数：

```text
fill_strategy ∈ {seq, alt, greedy, parallel}
```

每种策略的细节见后文。

### 6.4 模型建议

Filler 模型结构：

```text
fen_model backbone
      ↓
candidate complete image
      ↓
global score head
      ↓
scalar score
```

Filler 的本质更接近一个 value / critic / scorer 网络：

- 输入一张“已经尝试填入某 piece 后”的完整候选图像
- 输出一个标量分数
- 分数越高，说明该候选填充越优

由于 Filler 的最终决策依赖比较多个候选方案，因此实际动作选择可在外部完成。

---

## 7. Filler 填充顺序策略

Filler 的填充顺序由超参数指定，推荐定义为：

```text
fill_strategy ∈ {seq, alt, greedy, parallel}
```

### 7.1 seq

顺序遵循“图一再图二”的固定顺序。

示例：

```text
先填图1的第1个空缺
再根据新图填图1的第2个空缺
再填图2的第1个空缺
再填图2的第2个空缺
```

特点：

- 严格按单图局部顺序推进
- 每次只评估当前空缺的候选分数
- 新状态会影响后续空缺选择

### 7.2 alt

顺序改成图1图2交替进行，其他规则不变。

示例：

```text
图1第1个空缺 -> 图2第1个空缺 -> 图1第2个空缺 -> 图2第2个空缺
```

特点：

- 强调双图交替协调
- 可减弱单图局部偏置

### 7.3 greedy

先计算每一块拼图对于每一个空缺的分数，再用 greedy 找到分数和最大的方案，且块不重复。

对于本方案中 4 个空位和 4 个 buffer piece：

- 先构建 piece-slot score matrix
- 再做贪心匹配
- 最终将 4 块一起填入

特点：

- 适合一次性全局匹配
- 近似求解最大总分 assignment

### 7.4 parallel

先生成所有可能的组合方式。

对于 4 个 piece 和 4 个空位，共有：

```text
4 * 3 * 2 * 1 = 24
```

种排列。

流程：

1. 枚举所有 piece 到空位的 bijection。
2. 对每种排列构造完整候选图像。
3. 用 Filler 对候选图像打分。
4. 每次评分仍是一张图的评分。
5. 总分可定义为：

```text
score = filler(image1) + filler(image2)
```

6. 选择总分最高的组合。

特点：

- 全局最优搜索的近似实现
- 计算量最大
- 适合离线推理或小规模验证

---

## 8. Reward 设计

### 8.1 Digger 奖励

Digger 和 Filler 不采用同一套联合奖励，而是使用各自独立的奖励信号。

Digger 的奖励目标很明确：

- 挖出不属于该绝对位置的块 -> 正奖励
- 挖出本来就在正确绝对位置的块 -> 惩罚

推荐定义：

```text
reward_digger =
    +r_remove_wrong   if removed_piece is not at its correct absolute slot
    -r_remove_right   if removed_piece is already at correct absolute slot
```

其中：

- 若移出的是 outsider 或错位 piece，则奖励更高。
- 若移出的是正确 piece，则给予明显惩罚。

建议同时加入轻微 step penalty，防止无意义反复挖取。

### 8.2 Filler 奖励

Filler 的奖励来自对填充后图像的评分。

评分参考环境中已有的图质量函数：

- `category`
- `pairwise`
- `consistency`
- `done`

mask 部分仍按空块 `-1` 处理。

推荐定义：

```text
score_filler = category + pairwise + consistency + done
```

其中：

- `done` 最高
- `category` 次之
- `consistency` 再次之
- `pairwise` 最基础

若某张图全对，则立即触发 done reward。

### 8.3 Reward 层级

保持以下关系：

```text
done reward >> category reward > consistency reward >= pairwise reward
```

这里不再要求 reward 适合 PPO，也不再以 PPO clipping 为设计前提。

### 8.4 Reward 结算方式

- Digger 只使用自己的挖取奖励信号。
- Filler 只使用自己的图评分奖励信号。
- 两者训练时互不共享 reward 公式。

---

## 9. RL 训练方案

### 9.1 总体原则

本方案不再采用 MAPPO，也不再围绕 PPO 设计。

两个 agent 采用不同体系训练，并且训练时只使用自己的奖励信号：

- Digger: classification / preference / Q-style 皆可
- Filler: 变种 Q-learning / scoring 模式

### 9.2 Digger 的训练体系

Digger 目标是学习“哪个位置更值得挖”。

推荐训练形式：

- classification
- ranking
- value-based score

训练信号只来自 Digger 奖励：

- 正确挖出错位或 outsider -> 正奖励
- 挖出正确位置的块 -> 惩罚

可把 Digger 看成一个 slot classifier：

```text
board -> 8-way logits
```

### 9.3 Filler 的训练体系

Filler 采用与 `sd2rl_torch.py` 同款的变种 Q-learning 模式：

- 列举所有动作
- 对所有动作评分
- 选择分数最高的方案

在这里，“动作”不是单纯的 slot，而是：

> 某块 buffer piece 填入某个空位后形成的候选完整图像

Filler 学习的是：

```text
Q(candidate_complete_image)
```

然后动作选择时：

1. 枚举所有合法填充组合
2. 构造对应候选图像
3. 对每个候选图像打分
4. 选总分最高的组合

这与 `sd2rl_torch.py` 的“先算所有候选，再 greedy 选最优”模式一致。

---

## 10. 实现流程

### 10.1 Env 新建

在 `env/` 下新建该任务环境：

```text
env/buffer_excavate_fill_env.py
```

建议类名：

```python
class BufferExcavateFillEnv:
    ...
```

关键职责：

- 初始化两张 board。
- 管理 empty mask 与 vacancy token。
- 管理公共 buffer。
- 执行 Digger 挖取动作。
- 执行 Filler 候选评分与落位。
- 计算 reward 与终止条件。

建议核心接口：

```python
reset()
get_digger_observation(board_id)
step_digger(board_id, slot_id)
get_filler_observation()
step_filler(candidate_index)
get_digger_mask(board_id)
get_candidate_mask()
get_metrics()
```

### 10.2 Agent 新建

在 `agent/` 下新建两个 agent：

```text
agent/digger_agent.py
agent/filler_agent.py
```

Digger Agent 职责：

- 接收 local board observation。
- 调用 Digger model 输出 8 个 slot logits。
- 应用 empty mask。
- 选择被挖取的位置。
- 记录 transition。

Filler Agent 职责：

- 接收候选完整图像序列。
- 调用 Filler model 输出单个 score。
- 由外部填充策略完成组合搜索与排序。
- 记录 transition。

### 10.3 Model 新建

在 `model_code/` 下新建两个模型文件：

```text
model_code/digger_model.py
model_code/filler_model.py
```

共同要求：

- backbone 使用 `model_code/fen_model.py` 中已封装好的模型。
- 通过 `model_name` 参数切换 backbone。
- 不在模型内部写死 backbone 类型。

Digger model 示例接口：

```python
class DiggerModel(nn.Module):
    def __init__(self, hidden_size=512, model_name="modulator"):
        ...

    def forward(self, board_image, empty_mask):
        ...
        return slot_logits  # [B, 8]
```

Filler model 示例接口：

```python
class FillerModel(nn.Module):
    def __init__(self, hidden_size=512, model_name="modulator"):
        ...

    def forward(self, candidate_complete_image, empty_mask=None):
        ...
        return score  # scalar
```

### 10.4 Pretrain 新建

在 `pretrain/` 下新增：

```text
pretrain/digger_pretrain.py
pretrain/filler_pretrain.py
```

Digger 预训练：

- 50% 样本为完整图。
- 50% 样本带一个空缺。
- 空缺位置用 learnable token 表示。
- Digger 输出错位位置的分类结果。
- 训练时对空缺位置 logits 做 mask。

Filler 预训练：

- 建议优先采用 candidate complete-image scoring。
- 也可采用 pairwise ranking / reward regression / teacher distillation。

### 10.5 训练主流程

建议后续训练入口为：

```text
train_buffer_excavate_fill.py
```

流程：

1. env reset。
2. Digger 依次处理 Board A 和 Board B。
3. buffer 构建完成。
4. Filler 根据 `fill_strategy` 评分与放置。
5. 计算各自 reward。
6. 更新 Digger / Filler 参数。

---

## 11. 预训练任务细化

### 11.1 Digger 预训练任务

#### 任务目标

Digger 直接分辨某张图的错位部分位置。

错位可以来自：

- outsider piece
- 本图内部 swap 导致的错位块

#### 输出形式

分类输出：

```text
label ∈ {0..7}
```

#### 输入图构造

每个样本：

- 50% 完整图
- 50% 带一个空缺

空缺由 learnable vacancy token 代替。

#### 训练目标

对 slot logits 做分类：

```text
CrossEntropy(masked_logits, label)
```

### 11.2 Filler 预训练候选目标

Filler 的预训练目标可采用以下任一方向：

1. 评分回归：输出候选完整图的 reward / value。
2. pairwise ranking：正确候选比分数更高。
3. candidate selection：在若干候选图中选最优者。
4. teacher distillation：拟合贪心或枚举 teacher 的偏好。

---

## 12. 开放设计问题

| # | 问题 | 选项 | 初步建议 |
|---|---|---|---|
| 1 | 空位表示 | learnable token / Otsu / 邻域颜色 | learnable token |
| 2 | Digger 每图挖取数 | 1 / 2 / curriculum | 先 1 后 2 |
| 3 | Digger 训练形式 | classification / ranking / value-based | classification 优先 |
| 4 | Filler 填充策略 | seq / alt / greedy / parallel | 全部支持，超参数切换 |
| 5 | Filler 训练形式 | scoring / ranking / regression / distillation | 先 scoring + ranking |
| 6 | 模型 backbone | ef / modulator / 其他已封装 backbone | 通过 `model_name` 指定 |

---

## 13. 一句话总结

本方案保持拼图任务目标不变，但将求解流程重构为：

> Digger 用分类式决策找出错位块并挖入公共 buffer，Filler 用变种 Q-learning / scoring 方式对所有合法填充组合打分，选择最优回填方案。

这样既能显式建模错误定位，也能将最终拼图恢复转化为评分式决策问题，更适合与现有 `sd2rl_torch.py` 风格的候选打分范式对接。
