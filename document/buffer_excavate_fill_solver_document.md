# Buffer Excavate-Fill Jigsaw Solver 需求与设计文档

> 版本: v0.1  
> 日期: 2026-04-26  
> 参考文档: `document/puzzle_solver_document.md`

---

## 修订记录

| 版本 | 日期 | 修改内容 |
|:---:|:---:|:---|
| v0.1 | 2026-04-26 | 初版设计，提出“挖取 agent + 填充 agent + 公共 buffer”的双阶段拼图求解方案 |

---

## 1. 项目概述

### 1.1 背景

在现有拼图环境中，两个 board 的目标、piece 编码方式、reward 定义和完成判定都已经比较明确。  
本方案 **不修改拼图任务本身的环境目标**，而是重新设计 agent 的行为方式，使求解流程从“直接交换/放置”改为：

1. 先从局部 board 中挖走若干块
2. 再从公共 buffer 中选择块，回填到空隙位置

这样做的核心动机是：

- 将“拿走错误块”和“把合适块放回去”拆成两个角色
- 让 agent 显式处理中间态 `buffer`
- 让环境中出现“空位”这一结构化状态，强化 agent 对归属与放置顺序的理解

### 1.2 核心目标

保持原有拼图目标不变：

- 两张 `3x3` 拼图图像
- 每张图中心 anchor 固定
- 外围 8 个非中心块可操作
- 最终仍需恢复正确 piece 排列

但重新定义 agent 流程：

- **Agent 1: Excavator / Digger**
  - 看到单个 local board
  - 从中挖走 2 个非中心块
  - 挖走后当前位置用背景色填充
  - 被挖走的块进入公共 buffer

- **Agent 2: Filler / Placer**
  - 在 Agent 1 完成两张图的挖取后开始工作
  - encode buffer 中全部 piece
  - 逐步选择一块 piece，放入某个空隙
  - 非法动作是将 piece 放入非空位置，这类动作必须 mask 掉

### 1.3 与现有方案的关键差异

| 维度 | 现有方案 | 本方案 |
|---|---|---|
| 环境目标 | 恢复完整拼图 | 不变 |
| 中间状态 | 通常直接交换/移动 | 显式引入空位和公共 buffer |
| Agent 角色 | 单类或对称双 agent | 功能拆分为 Digger + Filler |
| 动作语义 | 交换或选位放置 | 先挖取，再从 buffer 回填 |
| 非法动作 | 视原动作空间而定 | Filler 对已占位置放置必须 mask |

---

## 2. 环境设定

### 2.1 保持不变的部分

以下环境设定沿用现有拼图系统：

- 输入图像格式不变
- `3x3` 拼图结构不变
- 中心块 anchor 固定不动
- 每张图 8 个非中心 piece 的 ground-truth 定义不变
- pairwise / category / consistency / done reward 的定义不变
- 若一张图完全恢复正确，则触发 done reward

### 2.2 新增的环境状态

在保持目标不变的前提下，环境中增加两个概念：

#### 空隙 `empty slot`

当 Digger 挖走一个 piece 后，该位置不再放真实块，而是用“背景填充值”形成一个空隙状态。

#### 公共缓冲区 `public buffer`

所有被挖出的块进入同一个共享 buffer。

在本任务设定下：

- 每张 board 挖走 2 块
- 共两张 board
- 因此单轮挖取结束后，buffer 中共有 4 块

---

## 3. 挖取阶段设计

### 3.1 Digger 的观察

Digger 每次只观察一张 local board：

```text
o_d^i = (board_i_slots[0..7], anchor_i)
```

其中：

- `i ∈ {A, B}`
- board 中允许出现“空隙填充块”
- Digger 不直接观察另一张 board
- Digger 不直接观察 buffer 的内容

### 3.2 Digger 的动作

对每张 board，Digger 需要选择两个非中心位置进行挖取。

推荐两种实现方式：

#### 方案 A: 两步式选择

每个 board 上依次输出两个 slot：

```text
a_d^i = (slot_1, slot_2)
```

约束：

- `slot_1 != slot_2`
- 二者都属于 8 个非中心 slot

#### 方案 B: 顺序两次单动作

Digger 在一个 board 上先挖 1 次，再基于新的 board 状态挖第 2 次。

推荐优先采用 **方案 B**，因为：

- 实现更简单
- 更容易与 PPO / pointer 结构兼容
- 第二次选择可以条件化在第一次挖取结果上

### 3.3 挖取执行顺序

建议固定流程如下：

1. Digger 处理 Board A，挖走 2 块
2. Digger 处理 Board B，挖走 2 块
3. 两张 board 挖取结束后，buffer 共 4 块
4. Filler 开始回填

这是一种“先全部挖完，再开始填”的硬阶段切换。

---

## 4. 空隙背景填充设计

### 4.1 目标

挖走 piece 后，需要在原位置放入一个“空隙视觉占位符”，使网络知道该位置为空，同时避免直接用纯黑块或纯零张量带来过强的人造偏置。

### 4.2 候选方案

#### 方案 A: Otsu 自动阈值背景估计

思路：

1. 将待挖 piece 或其周边区域转换到灰度图
2. 使用 Otsu 自动阈值法分离高亮/低亮区域
3. 从被判定为背景的一侧估计背景颜色
4. 用估计出的 RGB 均值填满该 slot

优点：

- 传统 CV 方法，可解释性强
- 不需要额外训练

问题：

- 本数据集是无 gap 拼图，很多 piece 本身不一定有稳定“背景”
- Otsu 在纹理复杂、颜色分布连续时并不稳定
- 对油画/版画/器物图像可能误判严重

#### 方案 B: 邻域边界颜色估计

思路：

1. 挖掉某块后，取其上、下、左、右相邻位置边界的一圈像素
2. 将这些边界像素汇总
3. 用中位数或均值估计一个填充值
4. 用该 RGB 颜色填充该空位

优点：

- 更符合“该位置周边应该是什么颜色”的局部上下文
- 比 Otsu 更稳定
- 不依赖 piece 内部本身是否有背景

#### 方案 C: 固定 learnable empty token

思路：

- 不真的生成视觉背景色
- 直接将空位映射为一个 learnable embedding / 特殊 token

优点：

- 简洁
- 最适合 transformer 类状态编码

问题：

- 如果当前代码仍以图像块为主输入，而不是 slot token 直接输入，则需要额外改造

### 4.3 推荐方案

**推荐主方案：B 邻域边界颜色估计**  
**推荐备选：A Otsu 自动估计**

推荐原因：

- 当前数据不是天然存在大面积平坦背景的场景
- 邻域颜色更接近“局部上下文缺失”的真实视觉表现
- 对后续模型更稳定

### 4.4 推荐实现规则

对于被挖掉的 slot：

1. 收集该位置四周相邻块边缘 `k` 像素带，推荐 `k = 3 ~ 5`
2. 若某个方向不存在相邻块，则忽略该方向
3. 将所有候选边缘像素拼接
4. 用 RGB 通道中位数生成填充值
5. 整块 `96x96` 用该颜色填满

若相邻可用像素过少，则 fallback 到：

- 对原 piece 做 Otsu 灰度分割
- 选择面积更大的类作为背景类
- 用该类像素的 RGB 均值填充

---

## 5. 公共 Buffer 设计

### 5.1 Buffer 内容

buffer 中每个元素至少应包含：

```text
(
    piece_image,
    source_board,
    source_slot,
    piece_id
)
```

其中：

- `piece_image`: 该块图像
- `source_board`: 原属 board A 或 B
- `source_slot`: 原始位置
- `piece_id`: 全局唯一标识

### 5.2 Buffer 大小

在当前固定设定下：

- Board A 挖走 2 块
- Board B 挖走 2 块
- buffer 大小固定为 4

后续如扩展为可变挖取数量，则 buffer 大小可动态变化。

### 5.3 Buffer 的状态表示

Filler 需要 encode buffer 中所有块，因此建议：

- 将 buffer 中每块都单独编码为向量
- 再使用 attention 或 pooling 汇总 buffer 全局摘要

可表示为：

```text
buffer = {b0, b1, b2, b3}
e_k = PieceEncoder(b_k)
E_buffer = [e_0, e_1, e_2, e_3]
```

---

## 6. 填充阶段设计

### 6.1 Filler 的观察

Filler 在填充阶段观察：

- 当前两张 board 的状态
- 各 board 中哪些位置为空
- buffer 中当前剩余的所有 piece

可表示为：

```text
o_f = (
    board_A_slots,
    board_B_slots,
    anchor_A,
    anchor_B,
    buffer_pieces
)
```

### 6.2 Filler 的动作

每一步 Filler 需要做两件事：

1. 从 buffer 中选择一块 piece
2. 选择一个空隙位置进行放置

推荐联合动作表示为：

```text
a_f = (buffer_index, target_board, target_slot)
```

其中：

- `buffer_index ∈ {0 .. |buffer|-1}`
- `target_board ∈ {A, B}`
- `target_slot ∈ {0..7}`

### 6.3 合法动作约束

Filler 的动作必须满足：

- 只能把 piece 放到空位上
- 已经有 piece 的位置不能被选择
- 如果某个 board 当前无空位，则该 board 上所有 slot 都应被 mask

因此 mask 规则为：

```text
mask(target_board, target_slot) = 1  only if slot is empty
mask(target_board, target_slot) = 0  otherwise
```

### 6.4 放置执行

执行 `a_f` 后：

1. 取出 buffer 中对应 piece
2. 将其写入目标空隙
3. 从 buffer 中移除该 piece
4. 该位置状态从 empty 变为 occupied

当 buffer 清空后，本轮 `挖取 -> 填充` 完成。

---

## 7. 整体交互流程

### 7.1 单个 episode 的建议流程

1. 初始化两张 board
2. Digger 在 Board A 上挖 2 次
3. Digger 在 Board B 上挖 2 次
4. 此时生成 4 个空隙，buffer 中有 4 块
5. Filler 开始循环执行放置动作
6. 直到：
   - buffer 清空，或者
   - 达到最大步数，或者
   - 某种完成/终止条件满足

### 7.2 时序结构

可以表示为：

```text
Excavate A x2
Excavate B x2
Fill from buffer until empty
Evaluate board quality
```

这是一个明确分阶段的非同步系统，不是两个 agent 同步一步一步交替。

---

## 8. Reward 设计

### 8.1 保持不变的 reward 类型

reward 的定义沿用现有系统，不新引入额外奖励类型：

- `pairwise`
- `category`
- `consistency`
- `done reward`

### 8.2 pairwise reward

定义保持不变：

- 若相邻 piece 在正确 board 且相对位置正确，则给予 pairwise reward

### 8.3 category reward

定义保持不变：

- 若某个 piece 放在正确的 board 且/或正确位置，则给予 category reward

### 8.4 consistency reward

定义保持不变：

- 若一张 board 内更多 piece 来自同一目标图像，则 consistency 更高

### 8.5 done reward

如果 **一张图全对**，则触发 done reward。

这里保留你当前环境中的既有语义：

- 一张图完全恢复正确，即可触发该图的 done reward
- 若系统已有双图完成判定，也可同时保留 episode-level 终止逻辑

### 8.6 建议 reward 施加时机

由于 Digger 会先破坏局部结构，再由 Filler 修复，因此建议 reward 采用以下方式：

#### 方案 A: 只在填充阶段结算主奖励

- 挖取阶段不给即时奖励，或只给极小 step penalty
- 每次 Filler 放置后，根据新 board 状态计算 reward

优点：

- 避免 Digger 因为“暂时挖坏了图”而总是收到大量负反馈

#### 方案 B: 全阶段都结算，但最终用 team return 学习

- Digger 挖走后也会导致 pairwise/category 下降
- 但两个 agent 共享 episode return

优点：

- 更接近真实环境变化

**推荐优先使用方案 A**，即：

- 挖取阶段只记录动作，不强调局部即时 reward
- 填充阶段用现有 reward 体系结算

---

## 9. Agent 设计建议

### 9.1 Digger Agent

职责：

- 识别哪些块“值得先拿走”
- 为后续 Filler 提供更有价值的候选 piece 集合

输入：

- 单 board 图像状态
- anchor

输出：

- 要挖走的非中心 slot

建议结构：

- Piece Encoder
- Board Encoder
- Pointer over 8 non-center slots

### 9.2 Filler Agent

职责：

- 从 buffer 中挑选最适合当前某个空位的 piece
- 逐步恢复 board 结构

输入：

- 双 board 状态
- 空位 mask
- buffer piece embeddings

输出：

- `(buffer_index, target_board, target_slot)`

建议结构：

- Buffer Piece Encoder
- Board State Encoder
- Piece-to-slot matching / pointer network

### 9.3 参数共享建议

不建议 Digger 与 Filler 直接参数共享。

原因：

- 两者职责不同
- 一个做“移除决策”
- 一个做“放置决策”
- 输入结构和动作语义也不同

推荐：

- 两个 agent 独立参数
- 若需要可共享底层 piece encoder

---

## 10. 训练建议

### 10.1 训练范式

可采用多智能体 PPO / MAPPO，也可先分别训练。

推荐顺序：

1. 先固定 Digger 策略，用启发式挖取
2. 单独训练 Filler
3. 再联合微调 Digger + Filler

原因：

- 若两者同时随机探索，训练不稳定
- Filler 在固定 buffer 分布下更容易先学会放置

### 10.2 启发式 Digger baseline

为了先验证整体系统，建议先做一个简单 baseline：

- Digger 优先挖走当前 consistency 最差的两个非中心块
- 或者随机挖 2 块

然后观察：

- Filler 是否能学会从 buffer 恢复
- reward 是否正常增长

### 10.3 Curriculum 建议

推荐课程学习顺序：

1. 只挖 1 块 / 每图
2. 再扩展到 2 块 / 每图
3. 再训练 full buffer=4 的正式设定

---

## 11. 工程实现建议

### 11.1 建议新增模块

推荐新增：

- `envs/buffer_fill_env.py`
  - 在现有环境逻辑上增加 empty slot 和 buffer 机制
- `models/digger.py`
  - 挖取 agent
- `models/filler.py`
  - 填充 agent
- `training/train_buffer_fill.py`
  - 训练入口

### 11.2 环境层需要新增的关键函数

建议实现：

- `dig_piece(board_id, slot_id)`
- `estimate_empty_fill_color(...)`
- `push_to_buffer(piece)`
- `place_from_buffer(buffer_idx, board_id, slot_id)`
- `get_empty_slot_mask()`
- `encode_buffer_state()`

### 11.3 状态表示建议

board 中空位建议保留两种信息：

1. 图像层面的填充值块
2. 逻辑层面的 `is_empty` mask

理由：

- 图像输入可供 CNN/ViT 使用
- 显式 mask 可供策略头进行合法动作屏蔽

---

## 12. 开放设计问题

| # | 问题 | 选项 | 初步建议 |
|---|---|---|---|
| 1 | 空位填充策略 | Otsu / 邻域颜色 / learnable token | 优先邻域颜色 |
| 2 | Digger 每图挖取方式 | 一次输出两块 / 两次顺序输出 | 优先顺序输出 |
| 3 | reward 结算时机 | 全阶段结算 / 只在填充阶段结算主奖励 | 优先后者 |
| 4 | Filler 目标 | 一次选 piece+slot / 分两步选择 | 优先联合动作 |
| 5 | 参数共享 | 完全独立 / 共享 encoder | 优先共享底层 encoder，策略头独立 |

---

## 13. 一句话总结

该方案的本质是：

> 保持拼图目标不变，但把“修图”拆成“先挖错块，再从公共 buffer 里挑块回填”的两阶段多智能体过程。

相比直接交换，这种设计更强调中间状态建模、buffer 推理和空位放置决策，更适合作为新的多智能体拼图求解研究方向。
