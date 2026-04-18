# 2-Mixed JPwLEG 拼图求解 —— 需求与设计文档

> 版本: v0.4 | 更新日期: 2026-04-13
> 基础论文: SD²RL (AAAI 2023) — Siamese-Discriminant Deep Reinforcement Learning for Solving Jigsaw Puzzles with Large Eroded Gaps

---

## 修订记录

| 版本 | 日期 | 修改内容 |
|:---:|:---:|:---|
| v0.4 | 2026-04-13 | 1. 统一动作执行协议为同步执行（5.5 节重写）；2. Pointer 2 Query 融合 Board 全局摘要（6.4.6 节）；3. Critic 输入增加通信向量（6.4.7 节）；4. 意愿对齐奖励增加工程实现说明（6.7.2 节）；5. Masking 数值稳定性修正（全文） |
| v0.3 | 2026-04-11 | 新增第 6 节网络架构详细设计 |

---

## 1. 项目概述

### 1.1 背景

基于 SD²RL 的 JPwLEG-3 拼图求解框架，本项目将其扩展为 **2-Mixed** 场景：
两张来自不同类别的图像的拼图碎片混合后，由 RL Agent 通过碎片交换动作还原两个拼图。

### 1.2 核心目标

给定两张来自不同类别的 3×3 JPwLEG 拼图（各含 1 个固定 Anchor + 8 个可移动碎片），
将 16 个可移动碎片随机混合并均分至两个 Board（每个 Board 8 个碎片），
训练 RL Agent 通过一系列碎片交换动作，使每个 Board 恢复为各自图像的正确拼图。

### 1.3 与原论文的关键差异

| 维度 | SD²RL（原论文） | 2-Mixed（本项目） |
|------|----------------|-------------------|
| 图像数量 | 1 张 | 2 张（跨类别） |
| 可移动碎片总数 | 8 | 16 |
| Board 数量 | 1 | 2 |
| Agent 数量 | 1 | 2（参数共享，同步执行） |
| 动作类型 | Board 内交换 | Board 内交换 + 跨 Board 交换（需 Mutual Agreement） |
| 子任务 | 排列（placement） | 归属判断（ownership）+ 排列（placement） |

---

## 2. 数据集

### 2.1 来源与规模

基础数据集为 JPwLEG-3，源自 MET Museum of Art 开源图像，经过预处理后共 12,000 张图像，涵盖三个类别：Painting（绘画）、Engraving（版画）、Artifact（器物）。

### 2.2 图像格式

图像已从原始 398×398（含 48px gap、±7px jitter）预处理为 **288×288** 像素，即纯 3×3 内容网格，无间隔、无边距。每个 piece 为 96×96 像素。图像中的 piece 保持其**打乱后的位置**（gap 已去除，但空间排列仍为打乱状态）。

### 2.3 存储结构
E:\PuzzleSolving\MET_Dataset\select_image\shuffle_fragment_no_gap
├── painting
│ ├── {img_id}\_{p0}\_{p1}\_{p2}\_{p3}\_{p4}\_{p5}\_{p6}\_{p7}.jpg
│ └── ...
├── engraving
│ └── ...
└── artifact
└── ...
类别信息由子文件夹名称决定。

### 2.4 文件名编码

格式：`{img_id}_{p0}_{p1}_{p2}_{p3}_{p4}_{p5}_{p6}_{p7}.jpg`

| 字段 | 含义 |
|------|------|
| `img_id` | 图像唯一标识符（整数） |
| `p0 ~ p7` | 8 个可移动 slot 中实际放置的 piece 的 ground-truth 编号 |

每张图有且仅有一种打乱方式。Center piece 始终位于网格 (1,1)，不参与排列编码。

### 2.5 Piece 提取

**288×288 图像中的网格布局：**
(0,0) (0,1) (0,2)
(1,0) [Center] (1,2)
(2,0) (2,1) (2,2)

提取公式（无 gap、无 border）：
piece(r, c) = image[r × 96 : (r+1) × 96, c × 96 : (c+1) × 96]

### 2.6 Slot 编号与网格位置映射

跳过中心 (1,1)，8 个可移动 slot 按行优先编号：

| Slot Index | 网格位置 (row, col) |
|:----------:|:------------------:|
| 0 | (0, 0) |
| 1 | (0, 1) |
| 2 | (0, 2) |
| 3 | (1, 0) |
| — | (1, 1) = Anchor，固定不动 |
| 4 | (1, 2) |
| 5 | (2, 0) |
| 6 | (2, 1) |
| 7 | (2, 2) |

### 2.7 Piece 身份确定

由于 288×288 图像中的 piece 处于打乱状态，需通过文件名中的排列信息确定每个 piece 的 ground-truth 身份：

- Slot j 位置上的 piece 身份 = `p_j`（由文件名编码）
- 含义：该 piece 在 ground-truth 拼图中应放在 slot `p_j` 的位置
- Anchor 位于 (1,1)，始终为该图像的中心碎片，不编码

---

## 3. 数据划分

### 3.1 划分比例

| 集合 | 数量 | 比例 |
|------|------|------|
| Training | 9,000 | 75% |
| Validation | 1,000 | 8.3% |
| Test | 2,000 | 16.7% |

### 3.2 划分策略

在每个类别内部独立进行随机划分（stratified split），以保持训练、验证、测试集中各类别的比例一致。使用固定随机种子确保可复现。划分以图像为单位。

---

## 4. 2-Mixed 跨类别配对

### 4.1 跨类别约束

每个 2-Mixed episode 中的两张图像必须来自**不同类别**，共三种合法组合：

| 组合编号 | Board A 类别 | Board B 类别 |
|:--------:|:----------:|:----------:|
| 1 | Painting | Engraving |
| 2 | Painting | Artifact |
| 3 | Engraving | Artifact |

### 4.2 采样流程

1. 均匀随机选择一种跨类别组合（3 选 1）
2. 从组合对应的两个类别中各随机抽取 1 张图像
3. 两张图必须来自同一划分集（训练集配对训练集，测试集配对测试集）

### 4.3 设计动机

不同类别间的视觉风格差异（如油画的笔触 vs 版画的线条 vs 器物的纹理）为 Agent 提供了判断碎片归属的线索。跨类别约束使归属判断子任务具有适当难度——既非不可能（同类别内极难区分），也非平凡（仍需学习视觉特征）。Anchor 在此场景下提供了"本 Board 属于哪种风格"的关键锚定信号。

---

## 5. 环境设计

### 5.1 双 Board 结构
Board A                        Board B
┌──────┬──────┬──────┐ ┌──────┬──────┬──────┐
│  s0  │  s1  │  s2  │ │  s0  │  s1  │  s2  │
├──────┼──────┼──────┤ ├──────┼──────┼──────┤
│  s3  │  A_a │  s4  │ │  s3  │ A_b  │  s4  │
├──────┼──────┼──────┤ ├──────┼──────┼──────┤
│  s5  │  s6  │  s7  │ │  s5  │  s6  │  s7  │
└──────┴──────┴──────┘ └──────┴──────┴──────┘
A_a = Image A 的 Anchor A_b = Image B 的 Anchor

每个 Board 含 1 个固定 Anchor（中心）和 8 个可移动 Slot。初始状态下，16 个可移动碎片全局随机打乱后均分至两个 Board（每 Board 8 个）。目标状态下，Board A 的 8 个 slot 放置 Image A 的 8 个 piece 且位置正确，Board B 同理。

### 5.2 全局状态 $s^{\text{global}}_t$

全局状态编码两个 Board 上所有碎片的当前放置情况，仅在**集中训练阶段**对 Critic 可见：

$$
s^{\text{global}}_t = \big(\text{board\_A\_slots}[0..7],\ \text{board\_B\_slots}[0..7],\ \text{anchor}_A,\ \text{anchor}_B\big)
$$

其中每个 slot 元素为该位置上碎片的全局标识 $(\text{source} \in \{A, B\},\ \text{piece\_id} \in \{0..7\})$。

### 5.3 局部观测 $o^{i}_t$

Agent $i \in \{A, B\}$ 的局部观测仅包含**自身 Board** 的信息，在分散执行阶段使用：

$$
o^{i}_t = \big(\text{board\_}i\text{\_slots}[0..7],\ \text{anchor}_i\big)
$$

每个 slot 对应一个 96×96×3 的 piece 图像。Agent 无法直接观测对方 Board 的状态。

### 5.4 通信机制

为支持跨 Board 交换决策中的隐式协商，两个 Agent 在每个时间步的 Pointer 1 决策完成后同步交换通信向量。Agent $i$ 将其 Pointer 1 所选源 piece 的上下文表征作为通信消息广播给对方 Agent $j$：

$$
e^{i}_{\text{comm}} = h^i_{s^*_i} \in \mathbb{R}^d
$$

该通信向量承载了"我打算用这个 piece 进行交换"的隐式提案信息。对方 Agent 在 Pointer 2 决策阶段利用接收到的通信向量作为 "outside" 选项的表征，从而判断对方提出的 piece 是否对自身有价值。通信内容的语义无需预先定义，Agent 通过训练自然学会利用 piece 的视觉特征进行隐式协商。详见 6.4.5 节与 6.9.2 节。

### 5.5 动作空间与同步执行协议

**同步执行（Simultaneous Execution）：**

每个 time step，两个 Agent **同时**输出各自的动作，环境统一结算。这一设计与 Mutual Agreement 机制（6.6 节）配合，使跨 Board 交换需要双方在同一时间步同时提出 "outside" 请求才能执行。

**每个 Agent 的动作分解为两步指向（Pointer Network）：**

- **Pointer 1（源选择）**：Agent $i$ 从自身 Board 的 8 个可移动 slot 中指向 1 个作为 source
- **通信**：两个 Agent 同步交换所选源 piece 的表征
- **Pointer 2（目标选择）**：Agent $i$ 从候选目标集合中指向 1 个作为 target

**Pointer 2 的候选集与动作类型推断：**

Pointer 2 的候选集包含 8 个有效选项（排除源槽位后）：自身 Board 的另外 7 个 slot 加上 1 个特殊的 "outside" 选项。

| Pointer 2 指向 | 动作类型 | 效果 |
|:---:|:---:|:---:|
| 自身 Board 的另一个 slot（$m \in \{0,...,7\} \setminus \{s^*_i\}$） | Intra-board swap | 交换本 Board 两个 slot 的 piece |
| outside 选项（$m = 8$） | 提议 Cross-board swap | 向对方提议交换双方各自 Pointer 1 所选的 piece |

动作类型由 Pointer 2 的指向隐式决定，无需显式分类头。

**Mutual Agreement 规则：**

跨 Board 交换涉及两个 Agent 的 Board，需双方在同一时间步均选择 outside 才实际执行。若仅一方选择 outside，该方本轮为 no-op（动作无效），另一方正常执行 intra swap。此规则在保证跨 Board 交换合理性的同时，为 Agent 引入了协调学习的挑战。详见 6.6 节。

### 5.6 奖励函数 R

奖励为**全局共享奖励**（team reward），两个 Agent 在同一 time step 获得相同的奖励值，与 CTDE 范式兼容。

Agent $i$ 在时间步 $t$ 的总奖励由四个分量构成：

$$
r^i_t = R^i_{\text{progress}} + R^i_{\text{coord}} + R_{\text{step}} + R_{\text{terminal}}
$$

| 分量 | 含义 |
|------|------|
| $R^i_{\text{progress}}$ | **进度奖励**：基于正确放置 piece 数的增量变化，区分 intra swap（仅自身 Board）和 cross swap（全局）|
| $R^i_{\text{coord}}$ | **协调奖励**：激励有效的 cross swap 协调、惩罚协调失败的 no-op |
| $R_{\text{step}}$ | **步惩罚**：负常数，鼓励更少步数内完成 |
| $R_{\text{terminal}}$ | **终局奖励**：两个 Board 均完全正确时给予的大额正奖励 |

各分量的详细定义、参数设置和设计动机见 6.7 节。

### 5.7 终止条件

训练与测试采用**不同的终止逻辑**，核心区别在于测试时不使用 ground-truth oracle，
而是通过 Agent 自身 Critic 的价值估计推断终止时机。

**训练模式**

训练过程中环境可访问 ground-truth。终止条件按优先级排列：

1. **Cooldown 终止**：当 $C_t$ 首次达到 16 时，环境发放终局奖励 $R_{\text{terminal}} = \eta$，
   但**不立即终止**，而是进入 cooldown 阶段，继续运行 $K_{\text{cool}} = 5$ 步。
   Cooldown 期间一切正常执行（swap、progress reward、step penalty），
   $K_{\text{cool}}$ 步后终止 episode。此设计使 Critic 能够观察到"拼图已完成但 episode 未结束"的状态，
   学会为此类状态赋予低价值估计——这一信号在测试中被用作终止依据。
2. **达到最大步数**：$t = T_{\max} = 20{,}000$，终止。

**测试模式**

测试过程中，环境**不使用 ground-truth 判断终止**。终止依据为 Centralized Critic 的
价值估计变化和 Board 配置稳定性。终止条件按优先级排列：

1. **Value Drop 终止**：在每个时间步 $t$，Board State Encoder 输出完成后、Pointer 1
   决策之前，计算 $V(s^{\text{global}}_t)$。维护 episode 内的价值峰值
   $V_{\text{peak}} = \max_{t' \leq t} V(s^{\text{global}}_{t'})$。
   当 $V(s^{\text{global}}_t) < V_{\text{peak}} - \Delta_{\text{drop}}$ 连续 $P$ 步满足时，
   episode 终止，跳过本步的 Pointer 和 Swap 执行。
   终止后使用 ground-truth **仅用于评估指标计算**，不反馈给 Agent。
2. **Board Cycling 终止**：对每个 Board 维护最近 $W$ 步的 piece 排列哈希。
   若两个 Board 的当前配置均在各自窗口内出现过（双方同时 cycling），episode 终止。
3. **达到最大步数**：$t = T_{\max} = 100$，终止，评估最终 Board 状态。

**设计动机**

在实际部署场景中没有外部 oracle 告知"拼图已完成"。本方案利用 Critic 已有的状态评估能力推断终止时机，无需引入额外网络头或额外损失函数。
训练中的 Cooldown Protocol 使 Critic 系统性地观察完成后状态，学会区分"接近完成"（$V$ 高）和"已完成、无进展可期"（$V$ 低）。
测试时，$V$ 从峰值骤降的模式作为终止触发信号，Board Cycling 检测作为补充安全网确保即使 $V$ 信号不够清晰也能终止。

| 条件 | 训练 | 测试 |
|:---|:---:|:---:|
| Oracle（$C_t = 16$） | ✅ 发放 $R_{\text{terminal}}$，进入 cooldown | ❌ 不使用 |
| Cooldown 结束（$K_{\text{cool}}$ 步后） | ✅ 终止 | — |
| Value Drop（$V < V_{\text{peak}} - \Delta_{\text{drop}}$ 持续 $P$ 步） | ❌ 不用于终止 | ✅ 终止 → 事后评估 |
| Board Cycling（双方配置重复） | ❌ | ✅ 终止 → 事后评估 |
| 达到最大步数 | $T_{\max} = 20{,}000$ | $T_{\max} = 100$ |

### 5.7.1 Cooldown Protocol 详述

Cooldown Protocol 是一个仅存在于训练阶段的机制，目的是为 Critic 提供
"拼图已完成但 episode 未结束"的训练经验。

**触发**：当环境检测到 $C_t = 16$（双 Board 全部正确）时，在该时间步的奖励中
加入 $R_{\text{terminal}} = \eta$，同时将 episode 标记为 cooldown 状态。

**Cooldown 期间的行为**：

Agent 被迫继续执行 swap（没有 skip 动作）。由于 Board 已全部正确，
任何 intra swap 都会使 $C_t$ 下降，产生负的 progress reward。
Agent 面临的最优策略是 swap-and-undo：在连续两步中交换同一对 piece，
使 $C_t$ 在 16 和 14 之间振荡，净 progress reward 为 0，仅承担步数惩罚。
Agent 通过负向 progress reward 的梯度信号自然学会这一保守行为。

对于 cross swap：cooldown 期间 Agent 不应选择 outside，
因为跨 Board 交换会同时破坏两个 Board 且难以撤销。
协调失败惩罚 $R^i_{\text{fail}}$ 和负的 cross progress reward 足以抑制此行为。

**Critic 学习效果**：

Cooldown 经验使 Critic 学到以下价值函数模式：

- $V(s \mid C_t \approx 14\text{–}15) \approx \eta + \alpha_1 \times \mathbb{E}[\Delta C] - \text{few} \times \delta \approx 10+$
  （接近完成，预期获得终局奖励和剩余 progress）
- $V(s \mid \text{cooldown 阶段}) \approx -\text{remaining\_cooldown} \times \delta \approx -0.25$
  （已完成，仅剩步数惩罚）

从 $\sim 10$ 到 $\sim -0.25$ 的断崖式下降（$\Delta \approx 10$）远超解题过程中的正常波动（$\Delta \lesssim 2$），
在测试时构成一个高信噪比的终止信号。

**对训练效率的影响**：Cooldown 仅增加 $K_{\text{cool}} = 5$ 步，
相对于 $T_{\max} = 20{,}000$ 的训练上限可忽略。
Agent 在 cooldown 期间的行为（swap-and-undo）简单且快速收敛，不会显著消耗训练资源。

---

# 6. 网络架构

## 6.1 整体架构概览

本系统由两个结构对称、参数共享的 Agent（记为 Agent A 与 Agent B）组成，采用 **Centralized Training with Decentralized Execution（CTDE）** 范式。每个 Agent 拥有一个 3×3 网格的拼图 Board，其中中心位置（centre）放置一块固定不动的 **centre piece**，其余 8 个位置（slot）放置着属于两幅不同目标图像的可移动 piece。Agent 的目标是通过 Board 内部交换（intra swap）和跨 Board 交换（cross swap）将所有 piece 归位至正确位置，使两个 Board 各自还原为完整的目标图像。centre piece 在整个 episode 中始终固定于中心位置，为 Agent 提供关于目标图像最关键的视觉锚点信息。

每个 Agent 的策略网络由五个核心模块串联构成：Piece Encoder 负责将每个 piece（包括 centre piece）的图像编码为向量表征；Board State Encoder 利用 Transformer 架构对 9 个 piece（8 个可移动 piece + 1 个 centre piece）的表征进行自注意力建模，使每个可移动 piece 的表征融合 centre piece 提供的目标图像上下文信息，输出包含全局关系的 Board 表征；Pointer 1 Network 基于 Board 表征在 8 个可移动槽位中选择一个源槽位（source slot）；通信模块将所选 piece 的表征广播给对方 Agent；Pointer 2 Network 在接收到对方通信信息后，综合源 piece 表征与 Board 全局摘要，决定将源 piece 与自身 Board 内的其他槽位交换（intra swap），还是提议与对方进行跨 Board 交换（cross swap）。此外，训练阶段设置集中式 Value Network 用于估计全局状态价值。

两个 Agent 共享同一套网络参数（parameter sharing），仅通过各自观测到的不同 Board 状态和接收到的不同通信内容产生差异化行为。这种设计在保证策略对称性的同时大幅降低了参数量和训练难度。

---

## 6.2 观测空间设计

### 6.2.1 局部观测

在时间步 $t$，Agent $i$（$i \in \{A, B\}$）的局部观测 $o^i_t$ 包含以下信息：

其一，**可移动 piece 图像信息**：Agent $i$ 的 Board 上 8 个可移动槽位各自对应一张 piece 图像 $x^i_{s,t} \in \mathbb{R}^{H_p \times W_p \times 3}$，其中 $s \in \{0, 1, \dots, 7\}$ 为槽位索引，$H_p \times W_p$ 为单个 piece 的图像尺寸。

其二，**Centre piece 图像信息**：Agent $i$ 的 Board 中心位置固定放置的 centre piece 图像 $x^i_{\text{centre}} \in \mathbb{R}^{H_p \times W_p \times 3}$。centre piece 在整个 episode 中不变，为 Agent 提供目标图像最核心的视觉参考。Agent 可以通过 centre piece 的视觉内容推断各周围位置应放置何种 piece——与真实拼图中人类利用中心区域判断周围 piece 归属的认知过程一致。

其三，**3×3 网格位置信息**：每个槽位及 centre piece 在 3×3 网格中的空间位置，通过可学习的位置编码注入网络。位置编码使 Agent 能够理解每个 piece 与 centre piece 之间以及 piece 彼此之间的空间邻接关系。

### 6.2.2 通信观测

在每个时间步的 Pointer 2 决策阶段，Agent $i$ 额外接收到来自对方 Agent $j$ 的通信向量 $e^j_{\text{comm}} \in \mathbb{R}^d$，该向量为 Agent $j$ 通过 Pointer 1 所选 piece 的上下文表征，承载了"对方打算用什么 piece 来交换"的隐式信息。

### 6.2.3 集中式 Critic 观测（仅训练时）

在 CTDE 框架下，训练阶段的集中式 Value Network 可以访问全局状态 $s_t$，包含两个 Agent 的 Board 状态摘要 $h^A_{\text{board}}$ 与 $h^B_{\text{board}}$，以及双方的通信向量 $e^A_{\text{comm}}$ 与 $e^B_{\text{comm}}$。该全局信息仅在训练时用于 Critic 估值，执行时每个 Agent 仅依赖自身局部观测和通信做决策。

---

## 6.3 动作空间设计

每个 Agent 在每个时间步 $t$ 输出拼图操作：Pointer 1（源位置选择）+ Pointer 2（目标位置选择）。
不包含显式的终止动作——终止逻辑由 Critic 价值变化和 Board 配置稳定性在测试阶段自动推断。

### 6.3.1 Pointer 1（源位置选择）

Agent $i$ 从自身 Board 的 8 个可移动槽位中选择一个作为源位置，
输出为离散动作 $a^i_{\text{ptr1}} \in \{0, 1, \dots, 7\}$。
语义为"我选中这个 piece，打算移动它"。

### 6.3.2 通信阶段

两个 Agent 同步交换 Pointer 1 所选 piece 的上下文表征（详见 6.4.5 节）。

### 6.3.3 Pointer 2（目标位置选择）

在接收到对方通信后，Agent $i$ 从候选位置集合中选择一个作为目标。
候选集合为 $\{0, 1, \dots, 8\} \setminus \{a^i_{\text{ptr1}}\}$，共 8 个有效选项。
其中索引 0–7 中非源的 7 个对应自身 Board 的其他可移动槽位
（选择其一表示执行 intra swap），
索引 8 为特殊的 "outside" 选项（表示提议与对方进行 cross swap）。
通过强制排除源槽位，Agent 每步必须执行一次有效交换或提出 cross swap 提议，
避免策略退化为反复跳过（skip）。

### 6.3.4 联合动作与策略分解

两个 Agent 的完整联合动作为
$a_t = (a^A_{\text{ptr1}}, a^A_{\text{ptr2}}, a^B_{\text{ptr1}}, a^B_{\text{ptr2}})$，
环境根据 6.6 节的 Mutual Agreement 规则判定实际执行结果。

Agent $i$ 的策略可分解为两个条件概率的乘积：

$$
\pi^i(a^i_t \mid o^i_t) = \pi^i_{\text{ptr1}}(a^i_{\text{ptr1}} \mid o^i_t) \;\cdot\; \pi^i_{\text{ptr2}}(a^i_{\text{ptr2}} \mid o^i_t,\; a^i_{\text{ptr1}},\; e^j_{\text{comm}})
$$

---

## 6.4 网络组件详解

以下各子节详述每个网络模块的结构与数学定义。所有模块在两个 Agent 之间共享参数。定义嵌入维度 $d = 128$。

### 6.4.1 Piece Encoder

Piece Encoder 负责将每张 piece 图像映射为固定维度的特征向量。采用在 ImageNet 上预训练的 **EfficientNet-B0** 作为视觉骨干网络，移除其原始分类头，保留全局平均池化层输出的 1280 维特征，随后通过一个线性投影层降维至 $d$ 维：

$$
e = W_{\text{proj}} \cdot f_{\text{backbone}}(x) + b_{\text{proj}}
$$

其中 $f_{\text{backbone}}: \mathbb{R}^{H_p \times W_p \times 3} \to \mathbb{R}^{1280}$ 为 EfficientNet-B0 的特征提取部分，$W_{\text{proj}} \in \mathbb{R}^{d \times 1280}$ 与 $b_{\text{proj}} \in \mathbb{R}^d$ 为可学习参数。

该 Encoder 统一处理可移动 piece 和 centre piece。对于可移动 piece，编码为 $e^i_{s,t} \in \mathbb{R}^d$（$s \in \{0, \dots, 7\}$）；对于 centre piece，编码为 $e^i_{\text{centre}} \in \mathbb{R}^d$。由于 centre piece 在整个 episode 中不变，其嵌入在 episode 开始时计算一次并缓存，无需每步重复计算。

EfficientNet-B0 在训练 Phase 1 和 Phase 2 中冻结骨干网络参数，仅训练投影层；在 Phase 3 中以 0.1 倍基础学习率进行端到端微调。

### 6.4.2 位置编码与输入构建

**3×3 网格位置编码**

Board 的 9 个位置按行优先顺序排列在 3×3 网格中，网格位置索引 $g \in \{0, 1, \dots, 8\}$ 的空间布局如下：

$$
\begin{array}{|c|c|c|}
\hline
g=0 & g=1 & g=2 \\
\hline
g=3 & g=4\;(\text{centre}) & g=5 \\
\hline
g=6 & g=7 & g=8 \\
\hline
\end{array}
$$

其中网格位置 $g = 4$ 为 centre piece 的固定位置，其余 8 个网格位置 $g \in \{0,1,2,3,5,6,7,8\}$ 对应 8 个可移动槽位。定义槽位索引 $s \in \{0, \dots, 7\}$ 到网格位置的映射函数：

$$
\text{gridpos}(s) = \begin{cases} s & \text{if } s < 4 \\ s + 1 & \text{if } s \geq 4 \end{cases}
$$

为 9 个网格位置定义可学习的位置编码向量 $\text{PE}_g \in \mathbb{R}^d$，$g \in \{0, 1, \dots, 8\}$。位置编码捕获 3×3 网格中的二维空间关系——特别是每个可移动槽位与 centre piece（$g=4$）之间的邻接关系。Edge 位置（$g \in \{1,3,5,7\}$）与 centre piece 共享一条边，corner 位置（$g \in \{0,2,6,8\}$）与 centre piece 对角相邻，这些空间关系在训练中通过位置编码自然习得。

**输入向量构建**

对于可移动槽位 $s$，将 piece 嵌入与对应网格位置的位置编码相加：

$$
\hat{z}^i_s = e^i_{s,t} + \text{PE}_{\text{gridpos}(s)} \in \mathbb{R}^d
$$

对于 centre piece，将其嵌入与中心位置编码相加：

$$
\hat{z}^i_{\text{centre}} = e^i_{\text{centre}} + \text{PE}_4 \in \mathbb{R}^d
$$

最终，Board State Encoder 的输入由 9 个 $d$ 维向量组成：8 个可移动槽位输入 $\hat{z}^i_0, \dots, \hat{z}^i_7$ 加上 1 个 centre piece 输入 $\hat{z}^i_{\text{centre}}$。

### 6.4.3 Board State Encoder

Board State Encoder 采用标准 **Transformer Encoder** 架构，对 9 个输入向量（8 个可移动 piece + 1 个 centre piece）进行自注意力建模。该设计的核心在于：centre piece 作为目标图像的视觉锚点参与全局注意力计算，使每个可移动 piece 的输出表征自然融合 centre piece 提供的目标图像上下文信息。

具体而言，Transformer 的多头自注意力机制允许以下关键信息流动。其一，每个可移动 piece 可以 attend 到 centre piece，从而获知目标图像中心区域的视觉内容（颜色、纹理、语义结构），据此判断自身是否与 centre piece 在视觉上连续，是否属于当前 Board 的目标图像。其二，centre piece 的位置编码（$\text{PE}_4$）与各可移动 piece 的位置编码之间的关系隐式编码了空间邻接结构——与 centre 共享一条边的 edge 位置（上、下、左、右）以及对角相邻的 corner 位置各自具有不同的空间语义，网络可以学会利用这些关系判断某个 piece 是否属于特定位置。其三，可移动 piece 之间的相互注意力可以捕获 piece 间的视觉一致性、边缘匹配等全局拼图线索。

Transformer Encoder 由 $L = 3$ 层堆叠而成，每层包含多头自注意力（Multi-Head Self-Attention，头数 $N_h = 4$）和前馈网络（FFN，隐藏层维度 $d_{\text{ff}} = 256$），均配备残差连接与 LayerNorm：

$$
H^i = \text{TransformerEncoder}(\hat{z}^i_0, \dots, \hat{z}^i_7, \hat{z}^i_{\text{centre}}) \in \mathbb{R}^{9 \times d}
$$

输出为 9 个上下文化的表征：8 个可移动槽位表征 $h^i_0, h^i_1, \dots, h^i_7 \in \mathbb{R}^d$ 以及 centre piece 表征 $h^i_{\text{centre}} \in \mathbb{R}^d$。经过自注意力建模后，每个可移动 piece 的表征 $h^i_s$ 不再仅包含该 piece 自身的视觉特征，还融合了 centre piece 的目标图像信息和其他 piece 的上下文信息，形成了对"该 piece 在当前 Board 状态下的角色"的综合表征。

定义 Board 全局摘要为所有 9 个表征的均值池化，包含 centre piece 以确保全局摘要携带目标图像的锚点信息：

$$
h^i_{\text{board}} = \frac{1}{9} \Big(\sum_{s=0}^{7} h^i_s + h^i_{\text{centre}}\Big) \in \mathbb{R}^d
$$

### 6.4.4 Pointer 1 Network（源位置选择）

Pointer 1 Network 采用 Pointer Network 注意力机制，以 Board 全局摘要为查询（query），以 8 个可移动槽位的上下文表征为键（key），计算每个槽位被选为源位置的得分。centre piece 不参与 Pointer 1 的候选集，因为它是固定不可移动的：

$$
u^{(1)}_s = v_1^\top \tanh\!\big(W_{q1} \, h^i_{\text{board}} + W_{k1} \, h^i_s\big), \quad s \in \{0, 1, \dots, 7\}
$$

其中 $v_1 \in \mathbb{R}^d$，$W_{q1} \in \mathbb{R}^{d \times d}$，$W_{k1} \in \mathbb{R}^{d \times d}$ 均为可学习参数。策略概率通过 softmax 归一化获得：

$$
\pi^i_{\text{ptr1}}(s \mid o^i_t) = \frac{\exp(u^{(1)}_s)}{\sum_{s'=0}^{7} \exp(u^{(1)}_{s'})}
$$

训练时通过采样获得源槽位 $s^*_i \sim \pi^i_{\text{ptr1}}$，推理时可采用贪心选择或采样。

### 6.4.5 通信模块

Pointer 1 选择完成后，两个 Agent 同步交换通信信息。Agent $i$ 将其所选源 piece 的上下文表征广播给对方：

$$
e^i_{\text{comm}} = h^i_{s^*_i} \in \mathbb{R}^d
$$

Agent $j$（$j \neq i$）接收到此向量后，将其作为 Pointer 2 中 "outside" 选项的表征。通信内容本身就承载了"我打算用这个 piece 和你交换"的隐式提案信息——piece 的视觉特征即为最直接的协商语言。Agent 无需显式的意愿信号，仅通过 piece 表征的语义匹配即可完成隐式协商（详见 6.9 节分析）。

在训练 Phase 1（纯 intra 阶段，outside 选项被 mask）中，通信模块不启用，使用一个可学习的空通信向量 $e_{\text{null}} \in \mathbb{R}^d$ 替代：

$$
\hat{e}^i_{\text{out}} = \begin{cases} e^j_{\text{comm}} & \text{通信已启用（Phase 2/3）} \\[4pt] e_{\text{null}} & \text{通信未启用（Phase 1）} \end{cases}
$$

### 6.4.6 Pointer 2 Network（目标位置选择）

Pointer 2 Network 决定将源 piece 交换到何处。与 Pointer 1 仅使用 Board 全局摘要作为查询不同，Pointer 2 需要同时考虑"手中拿的是什么 piece"和"当前整个拼图的进度与状态"两方面信息，因此将源槽位表征与 Board 全局摘要拼接后通过线性投影融合为查询向量：

$$
q^{(2)}_i = W_{\text{fuse}} \big[h^i_{s^*_i} \;\|\; h^i_{\text{board}}\big] + b_{\text{fuse}}
$$

其中 $W_{\text{fuse}} \in \mathbb{R}^{d \times 2d}$，$b_{\text{fuse}} \in \mathbb{R}^d$ 为可学习参数，$[\cdot \| \cdot]$ 表示向量拼接。这一设计使 Pointer 2 在选择目标位置时能够综合局部信息（源 piece 的视觉特征和空间位置）与全局信息（Board 整体的完成进度、piece 分布格局），从而做出更具全局观的决策。例如，当 Board 上大部分 piece 已归位时，Agent 更应倾向于 intra swap 微调；当 Board 上存在大量不属于本 Board 的 piece 时，Agent 更应倾向于提议 cross swap。

Pointer 2 的候选键（key）集合由自身 Board 的 8 个可移动槽位表征和 1 个 outside 表征构成，共 9 个候选。对所有候选计算注意力得分，随后将源槽位 $s^*_i$ 对应的 logit 强制设为 $-\text{1e9}$（详见下方实现说明），确保 Agent 不会选择将 piece 交换回原位（即禁止 skip）：

$$
u^{(2)}_m = v_2^\top \tanh\!\big(W_{q2} \, q^{(2)}_i + W_{k2} \, \mathcal{K}_m\big), \quad m \in \{0, 1, \dots, 8\}
$$

其中键集合 $\mathcal{K}$ 定义为：

$$
\mathcal{K}_m = \begin{cases} h^i_m & \text{if } m \in \{0, 1, \dots, 7\} \\[4pt] \hat{e}^i_{\text{out}} & \text{if } m = 8 \end{cases}
$$

$v_2 \in \mathbb{R}^d$，$W_{q2} \in \mathbb{R}^{d \times d}$，$W_{k2} \in \mathbb{R}^{d \times d}$ 为可学习参数。定义有效候选集合 $\mathcal{V}$ 为排除源槽位后的集合：

$$
\mathcal{V} = \{0, 1, \dots, 8\} \setminus \{s^*_i\}
$$

策略概率在有效候选上通过 masked softmax 归一化：

$$
\pi^i_{\text{ptr2}}(m \mid o^i_t, s^*_i, \hat{e}^i_{\text{out}}) = \frac{\exp(u^{(2)}_m) \cdot \mathbb{1}[m \in \mathcal{V}]}{\sum_{m' \in \mathcal{V}} \exp(u^{(2)}_{m'})}
$$

其中 $m \in \{0, \dots, 7\} \setminus \{s^*_i\}$ 表示 intra swap（将源 piece 与槽位 $m$ 上的 piece 互换），$m = 8$ 表示提议 cross swap。有效候选总数恒为 8 个（7 个其他可移动槽位 + 1 个 outside 选项）。在 Phase 1 中，$m = 8$ 的 logit 额外被设为 $-\text{1e9}$，有效候选缩减为 7 个。

> **实现说明（Masking 数值稳定性）**：在 PyTorch 实现中，所有需要 mask 的 logit 应设为 `-1e9`（一个极大的负有限数）而非 `float('-inf')`。使用 `-inf` 会导致 softmax 输出中被 mask 的位置概率为精确的 0，在后续 `log_prob` 计算时产生 `-inf`，进而在反向传播中引发 NaN 梯度。使用 `-1e9` 可使被 mask 位置的概率极度接近 0（约 $e^{-10^9} \approx 0$）但在梯度计算中保持数值稳定。

### 6.4.7 Value Network（集中式 Critic）

训练阶段采用集中式 Critic 估计全局状态价值。Value Network 接收两个 Agent 的 Board 摘要以及双方通信向量作为输入，通过多层感知机输出标量估值：

$$
V(s_t) = \text{MLP}\!\big([h^A_{\text{board}} \;\|\; h^B_{\text{board}} \;\|\; e^A_{\text{comm}} \;\|\; e^B_{\text{comm}}]\big)
$$

其中 $[\cdot \| \cdot]$ 表示向量拼接，MLP 结构为 $\text{Linear}(4d, 256) \to \text{ReLU} \to \text{Linear}(256, 128) \to \text{ReLU} \to \text{Linear}(128, 1)$。

将通信向量纳入 Critic 输入的设计动机在于：通信向量承载了双方当前意图选择的 piece 信息，反映了当前时间步的"协商状态"。Critic 需要评估的不仅是"两个 Board 目前的拼图进度"，还包括"双方当前的协调意愿是否匹配"——例如双方同时提出合理的 cross swap 提议时，状态价值应更高。加入通信向量使 Critic 能更准确地估计这种协商维度的价值差异，从而为 Actor 提供更精确的优势估计（Advantage Estimation）。

由于 $h^i_{\text{board}}$ 已包含 centre piece 的信息（通过 6.4.3 的均值池化），Critic 可以感知两个 Board 各自的目标图像上下文。两个 Agent 共享同一个 Critic，因为 CTDE 中全局状态价值对双方一致。执行时 Value Network 不参与决策。

在 Phase 1（通信未启用）中，通信向量位置使用可学习的空向量 $e_{\text{null}}$ 填充，确保 Critic 输入维度一致。

---

## 6.5 单步前向传播流程

每个时间步 $t$ 的计算流程如下。训练模式与测试模式在 Value 检查后出现分支。

**阶段 1 — Piece 编码**：两个 Agent 各自将自身 Board 上 8 个可移动 piece 的图像
通过 Piece Encoder 编码为嵌入向量，并从缓存中取出 centre piece 的嵌入。
各嵌入与对应的 3×3 网格位置编码相加，形成 9 个输入向量。

**阶段 2 — Board 状态编码**：各 Agent 将 9 个输入向量送入 Board State Encoder（Transformer），
获得上下文化表征 $h^i_0, \dots, h^i_7, h^i_{\text{centre}}$ 及 Board 摘要 $h^i_{\text{board}}$。

**阶段 2.5 — 终止检查（仅测试模式）**：计算 Centralized Critic 的
$V(s^{\text{global}}_t)$，检查 Value Drop 条件和 Board Cycling 条件。
若任一满足，episode 终止，跳过后续阶段。
**训练模式下**跳过此检查，直接进入阶段 3。

**阶段 3 — Pointer 1 决策与通信**：各 Agent 通过 Pointer 1 Network 在 8 个可移动槽位中
选择源槽位 $s^*_i$，并将 $h^i_{s^*_i}$ 作为通信向量同步广播给对方。

**阶段 4 — Pointer 2 决策**：各 Agent 接收对方通信后，通过 Pointer 2 Network 选择目标 $m^*_i$。

**阶段 5 — 动作执行与环境更新**：环境根据 Mutual Agreement 规则判定执行结果，
更新两个 Board 的状态，计算奖励。**训练模式下**：检查 $C_t = 16$，
若满足则进入 cooldown 状态或（若已在 cooldown 中）检查 cooldown 步数是否耗尽。

---

## 6.6 动作执行规则（Mutual Agreement）

### 6.6.1 Swap 执行规则

Cross swap 涉及两个 Agent 的 Board，因此需要双方的协调同意。由于两个 Agent 在每个时间步**同步执行**动作，环境在收到双方的 Pointer 2 输出后联合判定执行逻辑：

| Agent A Ptr2 | Agent B Ptr2 | 执行结果 |
|:---:|:---:|:---|
| intra ($m \in 0\text{-}7$) | intra ($m \in 0\text{-}7$) | 各自独立执行 intra swap |
| outside ($m = 8$) | outside ($m = 8$) | ✅ cross swap 执行：Board A 的 $s^*_A$ 槽位 piece 与 Board B 的 $s^*_B$ 槽位 piece 互换 |
| outside ($m = 8$) | intra ($m \in 0\text{-}7$) | ⚠️ Agent A 为 no-op（浪费本轮），Agent B 正常执行 intra swap |
| intra ($m \in 0\text{-}7$) | outside ($m = 8$) | ⚠️ Agent B 为 no-op（浪费本轮），Agent A 正常执行 intra swap |

定义以下指示变量供奖励函数使用。Cross swap 成功执行的指示：

$$
\mathbb{1}_{\text{cross}} = \mathbb{1}\!\big[a^A_{\text{ptr2}} = 8 \;\wedge\; a^B_{\text{ptr2}} = 8\big]
$$

Agent $i$ 因单方面提议 outside 而导致 no-op 的指示：

$$
\mathbb{1}^{i}_{\text{fail}} = \mathbb{1}\!\big[a^i_{\text{ptr2}} = 8 \;\wedge\; a^{j}_{\text{ptr2}} \neq 8\big], \quad j \neq i
$$

### 6.6.2 测试终止规则

测试模式下，在 Board State Encoder 输出完成后、Pointer 1 决策之前（6.5 节阶段 2.5），
依次检查以下两个终止条件：

**条件 A — Value Drop**：

维护 episode 内的价值峰值 $V_{\text{peak}}$ 和耐心计数器 $n_{\text{patience}}$：

$$
V_{\text{peak}} \leftarrow \max(V_{\text{peak}},\; V(s^{\text{global}}_t))
$$

$$
n_{\text{patience}} \leftarrow \begin{cases}
n_{\text{patience}} + 1 & \text{if } V(s^{\text{global}}_t) < V_{\text{peak}} - \Delta_{\text{drop}} \\[4pt]
0 & \text{otherwise}
\end{cases}
$$

当 $n_{\text{patience}} \geq P$ 时触发终止。默认 $\Delta_{\text{drop}} = 3.0$，$P = 3$。

**条件 B — Board Cycling**：

对每个 Board 维护最近 $W$ 步的 piece 排列哈希集合
$\mathcal{H}^i = \{\text{hash}(B^i_{t-W}), \dots, \text{hash}(B^i_{t-1})\}$。
当双方当前配置均在各自窗口内出现过时触发终止：

$$
\mathbb{1}_{\text{cycle}} = \mathbb{1}\!\big[\text{hash}(B^A_t) \in \mathcal{H}^A\big] \;\wedge\; \mathbb{1}\!\big[\text{hash}(B^B_t) \in \mathcal{H}^B\big]
$$

默认 $W = 6$。Board 配置的哈希基于 piece ID 到 slot 的映射，
不依赖 ground-truth（piece ID 是环境中每个 piece 的唯一标识符，Agent 可观察）。

任一条件触发时，episode 终止，当前 Board 状态作为最终状态用于评估。

---

## 6.7 奖励函数设计

### 6.7.1 基础度量

定义每个 Board 上正确放置的 piece 数量：

$$
C^i_t = \sum_{s=0}^{7} \mathbb{1}\!\big[\text{slot } s \text{ 上的 piece 属于该位置}\big], \quad i \in \{A, B\}
$$

全局正确放置数为两个 Board 之和：

$$
C_t = C^A_t + C^B_t \in [0, 16]
$$

各维度的逐步变化量定义为：

$$
\Delta C^i_t = C^i_t - C^i_{t-1}, \qquad \Delta C_t = C_t - C_{t-1}
$$

### 6.7.2 奖励分解

Agent $i$ 在时间步 $t$ 的总奖励由四个分量构成：

$$
r^i_t = R^i_{\text{progress}} + R^i_{\text{coord}} + R_{\text{step}} + R_{\text{terminal}}
$$

（各分量定义与 v0.4 一致，此处仅说明与终止相关的变更。）

**分量 4 — 终局奖励（Terminal Reward）**

终局奖励在训练模式下于 $C_t$ 首次达到 16 的时间步发放：

$$
R_{\text{terminal}} = \begin{cases} \eta & \text{if } C_t = 16 \;\wedge\; \text{非 cooldown 状态（首次达到）} \\[4pt] 0 & \text{otherwise} \end{cases}
$$

其中 $\eta = 10.0$。注意：$R_{\text{terminal}}$ 仅发放一次，cooldown 期间不再重复发放。
Cooldown 期间 $C_t$ 可能因 swap 而下降并再次回到 16，但不触发第二次终局奖励。

在测试模式下不涉及奖励计算。

### 6.7.3 奖励汇总表

| 分量 | 符号 | 触发条件 | 奖励值 | 信号目的 | 实现位置 |
|:---|:---|:---|:---|:---|:---|
| Intra 进度 | $R^i_{\text{progress}}$ | intra swap 执行 | $\alpha_1 \cdot \Delta C^i_t$ | 基础任务驱动 | Environment |
| Cross 进度 | $R^i_{\text{progress}}$ | cross swap 执行 | $\alpha_1 \cdot \Delta C_t$ | 联合收益共享 | Environment |
| 协调成功（有效） | $R^i_{\text{success}}$ | 双方 outside 且 $\Delta C_t > 0$ | $+0.5$ | 激励有效协调 | Environment |
| 协调成功（无效） | $R^i_{\text{success}}$ | 双方 outside 且 $\Delta C_t = 0$ | $-0.1$ | 抑制无意义协调 | Environment |
| 协调失败 | $R^i_{\text{fail}}$ | 单方 outside | $-0.3$ | 惩罚误判对方意图 | Environment |
| 意愿对齐 | $R^i_{\text{intent}}$ | 每步（训练时） | $\lambda_{\text{align}} \cdot (\text{alignment})$ | 加速协调收敛 | **PPO update** |
| 步数惩罚 | $R_{\text{step}}$ | 每步 | $-0.05$ | 鼓励效率 | Environment |
| 终局奖励 | $R_{\text{terminal}}$ | $C_t = 16$ | $+10.0$ | 完成任务 | Environment |

---

## 6.8 协调学习的课程训练策略

直接从零开始学习 Mutual Agreement 的协调极为困难。两个 Agent 需要同时探索到 outside 选项并获得正反馈，这是一个典型的协调探索（coordinated exploration）问题。为此设计三阶段课程训练（curriculum training）。

**Phase 1：纯 Intra 训练（训练进度 0% – 30%）**

Pointer 2 的 index 8（outside）被强制 mask（logit 设为 $-\text{1e9}$），Agent 只能进行 intra swap，有效候选为 7 个其他可移动槽位。此阶段的训练目标是让 Agent 学会 piece 的视觉特征编码（EfficientNet-B0 投影层训练），学会利用 centre piece 的视觉信息判断 piece 归属（Board State Encoder 中的 centre piece 注意力机制），以及建立稳定的 Board 内拼图逻辑（Pointer 1 与 Pointer 2 的 intra 部分）。通信模块不启用，outside 表征使用可学习的空向量 $e_{\text{null}}$。奖励仅包含 $R^i_{\text{progress}}$（intra）、$R_{\text{step}}$ 和 $R_{\text{terminal}}$。Critic 的通信向量输入位置使用 $e_{\text{null}}$ 填充。

**Phase 2：引导式协调探索（训练进度 30% – 60%）**

解除 outside mask，但通过以下两个机制降低协调学习难度。

机制 2a 为**强制协调轮次（Forced Coordination Round）**。每个 episode 中，以概率 $p_{\text{force}}(t)$ 将某个时间步设为"强制协调轮"。在此轮中两个 Agent 的 Pointer 2 均被强制设为 outside（index 8），环境直接执行 cross swap。Agent 仍然通过 Pointer 1 自主选择 source，因此学习的是"选哪个 piece 进行 cross swap 最有价值"。强制概率随训练进度线性退火：

$$
p_{\text{force}}(t) = 0.5 \times \left(1 - \frac{t - t_{\text{P2\_start}}}{t_{\text{P2\_end}} - t_{\text{P2\_start}}}\right)
$$

从 50% 线性下降至 0%，逐步将协调决策权交还给 Agent。

机制 2b 为**协调失败惩罚退火**。Phase 2 前期将 $\gamma$ 设为较小值以避免过早抑制 outside 探索，后期逐步增加至目标值：

$$
\gamma(t) = 0.05 + 0.25 \times \frac{t - t_{\text{P2\_start}}}{t_{\text{P2\_end}} - t_{\text{P2\_start}}}
$$

**Phase 3：自由协调训练（训练进度 60% – 100%）**

移除所有强制机制，Agent 完全自主决策。此阶段 $p_{\text{force}} = 0$，$\gamma = 0.3$（完整惩罚），所有奖励分量均激活（包括 $R^i_{\text{intent}}$），EfficientNet-B0 骨干网络以 0.1 倍基础学习率进行端到端微调。

---

## 6.9 协调行为的理论分析

### 6.9.1 博弈论视角

在 Phase 2 的引导下，Agent 已积累了 cross swap 的经验——它们知道在某些 Board 状态下 cross swap 会带来正向 $\Delta C_t$。进入 Phase 3 后，Agent 在每个时间步面临的是一个对称协调博弈（symmetric coordination game）。假设当前状态下 cross swap 有益，其简化收益矩阵为：

| | Agent B: intra | Agent B: outside |
|:---:|:---:|:---:|
| **Agent A: intra** | $(\alpha_1 \Delta C^A,\; \alpha_1 \Delta C^B)$ | $(\alpha_1 \Delta C^A,\; -\gamma - \delta)$ |
| **Agent A: outside** | $(-\gamma - \delta,\; \alpha_1 \Delta C^B)$ | $(\alpha_1 \Delta C + \beta_1,\; \alpha_1 \Delta C + \beta_1)$ |

当 cross swap 确实有益时（$\alpha_1 \Delta C + \beta_1 > \alpha_1 \Delta C^i$），(outside, outside) 构成一个 Pareto 最优的纳什均衡。意愿对齐奖励 $R^i_{\text{intent}}$ 提供的梯度信号帮助打破初始的对称不确定性，引导双方策略概率向同一方向移动，加速收敛至该均衡点。

### 6.9.2 涌现通信机制

通信模块的设计使得 Agent 之间形成了一种**涌现通信（Emergent Communication）** 机制。具体而言，Agent A 通过 Pointer 1 选择 $s^*_A$ 后，将该 piece 的上下文表征 $h^A_{s^*_A}$ 广播给 Agent B。Agent B 在做 Pointer 2 决策时，outside 选项的表征即为 $h^A_{s^*_A}$——即 Agent A 打算拿来交换的 piece 的特征。由于该表征经过 Board State Encoder 处理，已经融合了 Agent A 的 centre piece 的视觉信息，Agent B 可以据此推断该 piece 是否属于自身 Board 的目标图像。如果这个 piece 恰好是 Agent B 的 Board 所需要的（与某个槽位在目标图像中的视觉内容高度匹配），Agent B 选择 outside 的概率自然会升高；反之亦然。

因此，通信内容的语义无需预先定义，Agent 通过训练自然学会了以 source piece 的选择作为"提案"，对方通过 Pointer 2 的 outside/intra 选择来"接受"或"拒绝"这一提案。piece 的视觉特征本身就是最直接、最信息丰富的协商语言，无需额外设计显式的消息协议。

---

## 6.10 超参数汇总

| 类别 | 参数 | 符号 | 值 | 说明 |
|:---|:---|:---|:---|:---|
| 网络结构 | 嵌入维度 | $d$ | 128 | Piece / Board / Pointer 统一维度 |
| 网络结构 | Transformer 层数 | $L$ | 3 | Board State Encoder |
| 网络结构 | 注意力头数 | $N_h$ | 4 | Multi-Head Self-Attention |
| 网络结构 | FFN 隐藏维度 | $d_{\text{ff}}$ | 256 | Transformer 前馈网络 |
| 网络结构 | Transformer 输入序列长度 | — | 9 | 8 可移动 piece + 1 centre piece |
| 网络结构 | Pointer 2 查询融合层 | $W_{\text{fuse}}$ | $\mathbb{R}^{d \times 2d}$ | 源表征 + Board 摘要 → 融合查询 |
| 网络结构 | Critic MLP 维度 | — | 4d → 256 → 128 → 1 | 集中式 Value Network（含通信向量） |
| 网络结构 | 3×3 网格位置编码数 | — | 9 | 可学习位置编码 $\text{PE}_0, \dots, \text{PE}_8$ |
| 网络结构 | Masking 常量 | — | $-\text{1e9}$ | 替代 $-\infty$ 以确保数值稳定性 |
| 奖励函数 | 进度奖励系数 | $\alpha_1$ | 1.0 | 主导信号 |
| 奖励函数 | 有效协调奖励 | $\beta_1$ | 0.5 | cross swap 且 $\Delta C > 0$ |
| 奖励函数 | 无效协调惩罚 | $\beta_2$ | −0.1 | cross swap 但 $\Delta C = 0$ |
| 奖励函数 | 协调失败惩罚 | $\gamma$ | 0.3（退火终值） | 单方 outside no-op |
| 奖励函数 | 意愿对齐系数 | $\lambda_{\text{align}}$ | 0.1 | 训练时辅助信号（PPO update 阶段计算） |
| 奖励函数 | 步数惩罚 | $\delta$ | 0.05 | 每步恒定惩罚 |
| 奖励函数 | 终局奖励 | $\eta$ | 10.0 | 双 Board 完成 |
| 动作空间 | Pointer 1 候选数 | — | 8 | 可移动槽位 |
| 动作空间 | Pointer 2 有效候选数 | — | 8 | 7 其他槽位 + 1 outside（排除源槽位） |
| 动作空间 | 执行协议 | — | 同步执行 | 两 Agent 每步同时输出动作 |
| Cooldown | Cooldown 步数 | $K_{\text{cool}}$ | 5 | 训练中 $C_t = 16$ 后继续运行的步数 |
| 测试终止 | Value Drop 阈值 | $\Delta_{\text{drop}}$ | 3.0 | $V$ 相对峰值的下降量 |
| 测试终止 | Value Drop 耐心 | $P$ | 3 | 连续满足下降条件的步数 |
| 测试终止 | Cycling 窗口 | $W$ | 6 | Board 配置哈希的滑动窗口大小 |
| 课程训练 | Phase 1 区间 | — | 0% – 30% | 纯 intra，outside mask |
| 课程训练 | Phase 2 区间 | — | 30% – 60% | 引导协调探索 |
| 课程训练 | Phase 3 区间 | — | 60% – 100% | 自由协调 + 端到端微调 |
| 课程训练 | 强制协调初始概率 | $p_{\text{force}}$ | 0.5 → 0 | Phase 2 内线性退火 |
| 课程训练 | 骨干网络微调学习率 | — | 基础 LR × 0.1 | Phase 3 启用 |

---

## 7. 训练流程

### 7.1 训练算法：MAPPO（Multi-Agent PPO）

采用 MAPPO 作为训练算法，与 CTDE 范式天然兼容：

- **Actor**（分散）：每个 Agent 的 Pointer Network 基于局部观测 + 通信消息选择动作
- **Critic**（集中）：Centralized Critic 基于全局状态（含 Board 摘要与通信向量）估计状态价值 $V(s^{\text{global}})$

### 7.2 Episode 构建流程

1. 随机选择一种跨类别组合（3 选 1）
2. 从对应类别中各随机抽取 1 张训练图像
3. 分别提取 Anchor 和 8 个 piece（共 2 个 Anchor + 16 个 piece）
4. 将 16 个 piece 全局随机打乱，前 8 个分配至 Board A，后 8 个分配至 Board B
5. 固定两个 Anchor 于各自 Board 中心
6. 双 Agent 同步执行动作（每步双方同时输出），直到满足终止条件
7. 收集完整 episode 的 trajectory（包含双方的 $p^i_{\text{out}}$ 概率值）用于 PPO 更新

### 7.3 MAPPO 更新流程

每个 training iteration：

**Step 1 — 数据收集**：运行 $N_{\text{env}}$ 个并行环境，每个环境执行 $T_{\text{horizon}}$ 步，收集 trajectory 数据。每步除标准的 $(o_t, a_t, r_t, o_{t+1})$ 外，额外存储双方的 Pointer 2 outside 概率 $p^A_{\text{out}}, p^B_{\text{out}}$ 以及通信向量 $e^A_{\text{comm}}, e^B_{\text{comm}}$。

**Step 2 — 内在奖励注入与 GAE 计算**：首先从 buffer 中读取 $p^A_{\text{out}}$ 和 $p^B_{\text{out}}$，计算意愿对齐奖励 $R^i_{\text{intent}}$ 并加到对应时间步的环境奖励中，得到完整奖励 $r^i_t$。然后使用 Centralized Critic 的 $V(s^{\text{global}}_t)$ 计算广义优势估计（Generalized Advantage Estimation）：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**Step 3 — Actor 更新（PPO-Clip）**：

$$
L^{\text{actor}} = -\mathbb{E}\left[\min\left(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中 $\rho_t = \frac{\pi_{\theta_{\text{new}}}(a_t \mid o_t, m_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t, m_t)}$。

由于参数共享，两个 Agent 的 trajectory 数据合并后统一更新同一套 Actor 参数。

**Step 4 — Critic 更新**：

$$
L^{\text{critic}} = \mathbb{E}\left[\left(V_\theta(s^{\text{global}}_t) - \hat{R}_t\right)^2\right]
$$

其中 $\hat{R}_t = \hat{A}_t + V_{\theta_{\text{old}}}(s^{\text{global}}_t)$ 为 GAE 目标回报。

**Step 5 — 熵正则化**：

$$
L^{\text{entropy}} = -\mathbb{E}\left[H(\pi(\cdot \mid o_t, m_t))\right]
$$

**总损失**：

$$
L = L^{\text{actor}} + c_1 \cdot L^{\text{critic}} - c_2 \cdot L^{\text{entropy}}
$$

### 7.4 训练参数

以下为参考起始值。第 6 节（特别是 6.10 节超参数汇总表）中的网络结构和奖励参数为权威定义，此处列出 MAPPO 训练层面的参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| $N_{\text{env}}$ | 32 | 并行环境数 |
| $T_{\text{horizon}}$ | 256 | 每次收集的步数 |
| PPO epochs | 4 | 每批数据的 PPO 更新轮数 |
| Mini-batch size | 128 | PPO mini-batch 大小 |
| $\epsilon$（PPO clip） | 0.2 | 策略比率裁剪范围 |
| $\gamma$（折扣因子） | 0.99 | 折扣因子 |
| $\lambda$（GAE） | 0.95 | GAE 参数 |
| Learning rate（Actor） | 3e-4 | Adam 优化器 |
| Learning rate（Critic） | 1e-3 | Adam 优化器 |
| $c_1$ | 0.5 | Critic 损失权重 |
| $c_2$ | 0.01 | 熵正则化权重 |
| $d$（嵌入维度） | 128 | 见 6.10 节 |
| $N_L$（Transformer 层数） | 3 | 见 6.10 节 |
| $N_H$（注意力头数） | 4 | 见 6.10 节 |
| $T_{\max}$（训练） | 20,000 | 单 episode 最大步数 |
| $T_{\max}$（评估） | 100 | 评估 episode 最大步数 |

### 7.5 训练阶段规划

**Phase 1 — SDN 预训练（可选）**：在 JPwLEG-3 数据上训练 H-SDN 和 V-SDN，学习邻居兼容性判断。如有原论文预训练权重则直接加载。

**Phase 2 — Piece Encoder 预训练（可选）**：用对比学习（contrastive learning）或分类任务预训练 Piece Encoder，使其能区分不同图像/类别的 piece。

**Phase 3 — 端到端 MAPPO 训练**：冻结或微调 SDN，端到端训练 Piece Encoder + Board State Encoder + Pointer Network Actor + Communication Module + Centralized Critic。按 6.8 节的三阶段课程训练策略（Phase 1 纯 intra → Phase 2 引导协调 → Phase 3 自由协调）进行。

---

## 8. 评估指标

### 8.1 单 Board 指标（沿用原论文）

| 指标 | 定义 |
|------|------|
| Perfect | 该 Board 全部 8 个 piece 完全正确（归属 + 位置）的 episode 比例 |
| Absolute | piece 位于正确 Board 的正确 slot 的比例 |
| Horizontal | 同一 Board 上水平相邻 piece 对相对位置正确的比例 |
| Vertical | 同一 Board 上垂直相邻 piece 对相对位置正确的比例 |

### 8.2 2-Mixed 新增指标

| 指标 | 定义 |
|------|------|
| Ownership Accuracy | 16 个 piece 中位于正确 Board 的比例（不要求 slot 正确） |
| Both-Perfect | 两个 Board 同时完全正确的 episode 比例 |
| Overall Absolute | 16 个 piece 中位于正确 Board 且正确 slot 的比例 |
| Avg Steps | 成功 episode（Both-Perfect）的平均步数 |

### 8.3 终止质量评估指标（仅测试模式）

由于测试模式下 Agent 通过 Value Drop 或 Board Cycling 自主终止，
需评估终止时机的质量：

| 指标 | 定义 |
|------|------|
| Correct Termination Rate | 非超时终止且 $C_t = 16$ 的 episode 比例 |
| Premature Termination Rate | 非超时终止但 $C_t < 16$ 的 episode 比例 |
| Timeout Rate | 达到 $T_{\max}$ 仍未触发终止的 episode 比例 |
| Termination Precision | 非超时终止中 $C_t = 16$ 的比例 |
| $C_t$ at Termination | 终止时刻 $C_t$ 的均值和分布 |
| Value Drop Trigger Rate | 由 Value Drop 触发终止的比例（vs. Board Cycling 触发） |
| Steps After Completion | 在 $C_t$ 首次达到 16 之后、终止之前的步数（仅限 Correct Termination） |

### 8.4 分组报告

按三种跨类别组合（P+E、P+A、E+A）分别报告以上所有指标，并报告加权平均值。

---

## 9. 建议文件结构

project/
├── data/
│ ├── dataset.py # 数据加载、Piece 提取、跨类别配对
│ └── split.py # Train/Val/Test 划分（stratified by category）
├── envs/
│ ├── dual_board_env.py # 2-Mixed 双 Board 环境（状态管理、同步动作执行、Mutual Agreement）
│ └── reward.py # 环境奖励函数实现（progress, coord_success/fail, step, terminal）
├── models/
│ ├── piece_encoder.py # 共享 Piece Encoder（EfficientNet-B0 + 投影层）
│ ├── board_encoder.py # Board State Encoder（Transformer + Positional Encoding）
│ ├── pointer_network.py # Pointer Network Actor（Ptr1 + Ptr2 含查询融合层）
│ ├── communication.py # 通信模块（selected piece 表征广播）
│ ├── critic.py # Centralized Critic（MLP，输入含 Board 摘要 + 通信向量）
│ └── sdn.py # SDN 辅助模块（H-SDN, V-SDN，可选）
├── agents/
│ └── mappo_agent.py # MAPPO Agent（PPO 更新逻辑、GAE 计算、内在奖励注入）
├── training/
│ ├── train.py # MAPPO 训练主循环（并行环境 + 数据收集 + 更新）
│ ├── curriculum.py # 课程学习调度器（Phase 1/2/3 切换、mask 控制、退火）
│ └── evaluate.py # 评估脚本
├── utils/
│ ├── visualize.py # 可视化工具（Board 状态、训练曲线、注意力热图）
│ └── metrics.py # 指标计算（Ownership、Perfect、Absolute 等）
├── configs/
│ └── default.yaml # 超参数配置
├── requirements.txt
└── README.md

---

## 10. 开放设计选项

| # | 问题 | 可选方案 | 初步建议 |
|---|------|---------|---------|
| 1 | Piece Encoder 骨干 | A. EfficientNet-B0 / B. ResNet-18 / C. 轻量 CNN | 先 A（6.4.1 节已采用） |
| 2 | SDN 整合方式 | A. 作为 Transformer attention bias / B. 作为辅助损失 / C. 不使用 | 先实现 A |
| 3 | 通信机制复杂度 | A. 单向广播 selected piece 表征 / B. 多轮通信 / C. 注意力通信（TarMAC 风格） | 先实现 A（6.4.5 节已采用） |
| 4 | ~~动作轮替 vs 同步~~ | ~~A. 严格轮替 / B. 同步~~ | **已确定为 B（同步执行）**，见 5.5 节与 6.3 节 |
| 5 | 参数共享程度 | A. 全部共享 / B. Encoder 共享 + Actor 独立 | 先实现 A |
| 6 | 课程学习 | A. 按三阶段（6.8 节） / B. 直接端到端 | 推荐 A，降低学习难度 |
| 7 | 奖励权重调参 | 需实验调参 | 起始值见 6.10 节 |
| 8 | ~~Pointer 2 训练用全局还是消息~~ | ~~A. 训练也用消息 / B. 训练用全局~~ | **已确定为 A**（通信一致性，见 6.4.6 节） |