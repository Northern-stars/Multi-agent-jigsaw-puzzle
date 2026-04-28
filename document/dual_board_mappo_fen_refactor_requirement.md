# Dual-Board MAPPO FEN 重构需求文档

> 版本: v0.1 | 更新日期: 2026-04-28
> 目标文件:
> - `model_code/dual_board_mappo_model.py`
> - `agent/mappo_dual_agent.py`
> - `env/dual_board_env.py`

---

## 1. 背景

当前 `dual_board_mappo_model.py` 中，单个 board 的处理逻辑分散在以下几个步骤中：

- `agent` 从环境观测中取出 `slot_images` 与 `anchor_image`
- `model` 内部先分别编码 8 个 movable slot 和 1 个 anchor
- 再把两者拼成 9-token 序列
- 最后送入 Transformer 得到 `slot_context` 与 `board_summary`

这套实现是可用的，但存在两个问题：

1. 单个 board 的视觉编码逻辑没有被封装成独立模块，复用性差
2. 输入形式与仓库中 `model_code/fen_model.py` 的使用习惯不一致，后续难以统一

在第一轮整图化之后，`fen_model` 虽然已经接管了 board 级输入，但 patch 级视觉编码仍然依赖独立 `PieceEncoder`，说明单 board 编码链路还没有完全内聚。

因此，需要将 `dual_board_mappo_model.py` 中“负责当前 board 处理”的逻辑打包为一个独立的 `fen_model` 风格模块：

- 输入: 一整张 `288x288` 的 board 图像
- 输出: 一段 board 级特征张量

---

## 2. 重构目标

### 2.1 核心目标

将当前双 board MAPPO 模型中的单 board 编码流程改为：

- `agent` 只向 model 传入 `board_image`
- `dual_board_mappo_model` 中新增一个 `fen_model`
- `fen_model` 自行完成：
  - 切 patch
  - patch 编码
  - board 内上下文建模
  - 输出 board 特征张量

### 2.4 fen_model 内聚目标

进一步将 patch encoder 一并收进 `fen_model` 内部，使单 board 编码完整闭环都由 `fen_model` 自己负责：

- `fen_model` 直接持有 patch encoder backbone
- patch 归一化在 `fen_model` 内部完成
- 不再保留独立 `PieceEncoder` 作为外部可见结构

### 2.3 环境接口简化目标

在 `fen_model` 重构完成后，dual-board MAPPO 路径已经只依赖整图观测，因此环境侧也需要同步收敛接口：

- `DualBoardEnv._build_observation()` 仅返回 `board_image`
- 不再向 MAPPO 路径暴露：
  - `slot_images`
  - `anchor_image`
  - `piece_ids`

这样可以减少环境与 agent/model 之间的冗余耦合，避免继续维护已经不被消费的观测字段。

### 2.2 对齐目标

新的 `fen_model` 在接口形态上需要参考 `model_code/fen_model.py`：

- 输入为整图 `image`
- 输出为特征表示
- 内部自行负责 patch 切分与特征聚合

这里的“参考格式”主要指：

- 调用方式参考 `fen_model(image)`
- 不再要求外部预先传入 patch 级输入

---

## 3. 现状分析

### 3.1 现有 board 编码路径

当前 `DualBoardMAPPOModel.encode_boards()` 的流程是：

1. 输入 `slot_images: [B, 2, 8, 3, 96, 96]`
2. 输入 `anchor_images: [B, 2, 3, 96, 96]`
3. 用 `PieceEncoder` 分别编码 slot 和 anchor
4. 用 `BoardStateEncoder` 拼回 9 个 token
5. 输出：
   - `slot_context`
   - `board_summary`

### 3.2 当前问题

- board 编码依赖两路输入，接口偏碎
- board 级特征抽取逻辑没有模块化
- `agent` 和 `model` 对观测结构耦合较深
- `env` 仍保留旧的碎片级观测字段，与当前 MAPPO 实际消费接口不一致
- patch encoder 仍是独立类，导致 `fen_model` 不是完整的单 board 编码封装
- 不利于后续复用到其他 board 级策略模型

---

## 4. 目标结构

### 4.1 新的 board fen_model

在 `dual_board_mappo_model.py` 中引入一个新的 `fen_model`，其职责是：

- 输入 `board_images: [B, 3, 288, 288]`
- 将图像切分为 `3x3` 共 9 个 `96x96` patch
- 对每个 patch 进行视觉编码
- 在 patch token 上做上下文建模
- 输出：
  - `token_context: [B, 9, D]`
  - `slot_context: [B, 8, D]`
  - `board_summary: [B, D]`

### 4.2 DualBoardMAPPOModel 的新输入

`DualBoardMAPPOModel` 应改为接收：

- `board_images: [B, 2, 3, 288, 288]`

而不是：

- `slot_images`
- `anchor_images`

### 4.3 agent 的新调用方式

`DualBoardMAPPOAgent` 应改为：

- 从 observation 中读取 `board_image`
- 将两个 board 的整图堆叠为 `board_images`
- 调用 `model.evaluate_policy(board_images, ...)`

### 4.4 env 的新观测契约

`DualBoardEnv.get_observations()` 返回的单个 board 观测应改为：

```python
{
    "board_image": Tensor[3, 288, 288]
}
```

旧契约：

```python
{
    "slot_images": Tensor[8, 3, 96, 96],
    "anchor_image": Tensor[3, 96, 96],
    "board_image": Tensor[3, 288, 288],
    "piece_ids": Tensor[8],
}
```

新契约的目标不是改变环境语义，而是删除已经不再被当前 MAPPO 训练链路使用的冗余字段。

---

## 5. 具体修改流程

### 5.1 第一步：抽出 fen_model

在 `model_code/dual_board_mappo_model.py` 中：

- 删除 `BoardStateEncoder` 作为独立 board 主编码入口的地位
- 新增 `fen_model`
- 将 patch encoder 合并进 `fen_model`

`fen_model` 的内部流程：

1. 输入 `board_images`
2. 用 `unfold` 将图像切成 9 个 patch
3. 将 patch reshape 为 `[B*9, 3, 96, 96]`
4. 在 `fen_model` 内部完成 patch 归一化与编码
5. reshape 回 `[B, 9, D]`
6. 加位置编码并送入 Transformer
7. 输出：
   - 9-token 的 `token_context`
   - 去中心后的 `slot_context`
   - 全局 `board_summary`

### 5.2 第二步：改写 encode_boards

将 `DualBoardMAPPOModel.encode_boards()` 改为：

- 输入 `board_images: [B, 2, 3, 288, 288]`
- reshape 成 `[B*2, 3, 288, 288]`
- 调用新的 `fen_model`
- 再 reshape 回双 board 结构

### 5.3 第三步：保留 pointer actor 逻辑

`PointerActor` 的逻辑保持不变：

- `ptr1` 仍然基于 `slot_context`
- `ptr2` 仍然基于：
  - 本方 `selected_source`
  - 本方 `board_summary`
  - 对方 `selected_source` 作为消息

换句话说，通信机制不改，只改 board 特征提取方式。

### 5.4 第四步：修改 agent 的输入堆叠

在 `agent/mappo_dual_agent.py` 中：

- `_stack_obs()` 不再返回 `slot_images, anchor_images`
- 改为返回 `board_images`

旧接口：

```python
slot_images, anchor_images = self._stack_obs(observations)
output = self.model.evaluate_policy(slot_images, anchor_images)
```

新接口：

```python
board_images = self._stack_obs(observations)
output = self.model.evaluate_policy(board_images)
```

### 5.5 第五步：修改 buffer 内容

由于 agent 不再需要保存 `slot_images` 和 `anchor_images`，buffer 中应改为保存：

- `board_images`
- `actions`
- `log_probs`
- `values`
- `rewards`
- `dones`
- `outside_probs`

这样 PPO 更新阶段也直接基于整图 board 输入进行前向。

### 5.6 第六步：修改 PPO 更新中的 model 调用

在 update 阶段：

- 从 buffer 取出 `board_images`
- 直接喂给 `model.evaluate_policy(board_images, ptr1_actions=...)`
- 不再拼 `batch_slot` 和 `batch_anchor`

### 5.7 第七步：同步简化环境 observation

在 `env/dual_board_env.py` 中：

1. 保留 `reset()`、`step()`、`get_observations()` 的对外调用方式不变
2. 保留环境内部 piece 级状态与 reward/metrics 逻辑不变
3. 仅收敛 `_build_observation()` 的返回值
4. 将 dual-board MAPPO 路径的观测接口统一为 `board_image`

这样可以把“整图输入”这件事落实到环境、agent、model 三层，而不是只在 agent 里做一次兼容转换。

### 5.8 第八步：将 piece encoder 内聚到 fen_model

在 `model_code/dual_board_mappo_model.py` 中：

1. 删除独立 `PieceEncoder` 类
2. 在 `fen_model.__init__()` 中直接构建 patch encoder backbone
3. 在 `fen_model` 内部增加私有 patch 编码函数
4. 将 `uint8 -> float -> /255` 的预处理逻辑也收敛到 `fen_model`

这样修改后，`fen_model` 才真正成为从整图输入到 board 特征输出的完整模块。

---

## 6. 预期收益

完成该重构后，有以下收益：

- 单 board 编码逻辑更加模块化
- 单 board 编码链路完整收敛到 `fen_model`
- `fen_model` 形态与仓库内其他图像编码模型更统一
- `agent` 与环境观测结构耦合下降
- `env` 观测定义与实际训练消费接口保持一致
- 后续排查观测问题时，不需要再区分旧的 slot/anchor 冗余字段
- 后续更容易替换 board 编码器，例如：
  - 更深的 patch encoder
  - ViT 风格主干
  - cross-attention board encoder

---

## 7. 风险与注意事项

### 7.1 输入数据类型

环境中的 `board_image` 当前是 `uint8`，进入 `fen_model` 后必须显式转 `float` 并归一化，否则容易数值不稳定。

### 7.2 中心块语义

虽然现在输入改成整图，但中心块仍然是固定 anchor，它的语义不应丢失。新的 `fen_model` 需要保留中心 patch 在 token 序列中的位置。

### 7.3 slot_context 的定义

pointer actor 仍然只对 8 个 movable slot 做决策，因此：

- `slot_context` 必须继续保持 `[B, 8, D]`
- 中心 token 不应进入动作候选集

### 7.4 兼容 PPO 更新

当前 PPO 更新逻辑依赖：

- `ptr1_actions`
- `ptr2_actions`
- `log_prob`
- `entropy`
- `value`

因此重构时需要保证：

- `evaluate_policy(board_images, ptr1_actions=...)` 仍然能返回 `ptr2_logits`
- `agent` 仍然能在外部完成两阶段采样与 log-prob 计算

---

## 8. 验收标准

完成后应满足以下条件：

1. `DualBoardMAPPOModel` 的单 board 编码逻辑已封装成 `fen_model`
2. `fen_model` 输入为整张 `288x288` board 图像
3. patch encoder 也已内聚到 `fen_model` 内部
4. `agent` 调用 model 时不再传 `slot_images + anchor_images`
5. PPO 训练流程仍可正常跑通
6. Pointer 通信机制保持不变
7. 代码可以通过基本语法检查

---

## 9. 本次实施结果

本次重构按上述流程完成了以下修改：

- 在 `dual_board_mappo_model.py` 中新增 `fen_model`
- 将 patch encoder 合并进 `fen_model` 内部
- 将 board 主编码入口切到整图 `board_image`
- 将 `agent/mappo_dual_agent.py` 的输入堆叠从 `slot_images + anchor_images` 改为 `board_images`
- 将 PPO buffer 的观测缓存改为 `board_images`
- 将 `env/dual_board_env.py` 的 observation 接口收敛为仅返回 `board_image`
- 保持 `ptr1 -> 通信 -> ptr2` 的双阶段决策逻辑不变

因此，当前代码已经满足“将 dual_board_mappo_model 中负责当前 board 处理的逻辑打包成一个 fen_model，并参考 `model_code/fen_model.py` 的整图输入格式”的目标，同时完成了 patch encoder 内聚和 dual-board 环境观测接口的同步简化。
