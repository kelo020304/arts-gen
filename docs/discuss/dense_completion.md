# Discuss: Dense Completion

> 最后更新: 2026-05-06
> 范围: Part Flow 推理从 sparse SS labeling 对齐到 dense 64³ surface-to-solid completion 的设计讨论。

## Round 26 - Gradient checkpointing 配置收敛

日期: 2026-05-06

### 讨论背景
- 用户同意加入 gradient checkpointing config，默认关闭。

### 核心问题
- gradient checkpointing 是否暴露到配置。
- 默认行为是否保持第一版主路径简单。

### 观点与推理
- gradient checkpointing 是内存工程选项，不是 Part Flow 设计核心。
- 将其暴露到 config 可以在 H20 或 48G 4090 显存不足时打开，不需要改代码。
- 默认关闭可以避免训练变慢，也避免第一版 benchmark 混入重计算开销。

### 暂定结论 / 决策
- 决策：加入 `model.use_gradient_checkpointing: false`。
- 决策：默认关闭；只有显存不足或做显存/速度 trade-off 实验时开启。
- 决策：文档注释写明它会降低显存但增加训练时间。

### 未决问题
- 暂无。

### 后续动作
- 可以整理完整第一版设计草案，并进入实现计划。

## Round 25 - chunk config 归属、训练 forward 与 checkpointing

日期: 2026-05-06

### 讨论背景
- 用户决定 `voxel_chunk_size` 统一放在 `model.voxel_chunk_size`。
- 用户决定训练一次性 forward，不做训练 chunking。
- 用户询问 gradient checkpointing 是什么。

### 核心问题
- `voxel_chunk_size` 配置归属。
- 训练是否 chunked forward。
- gradient checkpointing 是否进入第一版设计。

### 观点与推理
- `model.voxel_chunk_size` 统一管理更清晰：
  - dense voxel decoder 是模型结构的一部分；
  - 推理和训练若需要 chunking，也都应使用同一个模型级配置；
  - 注释应写清楚该值影响显存/速度，不改变 dense 64³ 语义。
- 用户决定训练一次性 forward：
  - 即训练阶段把 full dense `64^3` coords 一次送进模型 forward；
  - 这更简单，避免 chunked training 的 loss 聚合、随机性、batch norm/norm 行为差异等复杂问题；
  - 在 H20 集群训练的前提下，第一版可以先按一次性训练 forward 设计。
- gradient checkpointing 是一种省显存技术：
  - 正常训练会保存每层 forward 的中间激活，backward 时直接用；
  - checkpointing 不保存部分中间激活，backward 时重新计算这些 forward；
  - 好处是显存下降；
  - 代价是训练变慢，因为 backward 时要重算。
- 对本项目第一版的建议：
  - 不把 gradient checkpointing 作为默认必需路径；
  - 可以加入 config，例如 `model.use_gradient_checkpointing: false`，默认关闭；
  - 如果 full dense 64³ + 4 层 decoder 在 H20 上显存不够，再打开；
  - 不应让 checkpointing 掩盖模型结构是否过重的问题，它只是内存工程选项。

### 暂定结论 / 决策
- 决策：chunk size 统一配置为 `model.voxel_chunk_size`。
- 决策：训练第一版一次性 forward full dense `64^3`，不做训练 chunking。
- 暂定建议：gradient checkpointing 作为 config 可选项，默认关闭。

### 未决问题
- 暂无。后续 Round 26 已决定加入 `model.use_gradient_checkpointing: false`，默认关闭。

### 后续动作
- 可以整理完整第一版设计草案，并进入实现计划。

## Round 24 - Decoder layer 顺序、condition pooling 与 chunk size config

日期: 2026-05-06

### 讨论背景
- 用户要求对 Round 23 的三个未决问题给出建议：
  - dense decoder 每层顺序。
  - condition pooling 具体实现。
  - chunk size 如何处理。
- 用户明确 chunk size 写入 config。

### 核心问题
- `DenseVoxelDecoderLayer` 内部应如何组织 part cross-attention、condition cross-attention 和 FFN。
- `compressed_cond_tokens` 的 attention pooling 应采用什么最小实现。
- dense 64³ 推理/训练 chunk size 应如何配置。

### 观点与推理
- dense decoder 每层建议固定为：
  - `AdaLN/time modulation`
  - `part cross-attention`
  - `condition cross-attention`
  - `FFN`
  - 每个子层使用 residual + norm/dropout。
- 先 attend part tokens 的原因：
  - part tokens 是主条件，定义当前 voxel 要在 `empty/part_id` slots 中如何解释；
  - 先把 voxel query 对齐到 part relation 空间，再读取 compressed visual context 更自然。
- 后 attend condition tokens 的原因：
  - compressed condition tokens 是辅助视觉上下文，帮助校正局部/全局形状；
  - 如果先看 condition 再看 part，模型可能更像视觉分类器，part slot 绑定变弱。
- condition pooling 建议使用 learnable queries per view：
  - 每个 view 有 3 个 learnable summary queries；
  - 这 3 个 queries cross-attend 该 view 的 patch tokens，得到 3 个 view summary tokens；
  - 再用 1 个 global learnable query cross-attend 所有 view summary tokens 或所有 projected condition tokens，得到 global token；
  - 加 view id embedding、summary token index embedding、global token type embedding。
- 不建议第一版用 top-M token selection：
  - top-M 需要定义打分方式，引入额外不稳定性；
  - attention pooling 是端到端可学习的压缩，更直接。
- chunk size 写入 config：
  - `inference.voxel_chunk_size` 控制推理时每次处理多少 dense voxels；
  - `train.voxel_chunk_size` 或 `model.voxel_chunk_size` 控制训练/forward chunking；
  - 第一版建议默认 `32768`，在 48G 4090 上可尝试 `65536`，H20 上可尝试更大；
  - config 注释写清楚：chunk size 只影响显存/速度，不改变 dense 64³ 枚举语义。

### 暂定结论 / 决策
- 暂定建议：每层顺序固定为 `part cross-attn -> condition cross-attn -> FFN`，带 time/AdaLN modulation 和 residual。
- 暂定建议：condition pooling 使用 learnable queries per view，每 view 3 个 summary tokens，再加 1 个 global token。
- 决策：chunk size 写入 config。
- 暂定建议：默认 `voxel_chunk_size: 32768`，后续在 48G 4090/H20 上实测调整。

### 未决问题
- chunk size config 放在 `inference.voxel_chunk_size` 和 `train.voxel_chunk_size` 两处，还是统一放在 `model.voxel_chunk_size`。
- dense training 是否一次性 forward 全 64³，还是训练也必须 chunked forward。
- 是否需要 gradient checkpointing 作为 config。

### 后续动作
- 下一轮讨论 chunking 在训练/推理中的边界，以及是否需要 gradient checkpointing。

## Round 23 - condition KV 数量、decoder 层数与 endpoint head 收敛

日期: 2026-05-06

### 讨论背景
- 用户同意 `compressed_cond_tokens` 作为 dense decoder 的 condition K/V。
- 用户对 Round 22 的三个未决问题给出决定：
  - 每个 view 输出 3 个 compressed condition tokens，再加 1 个 global token。
  - 做好 positional embedding。
  - dense decoder 使用 4 层。
  - 不保留额外 `Linear` fallback endpoint head。

### 核心问题
- condition compression 的 token 数量如何固定。
- dense voxel decoder 的深度如何固定。
- endpoint head 是否只保留 dot-product slot head。

### 观点与推理
- 每 view 3 个 tokens 比每 view 1 个 token 更稳：
  - 1 个 token 容易过度压缩单视角内部的局部形状/part 边界线索；
  - 3 个 tokens 仍然远小于原始 `V*T` patch tokens，计算可控；
  - 多个 tokens 允许 attention pooling 学到不同区域/语义摘要。
- 加 1 个 global token 是必要的：
  - per-view tokens 保留视角局部摘要；
  - global token 聚合跨视角整体形状上下文；
  - dense voxel query 可同时读取局部 view summary 和整体 object summary。
- positional embedding 应包含：
  - compressed token 的 view id embedding；
  - token-in-view index embedding，例如每 view 的 3 个 summary token；
  - global token type embedding，避免 global token 和 view tokens 混淆。
- dense decoder 使用 4 层是合理的第一版：
  - 2 层可能不足以反复融合 `x_t`、part relation、condition summary；
  - 4 层仍然比 dense voxel global attention 轻很多；
  - 后续可以 ablation `2/4/6`，但第一版默认 4。
- endpoint head 不保留额外 `Linear` fallback：
  - 输出只用 `voxel_query · part_tokens_refined`；
  - 这样类别权重和 sample-local part slots 绑定，变量 K 语义更清楚；
  - 不引入并行 fallback，避免模型绕开 part token 条件或产生冗余路径。

### 暂定结论 / 决策
- 决策：`compressed_cond_tokens` 使用每 view 3 个 tokens + 1 个 global token。
- 决策：condition K/V 加 positional/type embedding，包括 view id、summary token index 和 global type。
- 决策：dense voxel decoder 使用 4 层。
- 决策：endpoint head 不保留额外 `Linear` fallback，只使用 `voxel_query · part_tokens_refined`。
- 决策：chunk 内仍不做 voxel self-attention。

### 未决问题
- dense decoder 每层的顺序是否固定为 `part cross-attn -> condition cross-attn -> FFN`。
- condition compression 的 attention pooling 具体实现：learnable queries per view，还是轻量 attention pooling module。
- 推理 chunk size 的初始默认值仍需实测。

### 后续动作
- 下一轮讨论 dense decoder layer 顺序、condition pooling 实现和 chunk size 默认建议。

## Round 22 - Dense decoder 职责与 condition attention 建议

日期: 2026-05-06

### 讨论背景
- 用户追问 Round 21 的两个未决问题：
  - condition compression 用 attention pooling 还是 mean pooling。
  - dense decoder 到底是干什么的。

### 核心问题
- `DenseVoxelDecoder` 在 conditional Fisher flow 中的职责。
- `compressed_cond_tokens` 应通过 attention 还是 FiLM/concat 输入 dense decoder。
- 第一版如何保持结构清晰，同时不牺牲 dense completion 的条件表达。

### 观点与推理
- `DenseVoxelDecoder` 不是新的 flow，也不是 SLAT decoder。它是 conditional flow denoiser `f_theta` 的 voxel 端模块。
- 它每个 flow step 都会被调用，输入当前 dense voxel state 和条件，输出 endpoint logits：
  - 输入：`x_t[n]`、`coords[n]`、`is_on_surface[n]`、`t`、`part_tokens_refined[b]`、`compressed_cond_tokens[b]`；
  - 输出：`endpoint_logits[n, 0:K_b]`；
  - bridge 再把 `endpoint_logits -> endpoint_probs`，执行 Fisher step。
- dense decoder 的存在是为了避免把 dense voxel 直接塞进全局 transformer，同时仍然让每个 voxel 独立/分块地读取全局 part relation 和视觉条件。
- 第一版 dense decoder 应该是 chunked 的：
  - 对 full `64^3` coords 枚举；
  - 每次处理 `chunk_size` 个 voxels；
  - chunk 内不做 voxel-to-voxel self-attention；
  - 每个 voxel query 可以 cross-attend 到少量 condition tokens。
- 对 condition compression 的建议：
  - 第一版使用 attention pooling，而不是简单 mean pooling。
  - mean pooling 太弱，会把多视角和局部可见性平均掉；对于 surface-to-solid completion，局部 shape/part 边界线索仍然重要。
  - attention pooling 可以学会从 patch/view tokens 中提取少量 summary tokens，仍然比让 dense voxels attend 全量 `V*T` tokens 便宜。
- 对 dense decoder 使用 condition 的建议：
  - 主路径：voxel query cross-attend 到 `part_tokens_refined`，这是最重要的条件。
  - 辅助路径：`compressed_cond_tokens` 通过第二路 cross-attention 输入，而不是只用 FiLM。
  - FiLM 可作为后续轻量 ablation；第一版用 cross-attention 更直观，因为 voxel query 可以按位置和当前 `x_t` 选择性读取视觉 summary。
- 推荐最小结构：
  - `voxel_query = MLP([x_t_emb, coord_emb, surface_emb, time_emb])`
  - `voxel_query = CrossAttn(voxel_query, part_tokens_refined)`
  - `voxel_query = CrossAttn(voxel_query, compressed_cond_tokens)`
  - `voxel_query = FFN(voxel_query)`
  - `endpoint_logits = dot(voxel_query, part_tokens_refined)` 或 `Linear(voxel_query)`
- 输出 head 第一版建议仍使用 `voxel_query · part_tokens_refined`：
  - 这样 part slot 的分类权重与 part token 绑定，能更清楚表达“这个 voxel 属于哪个输入 part slot”；
  - padding slots 可以自然 mask；
  - 比完全独立 `Linear(H,K_max)` 更符合 variable-K sample-local slot 设计。

### 暂定结论 / 决策
- 决策：`DenseVoxelDecoder` 是 denoiser 的 voxel 端，负责输出 dense voxel endpoint logits。
- 暂定建议：condition compression 使用 attention pooling，不用简单 mean pooling 作为主线。
- 暂定建议：dense decoder 使用两路 cross-attention：先 attend `part_tokens_refined`，再 attend `compressed_cond_tokens`。
- 暂定建议：chunk 内不做 voxel self-attention。
- 暂定建议：endpoint head 使用 `voxel_query · part_tokens_refined`，保持 variable-K part slot 语义。

### 未决问题
- attention pooling 输出多少个 `compressed_cond_tokens`：例如每 view 1 个 token + 1 个 global token，还是每 view 2-4 个 tokens。
- dense decoder 堆叠几层 cross-attention：第一版 1 层还是 2 层。
- endpoint head 是否保留一个额外 `Linear` fallback 做 ablation；第一版倾向不保留，避免冗余。

### 后续动作
- 下一轮讨论 `compressed_cond_tokens` 数量、dense decoder 层数、endpoint head 是否只用 dot-product。

## Round 21 - Flow condition 的定义与本轮收敛

日期: 2026-05-06

### 讨论背景
- 用户指出按 RF 理解，`condition tokens` 和 `part tokens` 都应是 flow 的条件。
- 用户同意 slot id 加入 config。
- 用户同意 condition tokens 压缩后进入 dense voxel decoder。
- 用户表示不需要继续讨论 part coverage 统计/缺失处理。

### 核心问题
- Fisher 主线下，`condition tokens` 和 `part tokens` 是否属于 flow condition。
- 如何解释“condition 不直接给 bridge”与“condition 是 flow 条件”之间的关系。
- slot id embedding 应如何配置。

### 观点与推理
- 用户理解是正确的：`condition tokens` 和 `part tokens` 都是 conditional flow 的条件。数学上可以写为：
  - `p_theta(x_1 | x_t, t, C)` 或 `v_theta(x_t,t,C)`；
  - 其中 `C = {SS surface condition, image/condition tokens, part tokens, num_parts/masks}`。
- Round 20 中“condition/part tokens 不直接进入 FisherBridge”的意思只是工程分层：
  - `PartFlowPredictor / denoiser` 读取条件 `C`，输出 `endpoint_logits` / `endpoint_probs`；
  - `FisherBridge` 读取 `x_t` 和 `endpoint_probs`，执行 Fisher-Rao 几何 step；
  - 因为 `endpoint_probs` 已经是 `p_theta(x_1 | x_t,t,C)`，所以条件已经通过 denoiser 进入 flow。
- 换句话说，条件确实属于 flow，只是不属于 bridge 的低层几何 API。Bridge 是条件无关的几何积分器；条件依赖在 `f_theta` 中。
- slot id embedding 加入 config 的建议：
  - `model.use_slot_id_embedding: false` 作为默认，保持第一版主要依赖 mask-pooled visual part features；
  - `model.slot_id_embedding_scale: 0.1` 作为可选弱 residual scale；
  - `empty_token` 始终启用，不受 `use_slot_id_embedding` 控制，因为 slot 0 的 empty 语义是固定的；
  - 如果打开 slot id，只加到 valid part slots `1..K`，padding slots 保持 0；
  - 文档注释必须写清楚：slot id 是 sample-local slot identity hint，不是跨类别 semantic class。
- 对本轮问题 1 的建议：condition compression 使用 per-view attention pooling + global token 的轻量结构。
  - 如果 `cond` 来自多视角 `B,V*T,D`，先按 view 分组，把每个 view 的 patch tokens 压成 1 到少量 tokens；
  - 再加一个 all-view global token；
  - 最终得到 `compressed_cond_tokens [B, M, H]`，其中 `M` 远小于 `V*T`；
  - dense voxel chunk decoder 使用 `part_tokens_refined` 作为主 K/V，使用 `compressed_cond_tokens` 做辅助 cross-attention 或 FiLM。
- 不继续处理 coverage 统计/缺失 part coverage 问题；后续实现保持已有契约，不新增该方向的设计分支。

### 暂定结论 / 决策
- 决策：`condition tokens` 和 `part tokens` 都是 Fisher conditional flow 的条件。
- 决策：工程 API 中条件进入 denoiser，bridge 只接收条件化后的 `endpoint_probs`。
- 决策：slot id embedding 加入 config，默认关闭或弱 residual；`empty_token` 始终保留。
- 决策：condition tokens 第一版采用压缩后输入 dense voxel decoder，不让 dense voxel chunk 直接 cross-attend 全量 `V*T` tokens。
- 决策：不继续展开 part coverage 统计/缺失处理问题。

### 未决问题
- condition compression 的具体模块用 attention pooling 还是简单 mean pooling；当前建议 attention pooling，但实现复杂度略高。
- dense voxel decoder 中 `compressed_cond_tokens` 更适合作为 FiLM 条件，还是作为第二路 cross-attention K/V。

### 后续动作
- 下一轮讨论 condition compression 和 dense voxel decoder 的最小结构，确认后即可进入实现计划。

## Round 20 - condition/part tokens 如何进入 Fisher flow

日期: 2026-05-06

### 讨论背景
- 用户同意 part-token transformer 的设计原则：非 causal，只在 `K+1` 个 part tokens 上建模 part-part relation，不做 dense voxel global self-attention。
- 用户追问 `condition tokens` 和 `part tokens` 分别如何给 flow。
- 用户要求解释 Round 19 的未决问题 1，并对未决问题 2 给出建议。

### 核心问题
- Fisher flow 中 bridge 和 denoiser 的职责边界。
- `condition tokens` 与 `part_tokens_refined` 应如何进入 denoiser。
- 是否加入 slot id embedding。
- condition tokens 是否全量参与 voxel decoder cross-attention，还是先压缩。

### 观点与推理
- Fisher bridge 的几何层不直接消费 `condition tokens` 或 `part tokens`。它只负责：
  - 从 source 采样 `x_t`；
  - 根据模型输出的 `endpoint_probs` 做 Fisher-Rao / sphere geodesic step；
  - 维持 valid simplex 和 padding mask 契约。
- `condition tokens` 和 `part tokens` 是给 denoiser `f_theta` 的条件输入。整体关系应写成：
  - `cond_tokens = rgb_proj(cond)`，保留图像/多视角 patch 级视觉信息；
  - `part_tokens = pool(cond_tokens, mask_token_labels)`，得到 sample-local `K+1` part slots；
  - `part_tokens_refined = PartTokenTransformer(part_tokens)`，建模 part-part relations；
  - `voxel_query = embed(x_t, coords, is_on_surface, t)`；
  - `endpoint_logits = DenseVoxelDecoder(voxel_query, part_tokens_refined, cond_tokens)`；
  - `endpoint_probs = masked_softmax(endpoint_logits)`；
  - `x_{t+dt} = FisherBridge.step(x_t, endpoint_probs, ...)`。
- 换句话说，flow state 是 dense voxel categorical state `x_t`；condition/part tokens 不属于 flow state，而是每一步 denoiser 预测 endpoint posterior 的条件。
- 未决问题 1：slot id embedding 的含义。
  - slot id embedding 是给 slot `j` 一个 learned embedding，例如 `slot_emb[j]`，帮助模型区分“这是第 j 个局部 part slot”。
  - 风险是 slot id 容易被误解成跨类别语义 class，例如错误地让模型以为 `slot 1` 总是某种固定语义部件。
  - 在本项目里，part slots 应是 sample-local instance slots，真正语义来自 `mask_token_labels` 对应的视觉区域；因此 slot id 只能是弱位置/身份提示，不能作为主语义来源。
- 对 slot id embedding 的建议：
  - 第一版不加入强 slot id embedding。
  - 保留明确的 `empty_token`，因为 slot 0 的 empty 语义是全局固定的。
  - 对 part slots `1..K`，优先依赖 mask-pooled visual token；若要加入 slot id，也应作为 config ablation，默认关闭或使用很小 residual scale。
  - 如果 valid part slot 没有 2D coverage，不建议静默用 slot id 当作语义兜底；应暴露 `missing_part_coverage` 统计或 fail/warn，由数据契约决定。
- 未决问题 2：condition tokens 是否全量进入 voxel decoder。
  - 全量 cross-attention 到所有 `V*T` condition tokens 信息最完整，但 dense `64^3` chunk 每步都 cross-attend 全量视觉 tokens，时间和显存压力较大。
  - 只用全局 pooled condition 最省，但会损失局部视觉细节，可能影响 surface-to-solid completion 对局部形状和 part 边界的判断。
  - 更稳的第一版是“两路条件”：
    - 主条件：`part_tokens_refined`，每个 voxel chunk cross-attend 到 `K+1` tokens，便宜且直接提供 part relation；
    - 辅助条件：从 `cond_tokens` 压缩出少量 global/view tokens，作为 FiLM 或少量 cross-attention K/V，避免每个 voxel chunk 直接看全部 `V*T` tokens。

### 暂定结论 / 决策
- 决策：`condition tokens` 和 `part tokens` 都进入 denoiser，不直接进入 Fisher bridge。
- 决策：每个 flow step 中，denoiser 用 `x_t + coords + is_on_surface + t + part_tokens_refined + compressed condition` 预测 endpoint logits。
- 暂定建议：第一版不加入强 part slot id embedding；slot 0 保留 `empty_token`，part slots 主要依赖 mask-pooled visual features。
- 暂定建议：condition tokens 第一版不要全量给 dense voxel cross-attention；先压缩成少量 global/view tokens，与 `part_tokens_refined` 一起给 chunk decoder。

### 未决问题
- condition compression 的具体形式：mean/max pooling、per-view pooling、attention pooling，还是保留 top-M tokens。
- 是否需要统计每个 valid part slot 的 2D coverage，作为训练/推理 contract 指标。
- 如果某个 valid part slot 没有 mask coverage，第一版是 hard fail、warning，还是使用显式 `missing_part_token`。

### 后续动作
- 下一轮讨论 `DenseVoxelDecoder` 最小结构：只 cross-attend part tokens，还是 `part tokens + compressed condition tokens` 双路；以及缺失 part coverage 的契约。

## Round 19 - Part token、condition token 与 Omni causal planner 的区别

日期: 2026-05-06

### 讨论背景
- 用户追问当前方案中 `part token` 和 `condition token` 分别是什么。
- 用户追问 small transformer 是否类似 OmniPart 的 causal transformer。

### 核心问题
- `part_tokens` 与 `cond/condition tokens` 的职责边界。
- 本项目的 part-token transformer 是否应该自回归/causal。
- 与 OmniPart 的 autoregressive structure planner 的相同点和不同点。

### 观点与推理
- 当前代码里的 `condition tokens` 是 `cond [B, V*T, cond_dim]`，即来自图像/多视角/ViT-DINO 分支的视觉 token。它们是观测条件，保留空间/视角/patch 级视觉信息，不是 part 槽位。
- 当前代码里的 `part_tokens [B, k_max, H]` 是从 `condition tokens` 按 `mask_token_labels` 做 mask-guided pooling 得到的对象内 part 槽位：
  - slot 0 是 learnable `empty_token`，对应 dense voxel label 的 empty 类；
  - slot `1..K` 是当前样本的局部 part slots，由 `mask_token_labels == j` 的视觉 tokens 平均池化得到；
  - padding slots `j >= num_parts[b]` 无效，必须 mask。
- `part_tokens` 不是跨类别固定语义 class。也就是说，不应默认“slot 1 永远是 chair leg”或“slot 2 永远是 back”。它们是当前样本内的 part instance slots，语义来自对应 mask 区域和视觉特征。
- small transformer 应该是 bidirectional transformer encoder / Set Transformer 风格，在 `K+1` 个 part tokens 上做 full self-attention，用来建模 part-part relation。它不是 causal transformer。
- 不建议使用 causal transformer 的原因：
  - 本项目不是生成 variable-length part box sequence，`K` 和 `mask_token_labels` 已经由输入给定；
  - dense completion 需要所有 part slots 同时互相可见，causal mask 会人为规定一个 part 顺序；
  - part slot 顺序多半是局部/实例顺序，不应引入强 next-token semantics；
  - 自回归会增加推理串行开销，但不能直接解决 dense voxel completion。
- OmniPart 的 causal transformer 是 structure planning 模块：根据 2D part masks 等条件，自回归生成 variable-length 3D part bounding boxes/layout sequence；之后再用 spatially-conditioned rectified flow 同时生成 parts。它的 causal 性来自“下一个 box/token 预测”任务。
- 本项目可参考 OmniPart 的 part-level reasoning 思想，但不复制 causal planner。我们的不同点是：不预测 part boxes 作为主输出，而是在 SLAT 前预测 dense solid part label field，并且有明确 `empty/part_id` dense completion 目标。

### 暂定结论 / 决策
- 决策：`condition tokens` 指原始视觉/图像 token `cond`，提供观测条件。
- 决策：`part_tokens` 指由 `mask_token_labels` 从 `cond` 池化出的 `K+1` 个局部 part slots，slot 0 为 empty。
- 决策：part-token small transformer 使用非 causal/bidirectional self-attention。
- 决策：不采用 OmniPart 式 autoregressive causal planner 作为第一版 Part Flow 模块。

### 未决问题
- part-token transformer 是否加入 slot id embedding；若加，应只作为局部槽位提示，不应把 slot id 当成跨类别语义 class。
- condition tokens 是否全部参与 voxel decoder cross-attention，还是先做全局/局部压缩以降低计算量。

### 后续动作
- 下一轮讨论 dense voxel decoder 如何使用 `part_tokens_refined` 和 `condition tokens`：cross-attention、FiLM、concat-MLP 或混合结构。

## Round 18 - 收敛已决策项：dirichlet_alpha 与无 voxel global attention

日期: 2026-05-06

### 讨论背景
- 用户指出 Round 17 中第三条“dense decoder 避免 voxel-level global self-attention”此前已经讨论并达成一致，不应再写成仍需选择的问题。
- 用户明确 `flow.dirichlet_alpha` 作为配置名。

### 核心问题
- 哪些内容已经是设计决策，不应在后续讨论中反复作为开放问题。
- dense completion 第一版的 attention 边界应如何固定。

### 观点与推理
- 已决策：放弃 dense voxel 之间的 global self-attention。原因此前已经明确：
  - `64^3=262144` 个 voxel 做全局 self-attention 在计算和显存上不可接受。
  - voxel 初始特征主要是 `x_t`、坐标、surface flag、time，本身不适合用全局 voxel attention 学 part relation。
  - part-part relation 更应该发生在 `K+1` 个 part tokens 上。
- 已决策：使用 small transformer 只在 part tokens 之间做 self-attention，用来建模 part-part relations。
- 已决策：dense voxel 端负责逐 voxel / chunked 解码，读取 refined part tokens、image/DINO condition、坐标、`is_on_surface` 和 `x_t`，输出 endpoint logits；它不做 voxel-to-voxel global attention。
- `flow.dirichlet_alpha` 是更清晰的配置名，比 `source_alpha` 更直接说明 source distribution 是 Dirichlet。

### 暂定结论 / 决策
- 决策：配置名使用 `flow.dirichlet_alpha`，默认 `1.0`。
- 决策：第一版不使用 voxel-level global self-attention。
- 决策：part-token small transformer 是全局 relation 模块；dense voxel decoder 是 chunked endpoint logits decoder。
- 决策：后续讨论不再把“是否放弃 voxel attention”作为开放问题，只讨论 dense voxel decoder 的具体实现形态和 chunk size。

### 未决问题
- dense voxel decoder 第一版具体采用：
  - voxel query cross-attention 到 part tokens + condition tokens；
  - 还是 voxel query 与 pooled/global condition concat 后 MLP；
  - 是否加入轻量 local 3D conv 作为后续增强。
- 推理/训练 chunk size 的默认值需要结合 H20 和 48G 4090 实测。

### 后续动作
- 下一轮直接讨论 `PartTokenTransformer + DenseVoxelChunkDecoder` 的接口和最小实现，不再回到 voxel global attention 是否合理的问题。

## Round 17 - Fisher 速度参数化与四个实现建议

日期: 2026-05-06

### 讨论背景
- 用户追问为什么 Fisher 主线下不让网络直接输出 velocity。
- 用户要求对 Round 16 的四个未决问题给出明确建议。

### 核心问题
- Fisher-Rao categorical flow 中，速度是否仍然存在。
- 如果速度存在，为什么第一版推荐网络输出 endpoint logits，而不是直接回归 tangent velocity。
- Fisher 主线下 solver、source、dense decoder、few-step acceleration 四个问题如何取舍。

### 观点与推理
- Fisher 并不是没有 velocity。Flow 仍然是从 source categorical state 到 target one-hot endpoint 的连续动力学；只是在 Fisher-Rao 几何下，状态先通过 `u=sqrt(p)` 映射到球面，速度是球面切空间里的 tangent vector。
- 第一版不建议网络直接输出 velocity，原因是 Fisher 的条件速度可由 endpoint 和当前状态通过几何公式确定：
  - 给定真实 endpoint `x_1`，`u_t -> u_1` 的 geodesic tangent 是确定的。
  - 训练时让网络预测 `p(x_1 | x_t,t,cond)`，bridge 可以把 endpoint posterior 聚合成 marginal update。
  - 因此网络输出 endpoint logits，bridge 负责从 endpoint probability 计算几何更新，是更清楚的 factorization。
- 若直接让网络输出 tangent velocity，需要额外处理：
  - velocity 必须落在 sphere tangent space，满足与当前 `u_t` 正交。
  - padding/variable-K 维度必须严格为 0。
  - 更新后还要 exp-map 回球面，再映射回 simplex。
  - velocity MSE 的监督尺度随 `t` 和 geodesic 距离变化，训练更容易出现数值和权重问题。
- endpoint logits 参数化更贴近离散任务本体：最终目标就是每个 voxel 的 `empty/part_id`，CE/focal CE 直接监督 clean endpoint；flow 的几何运动由 bridge 保证，而不是把几何约束塞进网络输出层。
- 四个问题建议：
  - Solver：第一版只把 Euler/geodesic step 作为主路径，保留 Heun 为配置 ablation，但不要在主叙事和默认训练评估中依赖 Heun。
  - Source：第一版固定并暴露 `source_alpha=1.0`，语义为 valid simplex 上 Dirichlet source 的浓度；默认不 sweep，后续 ablation 再测 `0.5/1.0/2.0`。
  - Dense decoder：必须移除/绕开 voxel-level global self-attention；推荐 `PartTokenTransformer(K tokens)` 做全局 part relation，dense voxel 端只做 chunked cross-attention/MLP/轻量局部模块，保证枚举全 64³ 但不产生 `N^2` attention。
  - Few-step acceleration：第一版不引入 Categorical Flow Maps / consistency distillation；先训练 Fisher baseline 并测 `20/10/5`。如果 `5/10` 太慢或质量曲线显示可蒸馏，再做第二阶段加速。

### 暂定结论 / 决策
- 暂定建议：Fisher 主线仍然是 flow；velocity 存在于 bridge 内部的 Fisher-Rao / sphere tangent update 中。
- 暂定建议：第一版网络输出 endpoint logits，不直接输出 velocity。
- 暂定建议：主 sampler 使用 Euler/geodesic step，Heun 只作为可选 ablation。
- 暂定建议：`source_alpha` 暴露到 config，默认 `1.0`。
- 暂定建议：dense 64³ 正确枚举，但模型结构用 part-token global reasoning + voxel chunked decoding，避免 voxel full self-attention。
- 暂定建议：few-step/self-distillation 放到第二阶段，不进入第一版核心实现。

### 未决问题
- `source_alpha` 的 config 名称是沿用 `flow.source_alpha`，还是更明确写成 `flow.dirichlet_alpha`。
- Dense voxel decoder 第一版用纯 chunked MLP/cross-attention，还是加入轻量 3D local conv。
- 推理 chunk size 默认值需要结合 48G 4090 和 H20 实测后确定。

### 后续动作
- 下一轮可以定 `PartFlowPredictor` 新结构的最小可实现版本：输入、模块边界、chunking、输出和训练/eval 指标。

## Round 16 - Fisher 作为更优雅的 categorical flow 主线

日期: 2026-05-06

### 讨论背景
- 用户追问最优雅的 dense categorical Part Flow 方案是什么，以及是否 Fisher 反而比 standard RF 更好。
- 本轮结合近期离散 flow 文献和当前代码实现，重新评估 Round 12-15 中“Euclidean simplex velocity RF + projection”的设计。

### 核心问题
- 对 dense 64³ 的 `empty/part_id` categorical field，主 flow 应该采用 Euclidean simplex RF、Fisher-Rao categorical flow，还是 Gumbel-Softmax flow。
- Fisher 是否能避免 Euclidean RF 的 projection fail 和状态几何不自然问题。
- 选择 Fisher 后，网络输出应继续是 endpoint logits 还是 velocity。

### 观点与推理
- `part_labels_solid_64` 本质是离散 categorical label field。把每个 voxel 的 label 表示成 `K+1` 类 simplex 上的 one-hot 后，simplex 不是 Euclidean 空间；直接做 `x_t + dt * v` 再 `clamp + renorm` 是可运行的工程方案，但几何上不自然。
- Fisher Flow Matching 的核心优势是把 simplex 通过 `u=sqrt(p)` 映射到正球面正交象限，在球面上走 geodesic/slerp，再映射回 simplex。这比 Euclidean RF 更符合 categorical probability 的 Fisher-Rao 几何。
- 对本项目尤其重要的是：Fisher 的 `step` 在球面上用 log/exp map 朝模型预测的 endpoint 前进，天然保持 valid simplex 语义，不需要把 `clamp + renorm` 当成主动力学的一部分。因此 Round 15 讨论的 projection fail 不再是主线风险。
- Fisher 主线下，模型不必直接输出 velocity。更清晰的接口是模型输出 `endpoint_logits = p(x_1 | x_t, t, condition)`，bridge 根据 endpoint probability 和 Fisher-Rao geometry 组装下一步更新。
- 这会推翻 Round 12 中“主输出 velocity + endpoint CE 辅助头”的一部分：那是 Euclidean RF 方案下合理；若选择 Fisher，endpoint logits 应成为主参数化，CE/focal CE 是主训练目标，而不是辅助 loss。
- Gumbel-Softmax Flow 也合理，尤其在类别数很高时有优势；但本项目 `K+1` 是 object parts，通常远小于语言/protein alphabet 的组合规模。第一版优先 Fisher 更符合“优雅、无 trick、几何正确”的目标。
- Categorical Flow Maps 这类 2026 工作说明 discrete/categorical flow 的少步生成正在走向 endpoint-constrained / distillation / few-step acceleration。它更适合作为后续加速路线，而不是第一版 dense completion 的基础设计。
- 当前代码已经存在 `FisherBridge`、`GumbelSoftmaxBridge`、endpoint logits loss 和 bridge-driven sampler，并且 `base.yaml` 默认 `flow.type: fisher`。这说明代码方向已经部分接近本轮结论；真正需要重做的是 dense 64³ 推理输入和 voxel decoder 架构，而不是再发明一个 Euclidean velocity RF。

### 暂定结论 / 决策
- 暂定建议：第一版 dense completion 主线使用 `FisherBridge` / Fisher-Rao categorical flow。
- 暂定建议：保留 `GumbelSoftmaxBridge` 作为 ablation，不作为第一版默认。
- 暂定建议：暂不采用 Euclidean simplex velocity RF 作为主线；它可作为最低基线或历史参考。
- 暂定建议：网络输出采用 endpoint logits，由 bridge 负责几何更新；不再把 velocity head 作为第一版核心接口。
- 暂定建议：推理默认仍从 `num_steps=10` 起步，benchmark `20/10/5`；Fisher 下也可以后续探索 `2/1` 和 distillation。

### 未决问题
- Fisher 主线下是否需要保留 Heun solver，还是第一版只用 Euler/geodesic step。
- Fisher source 是否固定 Dirichlet(1)，还是暴露 `source_alpha` 做 ablation。
- Dense 64³ 枚举后，模型结构如何从当前 voxel full self-attention 改成 `part-token self-attention + dense voxel cross-attention/chunking`。
- 是否后续引入 Categorical Flow Maps/endpoint consistency distillation 做 few-step 或 one-step 加速。

### 后续动作
- 下一轮讨论 Fisher 主线下的具体模型接口：`PartFlowPredictor.forward` 输入/输出、dense coords chunking、`is_on_surface` dense condition、`mask_token_labels/num_parts` 契约。
- 后续实现时优先复用现有 `FisherBridge` 和 endpoint CE loss，避免新增冗余 Euclidean velocity 分支。

### 参考
- Davis et al., `Fisher Flow Matching for Generative Modeling over Discrete Data`, NeurIPS 2024 / arXiv:2405.14664。
- Gat et al., `Discrete Flow Matching`, arXiv:2407.15595。
- Tang et al., `Gumbel-Softmax Flow Matching with Straight-Through Guidance for Controllable Biological Sequence Generation`, arXiv:2503.17361。
- Roos et al., `Categorical Flow Maps`, arXiv:2602.12233。

## Round 15 - Projection fail 原理与 step 数建议

日期: 2026-05-06

### 讨论背景
- 用户要求更具体解释 projection fail 的原理。
- 用户决定 `ce_weight` 可以先固定为 `0.5`。
- 用户要求给出第一版推理 step 数建议。

### 核心问题
- 为什么 Euler 更新后的 simplex projection 会 fail。
- `ce_weight` 是否先固定，避免 schedule 复杂化。
- 第一版 dense RF 推理应默认跑多少 steps，benchmark 哪些 steps。

### 观点与推理
- Projection 的目标是把 Euler 更新后的 `x_raw` 拉回 probability simplex：
  - 当前 state: `x_t[n]` 是 valid part slots 上的概率，满足 `x>=0` 且 `sum(x)=1`。
  - Euler 更新: `x_raw = x_t + dt * v_pred`。
  - projection: padding slots 置 0；valid slots `clamp(min=0)`；再除以 valid slots 的和。
- Projection fail 的核心不是“投影函数坏了”，而是 `x_raw` 已经被模型速度推到一个无法可靠恢复的区域。最典型情况：
  - 对某个 voxel，valid slots 的 `x_raw` 全部小于等于 0。
  - clamp 后全变成 0。
  - 此时 valid sum = 0，无法归一化成概率分布。
- 具体例子：
  - 假设三类 valid slots 当前 `x_t=[0.2,0.7,0.1]`。
  - 若 step 太大或 `v_pred` 爆炸，更新后 `x_raw=[-0.5,-0.1,-0.3]`。
  - clamp 后 `[0,0,0]`，sum=0，无法判断该 voxel 应该朝哪个 part 走。
  - 强行改成 uniform `[1/3,1/3,1/3]` 会隐藏模型发散；这就是 silent fallback，不应作为默认。
- 另一类 fail 是 NaN/Inf：
  - `v_pred` 或 `x_raw` 出现 NaN/Inf 时，renorm 会传播坏值。
  - 这通常来自训练不稳、过大 `dt`、混合精度溢出、mask 错误或模型输出未约束。
- 还有 mask/contract fail：
  - 如果 `num_parts` 错，某个 voxel 没有 valid slot，projection 没有合法 simplex 可投。
  - 如果 padding slots 没清零，padding 概率可能进入 sum 或 argmax，破坏 variable-K 契约。
- 避免方式：
  - 训练和推理都把 `v_pred` 乘 `valid_per_voxel`，padding velocity 置 0。
  - Euler 更新后先检查 `isfinite(x_raw)`。
  - 使用较小 `dt`：steps 越少，`dt` 越大，越容易把 `x_t` 一步推过头。
  - 可选对 `v_pred` 做 value clipping，例如 `[-v_clip, v_clip]`，但第一版是否加需要谨慎；clip 是数值稳定手段，不改变任务语义。
  - projection 后统计 `bad_projection_frac = bad_voxels / N`。如果超过阈值 fail-loud；如果极少数是浮点边界，可以 eps renorm 并记录。
- `ce_weight` 第一版固定为 `0.5`：
  - 先不做 warmup/schedule，减少变量。
  - 若后续观察到 velocity 学不好或分类不 sharp，再做 ablation。
- Step 数建议：
  - 第一版默认 `num_steps=10`：速度和稳定性折中，`dt=0.1` 不算太大。
  - 必测 benchmark: `20, 10, 5`。`20` 作为质量上界，`10` 作为默认，`5` 作为快速模式。
  - `2` 和 `1` 不作为第一版稳定性要求，只作为 exploratory few-step 实验；如果 projection fail 或质量明显差，不阻塞第一版。
  - 训练/eval 记录同一组 steps 的速度、显存、endpoint accuracy、part/empty accuracy、bad_projection_frac。

### 暂定结论 / 决策
- 决策：`ce_weight` 第一版固定 `0.5`，暂不做 warmup/schedule。
- 暂定建议：推理默认 `num_steps=10`。
- 暂定建议：第一版正式 benchmark `20/10/5`，探索性测试 `2/1`。
- 暂定建议：projection fail 的判据包括 non-finite、valid sum 接近 0、bad_projection_frac 超阈值、valid mask 非法。

### 未决问题
- 是否加入 `v_clip` config；若加，默认值应谨慎，避免掩盖模型问题。
- `bad_projection_frac` 阈值具体用 `1e-4` 还是 `1e-3`。
- `num_steps=10` 是否作为训练中 eval 默认，还是只作为 inference 默认。

### 后续动作
- 下一轮可以整理完整 config 草案，并把 projection diagnostics 写进 inference/eval 指标。

## Round 14 - RF config 与 projection fail 条件

日期: 2026-05-06

### 讨论背景
- 用户同意 Round 13 的建议。
- 用户对 Round 13 的四个未决问题给出方向：
  - `α` 需要作为 config 暴露，并注释清楚功能。
  - `λ_ce` warmup 可以做。
  - 需要解释 projection 什么时候会 fail，以及能否避免。
  - endpoint CE 不直接复用旧函数，给 RF 写一个更清晰的 masked CE。

### 核心问题
- RF source/projection/loss 的配置和失败模式如何设计得清楚、可调、fail-loud。
- projection 数值失败是否可以通过训练/推理约束避免。

### 观点与推理
- `α` 应作为 config 暴露，例如：
  - `rf.source: dirichlet`
  - `rf.dirichlet_alpha: 1.0`
  - 注释：控制 `x_0` 在 valid simplex 上的随机性；`α=1` 为均匀 simplex，`α<1` 更接近 corner/categorical，`α>1` 更平滑靠近 uniform。
- `λ_ce` 可以 warmup 或 schedule：
  - 第一版可配置 `ce_weight`, `ce_warmup_steps`, `ce_final_weight`。
  - 比较稳的策略是前期 `λ_ce` 稍高帮助 endpoint 分类稳定，后期降低让 velocity 主导；例如从 `1.0 -> 0.5` 或 `0.5 -> 0.2`。
  - 但不要把 schedule 做太复杂；第一版 config 清晰优先。
- Projection fail 的主要情况：
  - Euler 更新后 valid slots 全部变成负数，clamp 后 sum 约为 0。
  - 模型输出 velocity 数值爆炸，导致 `x_t` 出现 NaN/Inf。
  - padding/valid mask 错误，某个 voxel 没有任何 valid slot。
  - `num_parts` 错误，例如小于 2 或超过 `k_max`，导致 valid mask 不合法。
- Projection fail 可以大部分避免：
  - 每次更新前后强制 padding slots 为 0。
  - 对 `v_pred` 做 valid mask，并可选 gradient/value clipping。
  - 使用较小步数间隔：few-step 时 `dt` 大，1-step 最容易出界；可以先 benchmark `20/10/5`，再看 `2/1`。
  - projection 时使用 `sum.clamp_min(eps)` 只能避免除零，但不能静默掩盖模型发散；应统计 `bad_projection_frac`。
  - 若 `bad_projection_frac` 超过阈值，应 fail-loud；若只是极少数浮点边界，可以记录 warning 并用 eps renorm。
- RF 应写独立清晰的 masked CE：
  - 输入 `endpoint_logits [N,k_max]`、`labels [N]`、`valid_per_voxel [N,k_max]`、`class_weights [k_max]`。
  - padding slots mask 到 `-1e4`。
  - ignore label `-1` 不参与 loss。
  - 支持 focal gamma，但函数命名和参数围绕 RF endpoint auxiliary loss，不复用旧 bridge loss 语义。

### 暂定结论 / 决策
- 决策：`dirichlet_alpha` 暴露到 config，并写清楚取值含义。
- 决策：`λ_ce` 支持 warmup/schedule，但第一版保持简单。
- 决策：projection 不做静默兜底；记录 `bad_projection_frac`，超过阈值 fail-loud。
- 决策：为 RF 新写清晰的 masked endpoint CE / focal CE，而不是直接复用旧函数。

### 未决问题
- `bad_projection_frac` 阈值具体设多少；初始建议 `1e-4` 或 `1e-3` 级别。
- `ce_weight` schedule 是高到低，还是固定 `0.5` 先跑 baseline。
- 1-step RF 是否作为第一版必测，还是等 5/10/20 稳定后再测。

### 后续动作
- 下一轮可以整理完整 config 草案，包括 `rf.*`, `model.part_token_encoder.*`, `model.voxel_decoder.*`, `loss.*`, `inference.*`。

## Round 13 - RF source/projection/loss 细节建议

日期: 2026-05-06

### 讨论背景
- 用户同意 Round 12 的三个建议：RF state 用 masked probability simplex，网络主输出 velocity + endpoint logits 辅助头，推理用 Euler few-step 且训练保留 endpoint CE 辅助 loss。
- 继续讨论 Round 12 留下的四个问题：
  - `x_0` source 具体选 Dirichlet、uniform-simplex，还是 logistic-normal 后 softmax。
  - `λ_ce` 权重如何设定，是否需要 focal CE 沿用当前 empty/part class imbalance 处理。
  - 每步 projection 是简单 clamp+renorm，还是使用 softmax/temperature。
  - velocity 输出是否需要在 padding slots 上强制为 0。

### 核心问题
- 如何定义稳定、可解释、符合 variable-K 契约的 RF source distribution。
- 如何同时优化 flow vector field 和 categorical sharpness。
- 如何保证推理积分过程中 `x_t` 不离开 valid simplex。

### 观点与推理
- `x_0` source 建议使用 **masked Dirichlet**，而不是 logits logistic-normal：
  - Dirichlet 直接生成 probability simplex 上的随机状态，和 `x_t=(1-t)x_0+t x_1` 的 simplex 语义完全对齐。
  - 对每个样本只在 valid slots `0..num_parts-1` 上采样，padding slots 固定为 0。
  - 初始可用对称 Dirichlet `α=1.0`，等价于 uniform over simplex；后续可 ablate `α<1` 让 source 更接近 categorical corners，或 `α>1` 让 source 更平滑。
  - logistic-normal 后 softmax也可行，但多了无界 logits 和 softmax 温度选择，不适合第一版。
- `λ_ce` 建议保留当前 class imbalance 思路，使用 **endpoint focal CE / weighted CE**：
  - 速度损失 `loss_v` 是主任务，初始设 `λ_v=1.0`。
  - endpoint 辅助损失初始可设 `λ_ce=0.5`；若训练早期分类不稳可升到 `1.0`，若发现模型过度依赖 endpoint head、velocity 学不好可降到 `0.1`。
  - empty/part imbalance 仍然存在，因为 dense 64³ 中 empty 数量通常远多于 part voxels；建议沿用当前 `empty_weight=0.05, part_weight=1.0, focal_gamma=2.0` 作为第一版默认。
  - CE 辅助 head 应只在 valid slots 上计算，padding slots masked 到 `-inf/-1e4`。
- 每步 projection 建议使用 **valid-slot clamp + renormalize**，不要用 softmax/temperature：
  - Euler 更新后 `x_t` 可能出现负值或 sum 不为 1；对 valid slots clamp 到非负，再除以 valid sum，padding slots 置 0。
  - 这样 projection 是最直接的 simplex projection 近似，保留 RF state 的 probability 语义。
  - softmax 会把 state 当 logits 处理，改变几何含义；temperature 还会引入额外超参，不适合第一版。
  - 若 valid sum 接近 0，不能 silent fallback 到 uniform；应记录/报错或在数值层面用 eps 后 renorm，同时统计异常比例，避免隐藏模型发散。
- velocity 输出必须在 padding slots 上 **强制为 0 并 mask loss**：
  - padding slots 不属于当前样本的 part space，不能让网络在这些维度上学习任何速度。
  - `v_target = x_1 - x_0` 在 padding slots 天然为 0；`v_pred` 也应乘 `valid_per_voxel`，loss 只对 valid dims 求。
  - 推理更新前后都要把 padding slots 清零，防止 padding 维度污染 renormalization 或 argmax。

### 暂定结论 / 决策
- 暂定建议：`x_0` 使用 per-sample masked Dirichlet，默认 `α=1.0`。
- 暂定建议：`λ_ce` 第一版默认 `0.5`，并保留当前 weighted/focal endpoint CE。
- 暂定建议：Euler 每步后做 valid-slot clamp + renormalize，padding slots 置 0。
- 暂定建议：velocity padding slots 强制为 0，velocity loss 只在 valid dims 上计算。

### 未决问题
- `α` 是否需要作为 config 暴露，并做 `0.5/1.0/2.0` ablation。
- `λ_ce` 是否需要 warmup：前期高一点稳定分类，后期降低让 velocity 主导。
- projection 异常时是 hard fail 还是只记录 warning；第一版倾向 fail-loud 或统计指标暴露。
- endpoint CE 是否直接复用当前 `weighted_focal_endpoint_ce` 实现，还是为 RF 新写更清晰的 masked CE。

### 后续动作
- 下一轮可以把目前共识整理成第一版 `PartTokenConditionedDenseRF` 的完整结构草案。
- 结构草案确认后，再进入实现计划，而不是继续零散改当前 Fisher/Gumbel 代码。

## Round 12 - Rectified Flow state/output/loss 建议

日期: 2026-05-06

### 讨论背景
- 用户要求继续深入三个 RF 设计问题：
  - RF state 使用 probability simplex 还是 logits。
  - 网络输出 endpoint logits、clean one-hot prediction，还是 velocity。
  - 推理是否使用 Euler few-step，训练是否保留 endpoint CE 辅助 loss。

### 核心问题
- 如何让 standard Rectified Flow 适配 dense categorical `empty/part_id` label field。
- 如何避免 RF 退化成普通分类器，同时保留 categorical label 的清晰监督。
- 如何让第一版既论文正确又方便少步推理 benchmark。

### 观点与推理
- 建议 RF state 使用 **probability/simplex-like continuous state**，不要用 logits 作为主 state：
  - endpoint `x_1` 是 one-hot categorical label，天然在 simplex 顶点上。
  - source `x_0` 可以从 Dirichlet / uniform-simplex / masked random simplex 采样，并只在 valid `num_parts` slots 上归一化。
  - 插值 `x_t = (1-t) x_0 + t x_1` 仍在 valid simplex 内，语义清楚。
  - logits state 虽然无界、适合神经网络输出，但插值和距离没有明确 categorical probability 语义，容易让 `x_t` 状态和最终 label 解释脱节。
- 建议网络主输出采用 **velocity**，同时带 **endpoint logits 辅助头**：
  - RF 主目标：预测 `v = x_1 - x_0`，使用 MSE / masked MSE。这样才是真正 standard RF / velocity field，方便 few-step Euler。
  - 辅助 endpoint logits：从同一个 voxel hidden 输出 `endpoint_logits`，对 `part_labels_solid_64` 做 masked CE/focal CE。这样保留 categorical sharpness，避免 velocity MSE 学到模糊概率而分类边界不清。
  - 不建议只输出 clean one-hot/probability：它更像 denoising classifier，会弱化 RF 的 vector-field 解释；可以作为 endpoint head，但不应是唯一主输出。
- 推理建议使用 **Euler few-step 作为第一版主 sampler**：
  - standard RF 最自然的推理是 `x_{t+dt} = x_t + dt * v_theta(x_t,t,cond)`。
  - 第一版直接 benchmark `num_steps=20/10/5/2/1`，观察质量-速度曲线。
  - Heun 可以作为可选更稳 sampler，但第一版不必依赖它。
  - DPM-Solver 不作为默认，因为当前不是标准 diffusion noise schedule；除非后续重写成 diffusion/VP/VE 形式，否则不硬套。
- 训练建议保留 endpoint CE 辅助 loss：
  - `loss = λ_v * masked_mse(v_pred, x_1 - x_0) + λ_ce * endpoint_ce(endpoint_logits, labels)`。
  - 初始可设 `λ_v=1.0, λ_ce=0.1~1.0` 做 sweep。
  - endpoint CE 是防止 categorical 任务“只看概率 MSE 不够 sharp”的安全锚点，不是冗余兜底。
- 对 valid slots 的处理必须延续当前 variable-K 契约：
  - `x_0, x_t, x_1, v_pred` 都只在 `0..num_parts-1` valid slots 上有效。
  - padding slots 必须 mask 掉，不能参与 MSE、CE、argmax。
  - 推理每步后建议对 valid slots 做 clamp + renormalize，保持 simplex state 合法。

### 暂定结论 / 决策
- 暂定建议：RF state 使用 masked probability simplex，而不是 logits。
- 暂定建议：网络主输出 velocity，辅以 endpoint logits CE head。
- 暂定建议：推理主用 Euler few-step；训练保留 endpoint CE 辅助 loss。
- 暂定建议：第一版不使用 DPM-Solver，后续可研究 reflow / distillation / consistency-style 加速。

### 未决问题
- `x_0` source 具体选 Dirichlet、uniform-simplex，还是 logistic-normal 后 softmax。
- `λ_ce` 权重如何设定，是否需要 focal CE 沿用当前 empty/part class imbalance 处理。
- 每步 projection 是简单 clamp+renorm，还是使用 softmax/temperature。
- velocity 输出是否需要在 padding slots 上强制为 0。

### 后续动作
- 下一轮讨论 `x_0` source distribution 和 projection 规则。
- 确认后再进入 `PartTokenConditionedDenseRF` 的模型/训练计划。

## Round 11 - Flow 仍在 dense voxel categorical state 上

日期: 2026-05-06

### 讨论背景
- 用户追问：如果引入 part-token transformer、去掉 voxel full self-attention，那么方案里的 flow 去哪里了。

### 核心问题
- part-token transformer 与 flow 的关系是什么。
- 新方案是否仍然是 Part Flow，而不是普通分类器。
- flow 的状态变量、条件变量和网络预测对象分别是什么。

### 观点与推理
- part-token transformer 不替代 flow。它只替代/削弱当前 `PartFlowDecoderLayer` 中昂贵的 voxel-level full self-attention，用来在 `K+1` 个 part tokens 上建模全局 part-part relation。
- Flow 仍然发生在 dense voxel categorical state 上：
  - 每个 voxel `n` 有一个状态 `x_t[n] ∈ R^{K_max}`，表示 empty/part slots 的 continuous categorical-simplex/logit state。
  - 训练 endpoint `x_1[n]` 来自 `part_labels_solid_64[n]` 的 one-hot label。
  - 初始 `x_0[n]` 是噪声或随机 simplex/logit source。
  - 模型学习从 `x_t`、时间 `t`、voxel 坐标、SS surface condition、DINO tokens、part tokens 预测 velocity 或 endpoint。
- 新结构可以写成：
  - `part_tokens = MaskedDinoPool(cond, mask_token_labels, num_parts)`
  - `part_tokens_refined = PartTokenTransformer(part_tokens)`
  - `voxel_queries = f(x_t, coords, is_on_surface, t)`
  - `out = DenseVoxelCrossDecoder(voxel_queries, cond_tokens, part_tokens_refined)`
  - `flow_step(out)` 更新 `x_t`
- 因此，Part Flow 的核心没有消失：采样仍然是多步/少步 flow integration；每一步都对所有 dense voxels 的 categorical state 做更新。变化的是 denoiser/vector-field 网络的 attention 结构，不再让 dense voxels 互相 full-attend。
- 这也避免退化成 adapter/classifier：普通 classifier 只做一次 `condition -> label logits`；flow 模型在每个时间步看到 `x_t` 并学习 `x_t -> x_1` 或 velocity，推理从 source state 逐步生成 dense labels。

### 暂定结论 / 决策
- 暂定结论：新方案仍是 dense Part Flow；part-token transformer 是条件编码器/全局结构编码器，不是生成过程本身。
- 暂定结论：flow 的状态空间仍是 `[64^3, K_max]` 的 dense categorical-simplex/logit field。
- 暂定方向：第一版新架构应明确命名为 `PartTokenConditionedDenseFlow` 一类，避免被误解成 simple classifier。

### 未决问题
- RF state 使用 probability simplex 还是 logits。
- 网络输出 endpoint logits、clean one-hot prediction，还是 velocity。
- 推理是否使用 Euler few-step，训练是否保留 endpoint CE 辅助 loss。

### 后续动作
- 下一轮讨论 Rectified Flow 具体公式：`x_t=(1-t)x_0+t x_1`、target velocity `v=x_1-x_0`，以及 categorical output 如何投影/argmax。

## Round 10 - Voxel self-attention 与 part-token transformer

日期: 2026-05-06

### 讨论背景
- 用户提出：voxel 之间本身可能还没有什么特征，让它们互相做 full attention 是否本身就不合理。
- 用户希望理解 OmniPart planner 的原理，以及它和当前方案的对比。
- 用户决定暂时不考虑 `coverage_stats`。
- 用户认为需要一个 small transformer 在 `K` 个 part tokens 之间做 self-attention，用来建模 part-part relations，替代 voxel-level global self-attention。

### 核心问题
- 当前 dense voxel token 是否适合做全局 self-attention。
- OmniPart planner 的原理是否可迁移到本项目。
- `K` 个 part tokens 上的 small transformer 是否应成为第一版结构的一部分。

### 观点与推理
- 用户关于 voxel self-attention 的直觉是合理的：当前 Part Flow 的 voxel token 初始主要由 `x_t` 类别状态、voxel position embedding、`is_on_surface` embedding 组成。它不像 image patch token 那样天然携带丰富局部语义/纹理；在训练早期或采样初期，voxel-token 之间全局 self-attention 的信号可能很弱，且 `N=64^3` 时成本极高。
- Voxel-level full self-attention 的潜在作用是传播全局 shape/part consistency，但这件事不一定应该发生在 262k voxel token 之间。更合理的归纳偏置是：
  - part tokens 负责全局 part-part relation；
  - voxel queries 负责 dense spatial decoding；
  - voxel-to-part / voxel-to-image cross-attention 把全局 part/image 条件注入每个 voxel；
  - 局部空间一致性若需要，再用 window/local attention 或 3D conv，而不是 full voxel attention。
- OmniPart planner 的原理：
  - 把 part layout 表示成 variable-length box token sequence。
  - 用 causal / decoder-only transformer 做 next-token prediction，逐步生成 part boxes。
  - 这个 planner 处理的是 part-level global structure，token 数约为 part 数，而不是 voxel 数。
  - 后续 rectified flow 在 planned spatial layout 条件下并行生成 structured latents。
- 与当前方案的对比：
  - OmniPart planner: 生成显式 3D boxes，适合 part-level layout control，但会引入 bbox 表示和 box supervision/推理。
  - 当前方案: 已有 `mask_token_labels` 和 `part_info`，可以直接构造 `K` 个 part tokens；不需要先生成 boxes，也不改变 dense 64³ 输出。
  - 可迁移的是“part-level relation modeling”，不是 causal box generation 本身。
- 因此，第一版更推荐：
  - 不引入 OmniPart-style causal planner。
  - 使用现有 masked DINO pooling + slot/empty token 得到 `K+1` 个 part tokens。
  - 在 `K+1` part tokens 上加 small bidirectional transformer encoder，建模 part-part relation。这里不需要 causal mask，因为 part tokens 是一组已知条件，不是要按顺序生成。
  - dense voxel decoder 去掉 full voxel self-attention，改为 cross-attention 到 refined part tokens 和 DINO tokens；如需要局部一致性，后续再加 window/local attention 或轻量 3D conv。
- 这样做的复杂度更合理：part-token self-attention 是 `O(K^2)`，通常 `K<=128`，远小于 `O((64^3)^2)`；voxel-to-part cross-attention 是 `O(NK)`，voxel-to-image cross-attention 是 `O(NVT)`，仍然大但比 full voxel attention可控，且可以进一步按 block/chunk 做 exact cross-attention 因为不同 voxel queries 对同一条件 KV 独立，不破坏全局条件语义。

### 暂定结论 / 决策
- 暂定接受：full voxel self-attention 不适合作为第一版 full dense 64³ 的核心结构，既贵也不一定是正确归纳偏置。
- 决策倾向：需要一个 small part-token transformer，在 `K+1` 个 part tokens 上建模 part-part relation，替代 voxel-level global self-attention 的全局结构作用。
- 暂定接受：第一版不引入 OmniPart-style causal planner；用 bidirectional part-token transformer 更适合当前已知 part slots 的条件建模。
- 暂定拒绝：本轮暂不加入 `coverage_stats`，避免扩展输入契约。

### 未决问题
- Dense voxel decoder 是否完全去掉 self-attention，还是保留 window/local attention。
- Voxel-to-image cross-attention 是否也需要优化；`N*VT` 在 dense 64³ 下仍然可能很大。
- RF 状态表示和 loss 还需要单独讨论：continuous one-hot/simplex、logits、velocity target 或 endpoint target。

### 后续动作
- 下一轮讨论第一版模型结构草案：`PartTokenEncoder + DenseVoxelCrossDecoder + RectifiedFlowBridge`。
- 需要明确哪些模块可复用当前代码，哪些模块需要新建。

## Round 9 - Part tokens 的来源与合理性

日期: 2026-05-06

### 讨论背景
- 用户认为“part-level tokens + dense voxel decoder”方案看起来合理，希望继续深入讨论 token 怎么来，以及这个设计是否正确合理。
- 当前代码中 `PartFlowPredictor.build_part_tokens` 已经实现了一版 part token 构造：用 `mask_token_labels` 对 DINOv2 tokens 做 masked average pooling。

### 核心问题
- `K` 个 part tokens 应该从哪里来。
- 现有 `mask_token_labels` pooling 是否足够表达每个 part。
- 这些 tokens 是否能承担全局 part structure reasoning，从而减少 dense voxel self-attention。

### 观点与推理
- 当前代码已有的 part token 来源：
  - 输入 `cond_proj=[B, V*T, H]`，来自多视角 DINOv2 tokens 投影。
  - 输入 `mask_token_labels=[B,V*T]`，每个 2D token 标注 `0=bg/CLS, 1..K=part id`。
  - 对每个 part id `j`，取所有 `mask_token_labels==j` 的 DINO token 平均，得到 `part_tokens[b,j]`。
  - slot 0 不从图像 pooling，而是 learnable `empty_token`。
  - 对 valid 但 2D mask 没覆盖到的 part，用 learnable `slot_emb[j]` fallback。
- 这套来源是合理的一阶方案，因为它把“part identity”直接绑定到 2D mask 证据：每个 part token 表示该 part 在多视角图像中的视觉/语义特征，而不是任意 learnable query。
- 但 masked average pooling 也有局限：
  - 平均会丢掉空间分布、视角差异、part 形状和大小信息。
  - 如果某个 part 在所有视角被遮挡或 mask 缺失，只能靠 slot embedding fallback；这提供 identity placeholder，但没有真实视觉证据。
  - 仅靠 part token 不能自动给出 dense 64³ 内部形状；dense voxel query 仍需要 position、surface condition，以及局部空间建模。
- 因此，第一版最稳的 token 来源可以是“现有 pooling + 少量增强”，而不是立刻上 OmniPart-style causal planner：
  - `visual_part_token`: masked pooling DINO tokens。
  - `slot_token`: learnable part slot embedding，提供稳定 id / fallback。
  - `geometry_summary_token`: 从 SS surface 对每个 part 的粗几何还无法直接得到，因为当前 SS surface 没有 part id；除非使用 2D-3D projection 或额外 predictor，否则第一版不强行加。
  - `empty_token`: 单独 learnable。
- 更合理的融合形式不是二选一，而是：
  - `part_token_j = MLP([masked_dino_pool_j, slot_emb_j, coverage_stats_j])`
  - 其中 `coverage_stats_j` 可以包括该 part 在多少 views 可见、2D mask token count、归一化面积等轻量统计。
- 这个设计与 OmniPart 的可参考点一致：全局 part 信息通过少量 part-level tokens 进入模型；但不同于 OmniPart，本项目不生成 boxes，不改变 dense 64³ 输出。

### 暂定结论 / 决策
- 暂定接受：第一版 part tokens 直接基于当前 `mask_token_labels` masked pooling，是正确且最贴合现有数据契约的来源。
- 暂定接受：可以增强 token 表达，但不应引入需要新标注或复杂投影的 token 来源。
- 暂定方向：用 part tokens 替代或削弱 full voxel self-attention 时，必须保留 dense voxel position embedding 和 SS `is_on_surface` condition；part tokens 负责 part identity/global semantics，voxel queries 负责空间位置和 surface-to-solid decoding。

### 未决问题
- 是否在第一版模型中完全去掉 voxel self-attention，只保留 voxel-to-part / voxel-to-image cross-attention。
- `coverage_stats` 是否值得加入，还是先保持当前 pooling 以降低改动。
- 是否需要一个 small transformer 在 `K` 个 part tokens 之间做 part-token self-attention，用来建模 part-part relations，替代 voxel-level global self-attention。

### 后续动作
- 下一轮讨论具体模型结构：`part-token self-attention + voxel cross-attention decoder` 是否作为第一版新 Part Flow 架构。
- 如果用户认可，再进入设计文档/计划阶段，而不是立即修改代码。

## Round 8 - OmniPart 的 K 个 part tokens 与 flow 不是同一阶段

日期: 2026-05-06

### 讨论背景
- 用户追问：如果 OmniPart 只有 `K` 个 part token，那么它是怎么做 flow 的；这是否意味着它其实是在自回归预测。

### 核心问题
- OmniPart 中 autoregressive causal transformer 和 rectified flow 的边界是什么。
- “K 个 part tokens/boxes” 与 “flow token 数量” 是否相同。
- 这对本项目 dense 64³ Part Flow 有什么启发。

### 观点与推理
- OmniPart 不是只用 `K` 个 token 做全部 geometry flow。它有两个阶段：
  - Stage 1: autoregressive structure planning，用 causal / decoder-only transformer 生成 variable-length part bounding box sequence。这里是自回归，序列长度约等于 part 数。
  - Stage 2: structured part latent generation，用 spatially-conditioned rectified flow 同时生成所有 parts 的 structured latents。这里不是按 part box 自回归逐个生成，而是在 planned layout 条件下并行 denoise / flow。
- 因此，“K 个 part tokens/boxes”主要是结构规划和条件表达，不等于 flow 只在 `K` 个 token 上运行。第二阶段仍然有许多 TRELLIS structured latent voxels/tokens，只是这些 tokens 位于 sparse structured latent / part boxes 内，而不是 full dense 64³ categorical label grid。
- OmniPart 的计算开销降低来自表示和阶段拆分：
  - 全局结构用少量 part-level tokens 自回归规划；
  - 几何生成在 TRELLIS sparse structured latent 上做；
  - part position embedding / spatial conditioning 让 flow 知道每个 latent token 属于哪个 part 或整体。
- 对本项目的可迁移思想是：可以引入 part-level tokens 承担全局结构条件，但不能误解为“只用 K 个 token 就能直接输出 dense 64³ label”。如果本项目保留完整 dense label field，仍需要一个 dense decoder 把每个 voxel 映射到 `empty/part_id`。
- 可能的本项目结构借鉴：
  - `mask_token_labels` pooling 得到 `K` 个 part tokens；
  - dense voxel queries 用 position + surface condition 表示；
  - voxel queries cross-attend 到 part tokens / DINO tokens；
  - 避免或替代 full voxel self-attention，从而减少 `N^2`；
  - 最终仍对 `64^3` 全体 voxel 输出 labels。

### 暂定结论 / 决策
- 暂定结论：OmniPart 的自回归只用于 part layout / box planning，不是整个 geometry flow 都自回归。
- 暂定结论：OmniPart 的 K 个 part-level tokens/boxes 可以作为条件和全局结构先验，但不能直接替代 dense voxel-level prediction。
- 暂定方向：本项目若参考 OmniPart，应参考“part-level conditioning + 并行 dense/latent decoding”的分工，而不是把 dense completion 改成自回归逐 part 生成。

### 未决问题
- 本项目第一版是否需要显式 part planner，还是先使用现有 `mask_token_labels` pooling 得到 part tokens。
- Dense decoder 是否去掉 voxel self-attention，仅靠 voxel-to-part / voxel-to-image cross-attention 是否足够。
- 如果保留 full dense output，是否需要 window/local attention 来补充局部空间一致性。

### 后续动作
- 下一轮讨论具体 dense decoder 结构：无 voxel self-attention、window/local self-attention、或 part-token planner + decoder。

### 调研参考
- OmniPart abstract / project: autoregressive structure planning + spatially-conditioned rectified flow。
- OmniPart GitHub: fine-tunes TRELLIS `slat_flow_img_dit_L_64l8p2_fp16` style denoiser checkpoint。

## Round 7 - 聚焦功能代码与 OmniPart 开销解法

日期: 2026-05-06

### 讨论背景
- 用户明确：暂时不要考虑论文怎么写，先把功能和代码设计改正确。
- 用户认为：相对 OmniPart，本项目的差异在于有 surface-to-solid 补全，以及明确 dense voxel part class / part id 预测。
- 用户不理解 “label-independent slots vs 跨类别语义 class” 的含义。
- 用户强调：参考 OmniPart 的目标是解决 dense voxel 计算开销，而不是引入 SLAT 路线。

### 核心问题
- OmniPart 到底如何降低计算开销，是否能迁移到本项目 dense 64³ Part Flow。
- 本项目所谓“明确 class 预测”应如何理解。
- 如果不使用 SLAT、不使用 bbox/candidate 裁剪，OmniPart 还有哪些机制能启发代码设计。

### 观点与推理
- 用户对主差异的判断基本正确：本项目相对 OmniPart 的核心不同是：
  - 本项目有从 SS surface 到 dense solid voxel 的补全任务，输出 `[64,64,64]` 的 `empty/part_id` label field。
  - OmniPart 没有输出 dense 64³ part-label volume；它是先规划 part boxes，再在 TRELLIS structured latent space 里生成 part-specific geometry latent。
- “明确 class / label” 需要区分两层：
  - 当前代码预测的是每个 object 内部的明确 part slot / part id，例如 `0=empty, 1..K=该物体的 parts`，并通过 `part_info` 和 `mask_token_labels` 对齐。这是明确的 per-object part label。
  - “跨类别语义 class” 指全数据集统一 taxonomy，例如 slot/class 永远表示 `seat/back/leg/handle`。当前 variable-K 设计不要求 slot 1 在不同物体间语义相同；它是 label-independent slot，但每个样本内部仍然有明确 part id。用户若说“明确 class”指 per-object 明确 part id，则当前设计满足；若指全局语义 taxonomy，则当前代码还不是这个方向。
- OmniPart 解决 dense voxel 计算开销的方式不是优化 dense 64³ self-attention，而是改变问题表示：
  - 用 autoregressive planner 在 part-level box 序列上做全局结构推理，序列长度是 part 数而不是 voxel 数。
  - 用 TRELLIS sparse structured latent 做 geometry generation，避免 dense 64³ categorical label grid。
  - 使用 part boxes / spatial conditioning / voxel validity score 处理 planned layout 内的 geometry，而不是让所有 dense voxel 做全局 self-attention。
- 因此，如果本项目坚持 “SLAT 前、dense 64³ label field、无 bbox/candidate 裁剪”，OmniPart 不能直接解决当前 full dense voxel self-attention 的 `N^2` 开销。
- 但 OmniPart 可迁移的原则是：把全局 part-structure reasoning 从 voxel-level 移到 part-level tokens。对应到本项目，可能的代码设计是：
  - 保留 full dense 64³ 输出，不裁剪 voxel。
  - 先从 DINO/mask/SS condition 得到 `K` 个 part-level tokens 或 coarse part layout tokens。
  - Dense voxel decoder 不再做全局 voxel self-attention，而主要做 voxel-to-part / voxel-to-image cross-attention，或 window/local voxel attention。
  - 这样全局语义由少量 part tokens 承担，dense voxel 只做局部/条件解码，避免 `262144^2` 的全局 voxel attention。
- 这不是照搬 OmniPart，而是吸收它的“part-level planning/conditioning”思想来重构本项目的 Part Flow model。

### 暂定结论 / 决策
- 决策：当前讨论优先服务功能和代码设计，不再把论文表述作为主要约束。
- 暂定接受：本项目应继续保持 SLAT 前 dense 64³ part-label completion，不采用 OmniPart 的 SLAT part generation。
- 暂定接受：OmniPart 对计算开销最有价值的参考不是 bbox 裁剪，而是 part-level tokens/planner 承担全局结构推理，从而减少或替代 full voxel self-attention。
- 暂定结论: 若第一版坚持论文正确 full dense 输出，最可能的代码方向是 full dense output + 非全局 voxel attention，而不是 full dense output + current global self-attention。

### 未决问题
- 第一版是否仍先跑当前 global self-attention 的 exact dense benchmark，还是直接设计去掉/替换 voxel self-attention 的 RF model。
- 是否需要引入 OmniPart-style causal planner 预测 part-level tokens，还是先用现有 `mask_token_labels` pooling 得到 part tokens。
- Dense voxel decoder 是否可以完全去掉 voxel self-attention，只保留 positional embedding + surface embedding + cross-attention to image/part tokens；还是必须加 window/local attention。

### 后续动作
- 下一轮讨论聚焦模型结构选择：A. 当前模型 exact dense benchmark；B. 去掉全局 voxel self-attn 的 dense decoder；C. window/local attention dense decoder；D. OmniPart-style part planner + dense decoder。
- 继续保持所有路线输出完整 `[64,64,64]` `empty/part_id`，不做 bbox/candidate/膨胀裁剪。

### 调研参考
- OmniPart project page: https://omnipart.github.io/
- OmniPart GitHub README: https://github.com/HKU-MMLab/OmniPart

## Round 6 - OmniPart 的 causal transformer 与本项目位置

日期: 2026-05-06

### 讨论背景
- 用户指出：本项目的 `part_flow` 在 SLAT 之前，因此不应直接使用 SLAT 路线。
- 用户认为：本项目和 OmniPart 都使用 TRELLIS 的 SS / sparse structure 相关 latent，但本项目把 SS decode 到 64³ 后，在 dense label grid 上做 Part Flow。
- 用户追问：OmniPart 中的 causal transformer 是做什么的；OmniPart 是否预测具体 part class；是否可以参考它的做法。

### 核心问题
- 本项目是否能直接采用 OmniPart 的 SLAT/structured latent generation 路线。
- OmniPart 的 autoregressive / causal transformer 在 pipeline 中承担什么角色。
- OmniPart 的 part 表示是否是语义 class，还是 label-independent part instances。

### 观点与推理
- 用户判断基本正确：本项目当前 pipeline 是 `SS Flow/Decoder -> Part Flow -> SLat Flow/Decoder`。`part_flow` 位于 SLAT 之前，任务是在 SS surface condition 之后生成 dense 64³ `empty/part_id` label volume。直接改成 SLAT 会把任务改到 TRELLIS second-stage structured latent generation，偏离当前设计。
- 本项目和 OmniPart 的共同点是都依赖 TRELLIS 的 sparse structure / active voxel 先验。差别是：
  - OmniPart 仍然在 TRELLIS structured latent space 里生成 part-specific latent codes，最终依赖 TRELLIS decoder 产出 mesh/3DGS/NeRF。
  - 本项目把 SS decode 成 64³ voxel 坐标后，在 explicit dense categorical label grid 上预测 solid part labels，再进入后续 SLAT stage。
- OmniPart 的 causal transformer / decoder-only transformer 是第一阶段 `Controllable Structure Planning`：把一组 3D part bounding boxes tokenized 成序列，用 autoregressive next-token prediction 生成 variable-length part boxes。它解决的是“有多少 part、每个 part 的大致空间 box 在哪里”的规划问题，不是对每个 voxel 做 part label completion。
- OmniPart 的 part boxes 是 label-independent：论文明确说 2D masks 和 3D boxes 没有一对一对应，并采用 non-one-to-one correspondence bounding boxes 来避免显式匹配；因此它不是在预测具体语义 class，如 `seat/back/leg`，而是在预测可控 part instances / boxes。
- OmniPart 可以参考的部分：
  - 使用 2D mask label embedding 作为 part-aware conditioning，这和本项目 `mask_token_labels` 的方向一致。
  - 使用 autoregressive planner 产生 variable-length part structures，这可作为后续高层 part prior，但不是第一版 dense completion 的必要条件。
  - 使用 part position embedding 区分各 part，这可启发本项目对 part tokens / slot embeddings 的设计。
  - 使用 validity score 丢弃 box 内冗余 voxels，这是 OmniPart 的 box 初始化清理机制；但它属于基于 box 的 structured latent generation，不适合作为本项目第一版 exact dense inference 的默认策略。

### 暂定结论 / 决策
- 决策：第一版不采用 SLAT/OmniPart second-stage latent generation 路线，因为本项目 `part_flow` 明确在 SLAT 前，目标是 dense 64³ categorical part label completion。
- 暂定接受：OmniPart 的 first-stage causal transformer 可作为后续“part structure prior / part count / coarse layout”参考，但不进入第一版 exact dense Part Flow。
- 暂定接受：OmniPart 的 label-independent part formulation 与本项目的 variable-K part slots 是兼容的；本项目不需要预测具体 semantic class 才能成立。

### 未决问题
- 是否需要在论文里明确写出：我们不是生成 part-specific SLAT，而是在 SLAT 前预测 dense solid part label field。
- 是否要后续引入 OmniPart-style autoregressive part layout planner，作为 `part_flow` 的额外 coarse prior。
- 本项目 `mask_token_labels` 的 part index 是否应继续保持 label-independent slots，而不是映射到跨类别语义 class。

### 后续动作
- 第一版仍保持 full dense 64³ exact inference + RF/categorical-simplex 设计讨论。
- 后续可以单独讨论是否引入 OmniPart-style planner，但必须和 dense completion 主线解耦。

### 调研参考
- OmniPart project page: https://omnipart.github.io/
- OmniPart arXiv/html: https://arxiv.org/abs/2507.06165 / https://ar5iv.labs.arxiv.org/html/2507.06165v1

## Round 5 - Rectified Flow 与 SLAT 路线

日期: 2026-05-06

### 讨论背景
- 用户同意第一版枚举完整 64³，不接受 candidate/bbox/膨胀等近似裁剪。
- 用户提出：第一版是否直接使用标准 Rectified Flow，把 Fisher/Gumbel 放到后续 ablation；这样似乎能接近 residual FM / DPM 类加速路线。
- 用户追问：如果用 SLAT 模式是否还需要训练 VAE，是否会降低分辨率；OmniPart 是如何实现的。

### 核心问题
- 标准 Rectified Flow 是否适合作为 Part Flow 第一版主路线。
- Fisher/Gumbel categorical bridge 与 standard RF 的取舍是什么。
- SLAT/TRELLIS/OmniPart 路线和本项目 dense 64³ part-label flow 是否是同一个问题。

### 观点与推理
- 标准 Rectified Flow 的优势是路径简单、工程成熟、天然支持少步采样实验：训练目标可以是从噪声/随机 simplex-like 状态到 one-hot label volume 的 velocity / endpoint prediction，推理用 Euler 少步推进。它更接近 TRELLIS/OmniPart 使用的 continuous rectified flow 家族，也便于后续尝试 reflow、few-step、distillation。
- 但 standard RF 原生是连续欧氏空间上的 flow；本任务的输出是 categorical label。若直接在 one-hot/probability simplex 上做 continuous RF，需要明确投影/归一化/argmax 的契约，否则模型可能产生不在 simplex 上的中间状态或把类别边界变模糊。Fisher/Gumbel 的价值就在于它们更“离散正确”：路径、mask、valid slots 都围绕 categorical simplex 设计。
- 因此一个合理论文路线是：主方法用 Rectified Flow over padded categorical-simplex endpoints，强调“surface-conditioned dense categorical RF”；Fisher/Gumbel 作为更严格 categorical bridge 的 ablation。这样叙事靠近 TRELLIS/OmniPart 的 RF 传统，同时保留离散桥作为技术对照。
- 需要注意：换 standard RF 不是单纯换推理 solver，而是训练目标/bridge 都要换；现有 Fisher/Gumbel checkpoint 不能直接用 DPM-Solver 或 residual FM 加速。
- DPM-Solver 是 diffusion ODE/SDE 噪声日程的专用 solver，不等于所有 flow 都能直接套；Rectified Flow 可以做 few-step Euler/Heun、reflow/distillation，但不是自动获得 DPM 的理论适配。
- SLAT 路线：TRELLIS 的 SLAT 是 sparse active voxel + continuous local latent，可 decode 到 mesh/3DGS/RF。若我们要把 dense part labels 放进 SLAT latent 空间，通常需要一个能把 part label/solid volume 编码进 latent 并解码回 part labels 的 VAE/autoencoder，或者训练一个新的 label decoder。否则只是借用 TRELLIS 的 surface latent，不会自然得到 dense internal part label。
- SLAT 不一定简单等于“降低分辨率”，但它会改变状态空间：从 dense 64³ categorical volume 变成 sparse surface-active continuous latent。对我们的内部 solid voxel completion 叙事来说，这会弱化或绕开 dense internal label 输出。
- OmniPart 的做法不是训练一个 dense part-label voxel VAE。公开描述显示它 built upon TRELLIS 的 spatially structured sparse voxel latent space：先用 autoregressive planner 预测 part bounding boxes，再在 TRELLIS 预训练 holistic 3D generator / structured latent 上 fine-tune 一个 spatially-conditioned rectified flow 来生成 part-specific structured latents；还带 voxel validity score 来丢弃 box 内冗余 voxels。也就是说，OmniPart 借用了 TRELLIS 的 SLAT/decoder 资产，而不是在 dense 64³ label grid 上做 categorical solid completion。

### 暂定结论 / 决策
- 暂定接受：如果要更贴近 TRELLIS/OmniPart 和 few-step 加速路线，可以把 standard Rectified Flow 作为第一版主方法候选。
- 暂定保留：Fisher/Gumbel 不删除，作为 categorical bridge ablation 或后续更离散正确的版本。
- 暂定拒绝：第一版不走 SLAT/VAE 路线，因为它会把问题改成 continuous sparse latent generation，并需要额外 encoder/decoder 或依赖 TRELLIS decoder，不再是当前 dense 64³ part-label completion 的最短路径。

### 未决问题
- standard RF 的状态表示应选择哪种：直接 continuous one-hot/probability vector、logits、还是 simplex projection 后的概率。
- RF 训练目标是 velocity regression、endpoint prediction，还是沿用 endpoint logits + RF path。
- 主文方法是否表述为 “Rectified Flow over categorical-simplex labels”，Fisher/Gumbel 作为 ablation。
- 是否需要同时实现 exact dense inference benchmark 与 `num_steps=20/10/5/1` 的速度-质量曲线。

### 后续动作
- 设计 standard RF bridge 的训练/推理契约，不先改代码。
- 对比 standard RF、Fisher、Gumbel 三者在论文叙事中的位置。
- 继续确认 dense 64³ full attention 是否需要架构级 window/local attention 作为第二阶段。

### 调研参考
- OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion, arXiv:2507.06165。
- TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation, arXiv:2412.01506。
- Flow Matching for Generative Modeling, Rectified Flow, DPM-Solver, Progressive Distillation, Consistency Models。

## Round 4 - 不接受候选裁剪后的论文正确路径

日期: 2026-05-06

### 讨论背景
- 用户明确：膨胀、candidate/bbox 这类推理域裁剪非常 trick，第一版需要论文正确版本，不接受其他妥协。
- 用户提出另一个可能方向：既然 SS 输出已经是表面，是否只需要考虑 SS surface 内部的 voxel，而不是整个 64³ grid。
- 用户要求调研现成论文，看是否有更原则性的方案解决 dense 3D attention 和多步 ODE 推理过慢的问题。

### 核心问题
- “只考虑 SS surface 内部”是否仍然符合论文正确的 surface-to-solid 设计。
- 如果第一版坚持 exact dense，不用膨胀/candidate 裁剪，现有论文通常如何处理 dense 3D 的 attention 开销。
- 多步 flow / diffusion 类采样慢的问题有没有成熟加速路线。

### 观点与推理
- 只考虑 SS surface 内部不是简单的膨胀 trick，但它会引入一个确定性几何前处理：先从 surface shell 求 inside/support，再只在 inside/support 上跑 Part Flow。这个选择是否“论文正确”取决于论文贡献定义：
  - 如果贡献是“由 SS surface 补全 solid occupancy + part label”，那么提前求 inside/support 会把 occupancy completion 的一部分从模型挪到几何算法里，叙事会变弱。
  - 如果贡献改成“SS surface 已给出 object support，Part Flow 负责内部 part label completion”，那 inside/support mask 是合理输入，但论文要诚实地把 occupancy support 和 part assignment 分开。
- 对当前主叙事“surface-to-solid categorical completion”而言，最干净的第一版仍是 full dense 64³ exact：所有 voxel 都参与采样，`is_on_surface` 只是 surface evidence，模型自己输出 empty/part label。
- 论文调研显示，解决 dense 3D token 过多的主流方式不是全局 dense attention，而是：
  - TRELLIS/SLAT：使用 sparse structure，只在 intersect object surface 的 active voxels 上建 latent，避免全 64³ dense token。
  - Swin Transformer：用 shifted window self-attention，把高分辨率视觉 token 的 attention 从全局二次复杂度降到近似线性复杂度，并通过 shifted windows 做跨窗口信息传播。
  - Stratified Transformer / Swin3D / SWFormer / DSVT：在 3D 点云或 sparse voxel 上使用 window/local/sparse attention，并通过 shifted window、stratified keys、multi-scale fusion 或 sparse window batching 扩大感受野。
  - XCube：用 sparse voxel hierarchy / VDB / coarse-to-fine latent diffusion 来生成高分辨率 3D voxel，避免一次性 dense 全局建模。
- 对多步 ODE / diffusion 推理慢的问题，常见方案是：
  - 更高阶/专用 solver，例如 DPM-Solver 将 diffusion ODE 采样压到约 10-20 次网络评估；但这类方法是 diffusion ODE 专用，不能直接无脑套到当前 categorical Fisher/Gumbel bridge。
  - Rectified Flow / straight flow 方向，让路径更直，从而减少 Euler steps，甚至支持 very few-step / one-step 近似。
  - Progressive distillation / Consistency Models，把多步 teacher 蒸馏成少步/一步 student；这是成熟加速路线，但需要额外训练。
- 因此，若第一版不接受推理域裁剪，最合理路线是：先实现 exact full dense benchmark，然后用 profiler/benchmark 量出实际瓶颈；如果不可接受，下一阶段不是加 heuristic candidate，而是改 Part Flow 架构或采样训练目标。

### 暂定结论 / 决策
- 决策：第一版 dense completion 设计不采用膨胀、bbox、candidate 裁剪作为默认路径。
- 暂定决策：第一版以 full dense 64³ exact inference 作为论文正确版本和 benchmark 路径。
- 暂定结论: “只考虑 SS surface 内部”可以作为后续可讨论的 support-mask 版本，但它会改变贡献叙事，暂不作为第一版默认。

### 未决问题
- 是否要在第一版 exact dense 后立刻规划 window/local attention 版 Part Flow，以便训练和推理都消除 `N^2`。
- 是否要把 few-step sampling 作为单独实验：比较 `num_steps=20/10/5` 的速度和质量。
- 是否需要在论文中明确区分 “solid occupancy completion” 和 “part label completion”，避免未来引入 inside/support mask 时叙事冲突。

### 后续动作
- 先实现或规划 full dense exact inference，并加显存/时间 benchmark。
- Benchmark 后再决定长期路线：window/local attention、hierarchical coarse-to-fine，或 few-step/distilled sampler。
- 在文档中保持 exact dense 与任何 approximate/support-mask 版本的命名隔离。

### 调研参考
- TRELLIS / Structured 3D Latents: sparse active surface voxel latent，避免 dense 全局 3D token。
- Swin Transformer: shifted window attention for high-resolution dense prediction。
- Stratified Transformer / Swin3D / SWFormer / DSVT: 3D local/window/sparse attention。
- XCube: sparse voxel hierarchy + coarse-to-fine 3D generation。
- DPM-Solver, Rectified Flow, Progressive Distillation, Consistency Models: 多步 generative ODE/diffusion 的采样加速路线。

## Round 3 - `is_on_surface` 语义与 attention/ODE 开销

日期: 2026-05-06

### 讨论背景
- 用户追问：在当前设计路径下，把 SS coords scatter 成 `is_on_surface=1`、其余 voxel 设为 0 是否真的有意义。
- 用户主要担心：真正贵的是模型内部 attention 和 ODE 多步循环，担心 dense 推理时间太长。

### 核心问题
- `is_on_surface` 是不是足够表达 TRELLIS SS 的几何条件。
- Dense 64³ 推理在当前 `PartFlowPredictor` 架构下是否可承受。
- 一般如何解决 full attention + 多步 ODE 带来的推理成本。

### 观点与推理
- `SS coords -> is_on_surface=1` 是有意义的，但它的角色只是条件信号：告诉模型哪些 dense voxel 是观测到的 surface evidence。completion 的关键不是这些 surface voxel 自身，而是其余 `is_on_surface=0` 的 dense voxels 仍然参与采样，让模型根据 surface shell、DINO tokens、2D mask part identity 推断内部 solid labels。
- 如果只对 SS sparse coords 采样，那么 `is_on_surface=1` 退化成“所有输入点都是 surface”，模型没有机会输出内部 voxel；这才是当前 wrapper 不是真正 dense completion 的原因。
- 但当前 `PartFlowPredictor` 有 voxel self-attention；full dense `N=262144` 会触发 `O(N^2)` attention。即使用 xformers/flash-attn 降低显存，计算量仍然非常大；再乘以 `num_steps`，推理时间会明显变长。
- 常见解决思路不是“硬跑全局 262k attention”，而是改变采样域或改变 attention 结构：只在候选区域推理、用窗口/局部/sparse attention、分层 coarse-to-fine、减少 ODE step/用蒸馏或更快 solver、缓存条件侧 token 计算。
- 对当前代码而言，最容易落地的是推理域裁剪和减少输出/步数；最根本的改法是把 voxel self-attention 改成 window/local attention 或训练一个与 chunk/candidate 推理一致的新架构。

### 暂定结论 / 决策
- 暂定接受：`is_on_surface` 仍然应该作为 dense inference 的 surface condition 输入；它本身有意义，但必须配合 dense/off-surface voxels 一起采样才体现 completion。
- 暂定接受：第一版如果沿用当前模型结构，full dense exact mode 可能主要用于 correctness benchmark，不一定适合作为默认日常推理路径。
- 暂定结论: 还未决定是否先实现 exact full dense，还是直接设计 candidate/window 近似推理路径。

### 未决问题
- 是否愿意接受 candidate/bbox 推理作为第一版默认模式，并把 full dense exact 作为 benchmark/ablation。
- 是否后续要改模型结构，使训练和推理都采用 window/local attention，从根上解决 `N^2`。
- 推理 `num_steps` 能否从训练/eval 的 20 降到 5/10 做质量-速度折中。

### 后续动作
- 下一轮讨论对比三个可行路线：A. full dense exact benchmark；B. SS bbox/candidate dense approximate；C. 改 Part Flow 架构为 window/local attention。
- 明确第一版代码要优先保证论文叙事正确，还是优先保证 4090 上日常推理可用。

## Round 2 - SS surface 语义与 64³ 计算开销

日期: 2026-05-06

### 讨论背景
- 用户确认：根据 TRELLIS 代码和论文，SS 输出应理解为表面 voxel；当前主要担心不是语义，而是 dense 64³ 枚举带来的计算开销。
- 代码核对：`trellis/pipelines/trellis_image_to_3d.py::sample_sparse_structure` 用 `decoder(z_s)>0` 取 sparse structure coords；TRELLIS 项目页/论文描述的 SLAT active voxels 是 intersect object surface 的 sparse voxel structure。
- Part Flow 代码核对：`PartFlowDataset` 训练时全枚举 `64^3=262144` voxel；`PartFlowPredictor` 每层包含 voxel self-attention、voxel->RGB cross-attention、voxel->part cross-attention。

### 核心问题
- 如果推理直接使用 full dense 64³，语义最正确，但每个 ODE step 都要对 `N=262144` voxel 做模型前向。
- 当前 `PartFlowDecoderLayer` 的 self-attention 是按样本 varlen full attention，不是局部窗口 attention；因此 full dense 的主要成本是 `O(N^2)` voxel self-attention，而不是保存 dense coords 或 hard labels。
- 需要决定第一版 dense inference 是做“精确训练契约对齐”，还是做“计算受控但有分布偏移”的工程近似。

### 观点与推理
- `64^3` 枚举本身很小：coords 约 262k 行，hard labels 约 2MB，`soft_probs [64,64,64,128]` 用 fp16 约 64MB。真正贵的是模型内部 attention 和 ODE 多步循环。
- Full dense one-shot 与训练契约最一致：`coords=dense_grid`，`is_on_surface` 由 SS coords scatter 得到，off-surface voxel 参与采样，输出完整 solid label volume。
- Chunked dense 可以降低峰值显存，但如果直接把 voxel 分块分别调用当前模型，self-attention 只能在 chunk 内发生，语义不再等价于训练时的全体 voxel attention；这是近似，不应伪装成 exact dense。
- Candidate/bbox dense 可以把 SS surface 的 padded bbox 或 flood-fill interior 作为候选区域，区域外直接设 empty。这能明显降低计算，但 completion 的几何空间被规则先验裁剪，也需要在文档中标明。

### 暂定结论 / 决策
- 暂定接受：SS decode coords 在 Part Flow 推理中可作为 surface mask 的来源。
- 暂定接受：当前需要优先围绕 full dense 与近似低开销 dense 的取舍继续讨论。
- 暂定结论: 还未决定具体实现路径。

### 未决问题
- 第一版代码是否必须 exact full dense，还是允许增加一个显式命名的近似模式。
- 如果做近似模式，是选择 chunked full-grid、bbox/candidate-grid，还是先做 benchmark 再决定。
- 是否默认保存 `soft_probs`；如果用户只关心可视化和后续 pipeline，默认保存 hard labels 可降低 IO 和内存压力。

### 后续动作
- 比较三种实现方案：full dense exact、chunked dense approximate、candidate/bbox dense approximate。
- 在修改代码前，先确定第一版 dense inference 的默认模式和允许的显存/耗时预算。

## Round 1 - 讨论启动

日期: 2026-05-06

### 讨论背景
- 训练侧 `PartFlowDataset` 已经是 dense 64³ 任务：`coords=[262144,3]` 全体 voxel，`is_on_surface=[262144]` 来自 `surface.npy`，监督目标是 `part_labels_solid_64.npy`。
- 当前推理侧 `TRELLIS-arts/inference.py::run_part_flow` 仍接收 TRELLIS SS decode 后的 sparse `coords`，并把这些 sparse coords 的 `is_on_surface` 全部设为 1。
- 因此当前 wrapper 的行为更接近“给 SS 输出的 occupied coords 打 part label”，还不是真正“由 SS surface condition 补全内部 solid voxels”。

### 核心问题
- 推理阶段是否应该枚举 dense 64³ coords，并用 TRELLIS SS 输出构造 `is_on_surface` 条件，再让 Part Flow 对全体 voxel 采样得到 solid labeled volume。
- 如何在 48G 4090 可接受显存内做 dense 64³ 推理，同时保持训练/推理契约一致。
- SS decode 输出的 sparse coords 在推理中应被解释为 surface shell、occupied shell，还是粗 occupancy evidence；这个语义会决定 `is_on_surface` 的构造方式。

### 观点与推理
- 直接 dense 全枚举最贴合训练契约，也最能支撑论文叙事中的 surface-to-solid completion。
- 当前 sparse wrapper 虽然省显存，但会丢掉 off-surface voxel 的采样机会，模型无法输出内部 voxel label，因此不能证明 dense completion 能力。
- 如果 SS 输出不是严格一层外表面，而是粗 occupancy 或含内部点，那么简单设为 `is_on_surface=1` 会改变训练时 `surface.npy` 的条件分布；可能需要先定义“SS evidence mask”和“surface condition”的关系。
- Dense 64³ 的 voxel 数是 262144；如果一次性送入模型显存不可控，可以考虑按 chunk 做 sampling，但需要确认 `flow_sample` / model 是否允许同一样本的 voxel_layout 被拆块而不改变条件语义。

### 暂定结论 / 决策
- 暂定结论: 暂无，仍需讨论。

### 未决问题
- 推理输入中的 SS coords 应该被当作外表面 mask，还是 coarse occupancy mask。
- Dense 推理优先目标是论文叙事正确，还是先做一个低显存可运行 wrapper。
- 是否需要保留 sparse labeling wrapper 作为 debug/ablation，还是直接把 `run_part_flow` 改成 dense contract。
- Dense 输出是否仍保存 `soft_probs.npz` 的 `[64,64,64,k_max]`，还是默认只保存 hard labels，避免大文件和显存/内存压力。

### 后续动作
- 先讨论 dense 推理的输入语义和输出契约。
- 再比较 2-3 个实现方案，包括 full dense、chunked dense、以及保留 sparse debug path 的方案。
- 等设计确认后，再进入代码修改计划。
