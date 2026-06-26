# Part Flow 20 样例过拟合 Checklist

> 最后更新: 2026-05-06
> 范围: 用户后续上传 20 个案例数据后，用于验证 dense Part Flow 能否在小数据上过拟合。

## 目标

- 使用 20 个案例数据验证新的 dense Part Flow 设计是否能过拟合。
- 重点验证：
  - full dense `64^3` surface-to-solid completion 路径正确。
  - `mask_token_labels` 和 `num_parts=K+1` 契约没有丢。
  - `is_on_surface` 是 SS surface condition，不是 solid target 泄漏。
  - Fisher endpoint logits 训练能在小数据上快速降低 loss。

## 数据准备

- [ ] 开发机已拉取最新代码。
- [ ] 20 个案例数据已上传到开发机或 vePFS。
- [ ] `data_root` 指向 20 个案例数据目录。
- [ ] 每个样例都有 `part_info.json`。
- [ ] 每个样例都有 `surface.npy`。
- [ ] 每个样例都有 `part_labels_solid_64.npy`。
- [ ] 每个样例都有对应 DINO condition tokens。
- [ ] 每个样例都有 `mask_token_labels` 所需的 mask 输入，或明确走 smoke/no-mask 行为。

## 训练配置

- [ ] 使用 `flow.type: fisher`。
- [ ] 使用 `flow.dirichlet_alpha: 1.0`。
- [ ] 使用 `model.num_layers: 4`。
- [ ] 使用 `model.condition_tokens_per_view: 3`。
- [ ] 使用 `model.voxel_chunk_size` 仅控制 eval/inference chunking。
- [ ] `model.use_gradient_checkpointing: false` 先默认关闭；若 H20 显存不足再打开。
- [ ] `training.batch_size: 1`。
- [ ] `training.grad_accum_steps` 根据显存设置。
- [ ] `training.eval_ode_steps` 先用 `10`，必要时对比 `20/10/5`。

## 过拟合观察指标

- [ ] `loss` 持续下降。
- [ ] `endpoint_acc` 持续上升。
- [ ] `part_acc` 不长期接近 0。
- [ ] `empty_acc` 不单独虚高掩盖 part 失败。
- [ ] eval `mIoU` 在 20 样例上明显上升。
- [ ] hard label volume 不全是 empty。
- [ ] hard label volume 不只在 SS surface 上有 part，内部也有 solid completion。
- [ ] padding slots 没有被预测出来。

## 最小通过标准

- [ ] 训练可以跑完至少一个短 overfit run，无 crash。
- [ ] 单样例或 20 样例训练集上的 endpoint accuracy 明显超过随机/empty baseline。
- [ ] 可视化或统计显示内部 voxels 被补全。
- [ ] 如果不能过拟合，必须记录失败模式：
  - [ ] loss 不下降。
  - [ ] 全 empty。
  - [ ] 只预测 surface。
  - [ ] part slot 混乱。
  - [ ] mask/num_parts contract 错。
  - [ ] OOM 或速度不可接受。

## 运行记录

### Round 1 - Checklist 初始化

日期: 2026-05-06

- 状态: 等待 20 个案例数据上传。
- 代码版本: 待填写。
- 数据路径: 待填写。
- 训练命令: 待填写。
- 结果: 待填写。

