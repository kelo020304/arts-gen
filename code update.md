# Code Update: EE-eval Snapshot Import

> 最后更新: 2026-07-14
> 范围: TOS EE-eval 代码快照导入、合并、契约测试同步与 GitHub 发布。

## Round 1 - 导入并验证 20260714T084329Z 快照

日期: 2026-07-14

### 改动摘要

- 从 TOS 下载时间戳快照及两个 SHA256 文件，本地计算值与发布值 `83e20687323326f3a7592a521a4f0655b6487392d4212e227bc7c7ac742a2f27` 一致。
- 创建独立快照提交并通过非快进 merge 合入，保留上一版快照历史，不删除发布包中缺失的既有 tracked 路径。
- 导入 103 个文件的有效 Git 变更，主要覆盖 kinematic agent、分阶段重建、fridge 3DGS 工作台、promptable part segmentation 训练/推理及对应测试和图像资产。
- 排除 `.claude/` 与快照内 `AGENTS.md`；尊重 `.gitignore`，未强制提交本地依赖源码和 GPU 动态库链接。
- 修正 3 个测试文件，使断言与 `arts_gen_kin_agent_v17`、严格 checkpoint 解析、缓存指纹和缺少 `manifold3d` 时的显式近似碰撞审计契约一致。

### 改动文件

- `post_process/kinematic_solver/`: 新增 kinematic agent、碰撞审计、范围先验、导出与 benchmark 实现及测试。
- `scripts/inference/reconstruct_stages.py`: 新增分阶段重建编排。
- `workbenches/fridge_3dgs/`: 扩展工作台 API、组件/kin-agent/voxel viewer 与回归测试。
- `TRELLIS-arts/trellis/models/part_seg/promptable_latent_seg.py`: 导入局部 refinement 与 joint query 相关更新。
- `scripts/train/part_promptable_seg/`: 导入 joint loss、评估、局部 refinement launcher 与数据拼接更新。
- `tests/kinematic_solver/test_decoded_collision_audit.py`: 同时验证精确 Manifold 路径和缺失后端时的显式降级路径。
- `tests/kinematic_solver/test_kin_agent.py`: 将 bundle 格式断言同步到 `arts_gen_kin_agent_v17`。
- `workbenches/fridge_3dgs/tests/test_app.py`: 补齐缓存指纹和严格 checkpoint resolver 测试隔离。
- `docs/specs/2026-07-14-ee-eval-snapshot-import-spec.md`: 记录本次导入、过滤、合并和验收契约。

### 设计意图

- 将发布快照作为可追溯上游提交导入，同时保持当前仓库用户工作区与本地代理配置不受影响。
- 对可选碰撞后端采取显式降级测试：缺少 `manifold3d` 时必须返回低置信度并要求 review，不能伪装成精确无碰撞。
- 保持生产代码来自给定快照；本地追加改动仅限修复快照内陈旧测试契约和仓库文档。

### 行为契约

- 改动前: `origin/main` 仅包含初始提交，本地 `main` 停留在 20260713 EE-eval 快照。
- 改动后: Git 历史包含 20260714 快照导入提交、非快进合并提交和验证文档提交；GitHub `origin/main` 更新到最终提交。
- 改动前: 部分测试仍断言 `v16`、旧缓存字段和无需真实 checkpoint 目录的宽松行为。
- 改动后: 测试严格匹配 `v17`、完整缓存指纹及 fail-loud checkpoint resolver；可选精确碰撞后端缺失时验证 `approximate_*` 降级。

### 验证

- `sha256sum /tmp/arts-gen-ee-eval-20260714T084329Z/arts-gen-code-ee-eval-20260714T084329Z.tar.gz`: 与发布 SHA256 一致。
- 归档路径检查: 4,621 个条目，单一 `arts-gen/` 根目录，无绝对路径或 `..` 路径穿越。
- `detect_changes(scope="staged")`: 175 个变更符号、17 条受影响执行流，整体风险 `critical`，范围与预期模块一致。
- 定向 `pytest`: `128 passed, 1 skipped, 11 warnings, 5 subtests passed`；跳过项为本机未安装 `manifold3d` 的精确 collision feedback 模块。
- `python -m py_compile` 覆盖所有本次变更 Python 文件: 通过。
- `node --check` 覆盖所有本次变更 JavaScript 文件: 通过。
- `bash -n` 覆盖所有本次变更 Shell 文件: 通过。
- JSON 解析检查: 7 个本次变更 JSON 文件全部通过。

### 风险 / 后续

- 未执行需要真实 GPU checkpoint、完整数据集或 MuJoCo 渲染环境的端到端推理。
- 本机未安装 `manifold3d`，精确碰撞相交体积路径未运行；近似路径和显式 review 契约已覆盖。
- 快照自带 `TRELLIS-arts/code_update/part_promptable_seg.md` 的历史 Markdown 表格含尾随空格；为保持发布内容原样未做无关格式化。
