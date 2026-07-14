# arts-reconstruction docs

文档按生命周期分类。新人入门请按 `specs/` → `designs/` → `explanation/` 顺序阅读。

## 一级目录

- `specs/` — 数据契约 / API 规范
  - `specs/code_arch.txt` — 用户原始架构蓝图（v1）
  - `specs/data/part_labels_solid_64.md` — Solid voxel 数据契约
  - `specs/development_documentation.md` — 默认开发文档与 plan 触发规则
  - `specs/joint_prompt_seg_local_refinement.md` — Joint 局部建模、边界监督与兼容规范
- `designs/` — Active 设计文档
  - `designs/aaai_pact_vs_my_pipeline.md` — 与 AAAI PACT 对比
  - `designs/part_flow_dinov2_mask_conditioning.md` — Part Flow DINOv2+mask 设计
  - `designs/part_predictor_dinov2_multiview_fusion.md` — Part Predictor 多视角融合
  - `designs/part_predictor_comparison.md` — Part Flow vs Part Predictor 三方案对比
- `reviews/` — 当前 milestone 的 peer review 记录（v0.1.0 历史 review 已归档到 `archive/`，本目录待 v1.0+ 新 review 入驻）
- `runbooks/` — 生产运行手册
  - `runbooks/ss_flow_art_h200_production.md` — SS Flow Art H200 生产
  - `runbooks/ss_flow_global_z_4view.md` — 4view tokens 到全局 z_global SS Flow 微调
  - `runbooks/slat_flow_art_h200.md` — SLat Flow Art H200
- `explanation/` — 教程性文档（保留单数命名以避免 link rot）
  - `explanation/01-overview.md` / `02-input-output.md` / `03-algorithm-flow.md` / `04-fisher-flow-matching.md`
- `figures/` — 图资产（PNG / SVG / GIF / PDF）
  - `figures/sources/` — 图生成器（`.py` / `.jsx` / `.html`）
- `summaries/` — Phase 完成总结
  - `summaries/phase09_refactor.md` — Phase 09 重构总结
- `archive/` — 历史 / 已被取代的文档，按 phase 编号查
  - `archive/phase02/` / `archive/phase03/` / `archive/phase04/` / `archive/phase07/` / `archive/phase08/`
  - `archive/v0.1.0_milestone.md` — v0.1.0 顶层 milestone（dev summary）
  - `archive/v0.1.0_milestone_peer_review.md` — v0.1.0 顶层 milestone（peer review）

## Active 快捷入口

- 新人入门：`explanation/01-overview.md` → `specs/code_arch.txt` → `designs/part_predictor_comparison.md`
- 数据契约：`specs/data/part_labels_solid_64.md`
- 最近重构：`summaries/phase09_refactor.md`
- 生产运行：`runbooks/ss_flow_art_h200_production.md` / `runbooks/ss_flow_global_z_4view.md` / `runbooks/slat_flow_art_h200.md`

## 约定

- 顶层 `docs/` 不留单独 `.md` 文件；本 README 是唯一例外。
- 命名风格：复数英文目录名，唯一例外是 `explanation/`。
- archive 内不再分类型；单 phase 子目录扁平存放。历史追溯用 `git log docs/archive/phaseXX/`。
- 默认开发过程只维护相关 `docs/specs/` 规范和 `TRELLIS-arts/code_update/` 更新日志。
- 仅当用户明确要求“计划”或 `plan` 时创建或更新计划文档；默认不生成 brainstorming、milestone、UAT 或 summary 文档。
