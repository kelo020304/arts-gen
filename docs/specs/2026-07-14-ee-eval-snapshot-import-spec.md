# EE-eval Code Snapshot Import Spec

日期: 2026-07-14

## 目标

将已发布的 EE-eval 代码快照导入当前 Git 仓库，保留可审计的快照提交与非快进合并提交，并将验证后的结果推送到 GitHub `origin/main`。

## 输入

- 快照: `tos://robot-data-lab/jzh/art-gen/releases/code/arts-gen-code-ee-eval-20260714T084329Z.tar.gz`
- 最新别名: `tos://robot-data-lab/jzh/art-gen/releases/code/latest-ee-eval-code.tar.gz`
- SHA256: `83e20687323326f3a7592a521a4f0655b6487392d4212e227bc7c7ac742a2f27`

两个 `.sha256` 文件必须与本地归档计算结果一致后才能导入。

## 导入契约

- 以上一版 `snapshot/ee-eval-20260713T094907Z` 为基线，覆盖快照中新增或修改的非忽略文件。
- 不删除新快照中缺失的既有 tracked 路径，避免把发布包过滤掉的历史资产误判为源码删除。
- 不导入 `.claude/`。该目录包含本机权限配置和无许可证来源声明的 GSD 插件副本，与仓库工作流约束不兼容。
- 不导入快照内 `AGENTS.md`，保留当前仓库本地代理指令。
- 尊重仓库 `.gitignore`；`sam3d_cu118_src_deps/`、`libnvidia-gpucomp.so.*` 等本地依赖路径不强制加入 Git 历史。
- 当前工作区已有未提交文件不得被覆盖、暂存或提交。

## Git 历史

- 快照分支: `snapshot/ee-eval-20260714T084329Z`
- 合并分支: `merge/ee-eval-20260714T084329Z`
- 合并方式: `git merge --no-ff`
- 推送目标: `origin/main`

## 验收标准

- 归档 SHA256 与发布值一致，且归档不存在绝对路径或 `..` 路径穿越。
- GitNexus `detect_changes()` 在提交前完成，受影响执行流与工作台、训练/评估、promptable segmentation 范围一致。
- 定向回归测试通过；可选精确碰撞后端缺失时，测试必须验证显式 `approximate_*` 降级，不得静默当作精确通过。
- 变更 Python 文件通过 `py_compile`，JavaScript 通过 `node --check`，Shell 通过 `bash -n`，JSON 可解析。
- GitHub `origin/main` 指向最终验证提交。

## 非目标

- 不重写快照的生产实现或模型算法。
- 不安装新的运行时依赖、模型权重或 checkpoint。
- 不提交与本次快照同步无关的用户工作区改动。
