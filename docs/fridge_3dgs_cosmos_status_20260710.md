# 冰箱 3DGS / Cosmos3 工作状态总结

日期：2026-07-10

## 当前目标

当前工作不是继续训练，也不是重新打包数据集，而是在验证一条系统工作台链路：

1. 从冰箱 3DGS 场景中选择和导出视角。
2. 用 SAM3 辅助手工得到 2D mask，并允许人工修正。
3. 导出后续模型需要的 4 view RGB 和 4 view int label mask。
4. 对接 SS flow、SS decoder、part-prompt-seg，先跑通 whole / part generation 和可视化。
5. 额外验证 Cosmos3 能否从一张不完整黑色冰箱图补出类似 part-prompt-seg 数据使用的四视角外观图。

所有大文件、输出和模型权重都应放在 `/robot/data-lab/jzh/art-gen/...`。不要再使用旧挂载前缀路径。

## 3DGS 工作台现状

repo 入口：

- `/root/code/arts-gen/workbenches/fridge_3dgs/run_workbench.py`

前端和服务端文件：

- `/root/code/arts-gen/workbenches/fridge_3dgs/static/index.html`
- `/root/code/arts-gen/workbenches/fridge_3dgs/static/viewer.html`
- `/root/code/arts-gen/workbenches/fridge_3dgs/static/css/workbench.css`
- `/root/code/arts-gen/workbenches/fridge_3dgs/static/js/app.js`
- `/root/code/arts-gen/workbenches/fridge_3dgs/static/js/viewer.js`
- `/root/code/arts-gen/workbenches/fridge_3dgs/server/app.py`
- `/root/code/arts-gen/workbenches/fridge_3dgs/server/sam3_server.py`

默认服务配置：

- 端口：`7865`
- 数据根目录：`/robot/data-lab/jzh/art-gen`
- 3DGS PLY：`/root/code/arts-gen/data/point_cloud.ply`

当前工作台输出目录：

- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud`

已经存在的 3DGS 四张 RGB：

- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud/rgb/view_0.png`
- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud/rgb/view_1.png`
- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud/rgb/view_2.png`
- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud/rgb/view_3.png`

已经调过的默认姿态判断：

- 用户确认更接近可用默认的是 `x=0, y=225, z=180`。
- 之前全 0 度会找不到或看不到正确点云。
- 工作台需要继续保留手动视角选择、缩放、距离调整、capture 四视角的能力。

## SAM3 / mask 工作台现状

SAM3 目标：

- 不删除 SAM3。
- SAM3 用于辅助出 2D mask。
- 用户希望 SAM3 结果直接在中间大图上以彩色 overlay 实时可视化，而不是只输出黑白图再让用户选择。

已接入资源：

- SAM3 code：`/robot/data-lab/jzh/art-gen/local-deps/sam3`
- SAM3 wheelhouse：`/robot/data-lab/jzh/art-gen/local-deps/wheelhouse_py310`
- 官方 SAM3 权重：`/robot/data-lab/jzh/art-gen/weights/sam3/sam3.pt`

已有 SAM3 输出痕迹：

- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud/sam3/text_view_0.json`
- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud/sam3/points_view_0.json`
- `/robot/data-lab/jzh/art-gen/workbench/fridge_3dgs/fridge_point_cloud/sam3/server.log`

还需要继续做：

- 把 SAM3 candidate / selected mask 的彩色 overlay 做成主图实时显示。
- 明确 prompt 输入方式：text prompt、positive / negative points、box prompt。
- mask 编辑需要支持至少：选择 label、刷子增减、擦除、撤销、清空当前 label、保存。
- 导出时必须是 4 张 int label mask，`0` 是背景，正整数是跨视角稳定 part id。

## SS flow / part-prompt-seg 输入契约

当前需要遵守的关键点：

- `scripts/inference/reconstruct.py` 要求 exactly 4 images 和 exactly 4 masks。
- mask 必须是 `[H, W]` 的整数 label map。
- label 约定：`0 = background`，正整数表示 part id，并且四个视角中同一个 part 必须使用同一个 id。
- part-prompt-seg 读取的是 2D part mask prompt，不是普通 RGB 涂色图。
- 对 whole object，用户理解是正确的：整体 mask 可以是冰箱整体为正、背景为 0。
- 对 part-prompt-seg，mask 需要按部件分 label，例如门体、铰链/把手/控制面板等后续定义的 part id。

SS flow 当前需要特别注意：

- TRELLIS SS flow 路径里不是直接拿 RGB 算 DINO，而是读取官方 prenorm DINO token 文件。
- 期望 token 形状是 `[V, 1374, 1024]`，再按 4 个 `view_indices` 选出 conditioning tokens。
- 之前 DINO 用法出过错，后续不能随手换成在线模型或临时 DINO 提取方式。
- 如果用 Cosmos 生成的新四视角继续往 SS flow 走，需要补一套与训练/eval一致的 DINO token 生成流程，否则只具备 RGB 可视化，不是完整 reconstruct 输入。

## Cosmos3-Nano 环境现状

Cosmos3 本地 Diffusers 推理环境已经跑通。

路径：

- Cosmos3 code：`/home/mi/jzh/AAAI2027/arts-gen/submodules/cosmos3`
- deps：`/robot/data-lab/jzh/art-gen/local-deps/cosmos3`
- venv：`/robot/data-lab/jzh/art-gen/local-deps/cosmos3/venv-diffusers-cu128-py313`
- weights：`/robot/data-lab/jzh/art-gen/weights/cosmos3/Cosmos3-Nano`
- smoke 输出：`/robot/data-lab/jzh/art-gen/outputs/cosmos3_smoke`

已经验证：

- `Cosmos3OmniPipeline` import OK。
- `torch.cuda.is_available()` 为 True。
- 使用 CUDA 12.8 / cu128 路线。
- 本地权重加载，不访问 HuggingFace。
- 成功导出 mp4 和四张采样图。

初始 smoke test：

- 输出视频：`/robot/data-lab/jzh/art-gen/outputs/cosmos3_smoke/cosmos3_i2v_fridge_orbit_smoke.mp4`
- 320 x 192，189 frames，24 fps，约 7.875 秒。
- CUDA peak allocated 约 31.336 GiB。

## Cosmos3 黑色冰箱视角补全结果

目标：

- 从 3DGS 场景里黑色冰箱的一张不完整图，生成绕物体旋转视频。
- 从视频采样出四个外观视角，尽量接近 part-prompt-seg 数据里的四视角输入风格。
- 不生成开门、不生成内部结构。

第一版问题：

- 输入 crop 里包含售货柜和厨房背景。
- Cosmos 把售货柜、墙角等也当成场景内容参与生成。
- 四象限不够稳定，后续帧有局部放大和裁切。

第二版 clean16：

- 输入先裁掉大部分售货柜干扰，并把左侧背景压平。
- Cosmos 用 16 step、192 x 320 生成 189 帧 orbit video。
- 结果明显更像单个黑色双开门冰箱的 front / right / back / left 外观。

输出目录：

- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16`

视频：

- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/black_fridge_cosmos3_orbit_clean_16step_192x320.mp4`

带标签四象限：

- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/black_fridge_cosmos3_clean_four_quadrants.png`

无标签四象限：

- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/part_promptseg_like/four_quadrants_rgb_nolabel.png`

按 part-prompt-seg 风格整理的四张 RGB：

- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/part_promptseg_like/rgb/view_0.png`
- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/part_promptseg_like/rgb/view_3.png`
- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/part_promptseg_like/rgb/view_8.png`
- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/part_promptseg_like/rgb/view_11.png`

manifest：

- `/robot/data-lab/jzh/art-gen/outputs/cosmos3_fridge_black_orbit_clean16/part_promptseg_like/manifest.json`

运行信息：

- 16 inference steps
- 192 x 320
- 189 frames
- 24 fps
- 约 7.875 秒视频
- 生成耗时约 55 秒
- CUDA peak allocated 约 31.336 GiB

质量判断：

- 这版可作为“四视角外观补全可行性”样例。
- 它不是严格几何一致的 3D 重建视角。
- back / side 是 Cosmos 的合理补全，不是真实 3DGS 渲染或真实 GT。
- 若要进入 reconstruct 链路，还缺 mask 和 DINO token。

## 当前未完成项

1. 还没有把 Cosmos 四视角转成完整 reconstruct 输入。
2. 还没有为 Cosmos 四视角生成 exactly 4 张 int label mask。
3. 还没有为 Cosmos 四视角生成与训练/eval一致的 DINO tokens。
4. 还没有用这组输入跑 SS flow。
5. 还没有用这组输入跑 part-prompt-seg。
6. 还没有跑 whole / part generation 和可视化结果。
7. 关节、hinge、开门 viewer 只应留接口，不能作为第一阶段 blocker。

## 2026-07-10 四图直输兼容更新

工作台第一阶段现在支持两种并列的 RGB 来源：

- `3DGS Capture`：保留原有 SuperSplat 视角选择和逐槽 capture。
- `4 Images`：不加载或依赖 3DGS，直接选择 exactly 4 张图片并导入标准 `rgb/view_0.png` 到 `view_3.png`。

四图模式会按文件名自然排序，导入前可用左右按钮调整槽位。类似 Cosmos clean16 的
`view_0.png / view_3.png / view_8.png / view_11.png` 会记录原文件名和源视角 id，但后续统一使用
canonical 槽位 `0 / 1 / 2 / 3`，避免用 `8 / 11` 索引只有四行的 conditioning token。
前端接受 PNG、JPEG 和 WebP；JPEG 会按 EXIF orientation 转正后再保存为标准 PNG。

服务端批量接口为 `POST /api/views/import`。它先验证四张图都可解码且槽位恰好为 `0..3`，再统一写入。
已有 RGB 的 session 必须在前端确认后显式发送 `replace_existing=true`。写入使用 staging + rollback，
session 读写受锁保护；重建任务运行期间拒绝修改 RGB 或 mask。
任何 RGB 被替换时，对应旧 mask、mask preview、SAM3 proposal、DINO input、全局 DINO token 和旧 manifest
都会失效，避免同尺寸旧 mask 让 4+4 contract 错误通过。
保存或修改任一 mask 时，也会使该视角 DINO input、全局 DINO token 和旧 manifest 失效。
mask 的正整数 label 必须全部出现在全局 `labels.json` 中，标签颜色变化会刷新所有已有视角的 mask preview。

导出 manifest 现在包含 `input_source`；纯四图上传时 `source_3dgs` 为 `null`。为避免覆盖已有
`fridge_point_cloud` 会话，实际导入外部四图时建议用独立会话启动，例如：

```bash
FRIDGE_3DGS_SESSION=fridge_four_view_upload python workbenches/fridge_3dgs/run_workbench.py
```

## 建议下一步

优先顺序：

1. 在工作台里加载 clean16 的四张 RGB，做 SAM3 + 手工 mask。
2. 先导出 whole-object mask：冰箱整体为 `1`，背景为 `0`，四视角严格对齐。
3. 再导出 part mask：每个 part 使用稳定整数 label，背景仍为 `0`。
4. 补 DINO token 流程，保证 SS flow 使用的 token shape 和 prenorm 方式正确。
5. 跑 SS flow whole object。
6. 跑 part-prompt-seg。
7. 把每一步输出接回工作台页面，按阶段可视化：
   - 3DGS view selection
   - 4 rendered / generated views
   - SAM3 / mask editing
   - exported model inputs
   - SS flow outputs
   - part-prompt-seg outputs

## 继续时的关键约束

- 不训练。
- 不微调。
- 不在线安装依赖。
- 不切换到 Docker / vLLM-Omni。
- 不把 Cosmos 输出当作真实内部结构恢复。
- 不生成开门或内部视角。
- 不覆盖用户已有输出，除非明确要求。
- 所有新的大输出继续写到 `/robot/data-lab/jzh/art-gen/...`。
