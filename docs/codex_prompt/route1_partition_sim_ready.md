# 第一步 · 后处理切分诊断：按 seg-voxel 切「好的整体 mesh」→ 光滑完整 part（v5 door+drawer，8卡并行）

> 几何打磨第一步 · **零训练** · 独立后处理脚本 · 在 **v5 里 door / drawer 类对象**上验证 · 8 卡并行快测 · 输出 `ee-eval/route-1-diag`。
> **根因(皇上 eval 看图已确认)**:整体一次解码 `overall.obj` **干净完整**;
> 但"按 `part_seg` 的 voxel 把整体 SLat 切成子集、再对每个子集独立 FlexiCubes 解码"——
> 子集对整体训练的 decoder 是 **OOD**,边界被凭空封/凸/碎 → part **不光滑、不完整**(见 19179 双抽屉碎成片、19855 抽屉边毛)。
> **本步不切 SLat 重解码**:直接**按 seg voxel 标签,把那张好的整体 mesh 切成各 part**。每 part = 好整体上的一块 → 天然光滑完整、壳不丢。只用 seg 做**标签**、几何取自好整体 → **对 seg 噪声鲁棒**(平滑步收干净糊边界,如 19179 上下抽屉那条糊边)。

---

## ★测试数据（v5,door+drawer,8 卡一波跑完）

- v5 packed index:`/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v5/index.json`
- **统一输出**:`/mnt/robot-data-lab/jzh/art-gen/ee-eval/route-1-diag`
- **选样**:从 v5 挑 **part 名含 `door` 或 `drawer`** 的对象,**门类、抽屉类各 ≥3**,凑 **~8 个**(正好一卡一个、一波并行跑完)。先写小脚本读 index、按 part 名过滤、打印 `(dataset_id, object_id, angle)` 候选,再选。
- **建议固定 8 个(皇上已看过图、覆盖 door+drawer + 压力样本)** —— dataset_id/angle 以 v5 index 或已有 eval run 为准:
  - drawer:`19855`(单抽屉桌)、`19179`(双抽屉,seg 边界糊·压力)、`101940`(下门+抽屉灶台)
  - door:`101584`(保险柜门)、`11231`(冰箱双门)、`10638`(双门)
  - door+drawer 混合:`05a035c3347645b8a7ceb6d65f825ac3`(门+抽屉+窗板,5 part·压力)、`101808`(灶台:上下门+多旋钮·压力)

---

## Step 0 · 8 卡并行产出脚本输入（ee-eval → route-1-diag）

partition 脚本需每对象三样:**整体 mesh + 每 part 64³ seg voxel + 整体 64³ voxel**。前两样来自 ee-eval,但 **ee_0617 默认只渲染整体 mesh 不落盘**,需先暴露:

**(0a) 最小改动:暴露整体 mesh**(`scripts/eval/tasks/ee_0617_single.py` 导出 part `.obj` 那段):当前 `if args.export_mujoco and label != "overall":` 跳过 overall;改成对 `label == "overall"` 也调一次
`save_decoded_slat_assets({"mesh": mesh}, mujoco_assets_dir, mesh_name="overall.obj")`。**仅此一处暴露,不动 eval 主逻辑、不改默认参数。**

**(0b) 8 卡并行 dispatch**(一对象一卡,backgrounded + wait):
```bash
OUT=/mnt/robot-data-lab/jzh/art-gen/ee-eval/route-1-diag
COMMON='PYTHONPATH=/root/code/arts-gen:/root/code/arts-gen/TRELLIS-arts \
  SPCONV_ALGO=native ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa SS_FLOW_FUSION_MODE=concat'
# OBJS: "dataset_id object_id angle"，~8 个 door/drawer 对象（上面选出的）
i=0
for spec in "${OBJS[@]}"; do
  read ds obj ang <<< "$spec"; gpu=$((i % 8))
  CUDA_VISIBLE_DEVICES=$gpu env $COMMON /opt/venvs/arts-gen/bin/python \
    scripts/eval/tasks/ee_0617_single.py \
    --out-dir "$OUT" --dataset-id "$ds" --object-id "$obj" --angle "$ang" --gpu 0 \
    --slat-token-source live --export-mujoco --force-export \
    > "$OUT/log_${obj}_${ang}.txt" 2>&1 &
  i=$((i+1))
done
wait   # 8 个一波并行，等全部完成
```
> `CUDA_VISIBLE_DEVICES=$gpu` 后该进程只见这一张卡为 device0,故 `--gpu 0` 正确。对象 >8 时自动绕回(分多波)。

产出(脚本输入路径,每对象一份):
- 整体 mesh:`$OUT/.../__mujoco/assets/overall.obj`(0a 暴露后)
- 每 part seg voxel:`$OUT/_platform_runs/.../parts/part_*_voxel.npz`
- 整体 voxel:`$OUT/_platform_runs/.../voxel.npz`
- 旧独立解码 part mesh(作 before 对照):`$OUT/.../__mujoco/assets/<part>.obj`

---

## Step 1 · partition 脚本（新增 `scripts/post_process/partition_whole_to_parts.py`，零训练、CPU、可对 8 对象循环秒级跑完）

CLI:`--overall-mesh / --whole-voxel / --part-voxels-glob / --part-meshes-glob(旧独立解码,作 before)/ --out-dir / --coacd-threshold(默认0.05) / --density(默认300)`。

### ★坐标自校验（唯一技术风险，先做，过不去就报错退出）
`overall.obj` 在 decoder mesh 帧(经 `_mesh_vertices_y_up`,**Y-up**);`*_voxel.npz` 在 64³ voxel 帧(**Z-up**)。两者间是**固定仿射 + Y-up↔Z-up 旋转**(无 per-object scale)。脚本第一步**自动定标+校验**:
1. 候选:voxel `(i,j,k)∈[0,64)` → `((i+0.5)/64-0.5)×尺度` → ±90°X 旋转(两方向都试)。
2. 校验:`overall.obj` 顶点反映射回 64³,与 `--whole-voxel` 求占据 IoU;取 IoU 最高那组。
3. **IoU < 0.5 → raise 退出**(禁止带错坐标硬跑);打印最终变换 + IoU。

### 算法
1. 加载 `overall.obj`→trimesh `W`(`process=False`);各 `part_*_voxel.npz`→`V_k`(64³ int coords)。
2. **逐面归属**:用校验变换把 `V_k` 映到 mesh 帧;`W` 每面取面心,`scipy.cKDTree` 求最近 part → `label(f)=argmin_k dist`。**每面必有 label → 不丢几何、壳不丢**。
3. **边界平滑(核心)**:面-邻接对偶图 8 轮加权多数票,边权 `exp(-dihedral/σ)`(折痕处权小→切口落折痕);只更新边界面,迭代到稳。消最近-邻锯齿。
4. **切分**:按 label `W.submesh` → `S_k`。
5. **封口**:`trimesh.repair.fill_holes` + `fix_normals/fix_winding/merge_vertices`;报 `is_watertight`(非水密保留+标注,不强行布尔)。
6. **CoACD**:每 `S_k`→`coacd.run_coacd`→N 个凸 `.obj`;薄片退化 OBB box 记日志。
7. **MJCF**:每 part 一 `<body>`:visual=`S_k`(group2 contype0 conaffinity0);collision=各凸 piece(group3 contype1 conaffinity1);`<inertial>` 按体积×密度。静态,默认固定 base(CLI 可 freejoint)。
8. **前后对比**:并排渲 **before**(`--part-meshes-glob` 旧独立解码,毛/碎)vs **after**(`S_k`)→ `compare.png`;有 mujoco python 则额外 settle 出 `settle_after.png`。
9. **`report.json`**:每 part `watertight_before/after`、`coacd_pieces`、切口边长、`unassigned_faces`(应=0)、切分后 part 间最近面距离、坐标校验 IoU。

**8 对象循环**(CPU,秒级):
```bash
for d in $(ls -d $OUT/_platform_runs/*/ 2>/dev/null); do
  base=$(dirname $(dirname "$d"))   # 定位该对象的 __mujoco/assets
  /opt/venvs/arts-gen/bin/python scripts/post_process/partition_whole_to_parts.py \
    --overall-mesh "$base"/__mujoco/assets/overall.obj \
    --whole-voxel  "$d"/voxel.npz \
    --part-voxels-glob "$d"/parts/part_*_voxel.npz \
    --part-meshes-glob "$base"/__mujoco/assets/*.obj \
    --out-dir "$OUT"/partition/$(basename "$d") --coacd-threshold 0.05
done
```
（实际相对路径以 ee-eval 输出布局为准,Codex 自行对齐 `_platform_runs/.../parts` 与 `__mujoco/assets` 的对应关系。）

### 代码落点
新增 `scripts/post_process/partition_whole_to_parts.py`(argparse、`__main__`);仅 `ee_0617_single.py` 暴露 `overall.obj`(0a,≤3 行)。**不动** reconstruct/inference/模型/训练/eval 主逻辑。失败暴露、禁静默兜底。依赖 `trimesh/numpy/scipy/coacd`(arts-gen 均有),mujoco python 可选。

---

## 验收 UAT（皇上 dev 跑完填）

```
# Step 0：8 卡一波跑完 → 检查 8 个对象都产出 overall.obj + part voxel
ls $OUT/*/**/__mujoco/assets/overall.obj | wc -l   # 期望 = 测试对象数
# Step 1：8 对象循环 partition（见上）
```
- [ ] 8 个对象 ee-eval 全成(每个有 `overall.obj` + `parts/part_*_voxel.npz`)
- [ ] 每对象 **坐标校验 IoU ≥ 0.5**(否则报错)、`report.json` 的 `unassigned_faces == 0`(壳/几何不丢)
- [ ] 每 part 有 `S_k.obj`(visual)+ ≥1 CoACD 凸 `.obj`(collision)
- 实际 [皇上填]:`成功对象数=__/8  最低校验IoU=__  有无 unassigned>0=__`

```
# 核心:逐个打开 $OUT/partition/<obj>/compare.png
```
- [ ] **19179 双抽屉**:after 两抽屉不再碎、上下交界=干净光滑切口
- [ ] **101584 / 11231 门**:after 门板边缘**不再毛**
- [ ] **05a035c / 101808 多 part 压力样本**:after 各 part 光滑、壳完整、part 间不残留互穿
- 观感 [皇上填]:19179 __；门类 __；多part __

**判定**:after 切口干净+壳完整 → 路线①奏效,"边缘整齐光滑 sim-ready" 第一步达成。
（若某对象 after **壳仍缺**:该对象 seg voxel 本身漏分了壳 → 记下,part_seg recall 的另一刀、正交、后面单独处理。）

---

## Codex 自检（实现说明里答，中文）
1. **核心逻辑**:坐标自校验(±90°X×尺度搜 IoU)稳不稳?折痕加权多数票能否把最近-邻锯齿收成沿折痕的干净切口?part≥5(05a035c)三交点处会不会退化?
2. **数据路径一致性**:`overall.obj`(Y-up)↔`voxel.npz`(Z-up 64³)变换是否真对上(以校验 IoU 为准)?8 卡 dispatch 的 `CUDA_VISIBLE_DEVICES`+`--gpu 0` 是否每进程独占一卡?`_platform_runs/.../parts` 与 `__mujoco/assets` 的对象对应关系有没有对齐?0a 暴露 overall.obj 有没有碰 eval 主逻辑/默认参数?
3. **副作用**:`fill_holes` 会不会把抽屉腔/门内侧本该开口的结构错误封死?CoACD 默认 threshold 对薄抽屉板/门板是否合理、退化 box 是否兜住?是否引入 arts-gen 之外依赖?
