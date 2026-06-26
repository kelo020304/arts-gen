# PhysX-Mobility Raw Data Format

本文档只总结当前 `data/PhysX-Mobility/raw/` 目录里已有数据本身的格式。新增数据如果想和当前数据保持一致，应按这里的目录、文件名和字段约定准备。

## 目录结构

每个物体使用一个稳定的数字或字符串 ID，例如 `45305`。同一个 ID 需要在 `finaljson/`、`partseg/`、可选的 `urdf/` 中保持一致。

```text
raw/
├── finaljson/
│   └── {object_id}.json
├── partseg/
│   └── {object_id}/
│       ├── objs/
│       │   ├── {mesh_stem}.obj
│       │   ├── {mesh_stem}.mtl
│       │   └── {mesh_stem}.obj.convex.stl
│       └── images/
│           └── texture files, optional
└── urdf/
    └── {object_id}.urdf
```

当前样本统计：

- `finaljson/*.json`: 2022 个。
- `partseg/{id}/`: 2022 个，与 `finaljson` 一一对应。
- `urdf/*.urdf`: 2024 个。

## 必需文件

从当前数据看，新增一个物体至少应提供：

1. `finaljson/{object_id}.json`
2. `partseg/{object_id}/objs/*.obj`
3. `parts[].obj` 中引用到的每个 mesh stem，都必须存在对应的 `partseg/{object_id}/objs/{mesh_stem}.obj`

当前数据里通常还会出现：

1. `{mesh_stem}.mtl`，如果 OBJ 依赖材质。
2. `images/*`，如果 MTL 引用了贴图。
3. `{mesh_stem}.obj.convex.stl`，作为对应 OBJ 的凸包 STL。
4. `urdf/{object_id}.urdf`，作为同一物体的 URDF 描述。

## finaljson 格式

`finaljson/{object_id}.json` 是核心元数据，必须是 UTF-8 JSON object，顶层字段如下：

| 字段 | 类型 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| `object_name` | string | 是 | 物体名称，例如 `Cabinet`。 |
| `category` | string | 是 | 物体类别，例如 `Storage Furniture`。 |
| `dimension` | string | 是 | 尺寸字符串，当前数据常见格式为 `L*W*H`，例如 `80*60*90`。 |
| `parts` | array | 是 | 部件列表。部件 label、语义、材质和 mesh 引用都在这里。 |
| `group_info` | object | 是 | 关节树和运动参数。必须包含根组 `"0"`。 |

### parts 字段

`parts` 是数组。当前数据中每个 part 都包含以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `label` | integer | 部件编号。必须唯一，通常从 `0` 开始，并与 `group_info` 引用一致。 |
| `name` | string | 部件名称，例如 `Cabinet Door (Left)`。 |
| `material` | string | 材料名称。 |
| `density` | string | 密度字符串，例如 `0.7 g/cm^3`。 |
| `Young's Modulus (GPa)` | number | 杨氏模量，单位 GPa。 |
| `Poisson's Ratio` | number | 泊松比。 |
| `priority_rank` | integer | 部件优先级，数值语义由数据侧定义。 |
| `Basic_description` | string | 基础形态描述。 |
| `Functional_description` | string | 功能描述。 |
| `Movement_description` | string | 运动描述。 |
| `obj` | array[string] | 组成该 part 的 mesh stem 列表，不带 `.obj` 后缀。 |

`obj` 示例：

```json
"obj": [
  "original-31",
  "original-35"
]
```

这表示必须存在：

```text
partseg/{object_id}/objs/original-31.obj
partseg/{object_id}/objs/original-35.obj
```

### group_info 字段

`group_info` 描述固定主体和可动关节，是当前 JSON 中表达物体运动结构的关键字段。

#### 根组 `"0"`

根组是固定基座，格式为 part label 列表：

```json
"0": [0, 1]
```

含义：part `0` 和 part `1` 属于固定主体。`group_info` 必须包含 `"0"`。

#### 可动组

除 `"0"` 外，每个可动组通常是长度为 4 的数组：

```json
"1": [
  [2],
  "0",
  [axis_x, axis_y, axis_z, origin_x, origin_y, origin_z, lower, upper],
  "C"
]
```

字段含义：

| 位置 | 类型 | 说明 |
| --- | --- | --- |
| `[0]` | integer 或 array[integer] | 当前组包含的 part label。 |
| `[1]` | string 或 integer | 父 group id。根父级通常是 `"0"`。 |
| `[2]` | array[number] | 运动参数。 |
| `[3]` | string | 关节类型代码。 |

关节类型代码：

| 代码 | 含义 | 参数解释 |
| --- | --- | --- |
| `B` | prismatic / 平移关节 | `params[0:3]` 是平移方向，`params[6:8]` 是位移范围。 |
| `C` | revolute / 旋转关节 | `params[0:3]` 是旋转轴方向，`params[3:6]` 是轴上一点，`params[6:8]` 是角度范围，单位为 `pi` 倍数。 |
| `A` | free_rotation / 自由旋转 | `params[0:3]` 是轴方向，`params[3:6]` 是轴上一点，采样范围为 `0..2*pi`。 |
| `D` | pivot / 枢轴旋转 | `params[0:3]` 是轴方向，`params[3:6]` 是轴上一点，采样范围为 `0..pi`。 |
| `CB` | compound / 复合关节 | 前半段状态做平移，后半段状态做旋转；还会使用 `params[8:11]` 作为平移方向。 |
| `E` | fixed / 固定 | 固定组，无运动。 |

当前数据中最常见的是：

- `B`: 平移关节，例如抽屉、按钮滑动。
- `C`: 旋转关节，例如门、盖子、铰链。
- `A`: 少量自由旋转关节。

## partseg 格式

`partseg/{object_id}/objs/` 存放实际几何网格。文件名 stem 需要与 `finaljson.parts[].obj` 完全一致。

常见文件组合：

```text
partseg/45305/objs/original-31.obj
partseg/45305/objs/original-31.mtl
partseg/45305/objs/original-31.obj.convex.stl
```

要求：

- OBJ 文件应是标准 Wavefront OBJ 文本格式。
- 如果 OBJ 引用 MTL，MTL 文件应在同目录。
- 如果 MTL 引用贴图，贴图通常放在 `partseg/{object_id}/images/`。
- 坐标需要与 `finaljson.group_info` 的轴位置、轴方向一致。

## URDF 格式

`urdf/{object_id}.urdf` 是 XML 格式，当前样本的基本结构如下：

```xml
<robot name="scene">
  <link name="l_world">...</link>
  <link name="l_0">
    <visual>
      <geometry>
        <mesh filename="./../partseg/{object_id}/objs/{mesh_stem}.obj" scale="1 1 1" />
      </geometry>
    </visual>
  </link>
  <joint name="joint_fixed_world0" type="fixed">
    <parent link="l_world" />
    <child link="l_0" />
  </joint>
</robot>
```

URDF 中常见 joint type 包括 `fixed`、`revolute`、`prismatic`，mesh 路径通常指向 `./../partseg/{object_id}/objs/*.obj`。

## 最小示例

```json
{
  "object_name": "Cabinet",
  "category": "Storage Furniture",
  "dimension": "80*60*90",
  "parts": [
    {
      "label": 0,
      "name": "Cabinet Frame",
      "material": "Plywood",
      "density": "0.7 g/cm^3",
      "Young's Modulus (GPa)": 10.0,
      "Poisson's Ratio": 0.3,
      "priority_rank": 4,
      "Basic_description": "Main fixed cabinet body.",
      "Functional_description": "Supports the object structure.",
      "Movement_description": "Fixed, non-moving part.",
      "obj": ["original-0"]
    },
    {
      "label": 1,
      "name": "Door",
      "material": "MDF",
      "density": "0.75 g/cm^3",
      "Young's Modulus (GPa)": 4.0,
      "Poisson's Ratio": 0.35,
      "priority_rank": 1,
      "Basic_description": "Flat front door.",
      "Functional_description": "Allows access to interior storage.",
      "Movement_description": "Rotates around a hinge.",
      "obj": ["original-1"]
    }
  ],
  "group_info": {
    "0": [0],
    "1": [
      [1],
      "0",
      [0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
      "C"
    ]
  }
}
```

配套文件：

```text
partseg/{object_id}/objs/original-0.obj
partseg/{object_id}/objs/original-1.obj
```

## 新数据检查清单

提交新数据前检查：

1. `finaljson/{id}.json` 存在，且顶层字段包含 `object_name`、`category`、`dimension`、`parts`、`group_info`。
2. `group_info` 包含 `"0"` 根组。
3. 所有 part 的 `label` 唯一。
4. `group_info` 引用的 part label 都存在于 `parts[].label`。
5. 每个 `parts[].obj` 引用的 `{mesh_stem}.obj` 都存在。
6. 可动组的父 group id 存在，整棵 group tree 能从 `"0"` 访问。
7. `B`、`C` 等运动参数长度和含义正确，数值有限。
8. OBJ、MTL、贴图相对路径可解析。
9. 如果提供 URDF，URDF mesh 路径也能解析到同一批 OBJ。
