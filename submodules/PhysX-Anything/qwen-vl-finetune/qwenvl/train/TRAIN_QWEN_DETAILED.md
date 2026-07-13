# `train_qwen.py` 详细说明

这份文档的目标不是简单“翻译代码”，而是帮你真正建立对
[`train_qwen.py`](./train_qwen.py)
这份训练入口脚本的完整心智模型。

读完之后，你应该能回答这些问题：

- 这个脚本到底在训练什么
- 它和同级的 [`argument.py`](./argument.py)、[`trainer.py`](./trainer.py) 是怎么配合的
- `data_flatten` / `data_packing` / `tune_mm_llm` / `tune_mm_mlp` / `tune_mm_vision` 分别控制什么
- 它为什么既像 Hugging Face Trainer 脚本，又掺杂了很多本项目自己的定制逻辑
- 当前这份代码里，哪些地方是“原始训练逻辑”，哪些地方是“兼容性适配”

---

## 1. 先给一个总览

`train_qwen.py` 是整个 Qwen-VL 微调子工程的“训练总控入口”。

它不负责实现模型本身，也不负责具体的数据处理细节，而是负责把以下几部分接起来：

1. 解析命令行参数
2. 根据模型路径判断到底加载 `Qwen2VL` 还是 `Qwen2.5-VL`
3. 给模型设置训练策略
4. 根据数据相关参数选择数据集和 collator
5. 创建 Hugging Face `Trainer`
6. 启动训练、断点恢复、保存模型

你可以把它理解成一个“装配脚本”：

```text
命令行参数
    ↓
ModelArguments / DataArguments / TrainingArguments
    ↓
加载模型 + tokenizer + image_processor
    ↓
设置哪些模块可训练
    ↓
构造 data_module
    ↓
创建 Trainer
    ↓
train() / resume_from_checkpoint()
    ↓
保存状态、处理器、模型权重
```

所以这份脚本最重要的价值，不在“算法创新”，而在“把训练系统真正跑起来”。

---

## 2. 同级文件各自做什么

先把同级目录里几个核心文件的职责讲清楚：

### [`train_qwen.py`](./train_qwen.py)

训练入口。负责：

- 参数解析
- 模型加载
- tokenizer / processor 加载
- 可训练参数设置
- 选择数据模块
- 创建 Trainer
- 训练与保存

### [`argument.py`](./argument.py)

定义三组 dataclass 参数：

- `ModelArguments`
- `DataArguments`
- `TrainingArguments`

也就是说，`train_qwen.py` 并不自己声明一堆 argparse 参数，而是通过 dataclass 交给 `transformers.HfArgumentParser` 来解析。

### [`trainer.py`](./trainer.py)

这不是标准 Hugging Face 自带的 trainer 文件，而是这个项目对 Trainer 行为的定制层。

它做了几件事：

- 定义自定义 `_sdpa_attention_forward`
- 定义 `_update_causal_mask`
- 提供 `replace_qwen2_vl_attention_class()` 做 monkey patch
- 重写 `Trainer.create_optimizer`
- 给 Qwen2/Qwen2.5 的视觉塔和语言模型打上 `print_trainable_parameters()` 方法

也就是说，`train_qwen.py` 本身很短，但它的很多关键行为其实来自同级的 `trainer.py`。

---

## 3. 代码结构总览

`train_qwen.py` 可以按逻辑切成 7 个部分：

1. import 和环境兼容处理
2. 全局变量与辅助打印
3. `safe_save_model_for_hf_trainer()`
4. `set_model()`
5. `train()`
6. `__main__` 入口
7. 当前版本的兼容性改动

下面按这个顺序详细解释。

---

## 4. 第一部分：import 和环境兼容处理

文件开头：

```python
import os
import logging
import pathlib
import torch
try:
    import torch_npu
except ImportError:
    pass
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import ipdb
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
```

### 4.1 `torch_npu` 的意义

这里的：

```python
try:
    import torch_npu
except ImportError:
    pass
```

是典型的“可选后端兼容写法”。

它的作用不是立即调用 NPU API，而是让这份脚本在 Ascend 环境下能够识别 `torch.npu` 相关能力，同时不破坏普通 CUDA/CPU 环境。

如果导入失败，脚本不会中断，说明这份脚本设计上允许同一套代码跑在：

- CUDA
- Ascend NPU
- 或至少能在无 `torch_npu` 环境下导入

### 4.2 为什么要手动 `sys.path.append(project_root)`

```python
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
```

这是为了保证从任意工作目录启动脚本时，都能正确导入项目包：

- `qwenvl.train.trainer`
- `qwenvl.data.data_qwen`
- `qwenvl.train.argument`

这里的 `project_root` 实际上指向：

```text
.../qwen-vl-finetune/
```

因为：

- `__file__` 在 `qwenvl/train/train_qwen.py`
- `parent` 是 `train`
- `parent.parent` 是 `qwenvl`
- `parent.parent.parent` 是 `qwen-vl-finetune`

把这个目录加到 `sys.path` 后，`qwenvl` 这个包才能稳定导入。

### 4.3 一个容易忽略的点：这里有双重导入

代码里既有：

```python
import qwenvl.train.trainer
```

又有：

```python
from trainer import replace_qwen2_vl_attention_class
```

这两句合起来说明两件事：

1. 这个脚本希望导入 `trainer.py` 后立即触发其中的全局 monkey patch
2. 它还要直接拿到 `replace_qwen2_vl_attention_class()` 这个函数在后面调用

也就是说，单单 `import qwenvl.train.trainer` 就已经不是“无副作用导入”了，因为
[`trainer.py`](./trainer.py)
底部会直接执行：

```python
Trainer.create_optimizer = create_optimizer
Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = ...
Qwen2VLModel.print_trainable_parameters = ...
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = ...
Qwen2_5_VLModel.print_trainable_parameters = ...
```

这意味着：

> 只要导入了 `trainer.py`，当前 Python 进程里的相关类行为就已经被改写了。

这对理解后面“为什么 `Trainer(...)` 的行为和官方不完全一样”非常重要。

---

## 5. 第二部分：全局变量与辅助打印

```python
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
```

这是分布式训练里很常见的写法。

### 5.1 `local_rank` 是什么

在 `torchrun` 启动的多卡训练里，每个进程都会有自己的 rank。

这里的 `local_rank` 一般表示“本机上的第几张卡/第几个设备”。

### 5.2 为什么只在 rank 0 打印

多卡训练时如果每个进程都输出同样的日志，会非常乱。

所以这个辅助函数规定：

- 只有 `local_rank == 0` 的那个进程负责打印主要信息
- 其他进程静默

这是为了让训练日志可读。

---

## 6. 第三部分：`safe_save_model_for_hf_trainer`

函数：

```python
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
```

它的作用是：**在不同训练后端下，用尽量安全的方式保存模型。**

### 6.1 为什么需要单独写保存逻辑

Hugging Face `Trainer` 在普通单卡/多卡下和在 DeepSpeed 下，模型保存方式不一样。

尤其是：

- 在普通 PyTorch / DDP 下，`trainer.model.state_dict()` 通常可以直接拿
- 在 DeepSpeed ZeRO 下，模型参数可能是分片的，不能想当然地自己拼

所以这段代码先判断：

```python
if trainer.deepspeed:
```

如果启用了 DeepSpeed，就直接：

```python
trainer.save_model(output_dir)
```

把保存逻辑交回给 DeepSpeed/HF 自己处理。

### 6.2 为什么要先 synchronize

代码里：

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
elif hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.synchronize()
```

这一步是为了在保存前确保异步计算已经完成。

如果不做同步，某些后端上可能会出现：

- 计算还没真正完成
- 参数还没稳定
- 保存时状态不一致

当前这份代码显式兼容：

- `torch.cuda.synchronize()`
- `torch.npu.synchronize()`

这说明它已经考虑了 Ascend 场景。

### 6.3 非 DeepSpeed 分支做了什么

如果不是 DeepSpeed：

```python
state_dict = trainer.model.state_dict()
```

然后把所有参数转到 CPU：

```python
cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
```

再调用：

```python
trainer._save(output_dir, state_dict=cpu_state_dict)
```

这么做的好处是：

- 节省设备显存
- 避免某些后端在保存 GPU/NPU tensor 时出问题
- 保证磁盘保存的是标准 CPU 权重

### 6.4 这个函数的心智模型

你可以把它记成一句话：

> DeepSpeed 交给 DeepSpeed 自己存；普通模式手动把权重搬回 CPU 再存。

---

## 7. 第四部分：`set_model`

这是整个脚本里非常关键的函数：

```python
def set_model(model_args, model):
```

它负责设置哪些模块参与训练，也就是控制 `requires_grad`。

### 7.1 这份模型一共分哪几块

从代码来看，这个多模态模型至少分成三块：

1. `model.model`
   语言模型主干
2. `model.visual`
   视觉编码器
3. `model.visual.merger`
   视觉特征和语言侧对接的 merger / projector

另外还有：

4. `model.lm_head`
   最终语言输出头

### 7.2 三个开关分别控制什么

来自 [`argument.py`](./argument.py) 的 `ModelArguments`：

| 参数 | 含义 |
| --- | --- |
| `tune_mm_llm` | 是否训练语言模型主干 `model.model` 和 `lm_head` |
| `tune_mm_vision` | 是否训练视觉编码器 `model.visual` |
| `tune_mm_mlp` | 是否训练视觉 merger / projector `model.visual.merger` |

也就是说，这个脚本支持多种微调策略：

- 只训 projector
- 训 projector + LLM
- 全量训
- 甚至只训视觉塔

### 7.3 为什么设置顺序这么重要

当前版本的注释写得非常关键：

```python
# NOTE: In transformers 4.53, model.model.named_parameters() includes
# visual params (visual is nested under model.model). So we must set
# LLM first, then override visual/merger to get correct freeze state.
```

这句话的意思是：

在你当前的 `transformers 4.53` 版本里，

```python
model.model.named_parameters()
```

并不只包含“纯语言模型参数”，还可能把视觉相关参数也带进来。

这会导致一个坑：

- 如果你先冻结 visual
- 再设置 LLM
- 后面的 LLM 设置可能又把 visual 的参数状态改掉

所以当前代码采取的顺序是：

1. 先统一设置 LLM
2. 再覆盖 visual
3. 最后再覆盖 merger

也就是：

```text
先大范围设语言
    ↓
再局部修正视觉
    ↓
再局部修正 merger
```

这个顺序是为了避免参数归属边界不清导致的误冻结/误解冻。

### 7.4 为什么 `lm_head` 单独处理

`lm_head` 不一定包含在 `model.model.named_parameters()` 里，所以代码单独写了：

```python
model.lm_head.requires_grad = True / False
```

这说明作者不想假设 `lm_head` 一定属于 `model.model` 的参数树，而是明确控制它。

### 7.5 一个最常见的训练配置

比如你命令里经常会写：

```bash
--tune_mm_vision False
--tune_mm_mlp True
--tune_mm_llm True
```

这意味着：

- 冻结视觉编码器
- 训练 merger
- 训练 LLM

也就是常见的“冻结视觉塔，只调 projector + 语言模型”策略。

---

## 8. 第五部分：`train()` 是整份脚本的主流程

函数签名：

```python
def train(attn_implementation="flash_attention_2"):
```

但实际 `__main__` 调用的是：

```python
train(attn_implementation="sdpa")
```

说明当前版本已经不是原始 CUDA `flash_attention_2` 路线，而是改成了 `sdpa`。

下面按执行顺序拆解。

---

## 9. `train()` 第一步：解析三组参数

```python
parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments)
)
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```

这里没有使用传统 argparse，而是直接把三组 dataclass 喂给 Hugging Face 的解析器。

### 9.1 为什么这样写

优点有几个：

- 参数定义集中在 [`argument.py`](./argument.py)
- 参数类型更清晰
- 默认值集中管理
- 后面访问参数时有字段名补全，代码更整洁

### 9.2 三组参数各自关心什么

#### `ModelArguments`

控制模型级策略：

- `model_name_or_path`
- `tune_mm_llm`
- `tune_mm_mlp`
- `tune_mm_vision`

#### `DataArguments`

控制数据和视觉预处理：

- `dataset_use`
- `data_flatten`
- `data_packing`
- `max_pixels`
- `min_pixels`
- 视频相关参数

#### `TrainingArguments`

继承自 `transformers.TrainingArguments`，再额外补充：

- `cache_dir`
- `optim`
- `model_max_length`
- `mm_projector_lr`
- `vision_tower_lr`
- `deepspeed`

### 9.3 当前版本的一个兼容点

在 [`argument.py`](./argument.py) 中：

```python
deepspeed: Optional[str] = field(default=None)
```

这是对 `transformers 4.53+` 的一个兼容处理。

原因是原版 `TrainingArguments` 中 `deepspeed` 的类型定义在某些版本下会让
`HfArgumentParser`
把 JSON 路径错误解析成别的类型。

这里手动固定成 `Optional[str]`，就是为了让：

```bash
--deepspeed scripts/zero3.json
```

这种命令能稳定工作。

---

## 10. `train()` 第二步：初始化输出目录和 rank

```python
local_rank = training_args.local_rank
os.makedirs(training_args.output_dir, exist_ok=True)
```

这里做了两件很基础但必要的事：

1. 记录当前分布式 rank，供日志输出使用
2. 确保输出目录存在

注意：

`training_args.local_rank` 这个字段一般由 `torchrun` 注入。

---

## 11. `train()` 第三步：判断模型是 Qwen2 还是 Qwen2.5

这一段很关键：

```python
_is_qwen2_5 = "qwen2.5" in model_args.model_name_or_path.lower()
if not _is_qwen2_5:
    _config_path = os.path.join(model_args.model_name_or_path, "config.json")
    if os.path.isfile(_config_path):
        with open(_config_path) as _f:
            _cfg = json.load(_f)
            _is_qwen2_5 = _cfg.get("model_type", "") == "qwen2_5_vl"
```

### 11.1 原始思路

最简单的判断方法是看路径里有没有 `qwen2.5`。

比如：

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

这种官方 Hugging Face 名称就很好判断。

### 11.2 为什么还要读 `config.json`

因为你现在经常用的是本地模型路径，比如：

```text
/home/ma-user/cfy/pretrain/vlm
```

这种路径里不包含 `qwen2.5` 字样。

所以脚本做了第二层判断：

- 读取本地 `config.json`
- 看 `model_type` 是否等于 `qwen2_5_vl`

这一步的意义非常大：

> 它把“靠路径名猜模型类型”改成了“优先靠模型配置判断模型类型”。

这比原始做法稳得多。

---

## 12. `train()` 第四步：加载模型和 image processor

根据 `_is_qwen2_5` 分两条分支。

### 12.1 Qwen2.5-VL 分支

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
data_args.image_processor = AutoProcessor.from_pretrained(...).image_processor
data_args.model_type = "qwen2.5vl"
```

这里做了三件事：

1. 加载 `Qwen2_5_VLForConditionalGeneration`
2. 用 `AutoProcessor` 取出其中的 `image_processor`
3. 记录 `model_type`

### 12.2 Qwen2-VL 分支

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(...)
data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(...)
data_args.model_type = "qwen2vl"
```

同理，只是对应旧版模型类。

### 12.3 `attn_implementation` 的作用

模型加载时传了：

```python
attn_implementation=attn_implementation
```

这会影响底层 attention backend。

原始 CUDA 方案一般是：

```python
flash_attention_2
```

但当前 `__main__` 里改成了：

```python
sdpa
```

这是一个非常重要的行为变化，因为：

- 它不只是“换个更慢的 attention 实现”
- 它可能影响整个 packed / flatten 路线的内部调用语义

### 12.4 `torch_dtype` 的作用

```python
torch_dtype=(torch.bfloat16 if training_args.bf16 else None)
```

意思是：

- 如果命令里传了 `--bf16`
- 模型权重会按 `bfloat16` 加载

这是大模型训练里常见的显存优化手段。

---

## 13. `train()` 第五步：是否启用 `data_flatten`

```python
if data_args.data_flatten:
    replace_qwen2_vl_attention_class()
```

这句非常短，但影响极大。

### 13.1 这句到底做了什么

它会调用同级 [`trainer.py`](./trainer.py) 里的：

```python
replace_qwen2_vl_attention_class()
```

而这个函数会做全局 monkey patch：

- 替换 `_sdpa_attention_forward`
- 替换 `Qwen2VLModel._update_causal_mask`
- 替换 `Qwen2_5_VLModel._update_causal_mask`

### 13.2 这意味着什么

一旦 `data_flatten=True`：

- 你不再是“只改了 collator”
- 而是连模型的 attention 行为都改了

所以 `data_flatten` 不只是“数据组织方式开关”，它实际上是：

> 数据形式 + attention 实现假设 + causal mask 假设 的组合开关

### 13.3 `data_flatten` 和 `data_packing` 的区别

很多人容易把这两个混淆。

#### `data_flatten`

在本项目里，它主要指：

- 使用 `FlattenedDataCollatorForSupervisedDataset`
- 把 `attention_mask` 变成类似 `cu_seqlens` 的格式
- 同时要求 attention 路径配合这种 packed 表示

#### `data_packing`

这个开关决定是否走：

```python
make_supervised_data_module_packed()
```

也就是使用另一套数据构造逻辑。

所以它们不是完全同义：

| 开关 | 控制层面 |
| --- | --- |
| `data_flatten` | collator 与 attention 路径假设 |
| `data_packing` | 数据模块构建方式 |

---

## 14. `train()` 第六步：关闭 KV cache

```python
model.config.use_cache = False
```

### 为什么训练时要关掉 `use_cache`

因为 KV cache 主要是为推理服务的：

- 自回归生成时复用过去的 key/value

但训练时：

- 一般不需要它
- 还会增加显存占用
- 有时和 gradient checkpointing 冲突

所以训练前关掉是正常做法。

训练结束前，脚本又会恢复：

```python
model.config.use_cache = True
```

这说明作者明确区分了：

- 训练期配置
- 最终保存后供推理使用的配置

---

## 15. `train()` 第七步：处理 gradient checkpointing

代码：

```python
if training_args.gradient_checkpointing:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
```

### 15.1 这是在干什么

gradient checkpointing 的本质是：

- 前向时不保存全部激活
- 反向时再重算一部分
- 用计算换显存

但这样做时，有些模型需要显式保证输入 embedding 张量带梯度。

### 15.2 两种兼容路径

#### 路径 A：模型自己支持

如果模型有：

```python
enable_input_require_grads()
```

那就直接调用官方接口。

#### 路径 B：模型没有这个接口

那就给输入 embedding 模块挂一个 forward hook：

```python
output.requires_grad_(True)
```

相当于手动保证输出 tensor 可参与反向。

### 15.3 这段代码的意义

它说明这份训练脚本并不假设所有 Qwen 版本 API 完全一致，而是做了兼容处理。

---

## 16. `train()` 第八步：加载 tokenizer

```python
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=False,
)
```

### 16.1 为什么用 `AutoTokenizer`

因为它可以根据模型配置自动选择具体 tokenizer 类。

### 16.2 为什么 `padding_side="right"`

说明训练数据采用右侧 padding，这和很多 causal LM 训练习惯一致。

### 16.3 为什么 `use_fast=False`

通常有几个原因：

- 多模态 chat template / 特殊 token 兼容性更稳
- 某些模型在 fast tokenizer 上行为不完全一致
- 作者更偏向稳定而非极致速度

### 16.4 `model_max_length` 会影响什么

它会控制：

- tokenizer 截断长度
- collator 最终保留的最大序列长度
- 训练显存占用上限

---

## 17. `train()` 第九步：设置可训练参数

```python
set_model(model_args, model)
```

这里就是前面第 7 节讲的冻结/解冻逻辑真正生效的地方。

做完这一步之后，模型的训练参数范围就固定了。

### 17.1 脚本如何验证冻结结果

接着会打印：

```python
if torch.distributed.get_rank() == 0:
    model.visual.print_trainable_parameters()
    model.model.print_trainable_parameters()
```

注意这里的：

- `model.visual.print_trainable_parameters()`
- `model.model.print_trainable_parameters()`

不是 Hugging Face 原生方法，而是
[`trainer.py`](./trainer.py)
底部 monkey patch 进去的。

也就是说：

> `train_qwen.py` 利用 `trainer.py` 提供的打印函数，来验证冻结策略是否真的生效。

---

## 18. `train()` 第十步：根据数据参数选择 data module

代码：

```python
if data_args.data_packing:
    data_module = make_supervised_data_module_packed(...)
else:
    data_module = make_supervised_data_module(...)
```

### 18.1 这两者都返回什么

按 [`data_qwen.py`](../data/data_qwen.py) 的定义，它们返回的是一个字典，大致包含：

- `train_dataset`
- `eval_dataset`
- `data_collator`

然后后面通过：

```python
Trainer(..., **data_module)
```

展开进去。

### 18.2 普通模块和 packed 模块差别在哪

从命名可以看出：

- `make_supervised_data_module`
  普通 SFT 数据模块
- `make_supervised_data_module_packed`
  针对 packed 训练组织的数据模块

而在普通数据模块内部，如果：

```python
data_args.data_flatten
```

又会进一步选择 `FlattenedDataCollatorForSupervisedDataset`。

所以最终路线大概是：

```text
data_packing=False
    ↓
make_supervised_data_module()
    ↓
如果 data_flatten=True:
    用 FlattenedDataCollatorForSupervisedDataset
否则:
    用 DataCollatorForSupervisedDataset
```

---

## 19. `train()` 第十一步：创建 Trainer

```python
trainer = Trainer(
    model=model, processing_class=tokenizer, args=training_args, **data_module
)
```

### 19.1 这里的 `Trainer` 是谁

从 import 看：

```python
from transformers import ... Trainer
```

它表面上是 Hugging Face 的 `Trainer`。

但由于前面已经：

```python
import qwenvl.train.trainer
```

所以这个 `Trainer` 的某些行为已经被同级 `trainer.py` 全局改写过了。

最重要的是：

```python
Trainer.create_optimizer = create_optimizer
```

也就是说，**虽然类名没变，但优化器创建逻辑已经不是原生 HF 的那套了。**

### 19.2 `processing_class=tokenizer` 是什么含义

当前新版本 `Trainer` 支持用 `processing_class` 替代一部分旧接口。

这里传 tokenizer 的作用主要是：

- 保存处理器相关信息
- 让训练器知道如何关联输入处理组件

注意这里传的不是 `AutoProcessor`，而是 `tokenizer`。

图像处理器的保存是后面单独做的：

```python
data_args.image_processor.save_pretrained(training_args.output_dir)
```

---

## 20. `train()` 第十二步：断点恢复或从头训练

代码：

```python
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    logging.info("checkpoint found, resume training")
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
```

### 20.1 逻辑很直接

如果输出目录里已经存在：

```text
checkpoint-xxxx
```

就恢复训练；否则就从头开始。

### 20.2 为什么不用显式传 checkpoint 路径

因为这里的策略是：

- 默认在 `output_dir` 下找已有 checkpoint
- 找到就自动续训

这是比较实用的工程化写法。

---

## 21. `train()` 第十三步：训练后保存什么

训练结束后依次执行：

```python
trainer.save_state()
data_args.image_processor.save_pretrained(training_args.output_dir)
model.config.use_cache = True
safe_save_model_for_hf_trainer(...)
```

### 21.1 `trainer.save_state()`

保存 Trainer 状态，比如：

- 随机数状态
- 优化器/调度器相关状态
- Trainer 内部训练进度信息

### 21.2 `image_processor.save_pretrained(...)`

单独保存视觉处理器配置。

为什么 tokenizer 不需要这里单独保存？

因为 Hugging Face Trainer / save_model 体系通常会顺带处理 tokenizer 或配置相关信息；
而 image processor 是这份脚本手动挂到 `data_args` 里的，所以这里显式保存更稳。

### 21.3 恢复 `use_cache`

训练时关掉，保存前恢复，表示最终导出的模型更偏向推理可用状态。

### 21.4 `safe_save_model_for_hf_trainer(...)`

最后真正把模型权重保存到磁盘。

---

## 22. `__main__`：当前版本默认走 `sdpa`

文件结尾：

```python
if __name__ == "__main__":
    train(attn_implementation="sdpa")
```

### 22.1 这说明什么

这说明当前这份代码已经明确偏向：

- 非 flash-attn
- 更通用的 `sdpa`
- 适配非 CUDA 场景

### 22.2 这和原始论文环境有什么差别

原论文/原仓库脚本通常默认会更偏向：

- CUDA
- flash_attention_2
- 8x A100 之类环境

而你当前这份脚本已经明显融入了：

- Ascend / NPU 兼容
- transformers 4.53 兼容
- 本地模型路径兼容

所以它不是“原封不动的作者训练脚本”，而是“在作者脚本基础上做过兼容性演化的版本”。

---

## 23. 结合 `trainer.py` 再理解一次这份脚本

如果只读 `train_qwen.py`，你会觉得它很短。

但实际上它背后依赖了 `trainer.py` 的这些改动：

### 23.1 改 optimizer 创建逻辑

`Trainer.create_optimizer` 被替换成项目自定义版本。

这意味着你在命令行里设置：

- `mm_projector_lr`
- `vision_tower_lr`

这些参数时，真正生效的是
[`trainer.py`](./trainer.py)
里那套参数分组逻辑。

### 23.2 给模型类打上打印函数

`print_trainable_parameters()` 不是模型原生接口，而是 monkey patch 上去的。

所以 `train_qwen.py` 里这两句：

```python
model.visual.print_trainable_parameters()
model.model.print_trainable_parameters()
```

本质上依赖 `trainer.py` 的副作用。

### 23.3 `data_flatten=True` 时 attention 行为被整体改写

如果你打开 `data_flatten`，`trainer.py` 里还会进一步影响 attention 路径。

这也是为什么理解 `train_qwen.py` 时不能只盯这一个文件。

---

## 24. 这份脚本的“真实职责”总结

如果只用一句话概括：

> `train_qwen.py` 是“训练系统组装器”，不是“模型算法实现文件”。

它的职责是：

- 把模型、参数、数据、训练器、保存逻辑装配成一个完整训练流程

它不负责：

- 定义 Qwen2/Qwen2.5 的网络结构
- 定义数据集底层样本格式
- 实现视觉处理器
- 实现优化器细节本身

这些工作分别散落在：

- `transformers`
- `qwenvl.data.*`
- `qwenvl.train.trainer`

---

## 25. 你真正需要记住的 10 个关键点

### 1. 它是训练入口，不是模型定义

核心工作是“装配”，不是“发明模型”。

### 2. 参数来自 dataclass

不是 argparse 手写，而是：

- `ModelArguments`
- `DataArguments`
- `TrainingArguments`

### 3. 它同时支持 Qwen2-VL 和 Qwen2.5-VL

并且当前版本已经改成可以从 `config.json` 识别模型类型。

### 4. `set_model()` 决定谁训练、谁冻结

这是最核心的训练策略控制器。

### 5. 当前版本非常在意 `transformers 4.53` 的兼容性

尤其是参数归属和 deepspeed 参数解析。

### 6. `data_flatten=True` 不是“小开关”

它会触发 attention 路径的 monkey patch。

### 7. 这个脚本依赖 `trainer.py` 的全局副作用

包括：

- 自定义 optimizer
- 打印函数
- attention patch

### 8. 训练时关闭 `use_cache`

训练完再恢复。

### 9. 保存逻辑区分 DeepSpeed 和普通模式

DeepSpeed 让 DeepSpeed 自己存，普通模式手动搬 CPU 再存。

### 10. 当前 `__main__` 走的是 `sdpa`

这已经不是最初那条 `flash_attention_2` 路线。

---

## 26. 建议你下一步怎么读代码

如果你已经看懂这份文档，推荐继续按下面顺序读：

1. 先读 [`argument.py`](./argument.py)
   目的：搞清楚所有命令行参数到底有哪些

2. 再读 [`trainer.py`](./trainer.py)
   目的：搞清楚 Trainer 被改了哪些行为

3. 再读 [`../data/data_qwen.py`](../data/data_qwen.py)
   目的：搞清楚一条样本是怎么变成模型输入的

4. 再看具体训练脚本
   比如 `scripts/sft_7b.sh` 或 Ascend 相关脚本

这样你就会从：

```text
训练入口
→ 参数
→ Trainer 改动
→ 数据输入
→ 实际启动命令
```

形成一个完整闭环。

---

## 27. 最后给一个“脑内执行过程”

你可以把运行这份脚本时的大脑动画想成这样：

```text
torchrun 启动多个进程
    ↓
train_qwen.py 解析三组参数
    ↓
根据模型路径 / config.json 识别 Qwen2 还是 Qwen2.5
    ↓
加载模型、tokenizer、image_processor
    ↓
如果 data_flatten=True，则给 attention 路径打补丁
    ↓
关闭 use_cache，开启 gradient checkpointing 兼容逻辑
    ↓
根据 tune_mm_* 开关设置 requires_grad
    ↓
根据 data_packing / data_flatten 选择数据模块和 collator
    ↓
创建已经被 trainer.py 改写过行为的 Hugging Face Trainer
    ↓
检查 output_dir 有没有 checkpoint
    ↓
有就恢复，没有就从头训练
    ↓
保存 trainer state、image_processor、模型权重
```

如果你能顺着这条链把每一步为什么存在讲出来，说明你已经真正看懂这份脚本了。

---

## 28. 附：本文件特别适合回答的几个问题

以后你再看这份脚本，如果脑子里冒出下面这些问题，可以直接回到对应章节：

- “为什么明明只 import 了 Trainer，行为却和官方 Trainer 不完全一样？”
  去看第 4 节、第 19 节、第 23 节

- “为什么冻结顺序不能随便改？”
  去看第 7 节

- “`data_flatten` 为什么影响这么大？”
  去看第 13 节

- “为什么训练结束前又把 `use_cache` 打开了？”
  去看第 14 节、第 21 节

- “本地模型路径下，脚本怎么知道这是 Qwen2.5？”
  去看第 11 节

- “为什么保存时还要区分 DeepSpeed 和普通模式？”
  去看第 6 节、第 21 节

---

如果你接下来愿意，我还可以继续在同一目录下再补两份说明：

1. `trainer.py` 逐段详细说明
2. `data_qwen.py` 输入管线详细说明
