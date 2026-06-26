# Dense Part Flow 修复实现计划

> **给 agentic workers 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`。所有任务使用 checkbox (`- [ ]`) 跟踪。

**目标：** 将 Part Flow 从 sparse SS coords labeling 修正为 dense `64^3` surface-to-solid conditional Fisher flow，并移除 dense voxel global self-attention。

**架构：** 第一版主线保留 Fisher categorical flow。模型输出 `endpoint_logits = p_theta(x_1 | x_t,t,C)`，`FisherBridge` 负责 Fisher-Rao 几何 step。条件 `C` 包含 SS surface mask、DINO condition tokens、mask-pooled part tokens、valid part masks。模型结构改为 `PartTokenTransformer + ConditionTokenCompressor + DenseVoxelDecoder`，dense voxel 端不做 voxel-to-voxel global self-attention。

**技术栈：** PyTorch、现有 `TRELLIS-arts` Part Flow 代码、`FisherBridge`、`FlowMatchingLoss`、pytest、YAML config。

---

## 已锁定设计决策

- `flow.type: fisher` 是第一版主线。
- 暴露 `flow.dirichlet_alpha: 1.0`，用于 `FisherBridge.sample_source`。
- 模型输出保持 endpoint logits，不新增 velocity head。
- `PartTokenTransformer` 是非 causal / bidirectional transformer，只作用在 `K+1` 个 part tokens 上。
- `DenseVoxelDecoder` 使用 4 层。
- 每层顺序固定为 `part cross-attn -> condition cross-attn -> FFN`。
- dense voxel chunk 内不做 voxel self-attention。
- condition compression 使用 learnable queries：每个 view 3 个 tokens，加 1 个 global token，并加入 view/token/type positional embeddings。
- compressed condition tokens 作为 dense decoder 的 K/V。
- endpoint head 只使用 `voxel_query · part_tokens_refined`，不加入额外 `Linear(H, K_max)` fallback。
- slot id embedding 写入 config，默认关闭：`model.use_slot_id_embedding: false`，`model.slot_id_embedding_scale: 0.1`。
- `model.voxel_chunk_size: 32768` 是 eval/inference chunking 的统一配置。训练主 forward 保持一次性 full dense `64^3`。
- `model.use_gradient_checkpointing: false` 写入 config，默认关闭。
- inference wrapper 必须枚举完整 dense `64^3` coords，并把 SS surface coords scatter 成 `is_on_surface`。不能只在 sparse SS coords 上预测 label 后再 densify。

## 文件边界

- 修改 `TRELLIS-arts/trellis/models/part_flow/bridges.py`
  - 为 `FisherBridge` 增加 `dirichlet_alpha`。
  - 保持 valid-slot / padding discipline 不变。

- 修改 `TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py`
  - 替换包含 voxel self-attention 和 full RGB cross-attention 的旧 `PartFlowDecoderLayer`。
  - 新增 `ConditionTokenCompressor`。
  - 新增 `PartTokenTransformer`。
  - 新增 `DenseVoxelDecoderLayer`。
  - 新增 slot id embedding、condition compression、chunk size、gradient checkpointing 相关 config。

- 修改 `TRELLIS-arts/trellis/trainers/arts/part_flow.py`
  - 将 `flow.dirichlet_alpha` 传给 bridge。
  - 保持训练 full dense 一次性 forward。
  - checkpoint 保存新 model config。

- 修改 `TRELLIS-arts/trellis/trainers/arts/part_flow_losses.py`
  - 为 eval/inference sampling 增加 chunked model call。
  - 不改变 `FlowMatchingLoss.forward` 的训练路径。

- 修改 `TRELLIS-arts/inference.py`
  - 将 `run_part_flow` 改成真正 dense completion。
  - 使用 full `64^3` coords 和 SS-derived `is_on_surface`。
  - sampling 使用 `model.voxel_chunk_size`。

- 修改 `TRELLIS-arts/configs/arts/part_flow/base.yaml`
  - 增加 `flow.dirichlet_alpha`。
  - 更新 model 默认配置。

- 新增或更新测试：
  - `TRELLIS-arts/tests/arts/part_flow/test_fisher_dirichlet_alpha.py`
  - `TRELLIS-arts/tests/arts/part_flow/test_condition_compressor.py`
  - `TRELLIS-arts/tests/arts/part_flow/test_dense_decoder_no_voxel_self_attention.py`
  - `TRELLIS-arts/tests/arts/part_flow/test_flow_sample_chunking.py`
  - 更新 `test_model_shape.py`
  - 更新 `test_inference_contract.py`

## Task 1: 增加 Fisher `dirichlet_alpha`

**文件：**
- 修改：`TRELLIS-arts/trellis/models/part_flow/bridges.py`
- 修改：`TRELLIS-arts/trellis/trainers/arts/part_flow.py`
- 修改：`TRELLIS-arts/inference.py`
- 修改：`TRELLIS-arts/configs/arts/part_flow/base.yaml`
- 新增：`TRELLIS-arts/tests/arts/part_flow/test_fisher_dirichlet_alpha.py`

- [ ] **Step 1: 写失败测试**

创建 `TRELLIS-arts/tests/arts/part_flow/test_fisher_dirichlet_alpha.py`：

```python
import torch

from trellis.models.part_flow.bridges import FisherBridge, build_bridge


def test_fisher_bridge_accepts_dirichlet_alpha():
    bridge = FisherBridge(k_max=5, dirichlet_alpha=0.5)
    assert bridge.dirichlet_alpha == 0.5


def test_build_bridge_forwards_dirichlet_alpha():
    bridge = build_bridge('fisher', k_max=5, dirichlet_alpha=2.0)
    assert isinstance(bridge, FisherBridge)
    assert bridge.dirichlet_alpha == 2.0


def test_fisher_source_respects_valid_simplex_with_alpha():
    torch.manual_seed(0)
    bridge = FisherBridge(k_max=6, dirichlet_alpha=0.5)
    x0 = bridge.sample_source(
        num_parts=[3, 5],
        n_per_sample=[7, 11],
        device=torch.device('cpu'),
    )
    assert x0.shape == (18, 6)
    assert torch.allclose(x0[:7, :3].sum(dim=-1), torch.ones(7), atol=1e-5)
    assert torch.allclose(x0[7:, :5].sum(dim=-1), torch.ones(11), atol=1e-5)
    assert x0[:7, 3:].abs().max().item() == 0.0
    assert x0[7:, 5:].abs().max().item() == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=TRELLIS-arts pytest TRELLIS-arts/tests/arts/part_flow/test_fisher_dirichlet_alpha.py -q
```

预期：失败，报 `FisherBridge.__init__` 不接受 `dirichlet_alpha`。

- [ ] **Step 3: 实现 `dirichlet_alpha`**

在 `FisherBridge.__init__` 增加参数：

```python
def __init__(
    self,
    k_max: int,
    t_max: float = 1.0,
    eps: float = 1e-6,
    dirichlet_alpha: float = 1.0,
):
    super().__init__(k_max, t_max)
    assert dirichlet_alpha > 0.0, f'dirichlet_alpha must be > 0, got {dirichlet_alpha}'
    self.eps = eps
    self.dirichlet_alpha = float(dirichlet_alpha)
```

在 `sample_source` 中将：

```python
alphas = torch.ones(n_b, K_b, device=device, dtype=dtype)
```

改为：

```python
alphas = torch.full(
    (n_b, K_b),
    self.dirichlet_alpha,
    device=device,
    dtype=dtype,
)
```

- [ ] **Step 4: train/inference bridge 构造传递该配置**

在 `TRELLIS-arts/trellis/trainers/arts/part_flow.py` 和 `TRELLIS-arts/inference.py` 的 bridge key 列表中加入：

```python
'dirichlet_alpha'
```

- [ ] **Step 5: 更新 config**

在 `TRELLIS-arts/configs/arts/part_flow/base.yaml` 的 `flow:` 下加入：

```yaml
  dirichlet_alpha: 1.0          # Fisher source Dirichlet concentration on valid simplex slots
```

- [ ] **Step 6: 验证**

```bash
PYTHONPATH=TRELLIS-arts pytest \
  TRELLIS-arts/tests/arts/part_flow/test_fisher_dirichlet_alpha.py \
  TRELLIS-arts/tests/arts/part_flow/test_bridges.py -q
```

预期：全部通过。

## Task 2: 实现 condition compression

**文件：**
- 修改：`TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py`
- 新增：`TRELLIS-arts/tests/arts/part_flow/test_condition_compressor.py`

- [ ] **Step 1: 写 compressor 形状测试**

创建 `TRELLIS-arts/tests/arts/part_flow/test_condition_compressor.py`：

```python
import torch

from trellis.models.part_flow.part_flow_predictor import ConditionTokenCompressor


def test_condition_compressor_outputs_three_tokens_per_view_plus_global():
    torch.manual_seed(0)
    comp = ConditionTokenCompressor(
        dim=32,
        num_heads=4,
        num_view_tokens=3,
        max_views=8,
    )
    cond = torch.randn(2, 4 * 17, 32)
    out = comp(cond, num_views=4)
    assert out.shape == (2, 13, 32)


def test_condition_compressor_uses_distinct_positional_embeddings():
    torch.manual_seed(1)
    comp = ConditionTokenCompressor(
        dim=16,
        num_heads=4,
        num_view_tokens=3,
        max_views=4,
    )
    cond = torch.ones(1, 2 * 5, 16)
    out = comp(cond, num_views=2)
    assert out.shape == (1, 7, 16)
    assert not torch.allclose(out[:, 0], out[:, 1])
    assert not torch.allclose(out[:, 0], out[:, -1])
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=TRELLIS-arts pytest TRELLIS-arts/tests/arts/part_flow/test_condition_compressor.py -q
```

预期：失败，因为 `ConditionTokenCompressor` 尚不存在。

- [ ] **Step 3: 新增 `ConditionTokenCompressor`**

在 `part_flow_predictor.py` 中新增：

```python
class ConditionTokenCompressor(nn.Module):
    """Compress per-view condition tokens to 3 tokens/view + 1 global token."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_view_tokens: int = 3,
        max_views: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.num_view_tokens = int(num_view_tokens)
        self.max_views = int(max_views)
        self.view_queries = nn.Parameter(torch.randn(num_view_tokens, dim) * 0.02)
        self.global_query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.view_id_emb = nn.Embedding(max_views, dim)
        self.view_token_emb = nn.Embedding(num_view_tokens, dim)
        self.global_type_emb = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.view_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.view_norm = nn.LayerNorm(dim, eps=1e-6)
        self.global_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, cond_proj: torch.Tensor, num_views: int) -> torch.Tensor:
        B, VT, H = cond_proj.shape
        assert H == self.dim
        assert 1 <= num_views <= self.max_views
        assert VT % num_views == 0, f'cond token count {VT} not divisible by num_views={num_views}'
        tokens_per_view = VT // num_views
        cond_by_view = cond_proj.view(B, num_views, tokens_per_view, H)

        view_outputs = []
        token_ids = torch.arange(self.num_view_tokens, device=cond_proj.device)
        token_pos = self.view_token_emb(token_ids).unsqueeze(0)
        for v in range(num_views):
            q = self.view_queries.unsqueeze(0).expand(B, -1, -1)
            q = q + token_pos
            q = q + self.view_id_emb.weight[v].view(1, 1, H)
            out, _ = self.view_attn(q, cond_by_view[:, v], cond_by_view[:, v], need_weights=False)
            view_outputs.append(self.view_norm(out))
        view_tokens = torch.cat(view_outputs, dim=1)

        q_global = self.global_query.unsqueeze(0).expand(B, -1, -1) + self.global_type_emb.unsqueeze(0)
        global_token, _ = self.global_attn(q_global, view_tokens, view_tokens, need_weights=False)
        global_token = self.global_norm(global_token)
        return torch.cat([view_tokens, global_token], dim=1)
```

- [ ] **Step 4: 验证**

```bash
PYTHONPATH=TRELLIS-arts pytest TRELLIS-arts/tests/arts/part_flow/test_condition_compressor.py -q
```

预期：通过。

## Task 3: 替换 voxel global self-attention

**文件：**
- 修改：`TRELLIS-arts/trellis/models/part_flow/part_flow_predictor.py`
- 修改：`TRELLIS-arts/tests/arts/part_flow/test_model_shape.py`
- 新增：`TRELLIS-arts/tests/arts/part_flow/test_dense_decoder_no_voxel_self_attention.py`

- [ ] **Step 1: 写无 voxel self-attention 测试**

创建 `TRELLIS-arts/tests/arts/part_flow/test_dense_decoder_no_voxel_self_attention.py`：

```python
from trellis.models.part_flow.part_flow_predictor import (
    DenseVoxelDecoderLayer,
    PartFlowPredictor,
    PartTokenTransformer,
)


def test_dense_decoder_layer_has_no_voxel_self_attention_weights():
    layer = DenseVoxelDecoderLayer(dim=32, num_heads=4, dropout=0.0)
    names = dict(layer.named_parameters()).keys()
    forbidden = ('self_q', 'self_k', 'self_v', 'self_o')
    assert not any(any(f in name for f in forbidden) for name in names)


def test_predictor_uses_part_token_transformer_and_four_dense_layers():
    model = PartFlowPredictor(
        k_max=8,
        hidden_dim=32,
        num_layers=4,
        num_heads=4,
        cond_dim=16,
        num_views=4,
    )
    assert isinstance(model.part_token_transformer, PartTokenTransformer)
    assert len(model.decoder_layers) == 4
    assert all(isinstance(layer, DenseVoxelDecoderLayer) for layer in model.decoder_layers)
```

- [ ] **Step 2: 新增 `PartTokenTransformer`**

```python
class PartTokenTransformer(nn.Module):
    """Bidirectional transformer encoder over sample-local valid part slots."""

    def __init__(self, dim: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, part_tokens: torch.Tensor, part_valid_mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~part_valid_mask
        out = self.encoder(part_tokens, src_key_padding_mask=padding_mask)
        out = out * part_valid_mask.unsqueeze(-1).to(out.dtype)
        return out
```

- [ ] **Step 3: 新增 `DenseVoxelDecoderLayer`**

```python
class DenseVoxelDecoderLayer(nn.Module):
    """Voxel decoder layer: part cross-attn -> condition cross-attn -> FFN."""

    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.norm_part = nn.LayerNorm(dim, eps=1e-6)
        self.part_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cond = nn.LayerNorm(dim, eps=1e-6)
        self.cond_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _modulate(self, x, shift, scale):
        return x * (1.0 + scale) + shift

    def forward(
        self,
        voxel_tokens: torch.Tensor,
        t_emb_per_voxel: torch.Tensor,
        part_tokens: torch.Tensor,
        part_valid_mask: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.ada_ln_modulation(t_emb_per_voxel)
        shift_part, scale_part, gate_part, shift_ffn, scale_ffn, gate_ffn = mod.chunk(6, dim=-1)

        q = self._modulate(self.norm_part(voxel_tokens), shift_part, scale_part).unsqueeze(0)
        part_out, _ = self.part_attn(
            q,
            part_tokens.unsqueeze(0),
            part_tokens.unsqueeze(0),
            key_padding_mask=(~part_valid_mask).unsqueeze(0),
            need_weights=False,
        )
        voxel_tokens = voxel_tokens + gate_part * self.dropout(part_out.squeeze(0))

        q = self.norm_cond(voxel_tokens).unsqueeze(0)
        cond_out, _ = self.cond_attn(
            q,
            cond_tokens.unsqueeze(0),
            cond_tokens.unsqueeze(0),
            need_weights=False,
        )
        voxel_tokens = voxel_tokens + self.dropout(cond_out.squeeze(0))

        h = self._modulate(self.norm_ffn(voxel_tokens), shift_ffn, scale_ffn)
        voxel_tokens = voxel_tokens + gate_ffn * self.dropout(self.ffn(h))
        return voxel_tokens
```

- [ ] **Step 4: 更新 `PartFlowPredictor.__init__`**

新增参数：

```python
num_views: int = 4,
part_token_layers: int = 2,
condition_tokens_per_view: int = 3,
voxel_chunk_size: int = 32768,
use_slot_id_embedding: bool = False,
slot_id_embedding_scale: float = 0.1,
use_gradient_checkpointing: bool = False,
```

新增模块：

```python
self.condition_compressor = ConditionTokenCompressor(
    hidden_dim,
    num_heads,
    num_view_tokens=condition_tokens_per_view,
    max_views=max(16, int(num_views)),
    dropout=dropout,
)
self.part_token_transformer = PartTokenTransformer(
    hidden_dim,
    num_heads,
    num_layers=part_token_layers,
    dropout=dropout,
)
self.decoder_layers = nn.ModuleList([
    DenseVoxelDecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_layers)
])
```

- [ ] **Step 5: 更新 forward 数据流**

在 `build_part_tokens` 后：

```python
if self.slot_id_emb is not None:
    slot_ids = torch.arange(self.k_max, device=device)
    slot_hint = self.slot_id_emb(slot_ids).unsqueeze(0).to(part_tokens.dtype)
    real_part_mask = part_valid_mask.clone()
    real_part_mask[:, 0] = False
    part_tokens = part_tokens + (
        self.slot_id_embedding_scale
        * slot_hint
        * real_part_mask.unsqueeze(-1).to(part_tokens.dtype)
    )

part_tokens = self.part_token_transformer(part_tokens, part_valid_mask)
part_tokens[:, 0, :] = self.empty_token.to(part_tokens.dtype).unsqueeze(0).expand(B, H)
part_tokens = part_tokens * part_valid_mask.unsqueeze(-1).to(part_tokens.dtype)
cond_tokens = self.condition_compressor(cond_proj, num_views=self.num_views)
```

decoder 对每个 sample 的 voxel rows 分别运行，不做 voxel self-attention：

```python
offset = 0
for b, n_b in enumerate(n_seqlen):
    sl = slice(offset, offset + n_b)
    h = voxel_tokens[sl]
    t_h = t_emb_per_voxel[sl]
    part_b = part_tokens[b]
    valid_b = part_valid_mask[b]
    cond_b = cond_tokens[b]
    for layer in self.decoder_layers:
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(layer, h, t_h, part_b, valid_b, cond_b, use_reentrant=False)
        else:
            h = layer(h, t_h, part_b, valid_b, cond_b)
    voxel_tokens[sl] = h
    offset += n_b
```

- [ ] **Step 6: 保持 dot-product endpoint head**

保留：

```python
logits[sl, :K_b] = (voxel_q[sl] @ part_k[b, :K_b].T) / scale
```

不要加入 `nn.Linear(hidden_dim, k_max)`。

- [ ] **Step 7: 验证**

```bash
PYTHONPATH=TRELLIS-arts pytest \
  TRELLIS-arts/tests/arts/part_flow/test_condition_compressor.py \
  TRELLIS-arts/tests/arts/part_flow/test_dense_decoder_no_voxel_self_attention.py \
  TRELLIS-arts/tests/arts/part_flow/test_model_shape.py \
  TRELLIS-arts/tests/arts/part_flow/test_empty_token.py \
  TRELLIS-arts/tests/arts/part_flow/test_variable_k.py -q
```

预期：全部通过。

## Task 4: 增加 eval/inference chunked sampling

**文件：**
- 修改：`TRELLIS-arts/trellis/trainers/arts/part_flow_losses.py`
- 新增：`TRELLIS-arts/tests/arts/part_flow/test_flow_sample_chunking.py`

- [ ] **Step 1: 写 chunking 测试**

创建 `TRELLIS-arts/tests/arts/part_flow/test_flow_sample_chunking.py`：

```python
import torch

from trellis.models.part_flow.bridges import build_bridge
from trellis.trainers.arts.part_flow_losses import flow_sample


class CountingModel(torch.nn.Module):
    def __init__(self, k_max):
        super().__init__()
        self.k_max = k_max
        self.voxel_chunk_size = 4
        self.calls = []

    def forward(self, x_t, t, coords, cond, mask_token_labels, num_parts, is_on_surface):
        self.calls.append(coords.shape[0])
        logits = torch.zeros(coords.shape[0], self.k_max, device=coords.device)
        logits[:, 0] = 1.0
        return {'endpoint_logits': logits}


def test_flow_sample_chunks_eval_model_forward():
    bridge = build_bridge('fisher', k_max=3)
    model = CountingModel(k_max=3)
    coords = torch.cat([
        torch.zeros(10, 1, dtype=torch.long),
        torch.randint(0, 64, (10, 3), dtype=torch.long),
    ], dim=1)
    labels, soft = flow_sample(
        model,
        bridge,
        coords=coords,
        cond=torch.randn(1, 8, 16),
        mask_token_labels=torch.zeros(1, 8, dtype=torch.long),
        voxel_layout=[slice(0, 10)],
        num_parts=[3],
        is_on_surface=torch.zeros(10, dtype=torch.long),
        num_steps=1,
        solver='euler',
    )
    assert model.calls == [4, 4, 2]
    assert labels.shape == (10,)
    assert soft.shape == (10, 3)
```

- [ ] **Step 2: 实现 `_model_endpoint_logits_chunked`**

在 `part_flow_losses.py` 中增加 helper：

```python
def _model_endpoint_logits_chunked(
    model: nn.Module,
    x_t: torch.Tensor,
    t_batch: torch.Tensor,
    coords: torch.Tensor,
    cond: torch.Tensor,
    mask_token_labels: torch.Tensor,
    voxel_layout: List[slice],
    num_parts: List[int],
    is_on_surface: torch.Tensor,
) -> torch.Tensor:
    chunk_size = int(getattr(model, 'voxel_chunk_size', 0) or 0)
    N_total = coords.shape[0]
    if chunk_size <= 0 or chunk_size >= N_total:
        return model(
            x_t, t_batch, coords, cond, mask_token_labels, num_parts, is_on_surface,
        )['endpoint_logits']

    assert len(voxel_layout) == 1, (
        'chunked flow_sample currently supports eval/inference batch size 1; '
        'training forward remains full dense and unchunked'
    )
    logits_chunks = []
    for start in range(0, N_total, chunk_size):
        end = min(start + chunk_size, N_total)
        out = model(
            x_t[start:end],
            t_batch,
            coords[start:end],
            cond,
            mask_token_labels,
            [slice(0, end - start)],
            num_parts,
            is_on_surface[start:end],
        )
        logits_chunks.append(out['endpoint_logits'])
    return torch.cat(logits_chunks, dim=0)
```

- [ ] **Step 3: 在 `flow_sample` 中使用 helper**

将 Euler 和 Heun 中直接调用 `model(...)` 的地方替换为 `_model_endpoint_logits_chunked(...)`。

- [ ] **Step 4: 验证**

```bash
PYTHONPATH=TRELLIS-arts pytest TRELLIS-arts/tests/arts/part_flow/test_flow_sample_chunking.py -q
```

预期：通过。

## Task 5: 将 inference wrapper 改成 true dense completion

**文件：**
- 修改：`TRELLIS-arts/inference.py`
- 修改：`TRELLIS-arts/tests/arts/part_flow/test_inference_contract.py`

- [ ] **Step 1: 更新 inference contract 测试**

将 `test_run_part_flow_forwards_mask_labels_and_num_parts` 改为 dense-grid 测试：

```python
def test_run_part_flow_enumerates_dense_grid_and_scatters_surface(monkeypatch):
    inference = _patch_part_flow_loader(monkeypatch)
    losses = importlib.import_module("trellis.trainers.arts.part_flow_losses")

    captured = {}

    def fake_flow_sample(
        model,
        bridge,
        coords,
        cond,
        mask_token_labels,
        voxel_layout,
        num_parts,
        is_on_surface,
        num_steps,
        solver,
    ):
        captured["coords"] = coords.detach().clone()
        captured["mask_token_labels"] = mask_token_labels.detach().clone()
        captured["num_parts"] = list(num_parts)
        captured["is_on_surface"] = is_on_surface.detach().clone()
        n = coords.shape[0]
        return torch.zeros(n, dtype=torch.long), torch.zeros(n, bridge.k_max)

    monkeypatch.setattr(losses, "flow_sample", fake_flow_sample)

    surface_coords = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    cond_tokens = torch.randn(4, 1024)
    mask_token_labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    inference.run_part_flow(
        surface_coords,
        cond_tokens,
        "dummy.pt",
        mask_token_labels=mask_token_labels,
        num_parts=3,
        num_steps=1,
    )

    assert captured["coords"].shape == (64 ** 3, 4)
    assert captured["coords"][0].tolist() == [0, 0, 0, 0]
    assert captured["coords"][-1].tolist() == [0, 63, 63, 63]
    assert captured["mask_token_labels"].shape == (1, 4)
    assert captured["num_parts"] == [3]

    surface = captured["is_on_surface"].reshape(64, 64, 64)
    assert int(surface.sum().item()) == 2
    assert surface[1, 2, 3].item() == 1
    assert surface[4, 5, 6].item() == 1
```

- [ ] **Step 2: 新增 dense coords helper**

在 `inference.py` 中新增：

```python
def _dense_part_flow_coords(device: torch.device) -> torch.Tensor:
    xs = torch.arange(64, dtype=torch.long, device=device)
    gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing="ij")
    xyz = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    batch_idx = torch.zeros((xyz.shape[0], 1), dtype=torch.long, device=device)
    return torch.cat([batch_idx, xyz], dim=1)
```

- [ ] **Step 3: scatter SS surface coords**

在 `run_part_flow` 中用 full dense coords 替代 sparse coords：

```python
surface_coords_xyz = coords.long().cuda()
if surface_coords_xyz.dim() != 2 or surface_coords_xyz.shape[1] != 3:
    raise ValueError(f"coords expected [N,3] SS surface coords, got {tuple(surface_coords_xyz.shape)}")
if surface_coords_xyz.numel() > 0:
    if surface_coords_xyz.min().item() < 0 or surface_coords_xyz.max().item() >= 64:
        raise ValueError("coords must be within [0, 63] for dense 64^3 Part Flow")

coords_dev = _dense_part_flow_coords(surface_coords_xyz.device)
N = coords_dev.shape[0]
is_on_surface = torch.zeros(N, dtype=torch.long, device=coords_dev.device)
flat = (
    surface_coords_xyz[:, 0] * 64 * 64
    + surface_coords_xyz[:, 1] * 64
    + surface_coords_xyz[:, 2]
)
is_on_surface[flat] = 1
```

- [ ] **Step 4: dense 输出直接 reshape**

将 sparse scatter 输出改成：

```python
k_max = bridge.k_max
soft_vol = soft.detach().float().cpu().numpy().reshape(64, 64, 64, k_max).astype(np.float16)
hard_vol = labels.detach().cpu().numpy().reshape(64, 64, 64).astype(np.int64)
```

- [ ] **Step 5: 验证**

```bash
PYTHONPATH=TRELLIS-arts pytest TRELLIS-arts/tests/arts/part_flow/test_inference_contract.py -q
```

预期：全部通过。

## Task 6: 更新 config 与 checkpoint 兼容

**文件：**
- 修改：`TRELLIS-arts/configs/arts/part_flow/base.yaml`
- 修改：`TRELLIS-arts/configs/arts/part_flow/smoke_4090.yaml`
- 修改：`TRELLIS-arts/inference.py`

- [ ] **Step 1: 更新 base config**

将 `model:` 更新为：

```yaml
model:
  hidden_dim: 512
  num_layers: 4                  # DenseVoxelDecoder layers
  num_heads: 8
  cond_dim: 1024                 # DINOv2-L/14-reg
  dropout: 0.1
  num_views: 4                   # cond is interpreted as [B, num_views*T, D]
  part_token_layers: 2           # non-causal transformer over K+1 part tokens
  condition_tokens_per_view: 3   # compressed condition K/V tokens per view
  voxel_chunk_size: 32768        # eval/inference chunk size; does not change dense 64^3 semantics
  use_slot_embedding_fallback: true
  use_slot_id_embedding: false   # sample-local slot hint, not semantic class id
  slot_id_embedding_scale: 0.1
  use_gradient_checkpointing: false # saves training memory when true, but slows training
```

- [ ] **Step 2: 更新 smoke config**

若 `TRELLIS-arts/configs/arts/part_flow/smoke_4090.yaml` 覆盖 `model.num_layers`，改为 4 或删除覆盖。可保留轻量设置：

```yaml
model:
  hidden_dim: 128
  num_layers: 4
  voxel_chunk_size: 8192
```

- [ ] **Step 3: 旧 checkpoint 兼容**

在 `_load_part_flow` 中给旧 checkpoint 增加默认值：

```python
model_cfg.setdefault("num_views", 4)
model_cfg.setdefault("part_token_layers", 2)
model_cfg.setdefault("condition_tokens_per_view", 3)
model_cfg.setdefault("voxel_chunk_size", 32768)
model_cfg.setdefault("use_slot_id_embedding", False)
model_cfg.setdefault("slot_id_embedding_scale", 0.1)
model_cfg.setdefault("use_gradient_checkpointing", False)
```

- [ ] **Step 4: config smoke 验证**

```bash
PYTHONPATH=TRELLIS-arts python - <<'PY'
from trellis.utils.arts.config_utils import load_config, config_to_dict
from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor
cfg = load_config('TRELLIS-arts/configs/arts/part_flow/base.yaml')
d = config_to_dict(cfg)
model_cfg = dict(d['model'])
model_cfg['k_max'] = int(d['flow']['k_max'])
model = PartFlowPredictor(**model_cfg)
print(type(model).__name__, model.num_layers, model.voxel_chunk_size)
PY
```

预期输出包含：

```text
PartFlowPredictor 4 32768
```

## Task 7: 集成验证

**文件：**
- 不新增文件；若测试暴露遗漏，再做最小修复。

- [ ] **Step 1: 运行 Part Flow focused tests**

```bash
PYTHONPATH=TRELLIS-arts pytest TRELLIS-arts/tests/arts/part_flow -q
```

预期：全部通过。

- [ ] **Step 2: CPU forward sanity check**

```bash
PYTHONPATH=TRELLIS-arts python - <<'PY'
import torch
from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor

torch.manual_seed(0)
model = PartFlowPredictor(
    k_max=6,
    hidden_dim=32,
    num_layers=4,
    num_heads=4,
    cond_dim=16,
    num_views=4,
    voxel_chunk_size=8,
)
N = 12
K = 4
x_t = torch.zeros(N, 6)
x_t[:, :K] = torch.softmax(torch.randn(N, K), dim=-1)
coords = torch.cat([
    torch.zeros(N, 1, dtype=torch.long),
    torch.randint(0, 64, (N, 3), dtype=torch.long),
], dim=1)
cond = torch.randn(1, 4 * 8, 16)
mask = torch.zeros(1, 4 * 8, dtype=torch.long)
mask[0, 0] = 1
mask[0, 1] = 2
mask[0, 2] = 3
out = model(
    x_t,
    torch.tensor([0.5]),
    coords,
    cond,
    mask,
    [K],
    torch.zeros(N, dtype=torch.long),
)
print(out['endpoint_logits'].shape)
assert out['endpoint_logits'].shape == (N, 6)
assert torch.isfinite(out['endpoint_logits'][out['valid_per_voxel']]).all()
PY
```

预期输出：

```text
torch.Size([12, 6])
```

- [ ] **Step 3: 实施后维护 code update 文档**

代码实现和测试完成后，调用 `code-update-log` skill，更新：

```text
code_update/code_update_flow.md
```

文档使用中文，最新轮次放最上面。

## 自检

- 需求覆盖：
  - Fisher 主线和 `flow.dirichlet_alpha`：Task 1。
  - 非 causal part-token transformer：Task 3。
  - 移除 voxel global self-attention：Task 3。
  - condition compression `3 tokens/view + 1 global`：Task 2 和 Task 6。
  - dense decoder 4 层和 layer 顺序：Task 3 和 Task 6。
  - dot-product endpoint head only：Task 3。
  - `model.voxel_chunk_size`：Task 4 和 Task 6。
  - `model.use_gradient_checkpointing`：Task 3 和 Task 6。
  - dense inference completion：Task 5。

- 完整性检查：
  - 每个实现任务都有明确文件、代码片段、命令和预期结果。
  - 没有遗留未决设计项。
  - `num_parts` 仍表示 `K+1`，包含 empty slot 0。
  - `mask_token_labels` 仍是 `[B, V*T]`，取值范围 `[0, num_parts-1]`。
  - `endpoint_logits` 仍是 `[N_total, k_max]`。
  - padding slots 仍 mask 到 `-1e4`。

## 执行选项

计划已保存到 `docs/designs/flow_fix_plan0.md`。下一步有两个执行方式：

1. **Subagent-Driven（推荐）**：每个任务派一个 fresh subagent 执行，任务间做 review。
2. **Inline Execution**：在当前会话按计划逐项执行，并在阶段之间检查。
