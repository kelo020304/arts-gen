"""L2-SP anchor: 梯度钩子实现，把参数拉向预训练初始化值。

论文: Xuhong Li et al., "Explicit Inductive Bias for Transfer Learning
with Convolutional Networks" (ICML 2018)。

实现选择 (CONTEXT Rule B-01): 用 torch.Tensor.register_hook，不改 TRELLIS
trainer MRO 以避免 CLAUDE.md Lessons Learned 中记录的 v0.1.0 review 坑。
DDP/FP16/LoRA 都天然兼容：hook 在每卡 grad 计算后、all-reduce 前触发，
(θ-θ_0) 在各 rank 上相同，all-reduce 平均后值不变。

推荐用法 (CONTEXT Rule B-04):
    from safetensors.torch import load_file
    pretrained_state = load_file(pretrained_ckpt_path)
    anchor = L2SPAnchor.from_state_dict(
        model, pretrained_state, lambda_=1e-4, target='trainable')
    anchor.attach()
    del pretrained_state

公开接口:
    L2SPAnchor.from_state_dict(model, pretrained_state, lambda_, target)  # 推荐构造入口
    L2SPAnchor(model, lambda_, target)                                    # fallback 入口 (deprecated)
    .attach()             # 注册 backward hook
    .detach()             # 移除 hook (可重复调用安全)
    .drift_summary()      # 返回 total/input/io/out/trunk/other 6 个 bucket 的 L2 范数
    .drift_per_layer()    # 返回每参数 L2 范数 (开销较大)
    .state_dict()         # 保存 θ_0 + lambda_ (Rule B-09: 本 Phase 不写入 ckpt)
    .load_state_dict()    # 恢复 θ_0 + lambda_
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class L2SPAnchor:
    """把可训练参数拉向它们的初始值 θ_0。

    推荐用法（CONTEXT Rule B-04）:
        from safetensors.torch import load_file
        pretrained_state = load_file(pretrained_ckpt_path)
        anchor = L2SPAnchor.from_state_dict(
            model, pretrained_state, lambda_=1e-4, target='trainable')
        anchor.attach()
        del pretrained_state
    """

    # 分桶规则 (Rule B-07): {input_layer}, {input_blocks, out_blocks}, {out_layer}, {blocks.}
    # 顺序敏感: input_blocks 先被 io 捕获，不会落到 input；trunk 的 "blocks." 带点，
    # 避免把 input_blocks / out_blocks 误归 trunk。
    _BUCKETS = [
        ("input", ["input_layer"]),
        ("io",    ["input_blocks", "out_blocks"]),
        ("out",   ["out_layer"]),
        ("trunk", ["blocks."]),
    ]

    @staticmethod
    def _candidate_state_keys(name: str) -> List[str]:
        """Return likely pretrained-state keys for a live parameter name.

        PeftModel wraps base-model params under `base_model.model.*`, and LoRA-targeted
        Linear layers expose their frozen base weights as `...base_layer.weight`.
        Strip those wrappers so `pretrained_state` from the original TRELLIS checkpoint
        can still match keep_trainable layers when LoRA + anchor are combined.
        """
        candidates = [name]
        if name.startswith("base_model.model."):
            candidates.append(name[len("base_model.model."):])
        expanded: List[str] = []
        for candidate in candidates:
            expanded.append(candidate)
            if ".base_layer." in candidate:
                expanded.append(candidate.replace(".base_layer.", "."))
        deduped: List[str] = []
        seen = set()
        for candidate in expanded:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    def __init__(
        self,
        model: nn.Module,
        lambda_: float = 1.0e-4,
        target: str = "trainable",
    ):
        if target != "trainable":
            raise NotImplementedError(
                f"target={target} 暂未支持，只实现了 'trainable' (Rule B-03)"
            )
        self.model = model
        self.lambda_ = float(lambda_)
        self.target = target
        self._theta_0: Dict[str, torch.Tensor] = {}
        self._handles = []
        self._snapshot()  # fallback 路径：从 model 当前权重快照

    @classmethod
    def from_state_dict(
        cls,
        model: nn.Module,
        pretrained_state: Dict[str, torch.Tensor],
        lambda_: float = 1.0e-4,
        target: str = "trainable",
    ) -> "L2SPAnchor":
        """推荐入口：从外部 state_dict（通常是 pretrained_ckpt 文件）构造 anchor。

        θ_0 来源与 model 当前权重完全解耦，resume 场景下 model 可从 checkpoint 恢复
        而 θ_0 仍锚定在 pretrained (Rule B-04)。
        """
        if target != "trainable":
            raise NotImplementedError(
                f"target={target} 暂未支持，只实现了 'trainable' (Rule B-03)"
            )
        obj = cls.__new__(cls)
        obj.model = model
        obj.lambda_ = float(lambda_)
        obj.target = target
        obj._theta_0 = {}
        obj._handles = []
        obj._load_from_state_dict(pretrained_state)
        return obj

    def _load_from_state_dict(self, pretrained_state: Dict[str, torch.Tensor]):
        """从外部 state_dict 加载 θ_0（仅对可训练参数保留对应项），dtype/device 对齐 p (Rule B-06)。"""
        missing = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:
                # Adapter 参数在 base pretrained checkpoint 中没有对应项；target=trainable
                # 允许它们无 anchor（Rule B-03 + LoRA 叠加场景）。
                continue
            matched_key = None
            for candidate in self._candidate_state_keys(name):
                if candidate in pretrained_state:
                    matched_key = candidate
                    break
            if matched_key is None:
                missing.append(name)
                continue
            t0 = pretrained_state[matched_key]
            # detach + to(dtype/device) + clone: 与 p 对齐，避免 fp16↔fp32 运算开销 (Rule B-06)
            self._theta_0[name] = t0.detach().to(dtype=p.dtype, device=p.device).clone()
        logger.info(
            f"L2SP snapshot from state_dict: {len(self._theta_0)} trainable "
            f"params anchored, lambda={self.lambda_}"
        )
        if missing:
            logger.warning(
                f"L2SP: {len(missing)} trainable params not found in pretrained "
                f"state_dict (anchor NOT applied to these). Examples: {missing[:5]}"
            )

    def _snapshot(self):
        """[Deprecated fallback] 从 model 当前权重快照 θ_0。

        仅供 __init__ 路径使用。推荐使用 from_state_dict (Rule B-04)，
        避免 resume 时 model 当前权重已偏离 pretrained 的问题。
        """
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self._theta_0[name] = p.detach().clone()
        logger.info(
            f"L2SP snapshot from model: {len(self._theta_0)} trainable params "
            f"anchored, lambda={self.lambda_}"
        )

    def attach(self):
        """为每个可训练参数注册 backward hook (Rule B-05)。

        hook 公式: grad_eff = grad_task + 2λ(θ − θ_0)
        等价于 loss 加 λ‖θ − θ_0‖²。
        """
        assert not self._handles, "attach() 不能重复调用"
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self._theta_0:
                continue
            theta_0 = self._theta_0[name]
            lam = self.lambda_

            # 闭包陷阱防御: 必须显式传入 (p, theta_0, lam)，不能直接用循环变量
            # 否则所有 hook 共享最后一次迭代的 p/theta_0/lam 引用
            def make_hook(p_ref, t0_ref, lam_ref):
                def hook(grad):
                    return grad + 2.0 * lam_ref * (p_ref.data - t0_ref)
                return hook

            h = p.register_hook(make_hook(p, theta_0, lam))
            self._handles.append(h)
        logger.info(f"L2SP attached {len(self._handles)} hooks")

    def detach(self):
        """移除所有 hook（可重复调用安全）。"""
        for h in self._handles:
            h.remove()
        self._handles = []

    def drift_summary(self) -> Dict[str, float]:
        """返回分桶 drift summary (L2 范数)，不打印、不 log。

        分桶规则 (Rule B-07):
            input  = [input_layer]
            io     = [input_blocks, out_blocks]
            out    = [out_layer]
            trunk  = [blocks.]        # 末尾带点，避免误匹配 input_blocks/out_blocks
            other  = 未命中上述任一模式的参数

        fp16 安全: `.float().pow(2).sum()` 先升 fp32 再平方求和，避免 fp16 溢出。

        Returns:
            {"total", "input", "io", "out", "trunk", "other"} → float (L2 范数，非平方)
        """
        sq = {k: 0.0 for k in ["input", "io", "out", "trunk", "other"]}
        total_sq = 0.0
        for name, p in self.model.named_parameters():
            if name not in self._theta_0:
                continue
            d = (p.data - self._theta_0[name]).float().pow(2).sum().item()
            total_sq += d
            bucket = "other"
            for b_name, patterns in self._BUCKETS:
                if any(pat in name for pat in patterns):
                    bucket = b_name
                    break
            sq[bucket] += d
        return {
            "total": total_sq ** 0.5,
            **{k: v ** 0.5 for k, v in sq.items()},
        }

    def drift_per_layer(self) -> Dict[str, float]:
        """返回每参数的 L2 范数（开销较大，调用方控制频率）。"""
        out = {}
        for name, p in self.model.named_parameters():
            if name not in self._theta_0:
                continue
            out[name] = (p.data - self._theta_0[name]).float().norm().item()
        return out

    def state_dict(self):
        """保存 θ_0 + lambda_ 供未来 resume 使用 (Rule B-09: 本 Phase 不写入 checkpoint，但接口先留)。"""
        return {"theta_0": self._theta_0, "lambda_": self.lambda_}

    def load_state_dict(self, state: dict):
        self._theta_0 = state["theta_0"]
        self.lambda_ = state["lambda_"]
