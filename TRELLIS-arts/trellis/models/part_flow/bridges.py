"""Categorical Flow Matching bridges for variable-K part prediction.

Each sample has its own ``num_parts`` (``K_b <= k_max``). Simplex tensors are
PADDED to ``k_max``; all operations respect a per-sample ``part_valid_mask``
so padding dims never enter any softmax, loss, projection, or sampling.

Active bridges (this revision)
==============================

- :class:`FisherBridge` — Fisher-Rao Flow Matching (Davis et al., NeurIPS 2024,
  arXiv:2405.14664). Maps the simplex to the positive orthant of a sphere via
  ``u = sqrt(p)``; conditional paths are spherical geodesics (slerp). Velocity
  is the geodesic tangent; step is a spherical exp-map toward the predicted
  endpoint. Exact: sphere geometry. Practical simplification: we simulate
  the ODE with exp-map-based Euler steps rather than integrating along the
  Fisher-Rao vector field analytically, which matches the paper's "geodesic
  interpolation" parameterization.

- :class:`GumbelSoftmaxBridge` — Gumbel-Softmax Flow Matching (Tang et al.,
  arXiv:2503.17361, "Gumbel-Softmax Flow Matching with Straight-Through
  Guidance for Controllable Biological Sequence Generation").

    * Interpolant (paper Eq. 9):
        x_t_i = softmax_i((delta_ik + g_i/beta) / tau(t))
      where ``g`` is Gumbel noise (i.i.d. per dim, ``g_i = -log(-log U_i)``,
      ``U_i ~ Uniform(0, 1)``), ``delta_ik`` is 1 iff ``i == k`` (target), and
      ``tau(t)`` is a time-dependent temperature.

    * Temperature schedule (paper Eq. 8):
        tau(t) = tau_max * exp(-lambda * t),   t ∈ [0, 1].

    * Conditional velocity (paper Eq. 13, with g=0 taken at inference):
        u_t(x_t | x_1=e_k)_i = (lambda / tau(t)) * x_{t,k} * (delta_ik - x_{t,i})

    * Marginal velocity given a model endpoint distribution ``x_θ`` (paper
      Eq. 12, expanded component-wise):
        u_t^θ(x_t)_i = (lambda / tau(t)) * x_{t,i} * (x_θ_i - <x_θ, x_t>)

    * Training loss (paper Eq. 11, equivalent to masked CE on endpoint):
        L = -E[log <x_θ(x_t, t), x_1>].

    * Defaults (paper §6.2-6.3): ``tau_max=10, lambda=3, beta=2``.

  Gumbel-Softmax is better suited to high-K than Dirichlet because the
  softmax-based interpolant does not suffer from Dirichlet's support-shrinkage
  pathology (Stark et al. 2024 Prop. 1). Time-dependent temperature controls
  how sharp the interpolant is without any closed-form Beta-function grid.

Parameterization
================
All active bridges use **endpoint parameterization**: the neural network
predicts ``p(x_1 | x_t, t)`` per voxel (logits over THAT sample's parts),
the bridge assembles the velocity / next state internally. Training loss is
masked cross-entropy on the endpoint logits (classifier FM), which for
Gumbel-Softmax FM exactly matches paper Eq. 11 (``-log <x_θ, x_1>``).

Padding discipline
==================
For every active bridge:

- ``sample_source`` produces ``x_0`` with ``sum=1`` over valid dims, ``0``
  over padding.
- ``sample_conditional_path`` preserves the invariant.
- ``step`` preserves the invariant.
- ``compute_loss`` masks padding out of the softmax denominator.

Tests in ``scripts/train/tests/part_flow/test_bridges.py`` assert these.

Legacy / deprecated
===================
Dirichlet legacy code has been removed from the active package. The old SFM
bridge remains below as an unregistered reference only; it is not exported and
is not selectable from YAML/CLI.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'BaseCategoricalFlowBridge',
    'FisherBridge',
    'GumbelSoftmaxBridge',
    'build_bridge',
]


# --------------------------------------------------------------------------- #
# Utilities (shared)                                                          #
# --------------------------------------------------------------------------- #


def _build_part_valid_mask(
    num_parts: List[int],
    k_max: int,
    device: torch.device,
) -> torch.Tensor:
    """Build per-sample ``part_valid_mask [B, k_max]`` bool from num_parts."""
    B = len(num_parts)
    idx = torch.arange(k_max, device=device).unsqueeze(0).expand(B, k_max)
    nps = torch.tensor(num_parts, device=device).unsqueeze(-1)
    return idx < nps  # [B, k_max]


def _expand_valid_per_voxel(
    part_valid_mask: torch.Tensor,
    voxel_layout: List[slice],
    N_total: int,
) -> torch.Tensor:
    """Expand ``[B, k_max]`` valid mask into ``[N_total, k_max]`` per-voxel."""
    B, K = part_valid_mask.shape
    out = torch.empty(N_total, K, dtype=torch.bool, device=part_valid_mask.device)
    for b, sl in enumerate(voxel_layout):
        out[sl] = part_valid_mask[b:b + 1].expand(sl.stop - sl.start, K)
    return out


def _project_to_simplex_masked(
    x: torch.Tensor,
    valid_per_voxel: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Clamp nonnegative + renormalize on valid dims, zero on padding."""
    x = x.clamp(min=0.0)
    x = x * valid_per_voxel.to(x.dtype)
    total = x.sum(dim=-1, keepdim=True).clamp(min=eps)
    return x / total


def _masked_softmax(
    logits: torch.Tensor,
    valid_per_voxel: torch.Tensor,
) -> torch.Tensor:
    """Softmax with padding dims forced to zero probability.

    Padding dims are set to a very negative value pre-softmax so they drop
    out of the denominator, then are explicitly zeroed post-softmax to guard
    against float drift.
    """
    logits = logits.masked_fill(~valid_per_voxel, -1e4)
    probs = F.softmax(logits, dim=-1)
    probs = probs * valid_per_voxel.to(probs.dtype)
    total = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs / total


# --------------------------------------------------------------------------- #
# Base class                                                                  #
# --------------------------------------------------------------------------- #


class BaseCategoricalFlowBridge(nn.Module):
    """Abstract interface for a categorical flow matching bridge.

    Subclasses MUST implement:
      - ``sample_source``
      - ``sample_conditional_path``
      - ``step``

    Default endpoint-parameterization ``compute_loss`` is masked
    cross-entropy on the model's predicted endpoint logits — works for all
    active bridges (Fisher and Gumbel-Softmax).
    """

    def __init__(self, k_max: int, t_max: float = 1.0):
        super().__init__()
        self.k_max = k_max
        self.t_max = t_max

    # ---- Inputs helpers ----

    def build_part_valid_mask(self, num_parts: List[int], device: torch.device) -> torch.Tensor:
        return _build_part_valid_mask(num_parts, self.k_max, device)

    def expand_valid_per_voxel(
        self, part_valid_mask: torch.Tensor, voxel_layout: List[slice], N_total: int,
    ) -> torch.Tensor:
        return _expand_valid_per_voxel(part_valid_mask, voxel_layout, N_total)

    # ---- Contract methods ----

    def sample_source(
        self,
        num_parts: List[int],
        n_per_sample: List[int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_conditional_path(
        self,
        x_1_one_hot: torch.Tensor,
        t_per_voxel: torch.Tensor,
        voxel_layout: List[slice],
        num_parts: List[int],
        x_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        endpoint_probs: torch.Tensor,
        t_val: float,
        dt: float,
        voxel_layout: List[slice],
        num_parts: List[int],
    ) -> torch.Tensor:
        raise NotImplementedError

    # ---- Endpoint-parameterization loss (default: masked CE) ----

    def compute_loss(
        self,
        endpoint_logits: torch.Tensor,
        x_1_idx: torch.Tensor,
        valid_per_voxel: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Masked cross-entropy on endpoint prediction.

        For Gumbel-Softmax FM this is paper Eq. 11 exactly; for Fisher FM it
        is the classifier-FM equivalent to the log-likelihood of x_1 under
        the model's posterior at x_t.
        """
        masked_logits = endpoint_logits.masked_fill(~valid_per_voxel, -1e4)
        loss = F.cross_entropy(masked_logits, x_1_idx, reduction='mean')
        with torch.no_grad():
            probs = F.softmax(masked_logits, dim=-1)
            gt_prob = probs.gather(1, x_1_idx.unsqueeze(-1)).squeeze(-1)
            mean_gt_prob = gt_prob.mean().item()
            pred_match = (probs.argmax(dim=-1) == x_1_idx).float().mean().item()
        metrics = {
            'loss': float(loss.item()),
            'gt_prob_mean': mean_gt_prob,
            'endpoint_acc': pred_match,
        }
        return loss, metrics


# --------------------------------------------------------------------------- #
# Fisher bridge (active)                                                      #
# --------------------------------------------------------------------------- #


class FisherBridge(BaseCategoricalFlowBridge):
    """Fisher-Rao (statistical manifold) FM on the simplex.

    Reference: Davis et al., "Fisher Flow Matching for Generative Modeling
    over Discrete Data", NeurIPS 2024 (arXiv:2405.14664).

    Parameterization (this implementation):
      - ``φ: p -> sqrt(p)`` maps Δ^{K-1} to the positive orthant of the unit
        sphere S^{K-1}_+ in R^K (Davis et al. Section 2.1 — this is the
        Cencov isometry between the Fisher-Rao metric on Δ and the round
        metric on S).
      - Source ``u_0 = φ(p_0)`` with ``p_0 ~ Dir(1,...,1)``; target
        ``u_1 = φ(e_c) = e_c`` (sqrt preserves one-hot).
      - Conditional path is the spherical geodesic (slerp) from ``u_0`` to
        ``u_1``.
      - Velocity is the geodesic tangent at ``u_t``; the sampling step is an
        exp-map from ``u_t`` toward the predicted endpoint ``u_1_hat``
        scaled by remaining time.

    Exact parts: Fisher-Rao geometry, slerp-based conditional paths, sqrt
    bijection between simplex and sphere-orthant.

    Practical simplification: the ODE step is Euler-in-tangent-space (log map
    + exp map) rather than a full geodesic integrator; at small dt this is a
    standard and well-behaved approximation.

    Padding dims: ``sqrt(0) = 0`` so the map is strict no-op on padding;
    spherical normalization is done on valid dims only.
    """

    def __init__(
        self,
        k_max: int,
        t_max: float = 1.0,
        eps: float = 1e-6,
        dirichlet_alpha: float = 1.0,
    ):
        super().__init__(k_max, t_max)
        assert dirichlet_alpha > 0.0, (
            f'dirichlet_alpha must be > 0, got {dirichlet_alpha}'
        )
        self.eps = eps
        self.dirichlet_alpha = float(dirichlet_alpha)

    def _to_sphere(self, p: torch.Tensor) -> torch.Tensor:
        return p.clamp(min=0.0).sqrt()

    def _from_sphere(self, u: torch.Tensor, valid_per_vox: torch.Tensor) -> torch.Tensor:
        p = (u ** 2) * valid_per_vox.to(u.dtype)
        total = p.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        return p / total

    def sample_source(self, num_parts, n_per_sample, device, dtype=torch.float32):
        """x_0 ~ Dir(1,...,1) on valid dims."""
        N_total = sum(n_per_sample)
        p_0 = torch.zeros(N_total, self.k_max, device=device, dtype=dtype)
        offset = 0
        for K_b, n_b in zip(num_parts, n_per_sample):
            if K_b == 0 or n_b == 0:
                offset += n_b
                continue
            alphas = torch.full(
                (n_b, K_b),
                self.dirichlet_alpha,
                device=device,
                dtype=dtype,
            )
            p_0[offset:offset + n_b, :K_b] = torch.distributions.Dirichlet(alphas).sample()
            offset += n_b
        return p_0

    def sample_conditional_path(
        self, x_1_one_hot, t_per_voxel, voxel_layout, num_parts,
        x_0: Optional[torch.Tensor] = None,
    ):
        """u_t = slerp(u_0, u_1, t / t_max); x_t = u_t^2, renormalized."""
        device = x_1_one_hot.device
        dtype = x_1_one_hot.dtype
        if x_0 is None:
            n_per = [sl.stop - sl.start for sl in voxel_layout]
            x_0 = self.sample_source(num_parts, n_per, device, dtype)

        valid = _build_part_valid_mask(num_parts, self.k_max, device)
        valid_per_vox = _expand_valid_per_voxel(valid, voxel_layout, x_0.shape[0])
        u_0 = self._to_sphere(x_0) * valid_per_vox.to(dtype)
        u_1 = self._to_sphere(x_1_one_hot) * valid_per_vox.to(dtype)

        dot = (u_0 * u_1).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(dot)
        sin_theta = theta.sin().clamp(min=self.eps)
        s = (t_per_voxel.unsqueeze(-1) / self.t_max).clamp(0.0, 1.0)
        a = ((1.0 - s) * theta).sin() / sin_theta
        b = (s * theta).sin() / sin_theta
        u_t = a * u_0 + b * u_1
        u_t = u_t / u_t.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self._from_sphere(u_t, valid_per_vox)

    @torch.no_grad()
    def step(self, x_t, endpoint_probs, t_val, dt, voxel_layout, num_parts):
        """Spherical exp-map step toward predicted endpoint.

        log_{u_t}(u_1_hat) = (θ / sin θ) * (u_1_hat - cos θ * u_t)
        velocity ≈ log_map / (t_max - t); step = dt * velocity; exp-map back.
        """
        device = x_t.device
        dtype = x_t.dtype
        valid = _build_part_valid_mask(num_parts, self.k_max, device)
        valid_per_vox = _expand_valid_per_voxel(valid, voxel_layout, x_t.shape[0])

        u_t = self._to_sphere(x_t) * valid_per_vox.to(dtype)
        u_t = u_t / u_t.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        u_1 = self._to_sphere(endpoint_probs) * valid_per_vox.to(dtype)
        u_1 = u_1 / u_1.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        dot = (u_t * u_1).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(dot)
        sin_theta = theta.sin().clamp(min=self.eps)
        log_map = (theta / sin_theta) * (u_1 - dot * u_t)
        t_remaining = max(self.t_max - float(t_val), self.eps)
        v_tan = log_map / t_remaining
        step_vec = dt * v_tan
        step_norm = step_vec.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        u_new = step_norm.cos() * u_t + step_norm.sin() * (step_vec / step_norm)
        u_new = u_new / u_new.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self._from_sphere(u_new, valid_per_vox)


# --------------------------------------------------------------------------- #
# Gumbel-Softmax bridge (active, default)                                     #
# --------------------------------------------------------------------------- #


class GumbelSoftmaxBridge(BaseCategoricalFlowBridge):
    """Gumbel-Softmax Flow Matching (Tang et al., arXiv:2503.17361).

    Paper formulas (quoted directly; adapted to variable-K via masked softmax):

    - Interpolant (Eq. 9):
          x_t_i = softmax_i((delta_ik + g_i/beta) / tau(t))
      where ``g`` is i.i.d. Gumbel noise ``g_i = -log(-log U_i)``,
      ``U_i ~ Uniform(0, 1)``.

    - Temperature schedule (Eq. 8):
          tau(t) = tau_max * exp(-lambda * t),    t ∈ [0, 1].

    - Conditional velocity (Eq. 13, ``g = 0`` at inference):
          u_t(x_t | x_1=e_k)_i = (lambda / tau(t)) * x_{t,k} * (delta_ik - x_{t,i})

    - Marginal velocity given endpoint probabilities ``x_θ`` (Eq. 12, expanded):
          u_t^θ(x_t)_i = (lambda / tau(t)) * x_{t,i} * (x_θ_i - <x_θ, x_t>)

    - Training loss (Eq. 11, exactly masked CE on endpoint):
          L = -E[log <x_θ(x_t, t), x_1>]

    - Paper defaults (§6.2-6.3): ``tau_max = 10``, ``lambda = 3``, ``beta = 2``.
      A clamp ``tau >= tau_min`` is added here for numerical stability near
      ``t = 1`` (not in the paper but standard for temperature-based methods).

    Variable-K adaptation (this implementation):

    - Gumbel noise is drawn over ``k_max`` dims; the ``(logits / tau)`` input
      to softmax is masked to ``-inf`` on padding so padding entries drop
      out of the softmax denominator (strict no-op).
    - Marginal velocity in ``step`` is zeroed on padding dims via
      ``_project_to_simplex_masked`` on the post-step state.

    Why this bridge is the new default:
      - No ``c_factor`` beta-CDF grid (Dirichlet's bottleneck at high K).
      - Softmax interpolant is naturally well-behaved when K grows.
      - Training loss is plain masked CE — identical objective across bridges,
        cleaner ablation.
    """

    def __init__(
        self,
        k_max: int,
        t_max: float = 1.0,
        tau_max: float = 10.0,
        decay_rate: float = 3.0,        # lambda in paper Eq. 8
        noise_scale: float = 2.0,       # beta in paper Eq. 9
        tau_min: float = 1e-2,          # numerical floor (not in paper)
        eps: float = 1e-8,
    ):
        super().__init__(k_max, t_max)
        assert tau_max > 0.0
        assert decay_rate > 0.0
        assert noise_scale > 0.0
        assert tau_min > 0.0
        self.tau_max = float(tau_max)
        self.decay_rate = float(decay_rate)
        self.noise_scale = float(noise_scale)
        self.tau_min = float(tau_min)
        self.eps = eps

    # --- Temperature schedule (Eq. 8) ---

    def tau_at(self, t) -> float:
        """tau(t) = tau_max * exp(-lambda * t), clamped to tau_min floor."""
        t_val = float(t) if not torch.is_tensor(t) else float(t.item())
        normalized = t_val / self.t_max  # s ∈ [0, 1]
        tau = self.tau_max * math.exp(-self.decay_rate * normalized)
        return max(tau, self.tau_min)

    # --- Gumbel noise helper ---

    @staticmethod
    def _sample_gumbel(shape, device, dtype, eps=1e-10) -> torch.Tensor:
        """g ~ Gumbel(0, 1): g = -log(-log(U)), U ~ Uniform(0, 1)."""
        u = torch.rand(shape, device=device, dtype=dtype).clamp(min=eps, max=1.0 - eps)
        return -(-u.log()).log()

    # --- Contract methods ---

    def sample_source(self, num_parts, n_per_sample, device, dtype=torch.float32):
        """Source ``x_0 = x_{t=0}`` per paper Eq. 9 at t=0.

        At ``t = 0``: ``tau = tau_max``, so
          x_0_i = softmax_i(g_i / (beta * tau_max))  (delta term vanishes since
        one-hot is weighted by 0 time — paper's interpolant at t=0 is a pure
        Gumbel-softmax with no target bias). Here we sample a neutral source
        by using a uniform label (no target bias): set ``e_k = 0`` implicitly
        and return ``softmax(g / (beta * tau_max))``.
        """
        N_total = sum(n_per_sample)
        x_0 = torch.zeros(N_total, self.k_max, device=device, dtype=dtype)
        valid = _build_part_valid_mask(num_parts, self.k_max, device)
        valid_per_vox = torch.empty(N_total, self.k_max, dtype=torch.bool, device=device)
        offset = 0
        for b, (K_b, n_b) in enumerate(zip(num_parts, n_per_sample)):
            valid_per_vox[offset:offset + n_b] = valid[b:b + 1].expand(n_b, self.k_max)
            offset += n_b

        g = self._sample_gumbel((N_total, self.k_max), device, dtype)
        logits = g / (self.noise_scale * self.tau_max)
        # Masked softmax over valid dims only
        x_0 = _masked_softmax(logits, valid_per_vox)
        return x_0

    def sample_conditional_path(
        self, x_1_one_hot, t_per_voxel, voxel_layout, num_parts,
        x_0: Optional[torch.Tensor] = None,
    ):
        """Gumbel-Softmax interpolant (paper Eq. 9).

        ``x_t_i = softmax_i((delta_ik + g_i/beta) / tau(t))``.

        Note: x_0 argument is ignored here — the Gumbel-Softmax path is
        defined by (Gumbel noise, target, temperature(t)) alone, not by a
        shared source-target pair. Consumers may still pass x_0 for API
        compatibility.
        """
        device = x_1_one_hot.device
        dtype = x_1_one_hot.dtype
        N_total = x_1_one_hot.shape[0]

        valid = _build_part_valid_mask(num_parts, self.k_max, device)
        valid_per_vox = _expand_valid_per_voxel(valid, voxel_layout, N_total)

        # Gumbel noise over all k_max dims; padding is masked out in softmax
        g = self._sample_gumbel((N_total, self.k_max), device, dtype)

        # Per-voxel temperature (vectorized over t_per_voxel)
        # tau(t) = tau_max * exp(-lambda * t/t_max), clamped to tau_min
        t_norm = (t_per_voxel / self.t_max).clamp(0.0, 1.0)   # [N_total]
        tau = (self.tau_max * torch.exp(-self.decay_rate * t_norm)).clamp(min=self.tau_min)
        tau = tau.unsqueeze(-1)  # [N_total, 1]

        # Interpolant pre-softmax logits (paper Eq. 9):
        # (delta_ik + g_i/beta) / tau(t)
        logits = (x_1_one_hot + g / self.noise_scale) / tau

        # Masked softmax: padding dims become -inf → drop from denominator,
        # then explicit zeroing guards against float drift.
        x_t = _masked_softmax(logits, valid_per_vox)
        return x_t

    @torch.no_grad()
    def step(self, x_t, endpoint_probs, t_val, dt, voxel_layout, num_parts):
        """Euler step along marginal velocity (paper Eq. 12, expanded).

        u_t^θ(x_t)_i = (lambda / tau(t)) * x_{t,i} * (x_θ_i - <x_θ, x_t>)

        (Eq. 12 in the paper writes the velocity as a sum over candidate
        targets e_k of Eq. 13's conditional velocity; that sum collapses to
        the expression above. See module docstring for derivation.)
        """
        device = x_t.device
        dtype = x_t.dtype
        valid = _build_part_valid_mask(num_parts, self.k_max, device)
        valid_per_vox = _expand_valid_per_voxel(valid, voxel_layout, x_t.shape[0])

        # Zero out any padding residual before computing velocity
        x_t = x_t * valid_per_vox.to(dtype)
        endpoint_probs = endpoint_probs * valid_per_vox.to(dtype)

        tau = self.tau_at(t_val)
        scale = self.decay_rate / max(tau, self.tau_min)

        # <x_θ, x_t> per voxel (only valid dims contribute, since padding is 0)
        dot = (endpoint_probs * x_t).sum(dim=-1, keepdim=True)  # [N, 1]
        v = scale * x_t * (endpoint_probs - dot)                # [N, k_max]
        # v is tangent to simplex: sum_i v_i = scale * (sum_i x_t_i * x_θ_i - sum_i x_t_i * dot) = 0
        # since dot = <x_θ, x_t> and sum_i x_t_i = 1 over valid dims.

        x_new = x_t + float(dt) * v
        return _project_to_simplex_masked(x_new, valid_per_vox)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


_REGISTRY: Dict[str, type] = {
    'fisher': FisherBridge,
    'gumbel': GumbelSoftmaxBridge,
}


def build_bridge(flow_type: str, **kwargs) -> BaseCategoricalFlowBridge:
    """Factory: build a bridge by name.

    YAML-facing entry point. ``flow_type`` in {'fisher', 'gumbel'}. kwargs
    not accepted by the chosen bridge's ``__init__`` are silently dropped
    (e.g. a Fisher run (flow.type=fisher CLI override) applied to ``base.yaml`` will see
    ``tau_max`` / ``decay_rate`` in its flow config and we drop them rather
    than error out — this keeps config inheritance ergonomic).

    Defaults:
      - fisher: ``t_max = 1.0``.
      - gumbel: ``t_max = 1.0``, ``tau_max = 10.0``, ``decay_rate = 3.0``,
        ``noise_scale = 2.0``, ``tau_min = 1e-2`` (paper §6.2-6.3 + numeric floor).

    Legacy bridges (``dirichlet``, ``sfm``) are no longer selectable — they
    live in the ``_legacy`` section below for reference / reproducibility
    only.
    """
    import inspect

    flow_type = flow_type.lower()
    if flow_type not in _REGISTRY:
        raise ValueError(
            f'Unknown flow_type {flow_type!r}; active options are '
            f'{sorted(_REGISTRY)}. Legacy dirichlet/sfm are no longer '
            f'selectable from YAML/CLI (see module docstring).'
        )
    cls = _REGISTRY[flow_type]
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {'self'}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = set(kwargs.keys()) - accepted
    if dropped:
        # Informational; cross-flow kwargs are expected when YAML inheritance
        # carries gumbel hyperparameters into a fisher override.
        warnings.warn(
            f'[build_bridge] {flow_type}: dropped unused kwargs {sorted(dropped)}',
            stacklevel=2,
        )
    return cls(**filtered)


# --------------------------------------------------------------------------- #
# Legacy bridges — kept for history only. NOT in __all__, NOT in the registry,#
# NOT imported by any active training code path. Do not use in new training.  #
# --------------------------------------------------------------------------- #


class _LegacySFMBridge(BaseCategoricalFlowBridge):
    """(LEGACY, NOT ACTIVE) Simplex Flow Matching — Lipman et al. linear FM.

    Kept for reference only; not selectable from YAML/CLI.
    """

    def __init__(self, k_max: int, t_max: float = 1.0):
        super().__init__(k_max, t_max)

    def sample_source(self, num_parts, n_per_sample, device, dtype=torch.float32):
        N_total = sum(n_per_sample)
        x_0 = torch.zeros(N_total, self.k_max, device=device, dtype=dtype)
        offset = 0
        for K_b, n_b in zip(num_parts, n_per_sample):
            if K_b == 0 or n_b == 0:
                offset += n_b
                continue
            alphas = torch.ones(n_b, K_b, device=device, dtype=dtype)
            x_0[offset:offset + n_b, :K_b] = torch.distributions.Dirichlet(alphas).sample()
            offset += n_b
        return x_0

    def sample_conditional_path(self, x_1_one_hot, t_per_voxel, voxel_layout, num_parts, x_0=None):
        device = x_1_one_hot.device
        dtype = x_1_one_hot.dtype
        if x_0 is None:
            n_per = [sl.stop - sl.start for sl in voxel_layout]
            x_0 = self.sample_source(num_parts, n_per, device, dtype)
        s = (t_per_voxel.unsqueeze(-1) / self.t_max).clamp(0.0, 1.0)
        x_t = (1.0 - s) * x_0 + s * x_1_one_hot
        valid = _build_part_valid_mask(num_parts, self.k_max, device)
        valid_per_vox = _expand_valid_per_voxel(valid, voxel_layout, x_t.shape[0])
        x_t = x_t * valid_per_vox.to(dtype)
        return x_t

    @torch.no_grad()
    def step(self, x_t, endpoint_probs, t_val, dt, voxel_layout, num_parts):
        device = x_t.device
        t_remaining = max(self.t_max - float(t_val), 1e-6)
        v = (endpoint_probs - x_t) / t_remaining
        x_new = x_t + dt * v
        valid = _build_part_valid_mask(num_parts, self.k_max, device)
        valid_per_vox = _expand_valid_per_voxel(valid, voxel_layout, x_t.shape[0])
        return _project_to_simplex_masked(x_new, valid_per_vox)
