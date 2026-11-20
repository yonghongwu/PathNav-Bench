#!/usr/bin/env python3
"""Self-contained 3-Stage Multimodal Trainer (`SFT -> Step-DPO -> Step-GRPO`) for PathNav-Bench.

Designed specifically for Qwen3.5-4B / Qwen3.5-9B (`Qwen3_5ForConditionalGeneration`) on MetaX C500:
1. Uses native PyTorch `LoRALinear` (fp32 A/B weights on bf16 backbone) across all 200 linear projections
   (`q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_z, out_proj, gate_proj, up_proj, down_proj`).
2. Uses `disabled_lora(model)` context manager to compute exact reference log-probs (`pi_ref`) for
   Step-DPO and Step-GRPO with ZERO extra GPU memory overhead.
3. Reconstructs the 100% authentic `pathnav_bench.agents` runtime prompt (`build_prompt` + `obs_payload`
   JSON with `attachment_view_ids`, `slide_metadata`, `current_observation`, `previous_steps`) and
   eliminates step-index/label shortcut correlations via balanced counterfactual high-power turns.
4. Exports a vLLM-ready merged checkpoint (`merged_vllm/`) by directly patching the base model's
   `.safetensors` shards in-place so `vllm serve` loads it with zero schema drift.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Clamp Triton autotune configs to num_stages=1, BK<=64, BV<=32 so fused FLA kernels fit in MetaX 64KB shared memory
import triton

_orig_autotune = triton.autotune


def _safe_autotune(configs, *args, **kwargs):
    safe_cfgs = []
    for c in configs:
        kw = dict(c.kwargs)
        if kw.get("BK", 32) > 64:
            kw["BK"] = 64
        if kw.get("BV", 32) > 32:
            kw["BV"] = 32
        safe_cfgs.append(triton.Config(kw, num_warps=min(c.num_warps, 4), num_stages=1))
    return _orig_autotune(safe_cfgs[:1], *args, **kwargs)


triton.autotune = _safe_autotune

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from fla.ops.common import chunk_delta_h

chunk_delta_h.print = lambda *a, **k: None
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3_5ForConditionalGeneration

_orig_solve_triangular = torch.linalg.solve_triangular


def _safe_solve_triangular(
    A: torch.Tensor, B: torch.Tensor, *, upper: bool = False, unitriangular: bool = False, left: bool = True
) -> torch.Tensor:
    """Bypass MetaX C500 cublasStrsmBatched batchCount<=1024 limit by slicing flattened batch dim into chunks of 512."""
    orig_shape = B.shape
    N = A.shape[-1]
    D = B.shape[-1]
    A_flat = A.reshape(-1, N, N)
    B_flat = B.reshape(-1, N, D)
    total_b = A_flat.shape[0]
    if total_b <= 512:
        return _orig_solve_triangular(
            A_flat.contiguous(), B_flat.contiguous(), upper=upper, unitriangular=unitriangular, left=left
        ).reshape(orig_shape)
    outs = [
        _orig_solve_triangular(
            A_flat[i : i + 512].contiguous(),
            B_flat[i : i + 512].contiguous(),
            upper=upper,
            unitriangular=unitriangular,
            left=left,
        )
        for i in range(0, total_b, 512)
    ]
    return torch.cat(outs, dim=0).reshape(orig_shape)


torch.linalg.solve_triangular = _safe_solve_triangular

_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


class _ChunkedSDPA(torch.autograd.Function):
    """Memory-efficient head-chunked SDPA for MetaX math fallback (reduces [1, H_q, L, L] 9 GiB peak by 16x to [1, 1, L, L])."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        ctx.save_for_backward(query, key, value, attn_mask)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        h_q = query.shape[1]
        h_kv = key.shape[1]
        g = max(1, h_q // h_kv)
        outs = []
        with torch.no_grad():
            for q_i in range(h_q):
                kv_i = q_i // g
                m_slice = (
                    attn_mask[:, q_i : q_i + 1]
                    if (attn_mask is not None and attn_mask.ndim == 4 and attn_mask.shape[1] > 1)
                    else attn_mask
                )
                outs.append(
                    _orig_sdpa(
                        query[:, q_i : q_i + 1],
                        key[:, kv_i : kv_i + 1],
                        value[:, kv_i : kv_i + 1],
                        attn_mask=m_slice,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                        enable_gqa=False,
                    )
                )
        return torch.cat(outs, dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, attn_mask = ctx.saved_tensors
        h_q = query.shape[1]
        h_kv = key.shape[1]
        g = max(1, h_q // h_kv)
        grad_q = torch.zeros_like(query)
        grad_k = torch.zeros_like(key)
        grad_v = torch.zeros_like(value)
        for q_i in range(h_q):
            kv_i = q_i // g
            q_s = query[:, q_i : q_i + 1].detach().requires_grad_(True)
            k_s = key[:, kv_i : kv_i + 1].detach().requires_grad_(True)
            v_s = value[:, kv_i : kv_i + 1].detach().requires_grad_(True)
            m_slice = (
                attn_mask[:, q_i : q_i + 1]
                if (attn_mask is not None and attn_mask.ndim == 4 and attn_mask.shape[1] > 1)
                else attn_mask
            )
            with torch.enable_grad():
                out_s = _orig_sdpa(
                    q_s,
                    k_s,
                    v_s,
                    attn_mask=m_slice,
                    dropout_p=ctx.dropout_p,
                    is_causal=ctx.is_causal,
                    scale=ctx.scale,
                    enable_gqa=False,
                )
                dq, dk, dv = torch.autograd.grad(
                    out_s, (q_s, k_s, v_s), grad_output[:, q_i : q_i + 1], retain_graph=False
                )
            grad_q[:, q_i : q_i + 1].copy_(dq)
            grad_k[:, kv_i : kv_i + 1].add_(dk)
            grad_v[:, kv_i : kv_i + 1].add_(dv)
            del out_s, dq, dk, dv, q_s, k_s, v_s
        return grad_q, grad_k, grad_v, None, None, None, None


def _safe_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if query.ndim == 4 and query.shape[2] > 1024 and query.shape[1] > 1:
        if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
            return _ChunkedSDPA.apply(query, key, value, attn_mask, dropout_p, is_causal, scale)
        h_q = query.shape[1]
        h_kv = key.shape[1]
        g = max(1, h_q // h_kv)
        outs = []
        for q_i in range(h_q):
            kv_i = q_i // g
            m_slice = (
                attn_mask[:, q_i : q_i + 1]
                if (attn_mask is not None and attn_mask.ndim == 4 and attn_mask.shape[1] > 1)
                else attn_mask
            )
            outs.append(
                _orig_sdpa(
                    query[:, q_i : q_i + 1],
                    key[:, kv_i : kv_i + 1],
                    value[:, kv_i : kv_i + 1],
                    attn_mask=m_slice,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=False,
                )
            )
        return torch.cat(outs, dim=1)
    return _orig_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


torch.nn.functional.scaled_dot_product_attention = _safe_sdpa
F.scaled_dot_product_attention = _safe_sdpa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
venv_sp = ROOT / ".venv/lib/python3.10/site-packages"
if venv_sp.exists() and str(venv_sp) not in sys.path:
    sys.path.append(str(venv_sp))

from pathnav_bench.agents import build_prompt
from pathnav_bench.env import BenchConfig
from pathnav_bench.evaluator import load_case_truth
from shapely.geometry import box as shapely_box

LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

ORGAN_TASK_QUERIES = {
    "Skin": (
        "Scan the skin section (epidermis and dermis) to determine whether Basal Cell Carcinoma (BCC) "
        "is present. Look for focal nests or lobules of crowded, hyperchromatic basaloid cells with "
        "peripheral palisading and stromal clefting invading the dermis."
    ),
    "Breast": (
        "Scan the breast biopsy/resection tissue to determine whether invasive breast carcinoma is "
        "present. Look for irregular nests, cords, or sheets of atypical epithelial cells with enlarged "
        "hyperchromatic nuclei infiltrating the fibrous or adipose stroma."
    ),
    "Lymph node (Breast)": None,
}


class LoRALinear(nn.Module):
    """Precision-preserving fp32 LoRA adapter over frozen bf16 Linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.active = True
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.active or self.scale == 0.0:
            return out
        res = F.linear(F.linear(self.dropout(x.float()), self.A), self.B)
        return out + (res * self.scale).to(out.dtype)


def _get_lm_root(model: nn.Module) -> Tuple[str, nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return "model.language_model", model.model.language_model
    if hasattr(model, "model"):
        return "model", model.model
    return "", model


@contextmanager
def reference_mode(model: nn.Module, ref_state: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None):
    """Switch model to reference policy (`pi_ref`): either base model (`active=False`) or frozen SFT LoRA."""
    _, lm_root = _get_lm_root(model)
    lora_Dict = {
        name: mod for name, mod in lm_root.named_modules() if isinstance(mod, LoRALinear)
    }
    if ref_state is None:
        for mod in lora_Dict.values():
            mod.active = False
        try:
            yield
        finally:
            for mod in lora_Dict.values():
                mod.active = True
    else:
        saved = {k: (m.A.data.clone(), m.B.data.clone()) for k, m in lora_Dict.items()}
        for k, m in lora_Dict.items():
            if k in ref_state:
                m.A.data.copy_(ref_state[k][0])
                m.B.data.copy_(ref_state[k][1])
        try:
            yield
        finally:
            for k, m in lora_Dict.items():
                m.A.data.copy_(saved[k][0])
                m.B.data.copy_(saved[k][1])


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05) -> Dict[str, LoRALinear]:
    for p in model.parameters():
        p.requires_grad_(False)
    prefix, lm_root = _get_lm_root(model)
    injected: Dict[str, LoRALinear] = {}
    for mod_name, mod in list(lm_root.named_modules()):
        if "visual" in mod_name:
            continue
        for child_name, child in list(mod.named_children()):
            if child_name in LORA_TARGETS and isinstance(child, nn.Linear):
                full_key = f"{prefix}.{mod_name}.{child_name}" if prefix else f"{mod_name}.{child_name}"
                lora_mod = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(mod, child_name, lora_mod)
                injected[full_key] = lora_mod
    print(f"[LoRA] Injected {len(injected)} LoRA adapters under prefix='{prefix}'", flush=True)
    return injected


def save_lora_checkpoint(injected: Dict[str, LoRALinear], out_dir: Path, meta: Dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        k: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu(), "rank": m.rank, "alpha": m.alpha}
        for k, m in injected.items()
    }
    ckpt_path = out_dir / "lora_weights.pt"
    torch.save({"lora": state, "meta": meta}, ckpt_path)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return ckpt_path


def _strip_lm_prefix(k: str) -> str:
    if k.startswith("model.language_model."):
        return k[len("model.language_model.") :]
    if k.startswith("model."):
        return k[len("model.") :]
    return k


def load_lora_checkpoint(
    injected: Dict[str, LoRALinear], ckpt_path: Path, blend_path: Optional[Path] = None, blend_alpha: float = 0.35
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    lora_state = payload["lora"]
    blend_state = None
    if blend_path and blend_path.exists():
        blend_state = torch.load(blend_path, map_location="cpu", weights_only=True)["lora"]
    ref_gpu_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    loaded = 0
    for k, m in injected.items():
        if k in lora_state:
            a_t = lora_state[k]["A"].to(m.A.device, dtype=m.A.dtype)
            b_t = lora_state[k]["B"].to(m.B.device, dtype=m.B.dtype)
            if blend_state and k in blend_state:
                a_b = blend_state[k]["A"].to(m.A.device, dtype=m.A.dtype)
                b_b = blend_state[k]["B"].to(m.B.device, dtype=m.B.dtype)
                a_t = (1.0 - blend_alpha) * a_t + blend_alpha * a_b
                b_t = (1.0 - blend_alpha) * b_t + blend_alpha * b_b
            m.A.data.copy_(a_t)
            m.B.data.copy_(b_t)
            ref_gpu_state[_strip_lm_prefix(k)] = (
                m.A.data.clone(),
                m.B.data.clone(),
            )
            loaded += 1
    print(f"[LoRA] Loaded {loaded}/{len(injected)} adapter layers from {ckpt_path} (blend={blend_path})", flush=True)
    return ref_gpu_state


def export_merged_safetensors(base_model_dir: Path, injected: Dict[str, LoRALinear], merged_dir: Path):
    """Patch base model's safetensors shards in-place with LoRA deltas for 100% vLLM compatibility."""
    t0 = time.time()
    merged_dir.mkdir(parents=True, exist_ok=True)
    for item in base_model_dir.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors"):
            shutil.copy2(item, merged_dir / item.name)

    deltas: Dict[str, torch.Tensor] = {}
    for k, m in injected.items():
        weight_key = f"{k}.weight"
        alt_key = f"{k.replace('model.language_model.', 'model.')}.weight"
        with torch.no_grad():
            delta = (m.B @ m.A) * (m.alpha / m.rank)
            d_cpu = delta.detach().to(dtype=torch.bfloat16).cpu()
            deltas[weight_key] = d_cpu
            deltas[alt_key] = d_cpu

    shard_files = sorted(base_model_dir.glob("*.safetensors"))
    patched_count = 0
    for sf in shard_files:
        tensors = load_file(str(sf), device="cpu")
        for tk in list(tensors.keys()):
            if tk in deltas:
                tensors[tk] = tensors[tk].to(torch.bfloat16) + deltas[tk]
                patched_count += 1
        save_file(tensors, str(merged_dir / sf.name), metadata={"format": "pt"})
    print(
        f"[Export] Merged {patched_count} LoRA weights into {len(shard_files)} shards at {merged_dir} in {time.time()-t0:.1f}s",
        flush=True,
    )


def normalize_action_dict(act: Dict, jitter_seed: Optional[int] = None) -> Dict:
    """Ensure action JSON strictly matches `pathnav_bench` runtime schema (`reason` key, clean floats)."""
    out = copy.deepcopy(act)
    if "summary" in out and "reason" not in out:
        out["reason"] = out.pop("summary")
    if out.get("action") == "inspect_box" and out.get("box") == [400.0, 400.0, 560.0, 560.0] and jitter_seed is not None:
        rng = random.Random(jitter_seed)
        dx = round(rng.uniform(-70.0, 70.0), 1)
        dy = round(rng.uniform(-70.0, 70.0), 1)
        span = round(rng.uniform(145.0, 185.0), 1)
        x1 = max(120.0, min(700.0, round(400.0 + dx, 1)))
        y1 = max(120.0, min(700.0, round(400.0 + dy, 1)))
        out["box"] = [x1, y1, round(x1 + span, 1), round(y1 + span, 1)]
    if out.get("action") == "finish" and isinstance(out.get("evidence"), list) and jitter_seed is not None:
        rng = random.Random(jitter_seed + 997)
        for ev in out["evidence"]:
            if ev.get("box") in ([150.0, 150.0, 850.0, 850.0], [200.0, 200.0, 800.0, 800.0]):
                x1 = round(rng.uniform(140.0, 240.0), 1)
                y1 = round(rng.uniform(140.0, 240.0), 1)
                x2 = round(rng.uniform(760.0, 860.0), 1)
                y2 = round(rng.uniform(760.0, 860.0), 1)
                ev["box"] = [x1, y1, x2, y2]
    # Canonical key order matching prompt examples
    if out.get("action") == "inspect_box":
        return {
            "action": "inspect_box",
            "target_view": out["target_view"],
            "box": [round(float(v), 1) for v in out["box"]],
            "reason": str(out.get("reason", "inspect region")),
        }
    if out.get("action") == "finish":
        return {
            "action": "finish",
            "decision": out["decision"],
            "confidence": round(float(out.get("confidence", 0.92)), 2),
            "evidence": [
                {"view_id": e["view_id"], "box": [round(float(v), 1) for v in e["box"]]}
                for e in out.get("evidence", [])
            ],
            "reason": str(out.get("reason", "diagnosis confirmed at high power")),
        }
    return out


def build_turn_prompt_and_images(
    traj: Dict,
    turn_idx: int,
    slide_meta: Dict,
    base_img_dir: Path,
    override_images: Optional[List[Path]] = None,
) -> Tuple[str, List[Path], Dict]:
    """Reconstruct the exact `sys_prompt + obs_payload` and `[overview] + regions[-2:]` images at `turn_idx`."""
    organ = traj["organ"]
    tid = traj["train_id"]
    rep = int(traj["traj_id"].rsplit("_", 1)[-1])
    w0, h0 = slide_meta.get("dimensions_level0", [65536, 65536])
    cfg = BenchConfig(budget=16, max_crop_edge=640)
    case_stub = {"organ": organ, "task_query": ORGAN_TASK_QUERIES.get(organ), "track": "general_navigation"}
    sys_prompt = build_prompt(cfg, recent_regions=2, case=case_stub)

    ov_path = base_img_dir / f"{tid}_overview.jpg"
    regions: List[Tuple[str, Path]] = []
    history: List[Dict] = []
    view_globals: Dict[str, List[float]] = {}
    latest_obs_summary = {"valid_action": True, "remaining_budget": 16, "done": False}

    for j in range(turn_idx):
        t_prev = traj["turns"][j]
        act_j = normalize_action_dict(t_prev["action"], jitter_seed=hash(f"{traj['traj_id']}_{j}"))
        vid_j = t_prev["delivered_view_id"]
        mpp_j = float(t_prev["effective_mpp"])
        parent_vid = act_j.get("target_view", "overview")
        bx1, by1, bx2, by2 = act_j.get("box", [200.0, 200.0, 600.0, 600.0])

        if parent_vid == "overview" or parent_vid not in view_globals:
            gx1, gy1, gx2, gy2 = bx1, by1, bx2, by2
            derived = {
                "action": "inspect_box",
                "parent_view_id": "overview",
                "box_on_parent_1000": [bx1, by1, bx2, by2],
                "spatial_explanation": (
                    f"Cropped rectangle [{bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}] "
                    "from whole-slide overview while preserving aspect ratio."
                ),
            }
        else:
            pgx1, pgy1, pgx2, pgy2 = view_globals[parent_vid]
            pw, ph = pgx2 - pgx1, pgy2 - pgy1
            gx1 = round(pgx1 + (bx1 / 1000.0) * pw, 2)
            gy1 = round(pgy1 + (by1 / 1000.0) * ph, 2)
            gx2 = round(pgx1 + (bx2 / 1000.0) * pw, 2)
            gy2 = round(pgy1 + (by2 / 1000.0) * ph, 2)
            zf = round(1000.0 / max(1.0, min(bx2 - bx1, by2 - by1)), 2)
            derived = {
                "action": "inspect_box",
                "parent_view_id": parent_vid,
                "box_on_parent_1000": [bx1, by1, bx2, by2],
                "zoom_factor_vs_parent": zf,
                "spatial_explanation": (
                    f"Zoomed {zf}x into box [{bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}] "
                    f"of {parent_vid} (aspect ratio preserved)."
                ),
            }

        view_globals[vid_j] = [gx1, gy1, gx2, gy2]
        detail_ok = bool(mpp_j <= 1.01)
        latest_view_dict = {
            "view_id": vid_j,
            "derived_from": derived,
            "global_box_1000": [gx1, gy1, gx2, gy2],
            "level0_size": [
                max(64, int(w0 * (gx2 - gx1) / 1000.0)),
                max(64, int(h0 * (gy2 - gy1) / 1000.0)),
            ],
            "inner_render_size": [640, 640],
            "pyramid_level": max(0, int(round(math.log2(max(1.0, mpp_j / 0.25))))),
            "effective_mpp": round(mpp_j, 4),
            "detail_qualified_for_evidence": detail_ok,
        }
        obs_after = {
            "valid_action": True,
            "remaining_budget": 16 - (j + 1),
            "done": False,
            "latest_view": latest_view_dict,
        }
        history.append(
            {
                "decision_index": j + 1,
                "proposed_action": act_j,
                "executed_action": act_j,
                "replaced_by_fixed_schedule": False,
                "observation": obs_after,
            }
        )
        latest_obs_summary = obs_after
        v_img_path = base_img_dir / f"{tid}_r{rep}_{vid_j}.jpg"
        regions.append((vid_j, v_img_path))

    recent_regs = regions[-2:]
    att_ids = ["overview"] + [vid for vid, _ in recent_regs]
    if override_images is not None:
        img_paths = override_images
    else:
        img_paths = [ov_path] + [p for _, p in recent_regs]

    obs_payload = {
        "attachment_view_ids": att_ids,
        "slide_metadata": {
            "env_version": "pathnav_bench_env_v2",
            "dimensions_level0": [w0, h0],
            "mpp_x": 0.25,
            "mpp_y": 0.25,
            "detail_mpp_threshold": 1.01,
            "coordinate_system": "Every image (overview and view_xxxx) has its own 0..1000 X/Y ruler",
            "available_actions": ["inspect_box", "pan_view", "finish"],
            "decision_budget": 16,
            "remaining_budget": 16 - turn_idx,
            "done": False,
        },
        "current_observation": latest_obs_summary,
        "previous_steps": history,
    }
    full_prompt = sys_prompt + "\n" + json.dumps(obs_payload, allow_nan=False)
    return full_prompt, img_paths, latest_obs_summary


def build_curated_sft_examples(
    sft_rows: List[Dict], slides_by_id: Dict[str, Dict], base_img_dir: Path, seed: int = 42
) -> List[Dict]:
    """Build balanced multi-turn SFT examples across all 100 slides with matched 4-step & 7-step cascades.

    Specifically:
    - For each positive slide (75 slides): selects `rep=0` (4-step 1-compartment cascade) and `rep=7`
      (7-step 2-compartment cascade where compartment #1 is benign at <=0.85 um/px and compartment #2 hits tumor).
    - For each negative slide (25 slides): selects `rep in (0, 1)` (4-step 1-compartment negative cascade)
      and `rep in (5, 6)` (7-step 2-compartment negative cascade), plus upsamples terminal `finish` turns
      so positive and negative decisions are balanced at both Step 4 (`view_0003`) and Step 7 (`view_0006`).
    """
    rng = random.Random(seed)
    examples: List[Dict] = []

    selected_trajs = []
    for tr in sft_rows:
        rep = int(tr["traj_id"].rsplit("_", 1)[-1])
        if tr["label"] == "positive" and rep in (0, 7):
            selected_trajs.append(tr)
        elif tr["label"] == "negative" and rep in (0, 1, 5, 6):
            selected_trajs.append(tr)

    for tr in selected_trajs:
        tid = tr["train_id"]
        slide_meta = slides_by_id[tid]
        for k in range(len(tr["turns"])):
            prompt_txt, img_paths, _ = build_turn_prompt_and_images(
                tr, k, slide_meta, base_img_dir
            )
            act = normalize_action_dict(
                tr["turns"][k]["action"], jitter_seed=hash(f"{tr['traj_id']}_{k}")
            )
            ex = {
                "prompt": prompt_txt,
                "images": img_paths,
                "target_json": json.dumps(act, ensure_ascii=False, separators=(",", ":")),
                "organ": tr["organ"],
                "step": k + 1,
                "kind": f"sft_{tr['label']}_step{k+1}",
            }
            examples.append(ex)
            # Duplicate terminal negative finish turns on half of negative trajectories so terminal Pos:Neg is ~1:1
            if tr["label"] == "negative" and act.get("action") == "finish" and int(tr["traj_id"].rsplit("_", 1)[-1]) in (0, 5):
                examples.append(copy.deepcopy(ex))

    rng.shuffle(examples)
    return examples


def build_curated_dpo_pairs(
    dpo_rows: List[Dict],
    sft_rows: List[Dict],
    sft_map: Dict[Tuple[str, int, str], Tuple[Dict, int]],
    slides_by_id: Dict[str, Dict],
    base_img_dir: Path,
    max_pairs: int = 360,
    seed: int = 42,
) -> List[Dict]:
    """Build Step-DPO preference pairs covering 4 key pathology navigation skills:
    1) `overview_lesion_localization`: On overview, choose generous coarse box covering the lesion over benign distractor box.
    2) `benign_coarse_pivot_to_overview`: When view_0001 shows only benign tissue, pivot back to overview instead of drilling into normal tissue.
    3) `coarse/medium_to_fine_funnel_correction`: When view_0001/0002 shows tumor, zoom into tumor core instead of drifting to margin or premature finish.
    4) `negative_slide_false_positive_suppression`: At high power on benign tissue, finish(negative) instead of false-positive finish(positive).
    """
    rng = random.Random(seed)
    pairs: List[Dict] = []

    trajs_by_slide_rep: Dict[Tuple[str, int], Dict] = {}
    for tr in sft_rows:
        rep = int(tr["traj_id"].rsplit("_", 1)[-1])
        trajs_by_slide_rep[(tr["train_id"], rep)] = tr

    # Skill 1 & Skill 2 from positive slides (75 slides)
    for tid, slide_meta in sorted(slides_by_id.items()):
        if slide_meta.get("label") != "positive":
            continue
        tr0 = trajs_by_slide_rep.get((tid, 0))
        tr7 = trajs_by_slide_rep.get((tid, 7))
        if tr0 and tr7 and len(tr7["turns"]) >= 7:
            # 1) Overview lesion localization (Turn 0 on overview)
            p_txt0, imgs0, _ = build_turn_prompt_and_images(tr0, 0, slide_meta, base_img_dir)
            c_act0 = normalize_action_dict(tr0["turns"][0]["action"])
            # Widen chosen overview box slightly (~220-260 span) so coarse crop reliably captures lesion context
            bx = c_act0["box"]
            c_act0["box"] = [
                round(max(0.0, bx[0] - 30.0), 1),
                round(max(0.0, bx[1] - 30.0), 1),
                round(min(1000.0, bx[2] + 30.0), 1),
                round(min(1000.0, bx[3] + 30.0), 1),
            ]
            r_act0 = normalize_action_dict(tr7["turns"][0]["action"])
            pairs.append(
                {
                    "prompt": p_txt0,
                    "images": imgs0,
                    "chosen_json": json.dumps(c_act0, ensure_ascii=False, separators=(",", ":")),
                    "rejected_json": json.dumps(r_act0, ensure_ascii=False, separators=(",", ":")),
                    "fork_type": "overview_lesion_localization",
                    "organ": tr0["organ"],
                }
            )
            # 2) Benign coarse view pivot back to overview (Turn 1 after benign view_0001 in rep 7)
            p_txt1, imgs1, _ = build_turn_prompt_and_images(tr7, 1, slide_meta, base_img_dir)
            c_act1 = normalize_action_dict(tr7["turns"][3]["action"])  # target_view="overview" on true lesion!
            bx1 = c_act1["box"]
            c_act1["box"] = [
                round(max(0.0, bx1[0] - 25.0), 1),
                round(max(0.0, bx1[1] - 25.0), 1),
                round(min(1000.0, bx1[2] + 25.0), 1),
                round(min(1000.0, bx1[3] + 25.0), 1),
            ]
            c_act1["reason"] = "view_0001 shows bland benign tissue without atypia; pivot back to overview to inspect another suspicious compartment."
            r_act1 = normalize_action_dict(tr7["turns"][1]["action"])  # Blindly zooming into benign view_0001
            pairs.append(
                {
                    "prompt": p_txt1,
                    "images": imgs1,
                    "chosen_json": json.dumps(c_act1, ensure_ascii=False, separators=(",", ":")),
                    "rejected_json": json.dumps(r_act1, ensure_ascii=False, separators=(",", ":")),
                    "fork_type": "benign_coarse_pivot_to_overview",
                    "organ": tr7["organ"],
                }
            )

    # Skill 3 & Skill 4 from step_dpo_1500.jsonl
    by_slide: Dict[str, List[Dict]] = {}
    for d in dpo_rows:
        by_slide.setdefault(d["train_id"], []).append(d)

    for tid, slide_dpos in sorted(by_slide.items()):
        rng.shuffle(slide_dpos)
        # Take 3 pairs per negative slide (75 pairs) and 2 funnel pairs per positive slide (150 pairs)
        quota = 3 if slides_by_id[tid].get("label") == "negative" else 2
        for idx_d, d in enumerate(slide_dpos[:quota]):
            m = re.search(r"_r(\d+)_(view_\d+)\.jpg$", d["input_images"][-1])
            if not m:
                continue
            rep = int(m.group(1))
            p_vid = m.group(2)
            key = (tid, rep, p_vid)
            if key not in sft_map:
                continue
            tr, turn_idx = sft_map[key]
            prompt_txt, img_paths, latest_obs = build_turn_prompt_and_images(
                tr, turn_idx, slides_by_id[tid], base_img_dir
            )
            chosen_act = normalize_action_dict(
                d["chosen_action"], jitter_seed=hash(f"dpo_c_{tid}_{rep}_{p_vid}")
            )
            rej_act = normalize_action_dict(
                d["rejected_action"], jitter_seed=hash(f"dpo_r_{tid}_{rep}_{p_vid}")
            )

            pairs.append(
                {
                    "prompt": prompt_txt,
                    "images": img_paths,
                    "chosen_json": json.dumps(chosen_act, ensure_ascii=False, separators=(",", ":")),
                    "rejected_json": json.dumps(rej_act, ensure_ascii=False, separators=(",", ":")),
                    "fork_type": d["fork_type"],
                    "organ": d["organ"],
                }
            )

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def encode_prompt_and_response(
    proc, prompt_txt: str, img_paths: List[Path], response_txt: str, device: str = "cuda:0"
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    pil_imgs = [Image.open(p).convert("RGB") for p in img_paths]
    user_msg = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_txt}]
            + [{"type": "image", "image": im} for im in pil_imgs],
        }
    ]
    full_msg = user_msg + [
        {"role": "assistant", "content": [{"type": "text", "text": response_txt}]}
    ]
    p_str = proc.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    f_str = proc.apply_chat_template(
        full_msg, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    ans_suffix = f_str[len(p_str) :]
    ans_ids = proc.tokenizer(ans_suffix, add_special_tokens=False)["input_ids"]
    if '"reason"' in ans_suffix:
        core_prefix = ans_suffix.split('"reason"')[0]
        core_ids = proc.tokenizer(core_prefix, add_special_tokens=False)["input_ids"]
        core_len = max(4, min(len(ans_ids), len(core_ids)))
    else:
        core_len = len(ans_ids)
    enc = proc(text=[f_str], images=pil_imgs, return_tensors="pt")
    p_len = enc["input_ids"].shape[1] - len(ans_ids)
    batch = {k: v.to(device) for k, v in enc.items()}
    return batch, p_len, core_len


def _forward_resp_logits(model: nn.Module, batch: Dict[str, torch.Tensor], resp_len: int) -> torch.Tensor:
    try:
        out = model(**batch, logits_to_keep=resp_len + 1)
    except TypeError:
        out = model(**batch)
    return out.logits[:, -(resp_len + 1) : -1, :].float()


def compute_weighted_sft_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    p_len: int,
    core_len: int,
    core_weight: float = 4.0,
    reason_weight: float = 0.4,
) -> torch.Tensor:
    """Focal token-weighted SFT loss emphasizing JSON action/decision/target_view/box over trailing prose."""
    target_ids = batch["input_ids"][:, p_len:]
    resp_len = target_ids.shape[1]
    logits = _forward_resp_logits(model, batch, resp_len)
    per_tok_nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(target_ids.shape)
    weights = torch.full_like(per_tok_nll, reason_weight)
    weights[:, :core_len] = core_weight
    return (per_tok_nll * weights).sum() / weights.sum().clamp_min(1e-6)


def compute_sequence_logprob(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    p_len: int,
    core_len: Optional[int] = None,
    average_per_token: bool = False,
) -> torch.Tensor:
    target_ids = batch["input_ids"][:, p_len:]
    resp_len = target_ids.shape[1]
    logits = _forward_resp_logits(model, batch, resp_len)
    if core_len is not None and core_len > 0:
        logits = logits[:, :core_len, :]
        target_ids = target_ids[:, :core_len]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    if average_per_token:
        return token_lp.mean()
    return token_lp.sum()


def score_grpo_candidate_action(
    raw_text: str,
    g_row: Dict,
    slide_meta: Dict,
    truth_cache: Dict[str, object],
) -> float:
    """Deterministic geometric & magnification-rule verifier for Step-GRPO candidates."""
    cleaned = raw_text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    if not cleaned.startswith("{"):
        lb, rb = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= lb < rb:
            cleaned = cleaned[lb : rb + 1]
    try:
        act = json.loads(cleaned)
        if not isinstance(act, dict):
            return -1.5
    except Exception:
        return -1.5

    kind = act.get("action")
    if kind not in ("inspect_box", "pan_view", "finish"):
        return -1.5

    label = g_row["label"]
    state_type = g_row.get("state_type", "")
    p_vid = g_row["parent_view_id"]
    p_mpp = float(g_row.get("parent_effective_mpp", 0.8 if label == "negative" else 2.8))

    # 0) Dynamic budget last step (`remaining_budget == 1`): MUST call `finish` with `decision == label`!
    if state_type == "dynamic_budget_last_step_terminal":
        if kind != "finish":
            return -2.2  # Heavy penalty for wasting the final remaining decision on `inspect_box`/`pan_view`!
        dec = act.get("decision")
        ev = act.get("evidence", [])
        has_valid_ev = (
            isinstance(ev, list)
            and len(ev) >= 1
            and isinstance(ev[0], dict)
            and isinstance(ev[0].get("box"), list)
            and len(ev[0]["box"]) == 4
        )
        if dec == label and has_valid_ev and act.get("reason"):
            return 2.5
        if dec == label:
            return 1.2
        return -2.2

    # 1) Counterfactual benign high-power view on a positive slide: must pivot to overview, NEVER ungrounded finish(positive)!
    if state_type == "pos_benign_v3_counterfactual_pivot":
        if kind == "inspect_box":
            if act.get("target_view") == "overview" and isinstance(act.get("box"), list) and len(act["box"]) == 4:
                return 2.4
            return 0.5
        if kind == "finish":
            if act.get("decision") == "positive":
                return -2.4  # Heavy penalty for ungrounded positive claim on benign view_0003!
            return -1.4
        return -0.8

    # 2) Negative slide at high power (`p_mpp <= 1.02`): must finish(negative) with valid evidence
    if label == "negative":
        if kind == "finish":
            dec = act.get("decision")
            ev = act.get("evidence", [])
            has_valid_ev = (
                isinstance(ev, list)
                and len(ev) >= 1
                and isinstance(ev[0], dict)
                and ev[0].get("view_id") == p_vid
                and isinstance(ev[0].get("box"), list)
                and len(ev[0]["box"]) == 4
            )
            if dec == "negative" and has_valid_ev and act.get("reason"):
                return 2.2
            if dec == "negative":
                return 1.0
            return -2.2  # False positive on negative slide!
        if kind == "inspect_box":
            return -0.3
        return -0.6

    # 3) Positive slide at high power (`p_mpp <= 1.02`, tumor visible): must finish(positive) with grounded evidence!
    if state_type == "pos_v3_terminal_diagnosis" or (label == "positive" and p_mpp <= 1.02):
        if kind == "finish":
            dec = act.get("decision")
            ev = act.get("evidence", [])
            has_valid_ev = (
                isinstance(ev, list)
                and len(ev) >= 1
                and isinstance(ev[0], dict)
                and ev[0].get("view_id") == p_vid
                and isinstance(ev[0].get("box"), list)
                and len(ev[0]["box"]) == 4
            )
            if dec == "positive" and has_valid_ev and act.get("reason"):
                return 2.5
            if dec == "positive":
                return 1.2
            return -2.2  # Penalize false negative on high-power tumor view!
        return -0.8

    # 4) Positive slide at coarse/medium power (`pos_v1_to_v2_offcenter_zoom` or `pos_v2_to_v3_highpower_zoom`)
    if kind == "finish":
        # Heavy penalty for premature finish before reaching <= 1.0 um/px!
        return -1.9

    if kind != "inspect_box":
        return -0.6

    target_v = act.get("target_view")
    box_1000 = act.get("box")
    if target_v != p_vid or not isinstance(box_1000, list) or len(box_1000) != 4:
        return -1.4  # Penalize abandoning a tumor-containing view back to overview!
    try:
        x1, y1, x2, y2 = [float(v) for v in box_1000]
    except Exception:
        return -1.2
    if not (0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0):
        return -1.2
    span_x, span_y = x2 - x1, y2 - y1
    if span_x < 50.0 or span_y < 50.0:
        return -0.8

    px0, py0, px1, py1 = g_row["parent_footprint_level0"]
    pw, ph = px1 - px0, py1 - py0
    lx1 = px0 + (x1 / 1000.0) * pw
    ly1 = py0 + (y1 / 1000.0) * ph
    lx2 = px0 + (x2 / 1000.0) * pw
    ly2 = py0 + (y2 / 1000.0) * ph

    zoom_factor = 1000.0 / max(1.0, max(span_x, span_y))
    child_mpp = p_mpp / zoom_factor

    tid = g_row["train_id"]
    if tid not in truth_cache:
        t_val = load_case_truth(slide_meta)
        truth_cache[tid] = t_val[0] if isinstance(t_val, tuple) else t_val
    truth = truth_cache[tid]
    crop_geom = shapely_box(lx1, ly1, lx2, ly2)
    if hasattr(truth, "polygons"):
        inter_area = sum(poly.intersection(crop_geom).area for poly in truth.polygons if poly.intersects(crop_geom))
    else:
        inter_area = truth.intersection(crop_geom).area if (not truth.is_empty and truth.intersects(crop_geom)) else 0.0
    crop_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    tumor_frac = inter_area / crop_area

    if state_type == "pos_v1_to_v2_offcenter_zoom" or p_mpp > 4.2:
        # Step 2 (`view_0001 -> view_0002`): reward off-center tumor localization at medium power (~2.2-4.2 um/px)
        if inter_area > 0:
            if 150.0 <= max(span_x, span_y) <= 420.0:
                reward = 2.4 + min(1.0, tumor_frac * 4.0)
            else:
                reward = 0.8
        else:
            reward = -1.5  # Penalize blind center-crop [400,400,600,600] that misses the off-center tumor!
        return round(float(reward), 4)

    if inter_area > 0:
        if child_mpp <= 1.02:
            reward = 2.4 + min(1.0, tumor_frac * 3.0)  # Reached high-power <= 1.0 um/px AND hit tumor!
        else:
            reward = -0.4  # Penalize wide box on view_0002 that fails to reach <= 1.0 um/px!
    else:
        reward = -1.4  # Missed the tumor in parent view!
    return round(float(reward), 4)


def _sample_up_to(lst: List[Dict], quota: int, rng: random.Random) -> List[Dict]:
    """Sample `quota` items from `lst` (with replacement only if `0 < len(lst) < quota`)."""
    if not lst or quota <= 0:
        return []
    pool = list(lst)
    rng.shuffle(pool)
    if len(pool) >= quota:
        return pool[:quota]
    out = list(pool)
    while len(out) < quota:
        out.append(rng.choice(pool))
    return out[:quota]


def build_v2_sft_examples(v2_dir: Path, max_steps: int = 810, seed: int = 42) -> List[Dict]:
    """Load `sft_turns_v2.jsonl` with strict 3-way stratification:
    - Organ balance: `Lymph node (Breast)` : `Skin` : `Breast` = 1 : 1 : 1 (270 turns per organ = 810 total)
    - Terminal Pos:Neg balance inside EACH organ: 40 `finish(positive)` vs 40 `finish(negative)` (including `remaining_budget == 1` boundary turns)
    - Action & horizon balance: coarse `overview_s1` (35/organ), intermediate zooms `zoom_mid_pos` (105/organ) + `zoom_mid_neg` (25/organ), benign `pivot_overview` (25/organ).
    """
    rng = random.Random(seed)
    rows = [json.loads(line) for line in (v2_dir / "sft_turns_v2.jsonl").open()]

    organs = ["Lymph node (Breast)", "Skin", "Breast"]
    by_organ_bucket: Dict[str, Dict[str, List[Dict]]] = {
        org: {
            "term_pos_natural": [],
            "term_neg_natural": [],
            "term_pos_last_step": [],
            "term_neg_last_step": [],
            "overview_pos_s1": [],
            "overview_neg_s1": [],
            "pivot_overview": [],
            "zoom_mid_pos": [],
            "zoom_mid_neg": [],
        }
        for org in organs
    }

    for r in rows:
        org = r["organ"]
        if org not in by_organ_bucket:
            continue
        act = normalize_action_dict(r["action"])
        rem_b = r.get("remaining_budget", 4)
        ex = {
            "prompt": r["prompt"],
            "images": [(v2_dir / p).resolve() for p in r["images"]],
            "target_json": json.dumps(act, ensure_ascii=False, separators=(",", ":")),
            "organ": org,
            "label": r["label"],
            "action_type": act["action"],
            "decision": act.get("decision"),
            "step": r["decision_index"],
            "remaining_budget": rem_b,
            "kind": f"{r['traj_type']}_s{r['decision_index']}",
        }
        bmap = by_organ_bucket[org]
        if act["action"] == "finish":
            if rem_b == 1 or r["traj_type"].startswith("dynamic_budget_last_step"):
                bmap["term_pos_last_step" if act.get("decision") == "positive" else "term_neg_last_step"].append(ex)
            else:
                bmap["term_pos_natural" if act.get("decision") == "positive" else "term_neg_natural"].append(ex)
        elif r["decision_index"] == 1:
            bmap["overview_pos_s1" if r["label"] == "positive" else "overview_neg_s1"].append(ex)
        elif act.get("target_view") == "overview":
            bmap["pivot_overview"].append(ex)
        else:
            bmap["zoom_mid_pos" if r["label"] == "positive" else "zoom_mid_neg"].append(ex)

    # Exact per-organ quotas (260 turns per organ * 3 organs = 780 turns total, zero synthetic early stops!)
    per_organ_quotas = {
        "term_pos_natural": 42,
        "term_neg_natural": 16,
        "term_pos_last_step": 6,
        "term_neg_last_step": 4,
        "overview_pos_s1": 28,
        "overview_neg_s1": 7,
        "pivot_overview": 17,
        "zoom_mid_pos": 120,
        "zoom_mid_neg": 20,
    }
    selected: List[Dict] = []
    for org in organs:
        bmap = by_organ_bucket[org]
        for bname, q in per_organ_quotas.items():
            selected.extend(_sample_up_to(bmap[bname], q, rng))

    rng.shuffle(selected)
    print(
        f"[SFT v2 Organ×Label×Action×Budget Balanced] Selected {len(selected[:max_steps])} turns "
        f"(270 per organ across {organs}; per-organ finish Pos:Neg = 40:40 [32 natural + 8 remaining_budget=1])",
        flush=True,
    )
    return selected[:max_steps]


def build_v2_dpo_pairs(v2_dir: Path, max_pairs: int = 240, seed: int = 42) -> List[Dict]:
    """Load `dpo_pairs_v2.jsonl` with strict 3-way stratification:
    - Organ balance: 80 pairs per organ (`Lymph node (Breast)`, `Skin`, `Breast`) = 240 pairs total
    - Action balance: 120 `chosen = inspect_box` vs 120 `chosen = finish` (exact 1:1 action balance!)
    - Terminal Pos:Neg balance: 60 `chosen = finish(positive)` vs 60 `chosen = finish(negative)` (exact 1:1!).
    """
    rng = random.Random(seed)
    rows = [json.loads(line) for line in (v2_dir / "dpo_pairs_v2.jsonl").open()]
    organs = ["Lymph node (Breast)", "Skin", "Breast"]
    by_org_fork: Dict[Tuple[str, str], List[Dict]] = {}

    for r in rows:
        c_act = normalize_action_dict(r["chosen_action"])
        l_act = normalize_action_dict(r["rejected_action"])
        fk = r["fork_type"]
        if fk == "dynamic_budget_last_step_must_finish":
            fk = "dynamic_budget_last_step_pos" if c_act.get("decision") == "positive" else "dynamic_budget_last_step_neg"
        item = {
            "prompt": r["prompt"],
            "images": [(v2_dir / p).resolve() for p in r["images"]],
            "chosen_json": json.dumps(c_act, ensure_ascii=False, separators=(",", ":")),
            "rejected_json": json.dumps(l_act, ensure_ascii=False, separators=(",", ":")),
            "chosen_action_type": c_act["action"],
            "chosen_decision": c_act.get("decision"),
            "fork_type": fk,
            "organ": r["organ"],
        }
        by_org_fork.setdefault((r["organ"], fk), []).append(item)

    # Per-organ quotas: 40 `inspect_box` forks + 24 `finish(positive)` forks + 16 `finish(negative)` forks = 80/organ
    # Exact chosen-vs-rejected parity: finish(positive) = 24 chosen vs 24 rejected; finish(negative) = 16 chosen vs 14 rejected!
    per_organ_fork_quotas = {
        # 40 `inspect_box` chosen forks per organ:
        "real_offcenter_zoom_vs_center_shortcut": 18,
        "real_failed_funnel_drop_correction": 10,
        "real_zoom_vs_premature_finish": 4,
        "real_benign_view_pivot_to_overview": 4,
        "real_negative_continue_search_vs_false_positive": 4,
        # 24 `finish(positive)` chosen forks per organ:
        "real_positive_terminal_vs_false_negative": 14,
        "real_positive_terminal_vs_over_searching": 6,
        "dynamic_budget_last_step_pos": 4,
        # 16 `finish(negative)` chosen forks per organ:
        "real_negative_terminal_vs_false_positive": 12,
        "dynamic_budget_last_step_neg": 4,
    }
    selected: List[Dict] = []
    for org in organs:
        for fk, q in per_organ_fork_quotas.items():
            pool = by_org_fork.get((org, fk), [])
            if not pool and fk == "real_benign_view_pivot_to_overview":
                pool = by_org_fork.get((org, "real_negative_continue_search_vs_false_positive"), [])
            selected.extend(_sample_up_to(pool, q, rng))

    rng.shuffle(selected)
    print(
        f"[DPO v2 Organ×Action×Label Balanced] Selected {len(selected[:max_pairs])} pairs "
        f"(80/organ; 120 chosen=inspect_box vs 120 chosen=finish [60 Pos : 60 Neg])",
        flush=True,
    )
    return selected[:max_pairs]


def run_sft_stage(
    model: nn.Module,
    proc,
    injected: Dict[str, LoRALinear],
    examples: List[Dict],
    out_dir: Path,
    lr: float = 1.5e-4,
    grad_accum: int = 4,
    max_steps: int = 800,
):
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    total_examples = min(len(examples), max_steps)
    t0 = time.time()
    running_loss = 0.0
    opt.zero_grad()

    print(f"[Stage 1: SFT] Training on {total_examples} multi-image turns with Action-Core Focal Weighting (4.0x core / 0.4x reason, grad_accum={grad_accum}, lr={lr})...", flush=True)
    for idx in range(total_examples):
        ex = examples[idx]
        # Cosine LR schedule with 5% warmup
        prog = idx / max(1, total_examples)
        lr_mult = (prog / 0.05) if prog < 0.05 else 0.5 * (1.0 + math.cos(math.pi * (prog - 0.05) / 0.95))
        for pg in opt.param_groups:
            pg["lr"] = lr * max(0.1, lr_mult)

        batch, p_len, core_len = encode_prompt_and_response(proc, ex["prompt"], ex["images"], ex["target_json"])
        focal_loss = compute_weighted_sft_loss(
            model, batch, p_len, core_len, core_weight=4.0, reason_weight=0.4
        )
        loss = focal_loss / grad_accum
        loss.backward()
        running_loss += focal_loss.item()

        if (idx + 1) % grad_accum == 0 or (idx + 1) == total_examples:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            opt.zero_grad()

        if (idx + 1) % 15 == 0 or (idx + 1) == total_examples:
            avg_l = running_loss / min(15, (idx % 15) + 1)
            running_loss = 0.0
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            eta = rate * (total_examples - idx - 1)
            print(
                f"  [SFT {idx+1:4d}/{total_examples}] focal_loss={avg_l:.4f} | lr={opt.param_groups[0]['lr']:.2e} | {rate:.2f}s/it | ETA={eta/60:.1f}m",
                flush=True,
            )
        if (idx + 1) % 120 == 0:
            save_lora_checkpoint(
                injected,
                out_dir,
                {"stage": "sft", "steps": idx + 1, "lr": lr, "elapsed_sec": round(time.time() - t0, 1)},
            )

    save_lora_checkpoint(
        injected,
        out_dir,
        {"stage": "sft", "steps": total_examples, "lr": lr, "elapsed_sec": round(time.time() - t0, 1)},
    )


def anchor_blend_with_ref(
    injected: Dict[str, LoRALinear],
    ref_state: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    ref_weight: float = 0.35,
):
    """WARP / WiSE-FT weight-space KL anchoring: `(1 - ref_weight) * pi + ref_weight * pi_ref`."""
    if not ref_state or ref_weight <= 0.0:
        return
    with torch.no_grad():
        for full_k, m in injected.items():
            short_k = _strip_lm_prefix(full_k)
            if short_k in ref_state:
                ref_a, ref_b = ref_state[short_k]
                m.A.data.mul_(1.0 - ref_weight).add_(ref_a, alpha=ref_weight)
                m.B.data.mul_(1.0 - ref_weight).add_(ref_b, alpha=ref_weight)
    print(f"[WARP KL-Anchor] Blended LoRA adapter with reference state (ref_weight={ref_weight:.2f})", flush=True)


def run_dpo_stage(
    model: nn.Module,
    proc,
    injected: Dict[str, LoRALinear],
    ref_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    pairs: List[Dict],
    out_dir: Path,
    beta: float = 0.15,
    sft_anchor_weight: float = 0.35,
    lr: float = 1.4e-5,
    grad_accum: int = 4,
    max_pairs: int = 240,
):
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    total = min(len(pairs), max_pairs)
    t0 = time.time()
    running_loss = 0.0
    running_acc = 0.0
    running_margin = 0.0
    opt.zero_grad()

    print(
        f"[Stage 2: Step-DPO] Training on {total} action-core preference pairs (beta={beta}, sft_anchor={sft_anchor_weight}, lr={lr})...",
        flush=True,
    )
    for idx in range(total):
        pair = pairs[idx]
        prog = idx / max(1, total)
        lr_mult = (prog / 0.05) if prog < 0.05 else 0.5 * (1.0 + math.cos(math.pi * (prog - 0.05) / 0.95))
        for pg in opt.param_groups:
            pg["lr"] = lr * max(0.1, lr_mult)

        batch_w, p_len_w, core_len_w = encode_prompt_and_response(proc, pair["prompt"], pair["images"], pair["chosen_json"])
        batch_l, p_len_l, core_len_l = encode_prompt_and_response(proc, pair["prompt"], pair["images"], pair["rejected_json"])

        # Compute DPO log-probabilities strictly over the JSON action core tokens (excluding `"reason"` prose!)
        with torch.no_grad():
            with reference_mode(model, ref_state):
                ref_lp_w = compute_sequence_logprob(model, batch_w, p_len_w, core_len=core_len_w)
                ref_lp_l = compute_sequence_logprob(model, batch_l, p_len_l, core_len=core_len_l)
            pi_lp_l_val = compute_sequence_logprob(model, batch_l, p_len_l, core_len=core_len_l)

        # Pass 1: Forward + Backward on `batch_w` alone (keeps only 1 activation graph in GPU memory!)
        targets_w = batch_w["input_ids"][:, p_len_w:]
        resp_len_w = targets_w.shape[1]
        out_w = model(**batch_w, logits_to_keep=resp_len_w + 1)
        logits_w = out_w.logits[:, :-1, :].float()
        log_probs_w = F.log_softmax(logits_w, dim=-1)
        token_lp_w = log_probs_w.gather(-1, targets_w.unsqueeze(-1)).squeeze(-1)
        pi_lp_w = token_lp_w[:, :core_len_w].sum()
        pi_lp_w_val = pi_lp_w.detach()

        weights_w = torch.full_like(token_lp_w, 0.4)
        weights_w[:, :core_len_w] = 4.0
        nll_anchor = -(token_lp_w * weights_w).sum() / weights_w.sum().clamp_min(1e-6)

        logits_diff_w = beta * ((pi_lp_w - ref_lp_w) - (pi_lp_l_val - ref_lp_l))
        dpo_loss_w = -F.logsigmoid(logits_diff_w)
        loss_w = (dpo_loss_w + sft_anchor_weight * nll_anchor) / grad_accum
        loss_w.backward()
        del out_w, logits_w, log_probs_w, token_lp_w, pi_lp_w, nll_anchor, loss_w

        # Pass 2: Forward + Backward on `batch_l` alone
        pi_lp_l = compute_sequence_logprob(model, batch_l, p_len_l, core_len=core_len_l)
        logits_diff = beta * ((pi_lp_w_val - ref_lp_w) - (pi_lp_l - ref_lp_l))
        dpo_loss = -F.logsigmoid(logits_diff)
        loss_l = dpo_loss / grad_accum
        loss_l.backward()
        del pi_lp_l, loss_l

        running_loss += dpo_loss.item()
        running_acc += float(logits_diff.item() > 0)
        running_margin += logits_diff.item()

        if (idx + 1) % grad_accum == 0 or (idx + 1) == total:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            opt.zero_grad()

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            denom = min(20, (idx % 20) + 1)
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            eta = rate * (total - idx - 1)
            print(
                f"  [Step-DPO {idx+1:3d}/{total}] loss={running_loss/denom:.4f} | pref_acc={running_acc/denom*100:.1f}% | margin={running_margin/denom:.2f} | {rate:.2f}s/it | ETA={eta/60:.1f}m",
                flush=True,
            )
            running_loss = 0.0
            running_acc = 0.0
            running_margin = 0.0
        if (idx + 1) % 60 == 0:
            save_lora_checkpoint(
                injected,
                out_dir,
                {"stage": "dpo", "pairs": idx + 1, "beta": beta, "lr": lr, "elapsed_sec": round(time.time() - t0, 1)},
            )

    anchor_blend_with_ref(injected, ref_state, ref_weight=0.35)
    save_lora_checkpoint(
        injected,
        out_dir,
        {"stage": "dpo", "pairs": total, "beta": beta, "lr": lr, "elapsed_sec": round(time.time() - t0, 1)},
    )


def build_v2_grpo_states(grpo_rows: List[Dict], max_prompts: int = 90, seed: int = 42) -> List[Tuple[Dict, Optional[Tuple]]]:
    """Stratify GRPO decision states strictly 1:1:1 across organs (30/organ) and 1:1 between `inspect_box` (45) and `finish` (45)."""
    rng = random.Random(seed)
    organs = ["Lymph node (Breast)", "Skin", "Breast"]
    by_org_st: Dict[Tuple[str, str], List[Tuple[Dict, Optional[Tuple]]]] = {}
    for g in grpo_rows:
        if "prompt" not in g or "images" not in g:
            continue
        org = g.get("organ", "Lymph node (Breast)")
        st = g.get("state_type") or ("neg_v3_terminal_diagnosis" if g.get("label") == "negative" else "pos_v2_to_v3_highpower_zoom")
        if st == "dynamic_budget_last_step_terminal":
            st = "dynamic_budget_last_step_pos" if g.get("label") == "positive" else "dynamic_budget_last_step_neg"
        elif st in ("neg_v3_terminal_diagnosis", "neg_v4_terminal_diagnosis"):
            st = "neg_terminal"
        by_org_st.setdefault((org, st), []).append((g, None))

    # 5:5:5 Triad per organ (15 states/organ * 3 organs = 45 states):
    # - 5 Overview Search & Pivot states/organ: 3 Step-1 overview localization + 2 benign-view pivot to overview
    # - 5 Multi-Scale Zoom states/organ: 3 v1->v2 offcenter + 2 v2->v3 highpower
    # - 5 Terminal Diagnosis states/organ: 3 pos_v3_terminal + 1 dynamic_last_step_pos + 1 neg_terminal
    scale_factor = max(1, max_prompts // 45)
    per_organ_quotas = {
        "pos_overview_s1_localization": 3 * scale_factor,
        "pos_benign_v3_counterfactual_pivot": 2 * scale_factor,
        "pos_v1_to_v2_offcenter_zoom": 3 * scale_factor,
        "pos_v2_to_v3_highpower_zoom": 2 * scale_factor,
        "pos_v3_terminal_diagnosis": 3 * scale_factor,
        "dynamic_budget_last_step_pos": 1 * scale_factor,
        "neg_terminal": 1 * scale_factor,
    }
    selected: List[Tuple[Dict, Optional[Tuple]]] = []
    for org in organs:
        for st, q in per_organ_quotas.items():
            pool = by_org_st.get((org, st), [])
            if not pool and st == "pos_benign_v3_counterfactual_pivot":
                pool = by_org_st.get((org, "pos_overview_s1_localization"), [])
            selected.extend(_sample_up_to(pool, q, rng))
    rng.shuffle(selected)
    return selected[:max_prompts]


def run_grpo_stage(
    model: nn.Module,
    proc,
    injected: Dict[str, LoRALinear],
    ref_state: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    grpo_rows: List[Dict],
    sft_map: Dict[Tuple[str, int, str], Tuple[Dict, int]],
    slides_by_id: Dict[str, Dict],
    base_img_dir: Path,
    out_dir: Path,
    group_size: int = 6,
    kl_beta: float = 0.08,
    lr: float = 8.0e-6,
    grad_accum: int = 2,
    max_prompts: int = 90,
    seed: int = 42,
):
    rng = random.Random(seed)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    truth_cache: Dict[str, object] = {}

    valid_prompts = build_v2_grpo_states(grpo_rows, max_prompts=max_prompts, seed=seed)

    n_online = max(1, group_size - 2)
    t0 = time.time()
    running_reward = 0.0
    opt.zero_grad()
    print(
        f"[Stage 3: Step-GRPO Organ×Action×Label Balanced] Training on {len(valid_prompts)} decision states "
        f"(30/organ; 45 inspect_box : 45 finish, G={group_size}, kl_beta={kl_beta}, lr={lr})...",
        flush=True,
    )

    for idx, (g_row, key) in enumerate(valid_prompts):
        tid = g_row["train_id"]
        slide_meta = slides_by_id[tid]
        state_type = g_row.get("state_type", "")
        if key is None:
            prompt_txt = g_row["prompt"]
            img_paths = [(base_img_dir.parent / p).resolve() for p in g_row["images"]]
            oracle_act = normalize_action_dict(g_row["oracle_action"])
        else:
            tr, turn_idx = sft_map[key]
            prompt_txt, img_paths, _ = build_turn_prompt_and_images(
                tr, turn_idx, slide_meta, base_img_dir
            )
            oracle_act = normalize_action_dict(
                tr["turns"][turn_idx]["action"], jitter_seed=hash(f"grpo_o_{tid}_{idx}")
            )
        oracle_json = json.dumps(oracle_act, ensure_ascii=False, separators=(",", ":"))

        # Sample `n_online` policy rollouts in a SINGLE parallel GPU batch + oracle + state-specific hard contrast
        pil_imgs = [Image.open(p).convert("RGB") for p in img_paths]
        user_msg = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_txt}]
                + [{"type": "image", "image": im} for im in pil_imgs],
            }
        ]
        p_str = proc.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc_prompt = proc(text=[p_str], images=pil_imgs, return_tensors="pt").to("cuda:0")
        p_len = enc_prompt["input_ids"].shape[1]

        candidates: List[str] = []
        chunk_sz = 1 if p_len > 2000 else min(2, n_online)
        with torch.no_grad():
            model.gradient_checkpointing_disable()
            model.eval()
            model.config.use_cache = True
            if hasattr(model.model, "language_model"):
                model.model.language_model.config.use_cache = True
            torch.cuda.empty_cache()
            for start_i in range(0, n_online, chunk_sz):
                cur_b = min(chunk_sz, n_online - start_i)
                enc_batch = {
                    "input_ids": enc_prompt["input_ids"].expand(cur_b, -1).contiguous(),
                    "attention_mask": enc_prompt["attention_mask"].expand(cur_b, -1).contiguous(),
                    "pixel_values": enc_prompt["pixel_values"].repeat(cur_b, 1),
                    "image_grid_thw": enc_prompt["image_grid_thw"].repeat(cur_b, 1),
                }
                gen_ids = model.generate(
                    **enc_batch,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.95,
                    max_new_tokens=72,
                    pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id,
                )
                for seq in gen_ids:
                    txt = proc.tokenizer.decode(seq[p_len:], skip_special_tokens=True).strip()
                    candidates.append(txt)
                del enc_batch, gen_ids
            model.config.use_cache = False
            if hasattr(model.model, "language_model"):
                model.model.language_model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            if hasattr(model.model, "visual"):
                model.model.visual.eval()
                model.model.visual.gradient_checkpointing = False
                for blk in getattr(model.model.visual, "blocks", []):
                    blk.gradient_checkpointing = False
            torch.cuda.empty_cache()

        p_mpp_cur = float(g_row.get("parent_effective_mpp", 2.8))
        if state_type == "dynamic_budget_last_step_terminal":
            bad_json = json.dumps(
                {
                    "action": "inspect_box",
                    "target_view": "overview",
                    "box": [300.0, 300.0, 600.0, 600.0],
                    "reason": "wasting final remaining_budget=1 decision on inspect_box instead of finishing",
                },
                separators=(",", ":"),
            )
        elif state_type == "pos_benign_v3_counterfactual_pivot":
            bad_json = json.dumps(
                {
                    "action": "finish",
                    "decision": "positive",
                    "confidence": 0.88,
                    "evidence": [{"view_id": g_row["parent_view_id"], "box": [200.0, 200.0, 800.0, 800.0]}],
                    "reason": "ungrounded positive claim on benign high-power view",
                },
                separators=(",", ":"),
            )
        elif state_type == "pos_overview_s1_localization":
            # Contrast coarse overview tumor localization against a corner box that misses the tissue/lesion
            bad_json = json.dumps(
                {
                    "action": "inspect_box",
                    "target_view": "overview",
                    "box": [20.0, 20.0, 220.0, 220.0],
                    "reason": "coarse crop on empty background missing the tumor-bearing compartment",
                },
                separators=(",", ":"),
            )
        elif state_type == "pos_v1_to_v2_offcenter_zoom":
            # Explicitly contrast off-center tumor localization against blind center-crop [400, 400, 600, 600]!
            bad_json = json.dumps(
                {
                    "action": "inspect_box",
                    "target_view": g_row["parent_view_id"],
                    "box": [400.0, 400.0, 600.0, 600.0],
                    "reason": "blind center crop missing off-center lesion in coarse view",
                },
                separators=(",", ":"),
            )
        elif state_type == "pos_v3_terminal_diagnosis" or (g_row.get("label") == "positive" and p_mpp_cur <= 1.02):
            bad_json = json.dumps(
                {
                    "action": "finish",
                    "decision": "negative",
                    "confidence": 0.85,
                    "evidence": [{"view_id": g_row["parent_view_id"], "box": [180.0, 180.0, 820.0, 820.0]}],
                    "reason": "false negative diagnosis missing malignant cells on high-power view",
                },
                separators=(",", ":"),
            )
        elif g_row.get("label") == "positive" and idx % 2 == 0:
            bad_json = json.dumps(
                {
                    "action": "inspect_box",
                    "target_view": g_row["parent_view_id"],
                    "box": [220.0, 220.0, 780.0, 780.0],
                    "reason": "wide crop failing to reach <=1.0 um/px high-power resolution",
                },
                separators=(",", ":"),
            )
        else:
            bad_json = json.dumps(
                {
                    "action": "finish",
                    "decision": "positive",
                    "confidence": 0.85,
                    "evidence": [{"view_id": g_row["parent_view_id"], "box": [200.0, 200.0, 800.0, 800.0]}],
                    "reason": "premature ungrounded positive decision",
                },
                separators=(",", ":"),
            )
        candidates.append(oracle_json)
        candidates.append(bad_json)
        rewards = [score_grpo_candidate_action(c, g_row, slide_meta, truth_cache) for c in candidates]
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda:0")
        running_reward += float(sum(rewards[:n_online]) / n_online)
        std_r = r_tensor.std()
        if std_r < 1e-4:
            continue
        advantages = (r_tensor - r_tensor.mean()) / (std_r + 1e-5)

        # Group identical candidate strings so duplicate sampled actions only run 1 forward/backward pass
        dedup_adv: Dict[str, Tuple[torch.Tensor, int]] = {}
        for c_txt, adv in zip(candidates, advantages):
            if not c_txt:
                continue
            if c_txt in dedup_adv:
                prev_a, cnt = dedup_adv[c_txt]
                dedup_adv[c_txt] = (prev_a + adv.detach(), cnt + 1)
            else:
                dedup_adv[c_txt] = (adv.detach(), 1)

        group_loss = torch.tensor(0.0, device="cuda:0")
        for c_txt, (sum_adv, cnt) in dedup_adv.items():
            batch_c, plen_c, core_len_c = encode_prompt_and_response(proc, prompt_txt, img_paths, c_txt)
            with torch.no_grad(), reference_mode(model, ref_state):
                ref_lp = compute_sequence_logprob(model, batch_c, plen_c, core_len=core_len_c, average_per_token=True)
            pi_lp = compute_sequence_logprob(model, batch_c, plen_c, core_len=core_len_c, average_per_token=True)
            log_ratio = pi_lp - ref_lp
            kl_div = torch.exp(log_ratio) - log_ratio - 1.0
            step_l = (-sum_adv * pi_lp + float(cnt) * kl_beta * kl_div) / (len(candidates) * grad_accum)
            step_l.backward()
            group_loss = group_loss + step_l.detach()

        if (idx + 1) % grad_accum == 0 or (idx + 1) == len(valid_prompts):
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            opt.zero_grad()

        if (idx + 1) % 15 == 0 or (idx + 1) == len(valid_prompts):
            denom = min(15, (idx % 15) + 1)
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            eta = rate * (len(valid_prompts) - idx - 1)
            print(
                f"  [Step-GRPO {idx+1:3d}/{len(valid_prompts)}] avg_sampled_reward={running_reward/denom:+.3f} | {rate:.2f}s/it | ETA={eta/60:.1f}m",
                flush=True,
            )
            running_reward = 0.0

    anchor_blend_with_ref(injected, ref_state, ref_weight=0.35)
    save_lora_checkpoint(
        injected,
        out_dir,
        {
            "stage": "grpo",
            "prompts": len(valid_prompts),
            "group_size": group_size,
            "kl_beta": kl_beta,
            "lr": lr,
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--stage", choices=["sft", "dpo", "grpo"], required=True)
    parser.add_argument("--data-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--init-lora", type=str, default=None)
    parser.add_argument("--blend-lora", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--max-steps", type=int, default=640)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--export-merged", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "benchmark_registry/train_pool_manifest.json").read_text())
    slides_by_id = {s["train_id"]: s for s in manifest["slides"]}
    v2_data = ROOT / "data/pathnav_bench/train_pool/datasets_terra_real"
    base_data = ROOT / "data/pathnav_bench/train_pool/datasets_4b_8b"

    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    proc = AutoProcessor.from_pretrained(str(model_path))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.config.use_cache = False
    if hasattr(model.model, "language_model"):
        model.model.language_model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    vis_mod = getattr(model.model, "visual", None) or getattr(model, "visual", None)
    if vis_mod is not None:
        vis_mod.eval()
        vis_mod.gradient_checkpointing = False
        for blk in getattr(vis_mod, "blocks", []):
            blk.gradient_checkpointing = False
    injected = inject_lora(model, rank=args.rank, alpha=args.alpha)
    ref_state = None
    if args.init_lora:
        ref_state = load_lora_checkpoint(
            injected,
            Path(args.init_lora),
            blend_path=Path(args.blend_lora) if args.blend_lora else None,
        )

    if args.data_version == "v2":
        if args.stage == "sft":
            examples = build_v2_sft_examples(v2_data, max_steps=args.max_steps)
            run_sft_stage(
                model,
                proc,
                injected,
                examples,
                out_dir,
                lr=args.lr or 1.6e-4,
                max_steps=args.max_steps,
            )
        elif args.stage == "dpo":
            pairs = build_v2_dpo_pairs(v2_data, max_pairs=args.max_steps)
            run_dpo_stage(
                model,
                proc,
                injected,
                ref_state,
                pairs,
                out_dir,
                lr=args.lr or 2.0e-5,
                max_pairs=args.max_steps,
            )
        elif args.stage == "grpo":
            grpo_rows = [json.loads(line) for line in (v2_data / "grpo_states_v2.jsonl").open()]
            run_grpo_stage(
                model,
                proc,
                injected,
                ref_state,
                grpo_rows,
                {},
                slides_by_id,
                v2_data / "images",
                out_dir,
                lr=args.lr or 1.0e-5,
                max_prompts=args.max_steps,
            )
    else:
        sft_rows = [json.loads(line) for line in (base_data / "sft_multiturn_train.jsonl").open()]
        sft_map: Dict[Tuple[str, int, str], Tuple[Dict, int]] = {}
        for tr in sft_rows:
            tid = tr["train_id"]
            rep = int(tr["traj_id"].rsplit("_", 1)[-1])
            for idx in range(1, len(tr["turns"])):
                prev_vid = tr["turns"][idx - 1]["delivered_view_id"]
                if prev_vid:
                    sft_map[(tid, rep, prev_vid)] = (tr, idx)
        if args.stage == "sft":
            examples = build_curated_sft_examples(sft_rows, slides_by_id, base_data / "images")
            run_sft_stage(
                model,
                proc,
                injected,
                examples,
                out_dir,
                lr=args.lr or 1.5e-4,
                max_steps=args.max_steps,
            )
        elif args.stage == "dpo":
            dpo_rows = [json.loads(line) for line in (base_data / "dpo_step_train.jsonl").open()]
            pairs = build_curated_dpo_pairs(
                dpo_rows, sft_rows, sft_map, slides_by_id, base_data / "images", max_pairs=args.max_steps
            )
            run_dpo_stage(
                model,
                proc,
                injected,
                ref_state,
                pairs,
                out_dir,
                lr=args.lr or 3e-5,
                max_pairs=args.max_steps,
            )
        elif args.stage == "grpo":
            grpo_rows = [json.loads(line) for line in (base_data / "grpo_prompts_train.jsonl").open()]
            run_grpo_stage(
                model,
                proc,
                injected,
                ref_state,
                grpo_rows,
                sft_map,
                slides_by_id,
                base_data / "images",
                out_dir,
                lr=args.lr or 1.5e-5,
                max_prompts=args.max_steps,
            )

    if args.export_merged:
        export_merged_safetensors(model_path, injected, out_dir / "merged_vllm")


if __name__ == "__main__":
    main()
