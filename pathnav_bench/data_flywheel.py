"""Trajectory-to-Training Data Flywheel (`pathnav_bench.data_flywheel`).

Converts evaluated `PathNav-Bench` trajectories + sealed ground truth into three
training-ready datasets / RL interfaces:

1. **Evidence-Grounded SFT (`sft_grounded.jsonl`)**:
   Filters trajectories to retain ONLY verified evidence-grounded trajectories:
   - Positive slides where the agent actually reached qualified detail (<=1.0 um/px),
     made a correct `positive` diagnosis, and submitted a geometrically supported box
     (`joint_support == True`), avoiding contamination from lucky ungrounded guesses.
   - Negative slides where the agent inspected multiple tissue regions at detail
     resolution and correctly concluded `negative`.

2. **Step-Level Fork DPO / Correction Pairs (`dpo_step_corrections.jsonl`)**:
   Specifically targets the "Coarse-to-Fine Funnel Drop" (where an agent's coarse/medium
   view `view_k` ALREADY captured the lesion, `gt_info_px > 0`, but its next action
   zoomed into a non-tumor sub-box or prematurely finished `negative`).
   At that exact step `k`, we pair:
   - `prompt / state`: The exact observation and `view_k` image with its 0..1000 ruler.
   - `rejected`: The agent's actual action that missed the lesion inside `view_k`.
   - `chosen`: The oracle-guided `inspect_box(target_view=view_k, box=[...])` that
     tightly frames the lesion inside `view_k`'s 0..1000 coordinate ruler.

3. **Dense Multi-Turn GRPO Reward Function (`compute_grpo_reward`)**:
   Computes a shaped, anti-hallucination reward for Group Relative Policy Optimization:
     R(tau) = R_format + R_coarse_hit + R_funnel_iou + R_detail_exposure
              + R_grounded_diagnosis - C_ungrounded_hallucination - C_step_cost
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
from audit_target_grid import components
from .evaluator import load_case_truth as load_truth


def compute_grpo_reward(target_geom, trace: List[Dict], score_row: Dict,
                        source_label: str) -> Dict:
    """Compute decomposed GRPO reward for a single trajectory."""
    reads = [r for r in trace if r.get('kind') == 'read']
    invalids = [r for r in trace if r.get('kind') == 'invalid']
    finish = next((r for r in trace if r.get('kind') == 'finish'), None)

    # 1. Format & protocol compliance (+0.2 if clean finish and 0 invalid actions)
    r_format = 0.2 if (finish is not None and len(invalids) == 0) else (-0.2 * len(invalids))

    # 2. Coarse & funnel progress on positive slides
    r_coarse = 0.0
    r_funnel = 0.0
    r_detail = 0.0
    r_grounded = 0.0
    penalty_hallucination = 0.0

    if source_label == 'positive' and not target_geom.is_empty:
        best_lesion_coverage = 0.0
        for r in reads:
            x0, y0, x1, y1 = r['clipped_footprint_level0']
            fov = box(x0, y0, x1, y1)
            inter = target_geom.intersection(fov)
            if inter.area > 0:
                r_coarse = 0.3
                # Measure how tightly the view focuses on the lesion
                focus_ratio = min(1.0, (inter.area / max(1.0, fov.area)) * 10.0)
                res_factor = min(1.0, 1.0 / max(0.25, r['effective_mpp']))
                best_lesion_coverage = max(best_lesion_coverage, focus_ratio * res_factor)
        r_funnel = round(0.5 * best_lesion_coverage, 4)

        if score_row.get('first_qualified_delivered_exposure_decision') is not None:
            r_detail = 1.0

        if score_row.get('joint_geometric_proxy') is True:
            r_grounded = 2.0
        elif score_row.get('decision') == 'positive':
            # Hallucinated / ungrounded positive on a positive slide without detail evidence!
            penalty_hallucination = -1.0
    else:
        # Negative slide
        detail_reads = sum(1 for r in reads if r.get('detail_qualified_for_evidence'))
        if score_row.get('decision') == 'negative':
            # Reward thorough negative exclusion (at least 3 detail reads)
            r_grounded = 1.5 if detail_reads >= 3 else 0.8
        elif score_row.get('decision') == 'positive':
            # False positive on negative slide
            penalty_hallucination = -1.5

    step_cost = -0.02 * score_row.get('decision_count', len(reads))
    total_reward = round(
        r_format + r_coarse + r_funnel + r_detail + r_grounded + penalty_hallucination + step_cost,
        4,
    )
    return {
        'total_reward': total_reward,
        'components': {
            'r_format': round(r_format, 4),
            'r_coarse_hit': round(r_coarse, 4),
            'r_funnel_progress': round(r_funnel, 4),
            'r_detail_exposure': round(r_detail, 4),
            'r_grounded_diagnosis': round(r_grounded, 4),
            'penalty_ungrounded_or_fp': round(penalty_hallucination, 4),
            'step_cost': round(step_cost, 4),
        },
    }


def compute_oracle_subbox_on_view(target_geom, view_row: Dict,
                                  margin_1000: float = 60.0) -> Optional[List[float]]:
    """Given a view that intersects `target_geom`, return a tight 0..1000 box on that view."""
    x0, y0, x1, y1 = view_row['footprint_level0']
    vw, vh = float(x1 - x0), float(y1 - y0)
    if vw <= 0 or vh <= 0:
        return None
    fov = box(x0, y0, x1, y1)
    inter = target_geom.intersection(fov)
    if inter.is_empty:
        return None
    # Pick the largest intersecting lesion component inside this view
    comps = components(inter)
    target_piece = max(comps, key=lambda g: g.area) if comps else inter
    bx0, by0, bx1, by1 = target_piece.bounds
    cx_1000 = ((0.5 * (bx0 + bx1) - x0) / vw) * 1000.0
    cy_1000 = ((0.5 * (by0 + by1) - y0) / vh) * 1000.0
    span_x = max(margin_1000 * 2.0, ((bx1 - bx0) / vw) * 1000.0 + margin_1000)
    span_y = max(margin_1000 * 2.0, ((by1 - by0) / vh) * 1000.0 + margin_1000)
    span_x = min(450.0, span_x)
    span_y = min(450.0, span_y)
    ox1 = max(0.0, min(1000.0 - span_x, cx_1000 - 0.5 * span_x))
    oy1 = max(0.0, min(1000.0 - span_y, cy_1000 - 0.5 * span_y))
    return [round(ox1, 1), round(oy1, 1), round(ox1 + span_x, 1), round(oy1 + span_y, 1)]


def extract_training_flywheel_from_run(run_dir: Path, cases_by_id: Dict[str, Dict],
                                       out_dir: Path) -> Dict:
    """Process all completed jobs in `run_dir` into SFT, Step-DPO, and GRPO datasets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_rows = []
    dpo_rows = []
    grpo_rows = []

    for jdir in sorted(run_dir.glob('job_*')):
        if not (jdir / 'result.json').exists():
            continue
        res = json.loads((jdir / 'result.json').read_text())
        trace = json.loads((jdir / 'env_trace.json').read_text())
        score = json.loads((jdir / 'score.json').read_text())
        cid = next((c for c in cases_by_id if c in jdir.name), None)
        if not cid:
            continue
        case = cases_by_id[cid]
        truth, _ = load_truth(case)
        reads_by_step = {r['decision_index']: r for r in trace if r.get('kind') == 'read'}
        reads_by_vid = {r['view_id']: r for r in trace if r.get('kind') == 'read'}

        # 1. Compute GRPO reward for every trajectory
        reward_info = compute_grpo_reward(truth, trace, score, case['label'])
        grpo_rows.append({
            'job_id': res['job_id'],
            'case_id': cid,
            'bench_split': case.get('bench_split', 'test'),
            'category': case['category'],
            'label': case['label'],
            'model': f"{res.get('provider')}/{res.get('model')}",
            'decision': score['decision'],
            'label_correct': score['label_correct'],
            'any_scale_hit': (score['delivered_exposure_recall_raw'] or 0) > 0 if case['label'] == 'positive' else None,
            'first_qualified_step': score['first_qualified_delivered_exposure_decision'],
            'joint_support': score['joint_geometric_proxy'],
            **reward_info,
        })

        # 2. Rejection-Sampled SFT: keep only verified evidence-grounded trajectories
        is_verified_pos = (case['label'] == 'positive' and score.get('joint_geometric_proxy') is True)
        is_verified_neg = (case['label'] == 'negative' and score.get('label_correct') is True
                           and sum(1 for r in reads_by_vid.values() if r.get('detail_qualified_for_evidence')) >= 3)
        if is_verified_pos or is_verified_neg:
            sft_rows.append({
                'job_id': res['job_id'],
                'case_id': cid,
                'category': case['category'],
                'label': case['label'],
                'source_model': f"{res.get('provider')}/{res.get('model')}",
                'total_steps': len(res['decisions']),
                'grpo_reward': reward_info['total_reward'],
                'trajectory_actions': [d['action'] for d in res['decisions']],
            })

        # 3. Step-Level DPO Fork Extraction on positive slides where coarse view hit lesion
        #    but agent failed to get joint_support
        if case['label'] == 'positive' and not score.get('joint_geometric_proxy'):
            decisions = res['decisions']
            for idx, dec in enumerate(decisions[:-1]):
                r = reads_by_step.get(dec['decision_index'])
                if not r:
                    continue
                x0, y0, x1, y1 = r['footprint_level0']
                inter = truth.intersection(box(x0, y0, x1, y1))
                if inter.area <= 0:
                    continue
                next_dec = decisions[idx + 1]
                next_act = next_dec.get('proposed_action') or {}
                next_r = reads_by_step.get(next_dec['decision_index'])
                next_hit = False
                if next_r:
                    nx0, ny0, nx1, ny1 = next_r['footprint_level0']
                    next_hit = truth.intersection(box(nx0, ny0, nx1, ny1)).area > 0

                # Case A: Coarse/medium view captured lesion, but next step missed the lesion!
                if not r['detail_qualified_for_evidence'] and not next_hit:
                    oracle_box = compute_oracle_subbox_on_view(truth, r)
                    if oracle_box:
                        dpo_rows.append({
                            'fork_type': 'coarse_to_fine_funnel_correction',
                            'job_id': res['job_id'],
                            'case_id': cid,
                            'category': case['category'],
                            'fork_at_step': next_dec['decision_index'],
                            'parent_view_id': r['view_id'],
                            'parent_view_mpp': r['effective_mpp'],
                            'rejected_action': next_act,
                            'chosen_action': {
                                'action': 'inspect_box',
                                'target_view': r['view_id'],
                                'box': oracle_box,
                                'reason': f"Zoom into hyperchromatic atypical cluster inside {r['view_id']} to verify metastasis at cellular resolution.",
                            },
                        })
                        break
                # Case B: Detail-qualified view ALREADY captured lesion (>=16 info px), but agent missed it!
                eff_px = inter.area / (r['effective_downsample'] ** 2)
                if r['detail_qualified_for_evidence'] and eff_px >= 16.0:
                    oracle_ev_box = compute_oracle_subbox_on_view(truth, r, margin_1000=40.0)
                    if oracle_ev_box:
                        dpo_rows.append({
                            'fork_type': 'detail_recognition_correction',
                            'job_id': res['job_id'],
                            'case_id': cid,
                            'category': case['category'],
                            'fork_at_step': next_dec['decision_index'],
                            'parent_view_id': r['view_id'],
                            'parent_view_mpp': r['effective_mpp'],
                            'lesion_info_px_in_view': round(eff_px, 1),
                            'rejected_action': next_act,
                            'chosen_action': {
                                'action': 'finish',
                                'decision': 'positive',
                                'confidence': 0.92,
                                'evidence': [{'view_id': r['view_id'], 'box': oracle_ev_box}],
                                'reason': f"Identified cohesive metastatic carcinoma nests in detail-qualified {r['view_id']}.",
                            },
                        })
                        break

    (out_dir / 'sft_grounded.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in sft_rows) + ('\n' if sft_rows else ''),
        encoding='utf-8',
    )
    (out_dir / 'dpo_step_corrections.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in dpo_rows) + ('\n' if dpo_rows else ''),
        encoding='utf-8',
    )
    (out_dir / 'grpo_rewards.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in grpo_rows) + ('\n' if grpo_rows else ''),
        encoding='utf-8',
    )
    summary = {
        'total_evaluated_trajectories': len(grpo_rows),
        'sft_verified_grounded_trajectories': len(sft_rows),
        'dpo_step_correction_pairs': len(dpo_rows),
        'dpo_by_fork_type': {
            k: sum(1 for r in dpo_rows if r['fork_type'] == k)
            for k in ('coarse_to_fine_funnel_correction', 'detail_recognition_correction')
        },
        'sample_dpo_pairs': dpo_rows[:3],
        'sample_grpo_rewards': grpo_rows[:4],
    }
    (out_dir / 'flywheel_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
