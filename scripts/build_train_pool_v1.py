"""Download the 100-slide disjoint Training Pool (`Train-Pool v1`) into `data/pathnav_bench/train_pool/`
and synthesize the 4B-8B VLM training datasets (`SFT`, `Step-DPO`, `Step-GRPO`) with 0 overlap against
the frozen 66-slide PathNav-Bench evaluation benchmark (`manifest_v3_private.json`).
"""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, List, Tuple
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import openslide
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathnav_bench.env import BenchConfig, WSIMicroscopeEnv, TASK_PROMPT
from pathnav_bench.evaluator import load_case_truth


def compute_oracle_box_1000(target_geom, footprint_level0, target_mpp: float, current_mpp: float):
    x0, y0, x1, y1 = footprint_level0
    vw, vh = float(x1 - x0), float(y1 - y0)
    if vw <= 0 or vh <= 0:
        return None
    inter = target_geom.intersection(box(x0, y0, x1, y1))
    if inter.is_empty:
        bx0, by0, bx1, by1 = target_geom.bounds
    else:
        bx0, by0, bx1, by1 = inter.bounds
    cx_1000 = max(0.0, min(1000.0, ((0.5 * (bx0 + bx1) - x0) / vw) * 1000.0))
    cy_1000 = max(0.0, min(1000.0, ((0.5 * (by0 + by1) - y0) / vh) * 1000.0))
    scale = max(0.15, min(0.55, float(target_mpp) / max(0.25, float(current_mpp))))
    span = scale * 1000.0
    ox1 = max(0.0, min(1000.0 - span, cx_1000 - 0.5 * span))
    oy1 = max(0.0, min(1000.0 - span, cy_1000 - 0.5 * span))
    return [round(ox1, 1), round(oy1, 1), round(ox1 + span, 1), round(oy1 + span, 1)]

REGISTRY_DIR = ROOT / "benchmark_registry"
SOURCES_DIR = REGISTRY_DIR / "sources"
TRAIN_POOL_DIR = ROOT / "data/pathnav_bench/train_pool"
WSI_POOL_DIR = TRAIN_POOL_DIR / "wsi"
ANN_POOL_DIR = TRAIN_POOL_DIR / "annotations"
SYNTH_DIR = TRAIN_POOL_DIR / "datasets_4b_8b"
SYNTH_IMG_DIR = SYNTH_DIR / "images"


def download_file_resumable(url: str, dest: Path, expected_bytes: int = 0) -> Dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_bytes <= 0 or dest.stat().st_size == expected_bytes):
        return {"path": str(dest), "bytes": dest.stat().st_size, "cached": True}

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 7):
        start_byte = part.stat().st_size if part.exists() else 0
        if expected_bytes > 0 and start_byte > expected_bytes:
            part.unlink()
            start_byte = 0
        headers = {"User-Agent": "PathNavBench-TrainPool/3.0"}
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            mode = "ab" if start_byte > 0 else "wb"
            with urllib.request.urlopen(req, timeout=180) as resp, part.open(mode) as out:
                while chunk := resp.read(8 * 1024 * 1024):
                    out.write(chunk)
        except Exception as e:
            if attempt == 6:
                raise RuntimeError(f"Failed downloading {url}: {e}") from e
            time.sleep(2.5 * attempt)
            continue
        if expected_bytes <= 0 or (part.exists() and part.stat().st_size == expected_bytes):
            break

    if expected_bytes > 0 and part.stat().st_size != expected_bytes:
        raise ValueError(f"Size mismatch for {dest.name}: got {part.stat().st_size}, expected {expected_bytes}")
    os.replace(part, dest)
    return {"path": str(dest), "bytes": dest.stat().st_size, "cached": False}


def normalize_tiger_xml(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = False
    for ann in root.findall(".//Annotation"):
        grp = (ann.attrib.get("PartOfGroup") or "").strip().lower()
        if grp in ("tumor-bulk", "invasive tumor", "in-situ tumor", "tumor", "roi"):
            if ann.attrib.get("PartOfGroup") != "Tumor":
                ann.attrib["PartOfGroup"] = "Tumor"
                changed = True
        for c in ann.findall(".//Coordinate"):
            for k in ("X", "Y"):
                if k in c.attrib and "," in c.attrib[k]:
                    c.attrib[k] = c.attrib[k].replace(",", ".")
                    changed = True
    if changed:
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def build_disjoint_100_candidates() -> Tuple[List[Dict], set]:
    v3_cases = json.loads((REGISTRY_DIR / "manifest_v3_private.json").read_text())["cases"]
    v2_cases = json.loads((REGISTRY_DIR / "manifest_private.json").read_text())["cases"]
    eval_slide_ids = (
        {c["source_slide_id"] for c in v3_cases}
        | {Path(c["wsi_path"]).stem for c in v3_cases}
        | {c["source_slide_id"] for c in v2_cases}
        | {Path(c["wsi_path"]).stem for c in v2_cases}
    )

    cobra = json.loads((SOURCES_DIR / "cobra_skin_candidates.json").read_text())
    tiger = json.loads((SOURCES_DIR / "tiger_breast_candidates.json").read_text())
    c16 = json.loads((SOURCES_DIR / "camelyon16_inventory.json").read_text())["slides"]

    cobra_pos = sorted(
        [x for x in cobra["focal_positive_candidates"] if x["source_slide_id"] not in eval_slide_ids],
        key=lambda x: x["declared_bytes"],
    )[:35]
    cobra_neg = sorted(
        [x for x in cobra["negative_candidates"] if x["source_slide_id"] not in eval_slide_ids],
        key=lambda x: x["declared_bytes"],
    )[:10]

    tiger_pos = sorted(
        [x for x in tiger["focal_positive_candidates"] if x["source_slide_id"] not in eval_slide_ids],
        key=lambda x: x["declared_bytes"],
    )[:19]
    bracs_neg = sorted(
        [x for x in tiger["negative_candidates"] if x["source_slide_id"] not in eval_slide_ids],
        key=lambda x: x["declared_bytes"],
    )[:5]

    cam16_pos = sorted(
        [
            (k, v)
            for k, v in c16.items()
            if k not in eval_slide_ids
            and v["category"] in ("macro", "micro")
            and (ROOT / "data/pathnav_bench/annotations/camelyon16" / f"{k}.xml").exists()
        ],
        key=lambda kv: kv[1]["declared_bytes"],
    )[:21]
    cam16_neg = sorted(
        [(k, v) for k, v in c16.items() if k not in eval_slide_ids and v["category"] == "negative"],
        key=lambda kv: kv[1]["declared_bytes"],
    )[:10]

    items: List[Dict] = []
    # 1. COBRA 45 (35 pos + 10 neg)
    for idx, x in enumerate(cobra_pos + cobra_neg, 1):
        sid = x["source_slide_id"]
        is_pos = x["label"] == "positive"
        items.append({
            "train_id": f"train_cobra_{idx:03d}",
            "source_dataset": "COBRA",
            "organ": "Skin",
            "source_slide_id": sid,
            "category": "cobra_train_pos" if is_pos else "cobra_train_neg",
            "label": x["label"],
            "declared_bytes": int(x["declared_bytes"]),
            "image_url": x["image_url"],
            "wsi_path": str(WSI_POOL_DIR / "cobra" / f"{sid}.tif"),
            "geojson_url": x.get("tumor_geojson_url") if is_pos else None,
            "geojson_path": str(ANN_POOL_DIR / "cobra" / f"{sid}.geojson") if is_pos else None,
            "xml_path": None,
            "track": "general_navigation",
            "task_mode": "focal_search",
        })

    # 2. TIGER + BRACS 24 (19 pos + 5 neg)
    for idx, x in enumerate(tiger_pos, 1):
        sid = x["source_slide_id"]
        items.append({
            "train_id": f"train_tiger_{idx:03d}",
            "source_dataset": "TIGER",
            "organ": "Breast",
            "source_slide_id": sid,
            "category": "tiger_train_pos",
            "label": "positive",
            "declared_bytes": int(x["declared_bytes"]),
            "image_url": x["image_url"],
            "wsi_path": str(WSI_POOL_DIR / "tiger" / f"{sid}.tif"),
            "xml_url": x["xml_url"],
            "xml_path": str(ANN_POOL_DIR / "tiger" / f"{sid}.xml"),
            "geojson_path": None,
            "track": "general_navigation",
            "task_mode": "focal_search",
        })
    for idx, x in enumerate(bracs_neg, 20):
        sid = x["source_slide_id"]
        items.append({
            "train_id": f"train_tiger_{idx:03d}",
            "source_dataset": "BRACS",
            "organ": "Breast",
            "source_slide_id": sid,
            "category": "tiger_train_neg",
            "label": "negative",
            "declared_bytes": int(x["declared_bytes"]),
            "image_url": x["image_url"],
            "wsi_path": str(WSI_POOL_DIR / "bracs" / f"{sid}.svs"),
            "xml_url": None,
            "xml_path": None,
            "geojson_path": None,
            "track": "general_navigation",
            "task_mode": "focal_search",
        })

    # 3. CAMELYON16 31 (21 pos + 10 neg)
    for idx, (sid, meta) in enumerate(cam16_pos + cam16_neg, 1):
        is_pos = meta["category"] in ("macro", "micro")
        src_xml = ROOT / "data/pathnav_bench/annotations/camelyon16" / f"{sid}.xml"
        dst_xml = ANN_POOL_DIR / "camelyon" / f"{sid}.xml"
        items.append({
            "train_id": f"train_cam16_{idx:03d}",
            "source_dataset": "CAMELYON16",
            "organ": "Lymph node (Breast)",
            "source_slide_id": sid,
            "category": f"cam16_train_{meta['category']}",
            "label": "positive" if is_pos else "negative",
            "declared_bytes": int(meta["declared_bytes"]),
            "image_url": meta["image_url"],
            "wsi_path": str(WSI_POOL_DIR / "camelyon" / f"{sid}.tif"),
            "src_local_xml": str(src_xml) if is_pos else None,
            "xml_path": str(dst_xml) if is_pos else None,
            "geojson_path": None,
            "track": "general_navigation",
            "task_mode": "focal_search",
        })

    train_ids = {it["source_slide_id"] for it in items}
    overlap = train_ids & eval_slide_ids
    assert len(overlap) == 0, f"CRITICAL: Overlap detected with evaluation benchmark: {overlap}"
    assert len(items) == 100, f"Expected 100 train pool slides, got {len(items)}"
    return items, eval_slide_ids


def download_and_validate_one(item: Dict) -> Dict:
    t0 = time.time()
    # 1. Annotation
    if item.get("geojson_url") and item.get("geojson_path"):
        download_file_resumable(item["geojson_url"], Path(item["geojson_path"]), expected_bytes=0)
    elif item.get("xml_url") and item.get("xml_path"):
        download_file_resumable(item["xml_url"], Path(item["xml_path"]), expected_bytes=0)
        normalize_tiger_xml(Path(item["xml_path"]))
    elif item.get("src_local_xml") and item.get("xml_path"):
        dst = Path(item["xml_path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(item["src_local_xml"], dst)

    # 2. WSI
    wsi_res = download_file_resumable(item["image_url"], Path(item["wsi_path"]), expected_bytes=item["declared_bytes"])

    # 3. Validate OpenSlide + Truth
    sl = openslide.OpenSlide(item["wsi_path"])
    w0, h0 = sl.dimensions
    sl.close()
    case_proxy = {
        "episode_id": item["train_id"],
        "label": item["label"],
        "wsi_path": item["wsi_path"],
        "xml_path": item.get("xml_path"),
        "geojson_path": item.get("geojson_path"),
        "track": "general_navigation",
        "task_mode": "focal_search",
    }
    truth, meta = load_case_truth(case_proxy)
    if item["label"] == "positive" and truth.is_empty:
        raise ValueError(f"Positive train slide {item['train_id']} has empty truth geometry")
    return {
        **item,
        "dimensions_level0": [w0, h0],
        "lesion_area_px2": round(float(truth.area), 1) if not truth.is_empty else 0.0,
        "cached": wsi_res["cached"],
        "elapsed_sec": round(time.time() - t0, 2),
        "validated": True,
    }


def _save_view_image(im, view_tag: str) -> str:
    SYNTH_IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTH_IMG_DIR / f"{view_tag}.jpg"
    if not out_path.exists():
        im.convert("RGB").save(out_path, format="JPEG", quality=88)
    return str(out_path.relative_to(ROOT))


def _find_tissue_boxes_1000(env: WSIMicroscopeEnv, truth, w0: int, h0: int, n_boxes: int = 8) -> List[List[float]]:
    """Find tissue-containing boxes in 0..1000 overview coordinates that have 0 intersection with truth."""
    thumb = np.array(env._overview_raw.convert("RGB"))
    th_h, th_w = thumb.shape[:2]
    gray = thumb.mean(axis=2)
    tissue = (gray < 218) & (gray > 25)
    ys, xs = np.where(tissue)
    if len(xs) == 0:
        return [[300.0, 300.0, 500.0, 500.0]]
    rng = random.Random((w0 * 31 + h0) & 0xFFFFFFFF)
    idxs = list(range(len(xs)))
    rng.shuffle(idxs)
    benign_boxes = []
    for i in idxs[:400]:
        cx = (xs[i] / th_w) * 1000.0
        cy = (ys[i] / th_h) * 1000.0
        span = rng.uniform(110.0, 200.0)
        x1 = max(0.0, min(1000.0 - span, cx - 0.5 * span))
        y1 = max(0.0, min(1000.0 - span, cy - 0.5 * span))
        x2, y2 = x1 + span, y1 + span
        b_lvl0 = box(x1 / 1000.0 * w0, y1 / 1000.0 * h0, x2 / 1000.0 * w0, y2 / 1000.0 * h0)
        if truth.is_empty or truth.intersection(b_lvl0).area == 0.0:
            benign_boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
            if len(benign_boxes) >= n_boxes:
                break
    if not benign_boxes:
        benign_boxes.append([250.0, 250.0, 420.0, 420.0])
    return benign_boxes


def synthesize_4b_8b_datasets_for_slide(
    item: Dict, reps: Tuple[int, ...] = tuple(range(10))
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Synthesize SFT trajectories, Step-DPO pairs, and Step-GRPO prompts from 1 validated train slide."""
    tid = item["train_id"]
    organ = item["organ"]
    label = item["label"]
    w0, h0 = item["dimensions_level0"]
    case_entry = {
        "episode_id": tid,
        "source_dataset": item["source_dataset"],
        "organ": organ,
        "category": item["category"],
        "label": label,
        "wsi_path": item["wsi_path"],
        "xml_path": item.get("xml_path"),
        "geojson_path": item.get("geojson_path"),
        "track": "general_navigation",
        "task_mode": "focal_search",
    }
    truth, _ = load_case_truth(case_entry)
    rng = random.Random(tid)

    sft_trajs: List[Dict] = []
    dpo_pairs: List[Dict] = []
    grpo_prompts: List[Dict] = []

    if label == "positive" and not truth.is_empty:
        polys = [truth] if truth.geom_type == "Polygon" else [g for g in truth.geoms if g.area > 0]
        polys.sort(key=lambda g: g.area, reverse=True)

        # Generate 10 diverse positive trajectories per positive slide (75 slides * 10 = 750 positive SFT trajectories)
        for rep in reps:
            target_poly = polys[rep % len(polys)]
            env = WSIMicroscopeEnv(case_entry, BenchConfig(budget=12, max_crop_edge=640))
            s0 = env.reset()
            ov_img_rel = _save_view_image(s0["image"], f"{tid}_overview")
            benign_ov_boxes = _find_tissue_boxes_1000(env, truth, w0, h0, n_boxes=6)

            turns = []
            # Mode A (rep 0..6): 4-step Coarse -> Medium -> High-Power -> Finish
            # Mode B (rep 7..9): 7-step 3-Zoom Benign Compartment -> Back out to True Coarse -> Medium -> High-Power -> Finish
            if rep >= 7 and benign_ov_boxes:
                b_box = benign_ov_boxes[rep % len(benign_ov_boxes)]
                act_d1 = {
                    "action": "inspect_box",
                    "target_view": "overview",
                    "box": b_box,
                    "reason": f"Inspect basophilic tissue region in {organ} overview to screen for atypical epithelial nests.",
                }
                sd1 = env.step(act_d1)
                vd1 = sd1["view"]
                vd1_img = _save_view_image(sd1["image"], f"{tid}_r{rep}_{vd1['view_id']}")
                turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel], "action": act_d1, "delivered_view_id": vd1["view_id"], "effective_mpp": vd1["effective_mpp"]})

                span_dm = max(140.0, min(380.0, (2.6 / max(1.0, vd1["effective_mpp"])) * 1000.0))
                cx_dm, cy_dm = rng.uniform(380.0, 620.0), rng.uniform(380.0, 620.0)
                x1_dm = round(max(60.0, min(940.0 - span_dm, cx_dm - 0.5 * span_dm)), 1)
                y1_dm = round(max(60.0, min(940.0 - span_dm, cy_dm - 0.5 * span_dm)), 1)
                sub_box_m = [x1_dm, y1_dm, round(x1_dm + span_dm, 1), round(y1_dm + span_dm, 1)]
                act_d2 = {
                    "action": "inspect_box",
                    "target_view": vd1["view_id"],
                    "box": sub_box_m,
                    "reason": f"Intermediate zoom inside {vd1['view_id']} to evaluate tissue architecture.",
                }
                sd2 = env.step(act_d2)
                vd2 = sd2["view"]
                vd2_img = _save_view_image(sd2["image"], f"{tid}_r{rep}_{vd2['view_id']}")
                turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, vd1_img], "action": act_d2, "delivered_view_id": vd2["view_id"], "effective_mpp": vd2["effective_mpp"]})

                span_dh = max(120.0, min(320.0, (0.68 / max(0.8, vd2["effective_mpp"])) * 1000.0))
                cx_dh, cy_dh = rng.uniform(380.0, 620.0), rng.uniform(380.0, 620.0)
                x1_dh = round(max(80.0, min(920.0 - span_dh, cx_dh - 0.5 * span_dh)), 1)
                y1_dh = round(max(80.0, min(920.0 - span_dh, cy_dh - 0.5 * span_dh)), 1)
                sub_box_h = [x1_dh, y1_dh, round(x1_dh + span_dh, 1), round(y1_dh + span_dh, 1)]
                act_d3 = {
                    "action": "inspect_box",
                    "target_view": vd2["view_id"],
                    "box": sub_box_h,
                    "reason": f"High-power zoom (<=1.0 um/px) inside {vd2['view_id']} to verify cellular morphology; if benign, return to overview.",
                }
                sd3 = env.step(act_d3)
                vd3 = sd3["view"]
                vd3_img = _save_view_image(sd3["image"], f"{tid}_r{rep}_{vd3['view_id']}")
                turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, vd2_img], "action": act_d3, "delivered_view_id": vd3["view_id"], "effective_mpp": vd3["effective_mpp"]})

            # Coarse box on overview covering target_poly with random jitter
            minx, miny, maxx, maxy = target_poly.bounds
            cx = 0.5 * (minx + maxx) + rng.uniform(-0.03, 0.03) * w0
            cy = 0.5 * (miny + maxy) + rng.uniform(-0.03, 0.03) * h0
            span_c = rng.uniform(0.16, 0.28)
            x1_c = max(0.0, min(1000.0 - span_c * 1000.0, (cx / w0 - 0.5 * span_c) * 1000.0))
            y1_c = max(0.0, min(1000.0 - span_c * 1000.0, (cy / h0 - 0.5 * span_c) * 1000.0))
            coarse_box = [round(x1_c, 1), round(y1_c, 1), round(x1_c + span_c * 1000.0, 1), round(y1_c + span_c * 1000.0, 1)]
            act1 = {
                "action": "inspect_box",
                "target_view": "overview",
                "box": coarse_box,
                "reason": f"Coarse scan of suspicious dense region in {organ} overview to localize focal lesion.",
            }
            s1 = env.step(act1)
            v1 = s1["view"]
            v1_img = _save_view_image(s1["image"], f"{tid}_r{rep}_{v1['view_id']}")
            turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel], "action": act1, "delivered_view_id": v1["view_id"], "effective_mpp": v1["effective_mpp"]})

            # Medium zoom inside v1
            med_box = compute_oracle_box_1000(target_poly, v1["footprint_level0"], target_mpp=rng.uniform(2.2, 3.8), current_mpp=v1["effective_mpp"])
            if not med_box:
                med_box = [350.0, 350.0, 650.0, 650.0]
            act2 = {
                "action": "inspect_box",
                "target_view": v1["view_id"],
                "box": med_box,
                "reason": f"Intermediate zoom inside {v1['view_id']} to refine boundaries of atypical epithelial cluster.",
            }
            s2 = env.step(act2)
            v2 = s2["view"]
            v2_img = _save_view_image(s2["image"], f"{tid}_r{rep}_{v2['view_id']}")
            turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, v1_img], "action": act2, "delivered_view_id": v2["view_id"], "effective_mpp": v2["effective_mpp"]})

            # Also harvest 2 Step-DPO pairs from v1 (coarse) and v2 (medium) on each repetition!
            if rep < 9:
                for p_view, p_img, chosen_b in [(v1, v1_img, med_box)]:
                    # Create a rejected box in the same parent view away from chosen_b
                    cx_c = 0.5 * (chosen_b[0] + chosen_b[2])
                    cy_c = 0.5 * (chosen_b[1] + chosen_b[3])
                    rx1 = 680.0 if cx_c < 500.0 else 80.0
                    ry1 = 680.0 if cy_c < 500.0 else 80.0
                    rej_b = [round(rx1, 1), round(ry1, 1), round(rx1 + 220.0, 1), round(ry1 + 220.0, 1)]
                    dpo_pairs.append({
                        "train_id": tid,
                        "organ": organ,
                        "fork_type": "coarse_to_fine_funnel_correction",
                        "parent_view_id": p_view["view_id"],
                        "parent_view_mpp": p_view["effective_mpp"],
                        "input_images": [ov_img_rel, p_img],
                        "chosen_action": {
                            "action": "inspect_box",
                            "target_view": p_view["view_id"],
                            "box": chosen_b,
                            "reason": f"Zoom directly into the atypical hyperchromatic tumor nests in {p_view['view_id']} to reach cellular resolution.",
                        },
                        "rejected_action": {
                            "action": "inspect_box",
                            "target_view": p_view["view_id"],
                            "box": rej_b,
                            "reason": f"Zoom into adjacent benign stromal/lymphoid background in {p_view['view_id']}, missing the focal lesion.",
                        },
                    })

            # High-power zoom <= 0.85 um/px inside v2
            hp_box = compute_oracle_box_1000(target_poly, v2["footprint_level0"], target_mpp=rng.uniform(0.55, 0.85), current_mpp=v2["effective_mpp"])
            if not hp_box:
                hp_box = [380.0, 380.0, 620.0, 620.0]
            act3 = {
                "action": "inspect_box",
                "target_view": v2["view_id"],
                "box": hp_box,
                "reason": f"High-power zoom (<=1.0 um/px) inside {v2['view_id']} to confirm nuclear pleomorphism and invasive growth.",
            }
            s3 = env.step(act3)
            v3 = s3["view"]
            v3_img = _save_view_image(s3["image"], f"{tid}_r{rep}_{v3['view_id']}")
            turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, v2_img], "action": act3, "delivered_view_id": v3["view_id"], "effective_mpp": v3["effective_mpp"]})

            # Second Step-DPO pair on v2 -> v3
            if rep < 9:
                cx_h = 0.5 * (hp_box[0] + hp_box[2])
                cy_h = 0.5 * (hp_box[1] + hp_box[3])
                rx2 = 700.0 if cx_h < 500.0 else 60.0
                ry2 = 700.0 if cy_h < 500.0 else 60.0
                rej_hp = [round(rx2, 1), round(ry2, 1), round(rx2 + 200.0, 1), round(ry2 + 200.0, 1)]
                dpo_pairs.append({
                    "train_id": tid,
                    "organ": organ,
                    "fork_type": "medium_to_highpower_funnel_correction",
                    "parent_view_id": v2["view_id"],
                    "parent_view_mpp": v2["effective_mpp"],
                    "input_images": [ov_img_rel, v2_img],
                    "chosen_action": act3,
                    "rejected_action": {
                        "action": "inspect_box",
                        "target_view": v2["view_id"],
                        "box": rej_hp,
                        "reason": f"Zoom into peripheral non-neoplastic stroma in {v2['view_id']}, losing the tumor cells at high power.",
                    },
                })

            # Tight evidence box inside v3
            ev_box = compute_oracle_box_1000(target_poly, v3["footprint_level0"], target_mpp=v3["effective_mpp"] * 0.5, current_mpp=v3["effective_mpp"])
            if not ev_box:
                ev_box = [200.0, 200.0, 800.0, 800.0]
            act_fin = {
                "action": "finish",
                "decision": "positive",
                "confidence": 0.95,
                "evidence": [{"view_id": v3["view_id"], "box": ev_box}],
                "summary": f"Confirmed malignant lesion in {organ} at high power ({v3['effective_mpp']:.2f} um/px) in {v3['view_id']}.",
            }
            env.step(act_fin)
            env.close()
            turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, v3_img], "action": act_fin, "delivered_view_id": None, "effective_mpp": v3["effective_mpp"]})

            sft_trajs.append({
                "traj_id": f"{tid}_sft_pos_{rep:02d}",
                "train_id": tid,
                "organ": organ,
                "label": "positive",
                "num_steps": len(turns),
                "turns": turns,
            })

            if rep < 6:
                grpo_prompts.append({
                    "prompt_id": f"{tid}_grpo_step_{rep:02d}",
                    "train_id": tid,
                    "organ": organ,
                    "label": "positive",
                    "parent_view_id": v2["view_id"],
                    "parent_footprint_level0": v2["footprint_level0"],
                    "parent_effective_mpp": v2["effective_mpp"],
                    "input_images": [ov_img_rel, v2_img],
                    "xml_path": item.get("xml_path"),
                    "geojson_path": item.get("geojson_path"),
                })

    else:
        # Negative slide: generate 10 thorough negative exclusion SFT trajectories (25 slides * 10 = 250 negative SFT trajectories)
        # + 6 anti-false-positive DPO pairs per negative slide (25 * 6 = 150 negative DPO pairs)
        for rep in reps:
            env = WSIMicroscopeEnv(case_entry, BenchConfig(budget=12, max_crop_edge=640))
            s0 = env.reset()
            ov_img_rel = _save_view_image(s0["image"], f"{tid}_overview")
            t_boxes = _find_tissue_boxes_1000(env, truth, w0, h0, n_boxes=6)
            turns = []
            last_hp_vid = "view_0003"
            last_hp_img = ov_img_rel
            last_hp_mpp = 0.72
            n_branches = 1 if rep < 5 else 2
            for b_idx in range(n_branches):
                b = t_boxes[(rep + b_idx) % len(t_boxes)]
                act_c = {
                    "action": "inspect_box",
                    "target_view": "overview",
                    "box": b,
                    "reason": f"Inspect tissue compartment #{b_idx + 1} in {organ} overview to screen for focal malignancy.",
                }
                sc = env.step(act_c)
                vc = sc["view"]
                vc_img = _save_view_image(sc["image"], f"{tid}_r{rep}_{vc['view_id']}")
                turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel], "action": act_c, "delivered_view_id": vc["view_id"], "effective_mpp": vc["effective_mpp"]})

                span_m = max(140.0, min(380.0, (2.6 / max(1.0, vc["effective_mpp"])) * 1000.0))
                cx_m, cy_m = rng.uniform(380.0, 620.0), rng.uniform(380.0, 620.0)
                x1_m = round(max(60.0, min(940.0 - span_m, cx_m - 0.5 * span_m)), 1)
                y1_m = round(max(60.0, min(940.0 - span_m, cy_m - 0.5 * span_m)), 1)
                box_m = [x1_m, y1_m, round(x1_m + span_m, 1), round(y1_m + span_m, 1)]
                act_m = {
                    "action": "inspect_box",
                    "target_view": vc["view_id"],
                    "box": box_m,
                    "reason": f"Intermediate zoom inside {vc['view_id']} to evaluate architectural organization.",
                }
                sm = env.step(act_m)
                vm = sm["view"]
                vm_img = _save_view_image(sm["image"], f"{tid}_r{rep}_{vm['view_id']}")
                turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, vc_img], "action": act_m, "delivered_view_id": vm["view_id"], "effective_mpp": vm["effective_mpp"]})

                span_h = max(120.0, min(320.0, (0.68 / max(0.8, vm["effective_mpp"])) * 1000.0))
                cx_h, cy_h = rng.uniform(380.0, 620.0), rng.uniform(380.0, 620.0)
                x1_h = round(max(80.0, min(920.0 - span_h, cx_h - 0.5 * span_h)), 1)
                y1_h = round(max(80.0, min(920.0 - span_h, cy_h - 0.5 * span_h)), 1)
                box_h = [x1_h, y1_h, round(x1_h + span_h, 1), round(y1_h + span_h, 1)]
                act_h = {
                    "action": "inspect_box",
                    "target_view": vm["view_id"],
                    "box": box_h,
                    "reason": f"Zoom to high power (<=1.0 um/px) inside {vm['view_id']} to verify benign cellular morphology and absence of atypia.",
                }
                sh = env.step(act_h)
                vh = sh["view"]
                vh_img = _save_view_image(sh["image"], f"{tid}_r{rep}_{vh['view_id']}")
                turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, vm_img], "action": act_h, "delivered_view_id": vh["view_id"], "effective_mpp": vh["effective_mpp"]})
                last_hp_vid = vh["view_id"]
                last_hp_img = vh_img
                last_hp_mpp = vh["effective_mpp"]

            act_fin = {
                "action": "finish",
                "decision": "negative",
                "confidence": 0.92,
                "evidence": [{"view_id": last_hp_vid, "box": [150.0, 150.0, 850.0, 850.0]}],
                "summary": f"Thoroughly inspected tissue compartment(s) at <=1.0 um/px ({last_hp_mpp:.2f} um/px) in {organ}; only normal architecture observed, negative for malignancy.",
            }
            env.step(act_fin)
            env.close()
            turns.append({"step": len(turns) + 1, "input_images": [ov_img_rel, last_hp_img], "action": act_fin, "delivered_view_id": None, "effective_mpp": last_hp_mpp})

            sft_trajs.append({
                "traj_id": f"{tid}_sft_neg_{rep:02d}",
                "train_id": tid,
                "organ": organ,
                "label": "negative",
                "num_steps": len(turns),
                "turns": turns,
            })

            if rep < 6:
                dpo_pairs.append({
                    "train_id": tid,
                    "organ": organ,
                    "fork_type": "negative_slide_false_positive_suppression",
                    "parent_view_id": last_hp_vid,
                    "parent_view_mpp": last_hp_mpp,
                    "input_images": [ov_img_rel, last_hp_img],
                    "chosen_action": act_fin,
                    "rejected_action": {
                        "action": "finish",
                        "decision": "positive",
                        "confidence": 0.85,
                        "evidence": [{"view_id": last_hp_vid, "box": [200.0, 200.0, 800.0, 800.0]}],
                        "summary": f"Misinterpreting normal {organ} epithelium/lobules as invasive carcinoma.",
                    },
                })
                grpo_prompts.append({
                    "prompt_id": f"{tid}_grpo_neg_{rep:02d}",
                    "train_id": tid,
                    "organ": organ,
                    "label": "negative",
                    "parent_view_id": last_hp_vid,
                    "parent_effective_mpp": last_hp_mpp,
                    "input_images": [ov_img_rel, last_hp_img],
                    "xml_path": None,
                    "geojson_path": None,
                })

    return sft_trajs, dpo_pairs, grpo_prompts


def main():
    items, _ = build_disjoint_100_candidates()
    total_gb = sum(x["declared_bytes"] for x in items) / 1e9
    print(f"[Train-Pool v1] Prepared 100 disjoint slides (0 overlap with Focal60/TCGA6), total = {total_gb:.2f} GB", flush=True)

    validated_items: List[Dict] = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(download_and_validate_one, it): it for it in items}
        for idx, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            validated_items.append(rec)
            if idx % 5 == 0 or idx == len(items):
                print(
                    f"  [Download & Validate {idx:3d}/100] latest={rec['train_id']} ({rec['source_dataset']}, {rec['declared_bytes']/1e6:.1f} MB, cached={rec['cached']})",
                    flush=True,
                )

    validated_items.sort(key=lambda x: x["train_id"])
    manifest_out = {
        "pool_version": "train_pool_v1_100slides",
        "created_at": "2026-11-14",
        "zero_overlap_with_manifest_v3_verified": True,
        "total_slides": len(validated_items),
        "positive_slides": sum(1 for x in validated_items if x["label"] == "positive"),
        "negative_slides": sum(1 for x in validated_items if x["label"] == "negative"),
        "by_source": {
            "COBRA": sum(1 for x in validated_items if x["source_dataset"] == "COBRA"),
            "TIGER": sum(1 for x in validated_items if x["source_dataset"] == "TIGER"),
            "BRACS": sum(1 for x in validated_items if x["source_dataset"] == "BRACS"),
            "CAMELYON16": sum(1 for x in validated_items if x["source_dataset"] == "CAMELYON16"),
        },
        "slides": validated_items,
    }
    (REGISTRY_DIR / "train_pool_manifest.json").write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False))
    print(f"[Train-Pool v1] Saved benchmark_registry/train_pool_manifest.json (100/100 validated)", flush=True)

    print("[Train-Pool v1] Synthesizing 4B-8B multi-turn SFT, Step-DPO, and Step-GRPO datasets...", flush=True)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    all_sft: List[Dict] = []
    all_dpo: List[Dict] = []
    all_grpo: List[Dict] = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for idx, (sft_l, dpo_l, grpo_l) in enumerate(pool.map(synthesize_4b_8b_datasets_for_slide, validated_items), 1):
            all_sft.extend(sft_l)
            all_dpo.extend(dpo_l)
            all_grpo.extend(grpo_l)
            if idx % 10 == 0 or idx == len(validated_items):
                print(f"  [Synthesize {idx:3d}/100] SFT={len(all_sft)} trajs, DPO={len(all_dpo)} pairs, GRPO={len(all_grpo)} prompts", flush=True)

    sft_turns = sum(t["num_steps"] for t in all_sft)
    (SYNTH_DIR / "sft_multiturn_train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_sft) + "\n")
    (SYNTH_DIR / "dpo_step_train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_dpo) + "\n")
    (SYNTH_DIR / "grpo_prompts_train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_grpo) + "\n")
    summary = {
        "pool_version": "train_pool_v1_100slides",
        "zero_overlap_with_benchmark": True,
        "total_train_slides": len(validated_items),
        "sft_trajectories": len(all_sft),
        "sft_step_level_turns": sft_turns,
        "dpo_step_pairs": len(all_dpo),
        "grpo_rollout_prompts": len(all_grpo),
        "output_dir": str(SYNTH_DIR.relative_to(ROOT)),
    }
    (SYNTH_DIR / "train_pool_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
