"""Pure open-loop policies: dimensions and RGB thumbnail only; no filesystem/GT."""
import numpy as np


def candidates(width, height, field):
    return np.array([(x, y) for y in range(0, height, field)
                     for x in range(0, width, field)], dtype=np.int64)


def uniform_order(xy, width, height):
    """Geometric farthest-first traversal, starting closest to slide centre."""
    centres = (xy + 0.5) / np.array([width, height])
    chosen = np.zeros(len(xy), dtype=bool)
    distance = np.full(len(xy), np.inf)
    index = int(np.argmin(np.sum((centres - 0.5)**2, axis=1)))
    order = []
    # Only a prefix is required. Return at most the fixed maximum budget.
    for _ in range(min(128, len(xy))):
        order.append(index)
        chosen[index] = True
        distance = np.minimum(distance, np.sum((centres-centres[index])**2, axis=1))
        distance[chosen] = -1
        index = int(np.argmax(distance))
    return np.asarray(order)


def tissue_scores(rgb, xy, width, height, field):
    """Fraction of non-white, non-black, chromatic thumbnail pixels per FOV."""
    a = np.asarray(rgb, dtype=np.float32)
    hi, lo = a.max(axis=2), a.min(axis=2)
    mask = (lo < 235) & (hi > 40) & ((hi-lo)/np.maximum(hi, 1) > 0.05)
    th, tw = mask.shape
    scores = []
    for x, y in xy:
        x0, y0 = int(x*tw/width), int(y*th/height)
        x1 = max(x0+1, int(np.ceil(min(x+field, width)*tw/width)))
        y1 = max(y0+1, int(np.ceil(min(y+field, height)*th/height)))
        scores.append(float(mask[y0:y1, x0:x1].mean()))
    return np.asarray(scores)


def select(strategy, width, height, field, thumbnail, seed=0):
    xy = candidates(width, height, field)
    if len(xy) < 128:
        raise ValueError('Fewer than 128 candidate FOVs: revise spec before running')
    if strategy == 'uniform':
        order = uniform_order(xy, width, height)
        score = np.zeros(len(xy))
    elif strategy == 'random':
        order = np.random.default_rng(seed).choice(len(xy), size=128, replace=False)
        score = np.zeros(len(xy))
    elif strategy == 'tissue':
        score = tissue_scores(thumbnail, xy, width, height, field)
        # Fixed geometric tie-breaking across all candidates; no GT involved.
        centres = (xy + field/2)/np.array([width, height])
        angle = np.mod(np.arctan2(centres[:, 1]-.5, centres[:, 0]-.5), 2*np.pi)
        order = np.lexsort((xy[:, 0], xy[:, 1], angle, -score))[:128]
    else:
        raise ValueError(strategy)
    return [{'x0': int(xy[i, 0]), 'y0': int(xy[i, 1]),
             'low_thumbnail_score': float(score[i])} for i in order]
