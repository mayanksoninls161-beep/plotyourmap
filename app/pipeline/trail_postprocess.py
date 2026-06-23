#!/usr/bin/env python3
"""
trail_postprocess.py – raster-based trail merging (v6, color-focused)
=====================================================================
Global, per-color raster pipeline:

  1. Sample dominant map color per predicted polygon.
  2. Cluster polygons by color (Lab distance, agglomerative).
  3. Per color cluster:
       a. union-rasterize all polygons into one binary mask
       b. morphological closing  -> solidifies dashed strokes
       c. skeletonize            -> 1-px centerline network
       d. graph extraction       -> edges between junction/end nodes
       e. spur pruning           -> drop tiny dead-end twigs
       f. continuation pairing   -> at junctions, pair edges whose tangents
                                    continue straight through (geometry only,
                                    no stroke-pattern classification)
  4. Direction-aware endpoint bridging between chains of the same color.
     Bridges are routed along the printed cluster-color ink, so they
     naturally follow the trail and skip over text/icons (different color).
  5. Smooth + simplify; emit one record per trail polyline.

All coordinates in/out are original-image pixel space.
"""

import colorsys
import logging
import math
from collections import Counter, defaultdict

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2lab
from skimage.graph import route_through_array
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# CONFIG (auto-scaled by image diagonal; reference diag = 2457 px)
# ──────────────────────────────────────────────────────────────────

_REF_DIAG = math.hypot(2100, 1275)

DEFAULT_CONFIG = {
    "proc_max_dim": 3200,        # downscale very large maps for raster ops
    "close_radius_ratio": 5.0 / _REF_DIAG,    # dash-filling closing radius
    "spur_len_ratio": 18.0 / _REF_DIAG,       # prune skeleton twigs shorter than this
    "min_chain_ratio": 30.0 / _REF_DIAG,      # drop merged chains shorter than this
    "bridge_dist_ratio": 110.0 / _REF_DIAG,   # max gap to bridge between chains
    "bridge_angle_deg": 55.0,    # max deviation between tangent and gap direction
    "junction_angle_deg": 65.0,  # max deviation for straight-through pairing
    "tangent_window": 12,        # px used to estimate endpoint tangents
    "max_color_clusters": 12,    # upper bound on distinct trail colors per map
    "color_merge_thresh": 0.22,  # post-KMeans merge: collapse clusters whose RGB
                                 # centroids are within this L2 distance (0-1 scale)
    "max_width_ratio": 30.0 / _REF_DIAG,      # drop mask blobs wider than this
    "min_comp_area_ratio": 9.0 / _REF_DIAG,   # drop specks smaller than this (px area ~ value^2/9)
    "simplify_eps": 0.8,         # RDP epsilon (px) – small; resampling provides density
    "smooth_window": 7,
    "min_polygon_pts": 3,
    # loop preservation: keep trail loops as rings (not centerlines)
    "loop_preserve": True,
    "loop_min_area_ratio": 20.0 / _REF_DIAG,  # min enclosed area (side, px@ref) to be a real loop
    # dense output coordinates: resample each trail to even spacing
    "resample_enable": True,
    "resample_spacing_ratio": 4.0 / _REF_DIAG,  # ~one point every N proc-px along the trail
    # ink detection: Lab distance to identify this cluster's printed ink pixels.
    # Used for fat-blob re-skeletonization and ink-guided bridge routing.
    # NOT used for stroke-pattern classification.
    "ink_delta_e": 26.0,
    # ink-routed bridging: bridges follow the printed ink, not straight chords.
    # The cost map is built from the cluster color ink, so bridges naturally
    # avoid text/icons whose ink is a different color.
    "bridge_route_enable": True,
    "bridge_route_weight": 0.6,   # cost slope per px of distance from ink
    "bridge_route_cap": 25.0,     # cap on that distance penalty (px)
    "bridge_route_gain": 0.75,    # take the routed path if cost < gain * chord cost
    "bridge_chord_max_ratio": 40.0 / _REF_DIAG,  # ink-less straight bridges capped at this
                                                 # (small: only icon/label occlusion gaps)
    # corner joins: a trail bending sharply at a junction/road crossing
    "bridge_corner_dist_ratio": 28.0 / _REF_DIAG,
    "bridge_corner_angle_deg": 100.0,
    # occlusion bridging: a trail broken by overlaid text/symbols. When two
    # endpoints face each other and the straight gap is covered by foreground
    # ink (label/icon), bridge it even without same-color ink.
    "occlusion_bridge_enable": True,
    "occlusion_bridge_dist_ratio": 170.0 / _REF_DIAG,
    "occlusion_min_cover": 0.40,
    # collinear merging: same-colour trail broken by a long model-miss gap.
    # If two ends point nearly straight at each other, join across the gap.
    "collinear_bridge_dist_ratio": 300.0 / _REF_DIAG,
    "collinear_bridge_angle_deg": 20.0,
    "merge_iterations": 2,
    # record reconciliation (across colour clusters)
    "dedup_enable": True,
    "dedup_tol_ratio": 16.0 / _REF_DIAG,
    "dedup_overlap_frac": 0.70,
    "group_color_delta_e": 33.0,
    "group_xcluster_touch_ratio": 18.0 / _REF_DIAG,
}


# ──────────────────────────────────────────────────────────────────
# COLOR SAMPLING & CLUSTERING
# ──────────────────────────────────────────────────────────────────

def _sample_polygon_color(poly_px, map_arr):
    """Dominant saturated color inside polygon (median of top-saturation 20%)."""
    h, w = map_arr.shape[:2]
    xs = poly_px[:, 0]
    ys = poly_px[:, 1]
    x0, x1 = max(0, xs.min()), min(w - 1, xs.max())
    y0, y1 = max(0, ys.min()), min(h - 1, ys.max())
    if x1 <= x0 or y1 <= y0:
        logger.debug("_sample_polygon_color: degenerate bbox, skipping polygon")
        return None
    local = poly_px - np.array([x0, y0])
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [local.astype(np.int32)], 1)
    pix = map_arr[y0:y1 + 1, x0:x1 + 1][mask.astype(bool)]
    if len(pix) == 0:
        logger.debug("_sample_polygon_color: no pixels in polygon mask")
        return None
    p = pix.astype(int)
    sat = p.max(axis=1) - p.min(axis=1)
    k = max(1, len(p) // 5)
    top = p[np.argsort(sat)[-k:]]
    color = tuple(int(v) for v in np.median(top, axis=0))
    logger.debug("_sample_polygon_color: sampled color RGB%s from %d pixels", color, len(pix))
    return color


def _cluster_colors(colors, max_clusters=12, color_merge_thresh=0.22):
    """
    KMeans on a 4D HSV-derived feature space, followed by a post-merge pass
    that collapses clusters whose RGB centroids are within color_merge_thresh
    (L2, normalised 0-1).

    Feature vector per polygon: [hx, hy, sat, val]
      hx = cos(2π·h) · sat   # hue as a unit circle weighted by saturation
      hy = sin(2π·h) · sat   # no 0°/360° wrap discontinuity,
                                and near-grey polygons (sat→0) collapse toward
                                the origin instead of splitting by hue noise.
      sat, val = saturation and brightness kept as separate axes.

    Automatic k via silhouette score (k=2..max_clusters). The post-merge step
    prevents over-splitting when the same trail color gets placed in two adjacent
    KMeans clusters.
    """
    n = len(colors)
    logger.debug("_cluster_colors: clustering %d polygon colors (max_clusters=%d)", n, max_clusters)
    if n <= 2:
        logger.debug("_cluster_colors: too few polygons, assigning all to cluster 0")
        return np.zeros(n, dtype=int)

    rgb_arr = np.array(colors, dtype=float) / 255.0

    features = []
    for r, g, b in rgb_arr:
        h, sat, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        hx = math.cos(h * 2 * math.pi) * sat
        hy = math.sin(h * 2 * math.pi) * sat
        features.append([hx, hy, sat, v])
    mat = np.array(features, dtype=float)

    best_score, best_labels = -1.0, None
    max_k = min(max_clusters, n - 1)

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(mat)
        if len(set(labels.tolist())) < 2:
            continue
        try:
            score = silhouette_score(mat, labels)
            logger.debug("_cluster_colors: k=%d silhouette=%.4f", k, score)
            if score > best_score:
                best_score, best_labels = score, labels.copy()
        except ValueError:
            continue

    if best_labels is None:
        logger.warning("_cluster_colors: KMeans produced no valid labels, defaulting to 1 cluster")
        best_labels = np.zeros(n, dtype=int)

    # Post-merge: collapse clusters whose RGB centroids are too similar
    unique = sorted(set(best_labels.tolist()))
    if len(unique) > 1:
        centroids = {}
        for cl in unique:
            mask = best_labels == cl
            centroids[cl] = rgb_arr[mask].mean(axis=0)

        parent = {cl: cl for cl in unique}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merges = 0
        for i, ci in enumerate(unique):
            for cj in unique[i + 1:]:
                dist = np.linalg.norm(centroids[ci] - centroids[cj])
                if dist < color_merge_thresh:
                    ri, rj = _find(ci), _find(cj)
                    if ri != rj:
                        parent[rj] = ri
                        merges += 1
                        logger.debug(
                            "_cluster_colors: merging clusters %d+%d (centroid dist=%.4f)",
                            ci, cj, dist)

        if merges:
            logger.info("_cluster_colors: post-merge collapsed %d cluster pair(s)", merges)

        canon = {cl: _find(cl) for cl in unique}
        new_ids = {old: new for new, old in
                   enumerate(sorted(set(canon.values())))}
        best_labels = np.array([new_ids[canon[lbl]]
                                for lbl in best_labels.tolist()])

    n_final = len(set(best_labels.tolist()))
    logger.info("_cluster_colors: best k=%d (silhouette=%.4f) -> %d clusters after merge",
                max_k, best_score, n_final)
    return best_labels


# ──────────────────────────────────────────────────────────────────
# INK MASK – cluster-color pixels (used for fat-blob recovery
# and ink-guided bridge routing, NOT pattern classification)
# ──────────────────────────────────────────────────────────────────

def _ink_mask(lab_img, color_rgb, delta_e):
    """Pixels whose Lab color is within delta_e of the cluster color.
    Returns (sample_mask, raw_mask): sample_mask is dilated 1 px so a
    skeleton running beside the stroke centerline still samples ink;
    raw_mask keeps the true ink shapes."""
    c_lab = rgb2lab(np.array(color_rgb, dtype=float).reshape(1, 1, 3) / 255.0).reshape(3)
    dist = np.sqrt(((lab_img - c_lab.astype(np.float32)) ** 2).sum(axis=2))
    raw = (dist < delta_e).astype(np.uint8)
    ink_px = int(raw.sum())
    logger.debug("_ink_mask: color RGB%s delta_e=%.1f -> %d ink pixels", color_rgb, delta_e, ink_px)
    return cv2.dilate(raw, np.ones((3, 3), np.uint8)), raw


# ──────────────────────────────────────────────────────────────────
# SKELETON GRAPH
# ──────────────────────────────────────────────────────────────────

_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_graph(skel):
    """
    Skeleton bool array -> (nodes, edges).
      nodes: {node_id: (cy, cx)} centroid of junction/endpoint pixel clumps
      edges: list of dicts {a, b, path}  path = list[(y, x)] node-to-node
    """
    H, W = skel.shape
    sk = skel.astype(np.uint8)
    deg = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - sk
    node_mask = (sk == 1) & (deg != 2)

    n_comp, comp_lab = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)
    nodes = {}
    for nid in range(1, n_comp):
        ys, xs = np.where(comp_lab == nid)
        nodes[nid] = (float(ys.mean()), float(xs.mean()))

    node_at = {}
    nys, nxs = np.where(node_mask)
    for y, x in zip(nys.tolist(), nxs.tolist()):
        node_at[(y, x)] = comp_lab[y, x]

    skel_set = set(zip(*[a.tolist() for a in np.where(sk == 1)])) if sk.any() else set()

    edges = []
    visited = set()   # directed (from_pixel, to_pixel) steps consumed

    def trace(start_px, first_px):
        """Walk from a node pixel through deg-2 pixels until another node pixel."""
        path = [start_px, first_px]
        prev, cur = start_px, first_px
        while cur not in node_at:
            nxt = None
            cy, cx = cur
            for dy, dx in _NB8:
                nb = (cy + dy, cx + dx)
                if nb != prev and nb in skel_set:
                    nxt = nb
                    break
            if nxt is None:
                return path  # dead end (shouldn't happen often)
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for (y, x), nid in node_at.items():
        for dy, dx in _NB8:
            nb = (y + dy, x + dx)
            if nb not in skel_set:
                continue
            if nb in node_at and node_at[nb] == nid:
                continue  # same junction clump
            if ((y, x), nb) in visited:
                continue
            path = trace((y, x), nb)
            end = path[-1]
            visited.add(((y, x), nb))
            visited.add((path[-1], path[-2]))
            b = node_at.get(end)
            if b is None:
                b = -1  # dangling
            edges.append({"a": nid, "b": b, "path": path})

    # pure cycles (no node pixels at all in a loop): trace remaining deg-2 px
    edge_px = set()
    for e in edges:
        edge_px.update(e["path"])
    leftovers = skel_set - edge_px - set(node_at.keys())
    leftover_done = set()
    for px in leftovers:
        if px in leftover_done:
            continue
        path = [px]
        leftover_done.add(px)
        prev, cur = None, px
        while True:
            cy, cx = cur
            nxt = None
            for dy, dx in _NB8:
                nb = (cy + dy, cx + dx)
                if nb != prev and nb in leftovers and nb not in leftover_done:
                    nxt = nb
                    break
            if nxt is None:
                break
            path.append(nxt)
            leftover_done.add(nxt)
            prev, cur = cur, nxt
        if len(path) > 8:
            edges.append({"a": -1, "b": -1, "path": path})

    logger.debug("_skeleton_graph: %d nodes, %d edges extracted", len(nodes), len(edges))
    return nodes, edges


def _path_len(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def _prune_spurs(nodes, edges, spur_len):
    """Iteratively remove short dead-end edges (twigs off the main line)."""
    n_start = len(edges)
    for _ in range(4):
        deg_count = defaultdict(int)
        for e in edges:
            deg_count[e["a"]] += 1
            deg_count[e["b"]] += 1
        keep = []
        removed = 0
        for e in edges:
            a_deg = deg_count[e["a"]] if e["a"] != -1 else 1
            b_deg = deg_count[e["b"]] if e["b"] != -1 else 1
            is_spur = (min(a_deg, b_deg) == 1 and max(a_deg, b_deg) > 1
                       and _path_len(e["path"]) < spur_len)
            if is_spur:
                removed += 1
            else:
                keep.append(e)
        edges = keep
        if not removed:
            break
    logger.debug("_prune_spurs: %d -> %d edges (spur_len=%.1f)", n_start, len(edges), spur_len)
    return edges


def _tangent(path, from_start, window):
    """Unit direction pointing INTO the path from one of its ends."""
    pts = np.array(path[:window] if from_start else path[::-1][:window], dtype=float)
    if len(pts) < 2:
        return np.array([0.0, 0.0])
    v = pts[-1] - pts[0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0])


def _pair_continuations(nodes, edges, max_angle_deg, window):
    """
    At each junction, pair incident edge-ends whose tangents continue
    straight through (geometry only – no stroke-pattern gate).
    Returns chains: lists of (edge_idx, flipped).
    """
    inc = defaultdict(list)
    for ei, e in enumerate(edges):
        if e["a"] != -1:
            inc[e["a"]].append((ei, 0))
        if e["b"] != -1:
            inc[e["b"]].append((ei, 1))

    pairing = {}
    cos_thresh = math.cos(math.radians(max_angle_deg))

    for nid, ends in inc.items():
        if len(ends) < 2:
            continue
        tans = {}
        for ei, end in ends:
            path = edges[ei]["path"]
            t = _tangent(path, from_start=(end == 0), window=window)
            tans[(ei, end)] = t
        cands = []
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                if ends[i][0] == ends[j][0]:
                    continue  # same edge looping at one junction
                score = -float(np.dot(tans[ends[i]], tans[ends[j]]))
                if score >= cos_thresh:
                    cands.append((score, ends[i], ends[j]))
        cands.sort(reverse=True)
        used = set()
        for score, e1, e2 in cands:
            if e1 in used or e2 in used or e1 in pairing or e2 in pairing:
                continue
            pairing[e1] = e2
            pairing[e2] = e1
            used.add(e1)
            used.add(e2)

    chains = []
    consumed = set()
    for ei in range(len(edges)):
        if ei in consumed:
            continue
        start = None
        for end in (0, 1):
            if (ei, end) not in pairing:
                start = (ei, end)
                break
        if start is None:
            start = (ei, 0)

        chain = []
        cur_edge, entry_end = start
        while True:
            consumed.add(cur_edge)
            flipped = (entry_end == 1)
            chain.append((cur_edge, flipped))
            exit_end = 1 - entry_end
            nxt = pairing.get((cur_edge, exit_end))
            if nxt is None:
                break
            ne, nend = nxt
            if ne in consumed:
                break
            cur_edge, entry_end = ne, nend
        chains.append(chain)

    logger.debug("_pair_continuations: %d edges -> %d chains", len(edges), len(chains))
    return chains


def _chain_to_polyline(chain, edges):
    pts = []
    for ei, flipped in chain:
        path = edges[ei]["path"]
        seg = path[::-1] if flipped else path
        if pts and math.hypot(pts[-1][0] - seg[0][0], pts[-1][1] - seg[0][1]) < 3.0:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts


# ──────────────────────────────────────────────────────────────────
# CHAIN-LEVEL ENDPOINT BRIDGING (gaps at icons / labels)
# ──────────────────────────────────────────────────────────────────

def _route_bridge(p1, p2, ink, weight, cap, gain):
    """Minimum-cost path from p1 to p2 ((x,y) proc px) over a cost map
    where this cluster's ink is cheap and blank background expensive –
    so a bridge follows the printed color ink instead of a straight chord.
    Returns the routed points, or None when no ink supports the route."""
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    margin = int(0.5 * d + 15)
    H, W = ink.shape
    x0 = max(0, int(min(p1[0], p2[0]) - margin))
    x1 = min(W - 1, int(max(p1[0], p2[0]) + margin))
    y0 = max(0, int(min(p1[1], p2[1]) - margin))
    y1 = min(H - 1, int(max(p1[1], p2[1]) + margin))
    sub = ink[y0:y1 + 1, x0:x1 + 1]
    if sub.size == 0:
        logger.debug("_route_bridge: empty subregion for gap d=%.1f", d)
        return None
    dt = cv2.distanceTransform((sub == 0).astype(np.uint8), cv2.DIST_L2, 3)
    costs = 1.0 + weight * np.minimum(dt, cap)
    sh, sw = costs.shape
    start = (min(sh - 1, max(0, int(round(p1[1])) - y0)),
             min(sw - 1, max(0, int(round(p1[0])) - x0)))
    end = (min(sh - 1, max(0, int(round(p2[1])) - y0)),
           min(sw - 1, max(0, int(round(p2[0])) - x0)))
    try:
        path, cost = route_through_array(costs, start, end,
                                         fully_connected=True, geometric=True)
    except Exception as exc:
        logger.debug("_route_bridge: route_through_array failed: %s", exc)
        return None
    n = max(int(d), 2)
    cxs = np.clip(np.linspace(p1[0], p2[0], n) - x0, 0, sw - 1).astype(int)
    cys = np.clip(np.linspace(p1[1], p2[1], n) - y0, 0, sh - 1).astype(int)
    chord_cost = float(costs[cys, cxs].sum())
    if cost >= gain * chord_cost:
        logger.debug("_route_bridge: routed cost %.2f >= gain*chord %.2f, skipping", cost, gain * chord_cost)
        return None
    logger.debug("_route_bridge: accepted bridge d=%.1f routed_cost=%.2f chord_cost=%.2f", d, cost, chord_cost)
    return [(float(x0 + c), float(y0 + r)) for r, c in path]


def _chord_occluded(p1, p2, occ_mask, min_cover):
    """Fraction of the straight gap p1->p2 that passes over foreground
    ink (text / symbols). High coverage means the trail is hidden under
    a label there, justifying a straight bridge across it."""
    H, W = occ_mask.shape
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    n = max(2, int(d))
    xs = np.clip(np.linspace(p1[0], p2[0], n).astype(int), 0, W - 1)
    ys = np.clip(np.linspace(p1[1], p2[1], n).astype(int), 0, H - 1)
    pad = min(n // 4, max(1, int(0.15 * n)))
    mid = occ_mask[ys[pad:n - pad], xs[pad:n - pad]]
    if mid.size == 0:
        return False
    cover = float((mid > 0).mean())
    result = cover >= min_cover
    logger.debug("_chord_occluded: cover=%.2f min=%.2f -> %s", cover, min_cover, result)
    return result


def _bridge_chains(polylines, max_dist, max_angle_deg, window,
                   route_ink=None, route_weight=0.6, route_cap=25.0,
                   route_gain=0.75, chord_max=None,
                   corner_dist=None, corner_angle_deg=100.0,
                   occ_mask=None, occ_dist=None, occ_min_cover=0.4,
                   collinear_dist=None, collinear_angle_deg=20.0):
    """
    Merge polylines whose endpoints face each other across a gap.
    Greedy nearest-first with port locking; direction-aware on both sides.

    Merging is color-based: all polylines in one call share the same color
    cluster, so no pattern gating is applied. Bridges are routed along the
    cluster-color ink (naturally avoiding text/icons of other colors).

    Corner joins: endpoints within corner_dist may join up to
    corner_angle_deg if ink connects them.

    Collinear merging: near-straight ends across a long model-miss gap
    join even with no detected ink between them.

    Occlusion bridging: endpoints facing each other across a foreground-
    ink-covered gap (label/icon) bridge straight across the occluder.
    """
    cos_thresh = math.cos(math.radians(max_angle_deg))
    n_input = len(polylines)

    def endpoint(pl, end):
        return np.array(pl[0] if end == 0 else pl[-1], dtype=float)

    def out_tangent(pl, end):
        t = _tangent(pl, from_start=(end == 0), window=window)
        return -t

    merged = [list(p) for p in polylines]
    active = list(range(len(merged)))

    cos_corner = math.cos(math.radians(corner_angle_deg))
    can_corner = corner_dist is not None and route_ink is not None
    cos_collin = math.cos(math.radians(collinear_angle_deg))
    can_collin = collinear_dist is not None and collinear_dist > 0

    total_merges = 0
    changed = True
    while changed:
        changed = False
        cands = []
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                i, j = active[ii], active[jj]
                lim = max_dist
                if occ_dist is not None:
                    lim = max(lim, occ_dist)
                if can_collin:
                    lim = max(lim, collinear_dist)
                for ei in (0, 1):
                    for ej in (0, 1):
                        p1 = endpoint(merged[i], ei)
                        p2 = endpoint(merged[j], ej)
                        gap = p2 - p1
                        d = float(np.linalg.norm(gap))
                        if d > lim or d < 1e-6:
                            continue
                        is_corner = (can_corner and d <= corner_dist)
                        gdir = gap / d
                        t1 = out_tangent(merged[i], ei)
                        t2 = out_tangent(merged[j], ej)
                        c1 = float(np.dot(t1, gdir))
                        c2 = float(np.dot(t2, -gdir))
                        thresh = cos_corner if is_corner else cos_thresh
                        is_collinear = (can_collin and d > max_dist
                                        and c1 >= cos_collin and c2 >= cos_collin)
                        if (c1 >= thresh and c2 >= thresh) or is_collinear:
                            score = d * (2.2 - c1 - c2)
                            cands.append((score, d, i, j, ei, ej,
                                          is_corner, is_collinear))
        if not cands:
            break
        cands.sort()
        used_ports = set()
        merged_ids = set()
        for score, d, i, j, ei, ej, is_corner, is_collinear in cands:
            if i in merged_ids or j in merged_ids:
                continue
            if (i, ei) in used_ports or (j, ej) in used_ports:
                continue
            a = merged[i] if ei == 1 else merged[i][::-1]
            b = merged[j] if ej == 0 else merged[j][::-1]
            mid = None
            if route_ink is not None:
                mid = _route_bridge(a[-1], b[0], route_ink,
                                    route_weight, route_cap, route_gain)
                if mid is None:
                    occluded = (occ_mask is not None and occ_dist is not None
                                and d <= occ_dist
                                and _chord_occluded(a[-1], b[0], occ_mask, occ_min_cover))
                    if occluded:
                        logger.debug("_bridge_chains: occlusion bridge accepted d=%.1f", d)
                    elif is_collinear:
                        logger.debug("_bridge_chains: collinear bridge accepted d=%.1f", d)
                    elif d > max_dist:
                        continue  # no ink to follow across a wide gap
                    elif is_corner and d > 8.0:
                        continue  # corner join needs ink continuity
                    elif chord_max is not None and d > chord_max:
                        continue  # long bridge with no ink support: don't merge
            merged[i] = a + (mid[1:-1] if mid else []) + b
            merged[j] = None
            active.remove(j)
            merged_ids.add(i)
            merged_ids.add(j)
            used_ports.add((i, ei))
            used_ports.add((j, ej))
            total_merges += 1
            changed = True

    keep = [k for k, m in enumerate(merged) if m]
    logger.info("_bridge_chains: %d polylines -> %d after %d merges",
                n_input, len(keep), total_merges)
    return [merged[k] for k in keep]


# ──────────────────────────────────────────────────────────────────
# POLYLINE CLEANUP
# ──────────────────────────────────────────────────────────────────

def _pt_polyline_dist(pt, pts):
    """Min distance from a point to a polyline (proper point-to-segment)."""
    if len(pts) < 2:
        return float(np.linalg.norm(np.asarray(pt, dtype=float) - pts[0]))
    p = np.asarray(pt, dtype=float)
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.sqrt(((p - proj) ** 2).sum(1).min()))


def _frac_within(a, b, tol):
    """Fraction of polyline a's vertices lying within tol of polyline b."""
    if len(b) < 2:
        return float((np.linalg.norm(a - b[0], axis=1) < tol).mean())
    s0, s1 = b[:-1], b[1:]
    ab = s1 - s0
    L2 = np.maximum((ab * ab).sum(1), 1e-9)
    cnt = 0
    for p in a:
        t = np.clip(((p - s0) * ab).sum(1) / L2, 0, 1)
        proj = s0 + t[:, None] * ab
        if np.sqrt(((p - proj) ** 2).sum(1)).min() < tol:
            cnt += 1
    return cnt / max(len(a), 1)


def _rgb_lab_dist(c1, c2):
    """Lab (CIE76) distance between two RGB tuples."""
    arr = np.array([c1, c2], dtype=float).reshape(-1, 1, 3) / 255.0
    lab = rgb2lab(arr).reshape(-1, 3)
    return float(np.linalg.norm(lab[0] - lab[1]))


def _dedup_overlaps(records, tol, overlap_frac):
    """Drop records that lie on top of a longer record (same physical
    trail traced twice – colour bleed at a crossing spawns a short
    mis-coloured fragment over a real trail). Keep the longest."""
    arrs = [np.array([[p["x"], p["y"]] for p in r["points"]], dtype=float)
            for r in records]
    order = sorted(range(len(records)), key=lambda k: -len(records[k]["points"]))
    kept = []
    drop = set()
    for k in order:
        a = arrs[k]
        if len(a) < 2:
            continue
        if any(_frac_within(a, arrs[kk], tol) >= overlap_frac for kk in kept):
            drop.add(k)
        else:
            kept.append(k)
    logger.debug("_dedup_overlaps: %d -> %d records dropped %d duplicates",
                 len(records), len(kept), len(drop))
    return [r for i, r in enumerate(records) if i not in drop]


def _smooth(pts, window):
    if len(pts) <= window:
        return pts
    arr = np.array(pts, dtype=float)
    kernel = np.ones(window) / window
    sm = np.copy(arr)
    half = window // 2
    for d in range(2):
        sm[half:-half, d] = np.convolve(arr[:, d], kernel, mode="valid")
    return [tuple(p) for p in sm]


def _rdp(pts, eps):
    if len(pts) <= 2:
        return pts
    arr = np.array(pts, dtype=float)
    out = cv2.approxPolyDP(arr.astype(np.float32).reshape(-1, 1, 2), eps, False)
    return [(float(p[0][0]), float(p[0][1])) for p in out]


def _resample(pts, spacing):
    """Resample a polyline to evenly-spaced vertices ~`spacing` apart by
    arc length. Endpoints are preserved."""
    if len(pts) < 2 or spacing <= 0:
        return pts
    arr = np.array(pts, dtype=float)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < spacing:
        return [tuple(arr[0]), tuple(arr[-1])]
    n = max(2, int(round(total / spacing)) + 1)
    targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, cum, arr[:, 0])
    ys = np.interp(targets, cum, arr[:, 1])
    return list(zip(xs.tolist(), ys.tolist()))


# ──────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ──────────────────────────────────────────────────────────────────

def postprocess_trails(predictions, map_image, config=None):
    """
    predictions : list of Roboflow seg predictions ({points:[{x,y}..], confidence,..})
                  coords in original image px space
    map_image   : PIL.Image (RGB) at original size
    returns     : list of merged trail records (points in original px space)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    W, H = map_image.size
    scale = min(1.0, cfg["proc_max_dim"] / max(W, H))
    pw, ph = int(round(W * scale)), int(round(H * scale))
    diag = math.hypot(pw, ph)

    logger.info("postprocess_trails: image %dx%d scale=%.3f proc=%dx%d diag=%.1f",
                W, H, scale, pw, ph, diag)

    close_r = max(2, int(round(cfg["close_radius_ratio"] * diag)))
    spur_len = cfg["spur_len_ratio"] * diag
    min_chain = cfg["min_chain_ratio"] * diag
    bridge_dist = cfg["bridge_dist_ratio"] * diag

    logger.debug("postprocess_trails: close_r=%d spur_len=%.1f min_chain=%.1f bridge_dist=%.1f",
                 close_r, spur_len, min_chain, bridge_dist)

    map_arr_full = np.array(map_image)
    if scale < 1.0:
        proc_img = cv2.resize(map_arr_full, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        proc_img = map_arr_full

    # Lab image used for ink-mask computation (fat-blob recovery + ink routing)
    lab_proc = rgb2lab(proc_img.astype(np.float32) / 255.0).astype(np.float32)

    # Foreground-ink mask (text / symbols / lines): dark OR saturated pixels
    # stand out from the pale terrain background. Used to bridge trail gaps
    # that are occluded by overlaid labels.
    occ_mask = None
    if cfg.get("occlusion_bridge_enable", True):
        hsv = cv2.cvtColor(proc_img, cv2.COLOR_RGB2HSV)
        occ_mask = ((hsv[..., 2] < 120) | (hsv[..., 1] > 90)).astype(np.uint8)
        logger.debug("postprocess_trails: occlusion mask coverage=%.2f%%",
                     100.0 * occ_mask.mean())
    occ_dist = (cfg["occlusion_bridge_dist_ratio"] * diag
                if cfg.get("occlusion_bridge_enable", True) else None)

    # ── 1. polygons + colors ─────────────────────────────────────
    polys = []
    skipped = 0
    for pred in predictions:
        pts = pred.get("points", [])
        if len(pts) < cfg["min_polygon_pts"]:
            skipped += 1
            continue
        poly = np.array([[p["x"] * scale, p["y"] * scale] for p in pts])
        color = _sample_polygon_color(poly.astype(int), proc_img)
        if color is None:
            skipped += 1
            continue
        polys.append({
            "poly": poly,
            "color": color,
            "confidence": pred.get("confidence", 0.0),
            "class": pred.get("class", "trails"),
            "class_id": pred.get("class_id", 0),
        })

    logger.info("postprocess_trails: %d predictions -> %d valid polygons (%d skipped)",
                len(predictions), len(polys), skipped)

    if not polys:
        logger.warning("postprocess_trails: no valid polygons, returning empty result")
        return []

    # ── 2. color clustering ───────────────────────────────────────
    labels = _cluster_colors([p["color"] for p in polys],
                             cfg["max_color_clusters"], cfg["color_merge_thresh"])
    n_clusters = len(set(labels.tolist()))
    logger.info("postprocess_trails: %d polygons -> %d color clusters", len(polys), n_clusters)
    print(f"   [postprocess] {len(polys)} polygons -> {n_clusters} color clusters")

    records = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))

    for cl in sorted(set(labels.tolist())):
        members = [p for p, l in zip(polys, labels) if l == cl]
        cl_color = tuple(int(np.median([m["color"][c] for m in members])) for c in range(3))
        best_conf = max(m["confidence"] for m in members)

        logger.info("postprocess_trails: cluster %d color=#%02x%02x%02x members=%d conf=%.4f",
                    cl, *cl_color, len(members), best_conf)

        # Cluster-color ink mask – used for fat-blob re-skeletonization and
        # ink-guided bridge routing. Ink is identified purely by Lab color
        # distance to the cluster centroid; no pattern classification is done.
        ink, ink_raw = _ink_mask(lab_proc, cl_color, cfg["ink_delta_e"])

        # ── 3a. union mask ────────────────────────────────────────
        # Build from cluster-color ink pixels bounded by each polygon,
        # NOT from fillPoly. fillPoly fills the interior of loop/ring
        # polygons → solid disk → skeletonize → single center point.
        # Ink pixels within the polygon naturally form the right shape:
        # a ring trail's ink is a ring, so the mask stays a ring.
        # Fallback to fillPoly only when no ink is found (very faint
        # or slightly mis-colored predictions).
        mask = np.zeros((ph, pw), np.uint8)
        _ink_min_px = 8
        ink_hits, fallbacks = 0, 0
        for m in members:
            poly_roi = np.zeros((ph, pw), np.uint8)
            cv2.fillPoly(poly_roi, [m["poly"].astype(np.int32)], 1)
            ink_within = poly_roi.astype(bool) & ink.astype(bool)
            if ink_within.sum() > _ink_min_px:
                mask[ink_within] = 1
                ink_hits += 1
            else:
                mask |= poly_roi  # fallback: no ink detected in this polygon
                fallbacks += 1
        logger.debug("cluster %d union mask: ink_hits=%d fillpoly_fallbacks=%d mask_px=%d",
                     cl, ink_hits, fallbacks, int(mask.sum()))
        mask_raw = mask.copy()

        # ── 3b. closing ───────────────────────────────────────────
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ── 3b'. loop preservation ────────────────────────────────
        # The closing step can fill the interior of small loops whose radius
        # is <= close_r. Re-open any filled hole that is large enough to be a
        # real loop interior (not just a small artifact gap).
        # Holes are detected in: (a) the raw ink mask, (b) the closed mask.
        # The ink-mask hole check catches loops that were already ring-shaped
        # before closing and whose interior the close step just filled.
        if cfg.get("loop_preserve", True):
            mb = mask.astype(bool)
            raw_bool = mask_raw.astype(bool)
            # holes in the raw ink mask (pre-closing ring shapes)
            holes_raw = binary_fill_holes(raw_bool) & ~raw_bool
            # holes that closing introduced
            holes_closed = binary_fill_holes(mb) & ~mb
            holes = (holes_raw | holes_closed).astype(np.uint8)
            if holes.any():
                min_loop_area = max(
                    (cfg["loop_min_area_ratio"] * diag) ** 2,
                    (2.5 * (2 * close_r)) ** 2)
                nh, hl, hstats, _ = cv2.connectedComponentsWithStats(
                    holes, connectivity=4)
                big = np.zeros_like(holes, dtype=bool)
                loops_preserved = 0
                for h in range(1, nh):
                    if hstats[h, cv2.CC_STAT_AREA] >= min_loop_area:
                        big |= (hl == h)
                        loops_preserved += 1
                if big.any():
                    mask = (mb & ~big).astype(np.uint8)
                    logger.debug("cluster %d loop_preserve: restored %d loop hole(s)", cl, loops_preserved)

        # ── 3c. skeleton ──────────────────────────────────────────
        skel = skeletonize(mask.astype(bool))
        if not skel.any():
            logger.debug("cluster %d: empty skeleton, skipping", cl)
            continue

        # ── 3c'. blob filter: trails are thin strokes ─────────────
        # A filled region skeletonizes into a tangle. Mean stroke width
        # (component area / skeleton length) flags fat blobs.
        # Fat blobs are re-skeletonized using only the cluster-color ink
        # inside them (thin trail ring), not the filled interior.
        max_w = cfg["max_width_ratio"] * diag
        min_area = (cfg["min_comp_area_ratio"] * diag) ** 2 / 9.0
        stroke_w = 2.0 * close_r
        n_lbl, lbl = cv2.connectedComponents(mask, connectivity=8)
        if n_lbl > 1:
            areas = np.bincount(lbl.ravel(), minlength=n_lbl).astype(float)
            skel_counts = np.bincount(lbl[skel].ravel(), minlength=n_lbl).astype(float)
            mean_w = areas / np.maximum(skel_counts, 1.0)
            fat = (mean_w > max_w)
            tiny = (areas < min_area)
            fat[0] = tiny[0] = False
            n_fat = int(fat.sum())
            n_tiny = int(tiny.sum())
            if n_fat or n_tiny:
                logger.debug("cluster %d blob filter: %d fat, %d tiny components removed",
                             cl, n_fat, n_tiny)
            if fat.any():
                fat_region = fat[lbl] & (mask > 0)
                ink_in = (fat_region & (ink > 0)).astype(np.uint8)
                if ink_in.any():
                    small_r = min(close_r, 3)
                    small_k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * small_r + 1, 2 * small_r + 1))
                    ink_in = cv2.morphologyEx(ink_in, cv2.MORPH_CLOSE, small_k)
                    reskel = skeletonize(ink_in.astype(bool))
                    skel = (skel & ~fat[lbl]) | reskel
                    logger.debug("cluster %d: re-skeletonized %d fat blob(s) from ink", cl, n_fat)
                else:
                    skel &= ~fat[lbl]
            if tiny.any():
                skel &= ~tiny[lbl]
            good = ~fat & ~tiny
            good[0] = False
            if good.any() and skel_counts[good].sum() > 0:
                stroke_w = float(areas[good].sum() / skel_counts[good].sum())
        if not skel.any():
            logger.debug("cluster %d: skeleton empty after blob filter, skipping", cl)
            continue

        logger.debug("cluster %d skeleton: skel_px=%d stroke_w=%.1f", cl, int(skel.sum()), stroke_w)

        # ── 3d-e. graph + spur pruning ────────────────────────────
        nodes, edges = _skeleton_graph(skel)
        if not edges:
            logger.debug("cluster %d: no skeleton edges, skipping", cl)
            continue
        edges = _prune_spurs(nodes, edges, spur_len)
        if not edges:
            logger.debug("cluster %d: no edges after spur pruning, skipping", cl)
            continue

        # ── 3f. continuation pairing -> chains (geometry only) ────
        chains = _pair_continuations(
            nodes, edges, cfg["junction_angle_deg"], cfg["tangent_window"])
        polylines = [_chain_to_polyline(c, edges) for c in chains]
        polylines = [p for p in polylines if _path_len(p) >= min_chain * 0.5]

        logger.info("cluster %d: %d chains -> %d polylines after min_chain filter",
                    cl, len(chains), len(polylines))

        if not polylines:
            logger.debug("cluster %d: no polylines after length filter, skipping", cl)
            continue

        # convert (y,x) -> (x,y)
        polylines = [[(x, y) for (y, x) in p] for p in polylines]

        # ── 4. iterate: bridge gaps ───────────────────────────────
        for _it in range(max(1, int(cfg["merge_iterations"]))):
            n_before = len(polylines)
            polylines = _bridge_chains(
                polylines, bridge_dist, cfg["bridge_angle_deg"], cfg["tangent_window"],
                route_ink=(ink if cfg["bridge_route_enable"] else None),
                route_weight=cfg["bridge_route_weight"],
                route_cap=cfg["bridge_route_cap"],
                route_gain=cfg["bridge_route_gain"],
                chord_max=cfg["bridge_chord_max_ratio"] * diag,
                corner_dist=cfg["bridge_corner_dist_ratio"] * diag,
                corner_angle_deg=cfg["bridge_corner_angle_deg"],
                occ_mask=occ_mask, occ_dist=occ_dist,
                occ_min_cover=cfg["occlusion_min_cover"],
                collinear_dist=(cfg["collinear_bridge_dist_ratio"] * diag
                                if cfg["collinear_bridge_dist_ratio"] else None),
                collinear_angle_deg=cfg["collinear_bridge_angle_deg"])
            logger.debug("cluster %d bridge iter %d: %d -> %d polylines",
                         cl, _it, n_before, len(polylines))
            if len(polylines) == n_before:
                break

        # ── 5. cleanup + records ──────────────────────────────────
        n_added = 0
        for pl in polylines:
            if _path_len([(y, x) for (x, y) in pl]) < min_chain:
                continue
            pl = _smooth(pl, cfg["smooth_window"])
            pl = _rdp(pl, cfg["simplify_eps"])
            if len(pl) < 2:
                continue
            if cfg.get("resample_enable", True):
                pl = _resample(pl, max(2.0, cfg["resample_spacing_ratio"] * diag))
            pl = [(x / scale, y / scale) for (x, y) in pl]
            xs = [p[0] for p in pl]
            ys = [p[1] for p in pl]
            records.append({
                "x": round((min(xs) + max(xs)) / 2, 3),
                "y": round((min(ys) + max(ys)) / 2, 3),
                "width": round(max(xs) - min(xs), 3),
                "height": round(max(ys) - min(ys), 3),
                "confidence": round(best_conf, 6),
                "class": members[0]["class"],
                "class_id": members[0]["class_id"],
                "points": [{"x": round(x, 3), "y": round(y, 3)} for x, y in pl],
                "_color_rgb": list(cl_color),
                "_color_hex": "#%02x%02x%02x" % cl_color,
                "_cluster": int(cl),
                "_n_segments": len(members),
            })
            n_added += 1
        logger.info("cluster %d: added %d trail records", cl, n_added)

    # ── 5b. reconcile across colour clusters ──────────────────────
    n_pre = len(records)
    if cfg.get("dedup_enable", True) and records:
        odiag = diag / scale
        records = _dedup_overlaps(
            records, max(4.0, cfg["dedup_tol_ratio"] * odiag),
            cfg["dedup_overlap_frac"])
        if len(records) != n_pre:
            logger.info("postprocess_trails: dedup %d -> %d records", n_pre, len(records))

    # ── 6. trail grouping ─────────────────────────────────────────
    # Give one trail identity (_trail_group) to polylines that touch,
    # including across colour clusters when colours are close.
    corner_orig = cfg["bridge_corner_dist_ratio"] * diag / scale
    parent = list(range(len(records)))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pts_cache = [np.array([[p["x"], p["y"]] for p in r["points"]], dtype=float)
                 for r in records]
    grp_de = cfg["group_color_delta_e"]
    xcluster_touch = cfg["group_xcluster_touch_ratio"] * diag / scale
    for a in range(len(records)):
        for b in range(a + 1, len(records)):
            same_cl = records[a]["_cluster"] == records[b]["_cluster"]
            if same_cl:
                tol = corner_orig
            else:
                if _rgb_lab_dist(records[a]["_color_rgb"],
                                 records[b]["_color_rgb"]) > grp_de:
                    continue
                tol = xcluster_touch
            A, B = pts_cache[a], pts_cache[b]
            touch = min(_pt_polyline_dist(A[0], B), _pt_polyline_dist(A[-1], B),
                        _pt_polyline_dist(B[0], A), _pt_polyline_dist(B[-1], A))
            if touch <= tol:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[rb] = ra

    roots = {}
    for k, r in enumerate(records):
        r["_trail_group"] = roots.setdefault(_find(k), len(roots))

    dd = f" (dedup {n_pre}->{len(records)})" if len(records) != n_pre else ""
    logger.info("postprocess_trails: done -> %d trails (%d polylines)%s",
                len(roots), len(records), dd)
    print(f"   [postprocess] -> {len(roots)} trails ({len(records)} polylines){dd}")
    return records
