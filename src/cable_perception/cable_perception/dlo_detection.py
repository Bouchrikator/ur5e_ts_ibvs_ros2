"""Marker-free detection of the cable centreline.

Implements the deformable one-dimensional object (DOO) detection pipeline of
Keipour, Bandari and Schaal, *Deformable One-Dimensional Object Detection for
Routing and Manipulation*, RA-L 2022 (arXiv:2201.06775):

    segmentation -> skeletonisation -> branch extraction -> fixed-length chain
    fitting -> pruning -> merging

The output is a chain of fixed-length segments connected by passive joints,
which is exactly the representation the reduced modal model needs: sampling it
at fixed normalised arc lengths gives the same ordered point set the coloured
fiducials used to provide, with no fiducials on the cable.

Why this matters here: coloured spheres give marker identity for free, but they
are a simulation artefact. A real cable has none, so a pipeline that depends on
them cannot transfer. Identity along a detected chain comes from arc length
measured from a known anchor (the clamped end), which a real cable does have.

Pure geometry and numpy: no ROS, no OpenCV, so every step is unit testable.
Segmentation itself is deliberately left to the caller — the paper is explicit
that any conservative segmentation method works.
"""

import numpy as np

# Zhang-Suen neighbour order P2..P9, clockwise from north (row, col) offsets.
_NEIGHBOURS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def zhang_suen_thin(mask, max_iterations=200):
    """Topological skeleton of a binary mask, one pixel wide.

    Zhang-Suen thinning satisfies the two requirements the paper places on the
    skeletonisation step: a connected component stays connected, and a real
    branch yields exactly one skeleton branch.
    """
    image = np.asarray(mask, dtype=bool).copy()
    if image.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {image.shape}")

    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1)
            p = [padded[1 + dr:padded.shape[0] - 1 + dr,
                        1 + dc:padded.shape[1] - 1 + dc]
                 for dr, dc in _NEIGHBOURS]
            occupied = sum(n.astype(np.int8) for n in p)
            ring = p + [p[0]]
            transitions = sum(((~ring[i]) & ring[i + 1]).astype(np.int8)
                              for i in range(8))
            if step == 0:
                first, second = p[0] & p[2] & p[4], p[2] & p[4] & p[6]
            else:
                first, second = p[0] & p[2] & p[6], p[0] & p[4] & p[6]
            remove = (image & (occupied >= 2) & (occupied <= 6)
                      & (transitions == 1) & ~first & ~second)
            if remove.any():
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


def _neighbour_map(thin):
    """Adjacency of the skeleton pixels, with redundant diagonals removed.

    A one-pixel-wide diagonal staircase has pixels with three 8-neighbours, and
    a plain 8-adjacency graph reads every one of them as a junction and cuts
    the branch there. A diagonal step is redundant whenever a 4-neighbour of
    the same pixel already touches it, so dropping those edges leaves a clean
    degree-2 path along the staircase.
    """
    rows, cols = np.nonzero(thin)
    pixels = set(zip(rows.tolist(), cols.tolist()))
    graph = {}
    for p in pixels:
        direct = [(p[0] + dr, p[1] + dc) for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1))
                  if (p[0] + dr, p[1] + dc) in pixels]
        diagonal = [(p[0] + dr, p[1] + dc) for dr, dc in ((-1, 1), (1, 1), (1, -1), (-1, -1))
                    if (p[0] + dr, p[1] + dc) in pixels]
        graph[p] = direct + [
            d for d in diagonal
            if not any(abs(d[0] - n[0]) + abs(d[1] - n[1]) == 1 for n in direct)]
    return graph


def skeleton_branches(thin, min_pixels=3):
    """Ordered pixel sequences, one per skeleton branch.

    A branch runs between two nodes of degree other than two (tips and
    junctions). This replaces the paper's contour-extraction step: the paper
    itself notes it "can be skipped" when an ordered pixel sequence per branch
    is already available, which is what a graph walk gives directly, without
    the duplicated out-and-back traversal a contour produces.
    """
    graph = _neighbour_map(np.asarray(thin, dtype=bool))
    if not graph:
        return []

    nodes = [p for p, n in graph.items() if len(n) != 2]
    branches, visited = [], set()

    def walk(start, first):
        path, previous, current = [start, first], start, first
        visited.add(frozenset((start, first)))
        while len(graph[current]) == 2:
            following = [n for n in graph[current] if n != previous]
            if not following:
                break
            previous, current = current, following[0]
            edge = frozenset((previous, current))
            if edge in visited:
                break
            visited.add(edge)
            path.append(current)
        return path

    for node in nodes:
        for neighbour in graph[node]:
            if frozenset((node, neighbour)) in visited:
                continue
            branches.append(walk(node, neighbour))

    # A closed loop has no degree-!=2 node; walk it from an arbitrary pixel.
    for pixel, neighbours in graph.items():
        if neighbours and frozenset((pixel, neighbours[0])) not in visited:
            branches.append(walk(pixel, neighbours[0]))

    return [np.asarray(b, dtype=float) for b in branches if len(b) >= min_pixels]


def fit_chain(points, segment_length, max_turn_rad=np.pi / 3):
    """Fit chains of fixed-length segments to an ordered point sequence.

    Algorithm 2 of the paper. Points are traversed and a new segment of length
    ``segment_length`` is emitted whenever the traversal has moved that far
    from the current start point. A segment whose direction turns by more than
    ``max_turn_rad`` from the previous one starts a new chain instead, which is
    what stops a skeleton artefact from being fused into the object.

    Returns a list of ``(K, D)`` vertex arrays, one per chain.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"points must be (N, D), got {points.shape}")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    chains, chain = [], [points[0]]
    start, direction = points[0], None
    for current in points[1:]:
        offset = current - start
        distance = float(np.linalg.norm(offset))
        if distance < segment_length:
            continue
        end = start + segment_length * offset / distance
        step = end - start
        if direction is not None:
            cosine = float(direction @ step) / (
                np.linalg.norm(direction) * np.linalg.norm(step))
            if np.arccos(np.clip(cosine, -1.0, 1.0)) > max_turn_rad:
                if len(chain) > 1:
                    chains.append(np.asarray(chain))
                chain, direction = [start], None
        chain.append(end)
        start, direction = end, step

    if len(chain) > 1:
        chains.append(np.asarray(chain))
    return chains


def _end_segments(chain):
    """``(point, outward direction)`` at the start and at the end of a chain."""
    return ((chain[0], chain[0] - chain[1]), (chain[-1], chain[-1] - chain[-2]))


def _angle(a, b):
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0.0:
        return np.pi
    return float(np.arccos(np.clip(float(a @ b) / norms, -1.0, 1.0)))


def merge_cost(end_a, end_b, weights=(1.0, 0.05, 0.05)):
    """Cost of joining two chain ends: equation (5) of the paper.

    Euclidean + direction + curvature. The direction cost discourages joining
    ends that do not face each other; the curvature cost discourages joins that
    need excessive bending. Euclidean alone would connect a far but perfectly
    aligned pair over a near, slightly misaligned one.
    """
    (point_a, dir_a), (point_b, dir_b) = end_a, end_b
    gap = point_b - point_a
    euclidean = float(np.linalg.norm(gap))
    direction = _angle(dir_a, -dir_b)
    curvature = max(_angle(dir_a, gap), _angle(dir_b, -gap))
    w_e, w_d, w_c = weights
    return w_e * euclidean + w_d * direction + w_c * curvature


def _hermite_fill(point_a, dir_a, point_b, dir_b, segment_length):
    """Gap-filling vertices that leave each end along its own direction.

    The paper fills the gap with two constant-radius arcs and a tangent line so
    the join is smooth and follows the object's natural bend. A cubic Hermite
    curve with the same boundary conditions is used here: identical C1
    behaviour at both ends, one closed form instead of the three-case arc
    construction, and no free radius to pick.
    """
    gap = float(np.linalg.norm(point_b - point_a))
    if gap <= segment_length:
        return np.empty((0, len(point_a)))

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0.0 else v

    tangent_a, tangent_b = gap * unit(dir_a), gap * unit(-dir_b)
    steps = max(int(np.ceil(gap / segment_length)), 1)
    t = np.linspace(0.0, 1.0, steps + 1)[1:-1][:, None]
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * point_a
            + (t ** 3 - 2 * t ** 2 + t) * tangent_a
            + (-2 * t ** 3 + 3 * t ** 2) * point_b
            + (t ** 3 - t ** 2) * tangent_b)


def merge_chains(chains, segment_length, weights=(1.0, 0.05, 0.05)):
    """Iteratively merge chains into a single object, filling the gaps.

    Each iteration joins the pair of ends with the lowest ``merge_cost`` and
    bridges the gap with a smooth fill, until one chain remains. Gaps come from
    occlusion (the gripper) or imperfect segmentation, and filling them is what
    keeps the arc-length parameterisation valid across the occluded part.
    """
    chains = [np.asarray(c, dtype=float) for c in chains if len(c) >= 2]
    if not chains:
        return np.empty((0, 2))

    while len(chains) > 1:
        best = None
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                for a, end_a in enumerate(_end_segments(chains[i])):
                    for b, end_b in enumerate(_end_segments(chains[j])):
                        cost = merge_cost(end_a, end_b, weights)
                        if best is None or cost < best[0]:
                            best = (cost, i, j, a, b)
        _, i, j, a, b = best
        # Orient both chains so the joined ends meet in the middle.
        first = chains[i][::-1] if a == 0 else chains[i]
        second = chains[j] if b == 0 else chains[j][::-1]
        fill = _hermite_fill(first[-1], first[-1] - first[-2],
                             second[0], second[0] - second[1], segment_length)
        merged = np.vstack([first, fill, second])
        chains = [c for k, c in enumerate(chains) if k not in (i, j)] + [merged]
    return chains[0]


def orient_chain(chain, anchor):
    """Order the chain so index 0 is the end nearest a known anchor.

    Arc length from the clamped end is what gives each sample a persistent
    identity once the coloured fiducials are gone, so the direction of travel
    has to be pinned to something physical rather than to detection order.
    """
    chain = np.asarray(chain, dtype=float)
    if len(chain) < 2:
        return chain
    anchor = np.asarray(anchor, dtype=float)[:chain.shape[1]]
    start = float(np.linalg.norm(chain[0] - anchor))
    end = float(np.linalg.norm(chain[-1] - anchor))
    return chain if start <= end else chain[::-1]


def resample_chain(chain, s_over_l):
    """Points at given normalised arc lengths along the chain.

    ``s_over_l`` values are fractions of the chain's own length, clamped to
    [0, 1]. This is the step that turns the continuous detected curve into the
    discrete feature set the reduced model consumes, without changing the
    message contract the coloured-marker tracker published.
    """
    chain = np.asarray(chain, dtype=float)
    if len(chain) < 2:
        raise ValueError("a chain needs at least two vertices to be resampled")

    steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(arc[-1])
    if total <= 0.0:
        raise ValueError("chain has zero length")

    targets = np.clip(np.asarray(s_over_l, dtype=float), 0.0, 1.0) * total
    return np.stack([np.interp(targets, arc, chain[:, axis])
                     for axis in range(chain.shape[1])], axis=1)


def detect_centreline(mask, segment_length_px, min_branch_pixels=3,
                      weights=(1.0, 0.05, 0.05)):
    """Full pixel-space pipeline: mask -> single ordered chain of vertices.

    Convenience wrapper for the case where the whole pipeline runs in the
    image. When the object lies on a known plane it is better to project the
    branches to metric coordinates first and call the steps individually, so
    the segment length is a real length and matches the cable's arc length.
    """
    branches = skeleton_branches(zhang_suen_thin(mask), min_branch_pixels)
    chains = [c for branch in branches
              for c in fit_chain(branch, segment_length_px)]
    return merge_chains(chains, segment_length_px, weights)
