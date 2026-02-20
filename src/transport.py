import numpy as np
import math
import heapq
import scipy.optimize
from utils import get_coords

def get_coords(index , width):
    return index % width, index // width

def compute_matrix(target_pixels: np.ndarray, weight_pixels: np.ndarray, result_pixels: np.ndarray) -> np.ndarray:
    N = target_pixels.shape[0]
    M = result_pixels.shape[0]
    width = int(math.sqrt(N))

    T = target_pixels.astype(np.float32)
    W = weight_pixels.astype(np.float32)
    R = result_pixels.astype(np.float32)

    idx = np.arange(N, dtype=np.int32)
    tx, ty = get_coords(idx, width)
    tx = tx.astype(np.float32)[None, :]
    ty = ty.astype(np.float32)[None, :]

    color_diff = T[None, :, :] - R[:, None, :]
    color_dist = np.sum(color_diff * color_diff, axis=2)

    m_idx = np.arange(M, dtype=np.int32)
    rx, ry = get_coords(m_idx, width)
    rx = rx.astype(np.float32)[:, None]
    ry = ry.astype(np.float32)[:, None]
    spatial_dist = (tx - rx)**2 + (ty - ry)**2

    costs = spatial_dist**2 + color_dist * W[None, :]
    return costs

# SINKHORN

def sinkhorn_transport(costs: np.ndarray,
                       source_colors: np.ndarray,
                       target_width: int,
                       verbose: bool = False):
    """
    Produce a strict 1:1 mapping from source pixels -> target grid cells
    that minimizes total cost (if SciPy is available) or a deterministic
    greedy fallback otherwise.

    Args:
        costs: (M, N) cost matrix, cost[i,j] = cost to move source i -> target j
               Typical case: M == N == target_width**2.
        source_colors: (M, C) array of source RGB colors (uint8 or floats).
        target_width: width of square target image (N == target_width**2).
        use_scipy: try to use scipy.optimize.linear_sum_assignment if True.
        verbose: print progress.

    Returns:
        dict with keys:
            - 'assignment': ndarray shape (M,) mapping source_index -> target_index (unique)
            - 'hard_image' : ndarray shape (target_width, target_width, C) dtype same as source_colors
    Notes:
        - This function ALWAYS returns a 1:1 mapping (each source assigned to exactly one unique target).
        - If M > N a ValueError is raised (can't create 1:1 mapping).
        - The SciPy Hungarian algorithm is O(n^3) and will be slow/huge memory for large n (e.g. > 10000).
    """
    M, N = costs.shape
    if source_colors.shape[0] != M:
        raise ValueError("source_colors must have same first dim as costs")
    if M > N:
        raise ValueError(f"Cannot assign {M} sources to {N} targets uniquely (M > N).")

    assignment = -np.ones(M, dtype=np.int64)

    try:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
        # SciPy returns matched pairs; row_ind length equals number of assigned rows (<= M)
        # Fill assignment array for those returned rows.
        for r, c in zip(row_ind, col_ind):
            assignment[r] = int(c)
        if verbose:
            print("Used scipy.optimize.linear_sum_assignment (optimal).")
    except Exception as e:
        if verbose:
            print("scipy not available or failed; falling back to greedy. Error:", e)
        assignment[:] = -1

    if verbose:
        total_cost = float(costs[np.arange(M), assignment].sum())
        print(f"[sinkhorn] finished, total cost {total_cost}")

    # Build the hard image (unique mapping). dtype matches source_colors dtype.
    C = source_colors.shape[1]
    dtype = source_colors.dtype
    hard_img = np.zeros((target_width, target_width, C), dtype=dtype)

    for s in range(M):
        t = int(assignment[s])
        y = t // target_width
        x = t % target_width
        hard_img[y, x, :] = source_colors[s]

    return {"assignment": assignment, "hard_image": hard_img}

# TOP-K GREEDY + REFINE (HILL CLIMB)

def topk_greedy_assignment(costs: np.ndarray, K: int = 32, verbose: bool = False
                          ) -> np.ndarray:
    """
    Approximate 1:1 assignment by limiting each source to its top-K targets,
    then performing a global greedy using a min-heap of (cost, source, rank_index).

    Args:
        costs: (M, N) cost matrix (minimization). Must satisfy M <= N.
        K: number of candidate targets kept per source (clamped to N).
        verbose: print progress.
    Returns:
        assignment: array shape (M,) mapping source i -> target j (unique).
    Notes:
        - Deterministic given identical numpy behaviour.
        - If some sources exhaust their K candidates, they are assigned in a final
          pass to the lowest-cost remaining targets (guarantees a full 1:1 mapping).
    """
    M, N = costs.shape
    if M > N:
        raise ValueError("topk_greedy_assignment requires M <= N")

    if K >= N:
        # nothing to prune: fallback to global greedy via argmin per-row with heap (still faster than O(n^3) in Python)
        K = N

    # For each source row, find top-K candidate columns (unsorted)
    # np.argpartition is O(N) per row; result not fully sorted. We'll sort the K slice by cost.
    K = min(K, N)
    topk_idx = np.argpartition(costs, K - 1, axis=1)[:, :K]  # (M, K) unsorted
    # Now sort each row's K candidates by cost to get increasing order
    # compute costs for the K candidates
    # costs_rows_k = costs[np.arange(M)[:, None], topk_idx]  # (M, K)
    # argsort within K
    order_within_k = np.argsort(costs[np.arange(M)[:, None], topk_idx], axis=1)
    # produce sorted candidate lists (M, K)
    sorted_topk = np.take_along_axis(topk_idx, order_within_k, axis=1)

    # For each source, we will maintain a pointer into its candidate list (rank index)
    ptr = np.zeros(M, dtype=np.int32)

    # Build min-heap of (cost, source, rank_index). Initially push each source's best candidate (rank=0)
    heap = []
    for i in range(M):
        j = int(sorted_topk[i, 0])
        c = float(costs[i, j])
        heap.append((c, int(i), 0))
    heapq.heapify(heap)

    assigned_target = -np.ones(N, dtype=np.int32)   # assigned_target[j] = source or -1
    assignment = -np.ones(M, dtype=np.int32)        # assignment[i] = j
    assigned_count = 0

    while heap and assigned_count < M:
        cost_ij, i, rank = heapq.heappop(heap)
        # if this source already assigned (maybe by earlier push), skip
        if assignment[i] != -1:
            continue

        # candidate j
        j = int(sorted_topk[i, rank])

        if assigned_target[j] == -1:
            # assign it
            assignment[i] = j
            assigned_target[j] = i
            assigned_count += 1
            # optionally continue
        else:
            # target taken: advance source i to its next candidate (if any) and push back
            next_rank = rank + 1
            if next_rank < K:
                j2 = int(sorted_topk[i, next_rank])
                c2 = float(costs[i, j2])
                heapq.heappush(heap, (c2, i, next_rank))
            else:
                # exhausted top-K candidates for this source; mark for final pass by leaving assignment -1
                # we'll fill these in a later pass
                pass

    # If some sources remain unassigned, assign them greedily to any free targets by lowest cost
    if assigned_count < M:
        if verbose:
            print(f"[topk] {M-assigned_count} sources exhausted top-K candidates; filling from remaining targets...")

        free_targets = np.where(assigned_target == -1)[0].tolist()  # remaining columns
        # We'll assign each unassigned source to its best among free_targets
        unassigned_sources = np.where(assignment == -1)[0]
        # For each such source, find argmin over free_targets (vectorized-ish)
        # To avoid huge memory usage when many free targets, do it in a loop which is OK because
        # usually unassigned count is small if K sufficiently large.
        taken = set()
        for i in unassigned_sources:
            # compute best among free_targets
            row = costs[i, free_targets]
            kbest = int(np.argmin(row))
            chosen = free_targets[kbest]
            assignment[i] = chosen
            assigned_target[chosen] = i
            taken.add(chosen)
            # remove chosen from free_targets list (pop by index)
            free_targets.pop(kbest)

    # At this point assignment should be full
    assert np.all(assignment >= 0), "assignment incomplete after final pass"

    return assignment

def local_pairwise_refinement(assignment: np.ndarray,
                              costs: np.ndarray,
                              iters: int = 20000,
                              verbose: bool = False,
                              seed: int = 12345) -> np.ndarray:
    """
    Randomized local pairwise swap refinement (2-opt):
    Repeatedly sample random pair (i1, i2). If swapping their targets reduces total cost, perform swap.
    Runs for `iters` attempts (not guaranteed to find optimum; cheap local improvement).

    Args:
        assignment: current assignment array (M,) mapping source->target.
        costs: (M, N) cost matrix.
        iters: number of random swap attempts.
    Returns:
        improved assignment array (copy).
    """
    rng = np.random.RandomState(seed)
    M = assignment.shape[0]
    assign = assignment.copy()
    # Precompute current cost per source
    current_costs = costs[np.arange(M), assign]

    for t in range(iters):
        i1 = int(rng.randint(0, M))
        i2 = int(rng.randint(0, M))
        if i1 == i2:
            continue
        t1 = assign[i1]
        t2 = assign[i2]
        # cost before
        before = current_costs[i1] + current_costs[i2]
        after = costs[i1, t2] + costs[i2, t1]
        if after < before - 1e-12:
            # swap
            assign[i1], assign[i2] = t2, t1
            # update cached costs
            current_costs[i1] = costs[i1, t2]
            current_costs[i2] = costs[i2, t1]
    if verbose:
        total_cost = float(current_costs.sum())
        print(f"[refine] finished, total cost {total_cost:.4f}")
    return assign


def topk_greedy_transport(costs: np.ndarray,
                          source_colors: np.ndarray,
                          target_width: int,
                          K: int = 32,
                          refine_iters: int = 5000,
                          verbose: bool = False) -> dict:
    """
    Complete pipeline: compute top-K greedy 1:1 assignment and build hard image.
    Args:
        costs: (M, N) cost matrix
        source_colors: (M, C) colors
        target_width: width of square target image (N == target_width**2)
        K: top-K candidates per source
        refine_iters: number of local refinement swap attempts (0 disables)
    Returns:
        {'assignment': assignment, 'hard_image': hard_img}
    """
    M, N = costs.shape
    if source_colors.shape[0] != M:
        raise ValueError("source_colors shape mismatch")
    if M > N:
        raise ValueError("M > N not supported")

    if verbose:
        print(f"[topk] computing top-{K} candidates for {M} sources / {N} targets...")

    assignment = topk_greedy_assignment(costs, K=K, verbose=verbose)

    if refine_iters and refine_iters > 0:
        if verbose:
            print(f"[topk] running local refinement {refine_iters} iters...")
        assignment = local_pairwise_refinement(assignment, costs, iters=refine_iters, verbose=verbose)

    # build image
    C = source_colors.shape[1]
    dtype = source_colors.dtype
    hard_img = np.zeros((target_width, target_width, C), dtype=dtype)
    for s in range(M):
        t = int(assignment[s])
        y = t // target_width
        x = t % target_width
        hard_img[y, x, :] = source_colors[s]

    return {"assignment": assignment, "hard_image": hard_img}


# SIMULATED ANNEALING

SEED = 12345

def set_temp(t: float) -> float:
    t *= 0.99999
    return t
def should_accept(cost_before: float, cost_after: float, temp: float, rng: np.random.RandomState) -> bool:
    if cost_after - cost_before < 1e-12:
        return True
    else:
        if temp < 1e-8:
            return False
        else:
            prob = math.exp((cost_before - cost_after) / temp)
            return rng.rand() < prob

def sim_anneal(costs: np.ndarray, assignment: np.ndarray, iters: int = 20000, verbose: bool = False) -> np.ndarray:
    """
    Transition is swapping two sources' assigned targets. If swap reduces total cost, perform it. Repeat for `iters` random swaps.
    """
    rng = np.random.RandomState(SEED)
    M = assignment.shape[0]
    assign = assignment.copy()
    # Precompute current cost per source
    current_costs = costs[np.arange(M), assign]

    temp = 1e-3
    for t in range(iters):
        if verbose and t % 50000 == 0:
            print(f"[sim_anneal] iteration {t}, current cost: {float(current_costs.sum()):.4f}")
        i1 = int(rng.randint(0, M))
        i2 = int(rng.randint(0, M))
        if i1 == i2:
            continue

        temp = set_temp(temp)

        t1 = assign[i1]
        t2 = assign[i2]
        before = current_costs[i1] + current_costs[i2]
        after = costs[i1, t2] + costs[i2, t1]
        if should_accept(before, after, temp, rng):
            assign[i1], assign[i2] = t2, t1
            current_costs[i1] = costs[i1, t2]
            current_costs[i2] = costs[i2, t1]
    if verbose:
        total_cost = float(current_costs.sum())
        print(f"[sim_anneal] finished, total cost {total_cost:.4f}")
    return assign

def sim_anneal_transport(costs: np.ndarray,
                     source_colors: np.ndarray,
                     target_width: int,
                     iters: int = 1000000,
                     verbose: bool = False) -> dict:
    """
    Complete pipeline: simulated annealing refinement, then build hard image.
    """
    M, N = costs.shape
    if source_colors.shape[0] != M:
        raise ValueError("source_colors shape mismatch")
    if M > N:
        raise ValueError("M > N not supported")

    if verbose:
        print("[sim_anneal] computing initial assignment...")

    # assignment = np.arange(M, dtype=np.int32)
    assignment = topk_greedy_assignment(costs, K=128, verbose=verbose)

    if iters and iters > 0:
        if verbose:
            print(f"[sim_anneal] running local refinement {iters} iters...")
        assignment = sim_anneal(costs, assignment, iters=iters, verbose=verbose)

    if verbose:
        total_cost = float(costs[np.arange(M), assignment].sum())
        print(f"[sim_anneal] finished, total cost {total_cost:.4f}")

    # build image
    C = source_colors.shape[1]
    dtype = source_colors.dtype
    hard_img = np.zeros((target_width, target_width, C), dtype=dtype)
    for s in range(M):
        t = int(assignment[s])
        y = t // target_width
        x = t % target_width
        hard_img[y, x, :] = source_colors[s]

    return {"assignment": assignment, "hard_image": hard_img}