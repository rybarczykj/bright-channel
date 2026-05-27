import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import cg
from skimage.segmentation import felzenszwalb
from sklearn.mixture import GaussianMixture


def bright_channel(img, kappa=15):
    """Eq. 1: I_bright(i) = max over channels of max over patch."""
    channel_max = np.max(img, axis=2)
    kernel = np.ones((kappa, kappa), np.uint8)
    return cv2.dilate(channel_max, kernel)


def dark_channel(img, kappa=15):
    """Dark channel prior (He et al. 2009): min over channels, min over patch."""
    channel_min = np.min(img, axis=2)
    kernel = np.ones((kappa, kappa), np.uint8)
    return cv2.erode(channel_min, kernel)


def estimate_atmospheric_light(img_float, dc):
    """He et al.: pick top 0.1% brightest pixels in dark channel,
    then find highest intensity pixel among those in original image."""
    flat_dc = dc.ravel()
    n_pixels = len(flat_dc)
    n_top = max(int(n_pixels * 0.001), 1)
    top_indices = np.argpartition(flat_dc, -n_top)[-n_top:]

    h, w = dc.shape
    ys, xs = np.unravel_index(top_indices, (h, w))
    intensities = np.sum(img_float[ys, xs], axis=1)
    best = np.argmax(intensities)
    A = img_float[ys[best], xs[best]]
    return A


def estimate_transmission(img_float, A, kappa=15, omega=0.95):
    """t(x) = 1 - omega * dark_channel(I / A)"""
    normalized = img_float / np.maximum(A[None, None, :], 1e-6)
    dc = dark_channel(normalized, kappa)
    t = 1.0 - omega * dc
    return t


def refine_transmission(img_float, t, radius=40, eps=0.001, color_guide=True):
    """Refine transmission with guided filter."""
    guide = img_float.astype(np.float32)
    if not color_guide and guide.ndim == 3:
        guide = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_f32 = t.astype(np.float32)
    t_refined = cv2.ximgproc.guidedFilter(guide, t_f32, radius=radius, eps=eps)
    return np.clip(t_refined.astype(np.float64), 0, 1)


MATTING_PROGRESS = {'pct': 0, 'stage': ''}


def _matting_laplacian(img, win_size=1, eps=1e-7):
    """Levin et al. closed-form matting Laplacian (vectorized)."""
    h, w, c = img.shape
    n = h * w
    win = 2 * win_size + 1
    nwin = win * win

    MATTING_PROGRESS['stage'] = 'Extracting patches'
    MATTING_PROGRESS['pct'] = 5

    # Extract all patches: (num_patches_y, num_patches_x, win, win, c)
    patches = np.lib.stride_tricks.sliding_window_view(img, (win, win, c)).squeeze(axis=2)
    ph, pw = patches.shape[0], patches.shape[1]
    N = ph * pw
    # Reshape to (N, nwin, c)
    P = patches.reshape(N, nwin, c).astype(np.float64)

    MATTING_PROGRESS['stage'] = 'Computing covariances'
    MATTING_PROGRESS['pct'] = 15

    mu = P.mean(axis=1)  # (N, c)
    P_centered = P - mu[:, None, :]  # (N, nwin, c)

    # Covariance: (N, c, c)
    cov = np.einsum('npc,npd->ncd', P_centered, P_centered) / nwin
    cov += (eps / nwin) * np.eye(c)[None]

    MATTING_PROGRESS['stage'] = 'Inverting covariances'
    MATTING_PROGRESS['pct'] = 30

    cov_inv = np.linalg.inv(cov)  # (N, c, c)

    MATTING_PROGRESS['stage'] = 'Computing window values'
    MATTING_PROGRESS['pct'] = 40

    # term[n, i, j] = P_centered[n,i,:] @ cov_inv[n] @ P_centered[n,j,:]^T
    tmp = np.einsum('npc,ncd->npd', P_centered, cov_inv)  # (N, nwin, c)
    term = np.einsum('npc,nqc->npq', tmp, P_centered)  # (N, nwin, nwin)

    eye = np.eye(nwin)[None]  # (1, nwin, nwin)
    win_vals = eye - (1.0 + term) / nwin  # (N, nwin, nwin)

    MATTING_PROGRESS['stage'] = 'Building sparse indices'
    MATTING_PROGRESS['pct'] = 60

    # Build pixel indices for each patch
    yi_range = np.arange(win_size, h - win_size)
    xi_range = np.arange(win_size, w - win_size)
    yi_grid, xi_grid = np.meshgrid(yi_range, xi_range, indexing='ij')
    # For each patch center, compute the win*win pixel indices
    dy, dx = np.meshgrid(np.arange(-win_size, win_size + 1),
                         np.arange(-win_size, win_size + 1), indexing='ij')
    dy = dy.ravel()  # (nwin,)
    dx = dx.ravel()

    centers_y = yi_grid.ravel()  # (N,)
    centers_x = xi_grid.ravel()
    # pixel indices: (N, nwin)
    pix_y = centers_y[:, None] + dy[None, :]
    pix_x = centers_x[:, None] + dx[None, :]
    pix_idx = pix_y * w + pix_x  # (N, nwin)

    # Build COO sparse matrix
    row_all = np.repeat(pix_idx, nwin, axis=1)  # (N, nwin*nwin)
    col_all = np.tile(pix_idx, (1, nwin)).reshape(N, nwin, nwin).transpose(0, 2, 1).reshape(N, nwin * nwin)

    # Wait — simpler: for each patch n, row[i]*nwin+j maps to (pix_idx[n,i], pix_idx[n,j])
    ri = np.repeat(pix_idx[:, :, None], nwin, axis=2)  # (N, nwin, nwin)
    ci = np.repeat(pix_idx[:, None, :], nwin, axis=1)  # (N, nwin, nwin)

    MATTING_PROGRESS['stage'] = 'Assembling matrix'
    MATTING_PROGRESS['pct'] = 75

    mask = np.abs(win_vals) > 1e-10
    rows = ri[mask]
    cols = ci[mask]
    vals = win_vals[mask]

    MATTING_PROGRESS['pct'] = 80
    L = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L


def refine_transmission_matting(img_float, t, lam=1e-4):
    """Refine transmission using Levin et al. closed-form matting Laplacian.
    Solves: (L + lambda*I) * t_refined = lambda * t"""
    MATTING_PROGRESS['pct'] = 0
    MATTING_PROGRESS['stage'] = 'Starting'
    h, w = t.shape
    n = h * w
    L = _matting_laplacian(img_float, win_size=1)
    MATTING_PROGRESS['pct'] = 85
    MATTING_PROGRESS['stage'] = 'Solving (conjugate gradient)'
    I_sp = sparse.eye(n)
    A = L + lam * I_sp
    b = lam * t.ravel()
    x, _ = cg(A, b, x0=t.ravel(), rtol=1e-5, maxiter=500)
    MATTING_PROGRESS['pct'] = 100
    MATTING_PROGRESS['stage'] = 'Done'
    return np.clip(x.reshape(h, w), 0, 1)


def recover_scene(img_float, A, t, t0=0.1):
    """J(x) = (I(x) - A) / max(t(x), t0) + A"""
    t_clamped = np.maximum(t, t0)[:, :, None]
    J = (img_float - A[None, None, :]) / t_clamped + A[None, None, :]
    return np.clip(J, 0, 1)


def transmission_to_depth(t):
    """d(x) = -log(t(x)), relative depth. Near=1 (white), far=0 (dark)."""
    depth = -np.log(np.maximum(t, 1e-6))
    depth = depth / (depth.max() + 1e-6)
    return 1.0 - depth


def dehaze(img_float, kappa=15, omega=0.95, t0=0.1, gf_radius=40, gf_eps=0.001,
           color_guide=True, use_matting=False):
    """Full He et al. dehazing pipeline. Returns dehazed image, transmission,
    depth map, and atmospheric light."""
    dc = dark_channel(img_float, kappa)
    A = estimate_atmospheric_light(img_float, dc)
    t_raw = estimate_transmission(img_float, A, kappa, omega)
    if use_matting:
        t_refined = refine_transmission_matting(img_float, t_raw)
    else:
        t_refined = refine_transmission(img_float, t_raw, gf_radius, gf_eps, color_guide)
    J = recover_scene(img_float, A, t_refined, t0)
    depth = transmission_to_depth(t_refined)
    return J, t_raw, t_refined, depth, A, dc


def normalize_bright_channel(bc, beta=0.1):
    """Eq. 6: normalize so top beta% of pixels map to 1.0, then erode."""
    flat = bc.flatten()
    flat_sorted = np.sort(flat)[::-1]
    idx = max(int(len(flat_sorted) * beta) - 1, 0)
    white_point = flat_sorted[idx]
    if white_point < 1e-6:
        return bc
    normalized = np.minimum(bc / white_point, 1.0)
    return normalized


def erode_bright_channel(bc, kappa=15):
    """Expand dark regions by kappa/2 to undo dilation artifact from Eq. 1."""
    half_k = kappa // 2
    if half_k < 1:
        return bc
    kernel = np.ones((half_k * 2 + 1, half_k * 2 + 1), np.uint8)
    bc_uint8 = (bc * 255).astype(np.uint8)
    eroded = cv2.erode(bc_uint8, kernel)
    return eroded.astype(np.float64) / 255.0


def compute_illumination_invariants(img_float):
    """Sec. 4.1-4.2: three illumination-invariant representations."""
    eps = 1e-6

    # 1. Normalized RGB: each channel / sum of channels
    channel_sum = np.sum(img_float, axis=2, keepdims=True)
    norm_rgb = img_float / np.maximum(channel_sum, eps)

    # 2. c1c2c3 (Eq. 11): c_k = arctan(rho_k / max(rho_{(k+1)mod3}, rho_{(k+2)mod3}))
    # OpenCV is BGR
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    c1 = np.arctan2(r, np.maximum(g, b) + eps)
    c2 = np.arctan2(g, np.maximum(r, b) + eps)
    c3 = np.arctan2(b, np.maximum(r, g) + eps)
    c1c2c3 = np.stack([c1, c2, c3], axis=2)
    c1c2c3 = (c1c2c3 - c1c2c3.min()) / (c1c2c3.max() - c1c2c3.min() + eps)

    # 3. Log-chromaticity (simplified 1d invariant, Eq. 12-13)
    log_rgb = np.log(img_float + eps)
    log_mean = np.mean(log_rgb, axis=2, keepdims=True)
    log_chrom = log_rgb - log_mean
    log_chrom = (log_chrom - log_chrom.min()) / (log_chrom.max() - log_chrom.min() + eps)

    return norm_rgb, c1c2c3, log_chrom



def to_u8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _per_segment_means(img_float, bc_refined, hue, labels, n_labels):
    """Vectorized per-segment mean computation for RGB, bright channel, and hue."""
    h, w = labels.shape
    flat_labels = labels.ravel()

    seg_rgb_sum = np.zeros((n_labels, 3), dtype=np.float64)
    seg_bc_sum = np.zeros(n_labels, dtype=np.float64)
    seg_hue_sum = np.zeros(n_labels, dtype=np.float64)
    seg_count = np.zeros(n_labels, dtype=np.float64)

    for c in range(3):
        np.add.at(seg_rgb_sum[:, c], flat_labels, img_float[:, :, c].ravel())
    np.add.at(seg_bc_sum, flat_labels, bc_refined.ravel())
    np.add.at(seg_hue_sum, flat_labels, hue.ravel())
    np.add.at(seg_count, flat_labels, 1)

    seg_count_safe = np.maximum(seg_count, 1)
    seg_rgb_mean = seg_rgb_sum / seg_count_safe[:, None]
    seg_bc_mean = seg_bc_sum / seg_count_safe
    seg_hue_mean = seg_hue_sum / seg_count_safe

    return seg_rgb_mean, seg_bc_mean, seg_hue_mean, seg_count


def _find_neighbor_pairs(labels):
    """Find all unique (a, b) neighbor segment pairs with a < b."""
    h, w = labels.shape
    pairs = set()
    if w > 1:
        mask = labels[:, :-1] != labels[:, 1:]
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            a, b = int(labels[y, x]), int(labels[y, x + 1])
            pairs.add((min(a, b), max(a, b)))
    if h > 1:
        mask = labels[:-1, :] != labels[1:, :]
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            a, b = int(labels[y, x]), int(labels[y + 1, x])
            pairs.add((min(a, b), max(a, b)))
    return pairs


def _fit_gmm_to_histogram(hist_values, max_components=3):
    """Fit a GMM to a set of values, selecting n_components via quasi-AIC."""
    if len(hist_values) < 10:
        return None
    X = hist_values.reshape(-1, 1)
    best_aic = np.inf
    best_gmm = None
    for k in range(1, min(max_components + 1, len(hist_values) // 5 + 1)):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  max_iter=50, random_state=0)
            gmm.fit(X)
            aic = gmm.aic(X)
            if aic < best_aic:
                best_aic = aic
                best_gmm = gmm
        except Exception:
            continue
    return best_gmm


def shadow_segmentation(img_float, bc_refined, felz_scale=200, felz_sigma=0.8,
                        felz_min_size=50, n_segmentations=3, theta_e=1.2):
    """TPAMI Section 5.2: shadow detection via segmentation + histogram confidence.

    Uses per-segment means instead of per-pixel semicircular patches for speed.
    For each neighboring segment pair, compares mean RGB, bright channel ratio,
    and hue difference — same features, vectorized over segments not pixels.

    Returns:
        confidence_map: per-pixel shadow confidence [0, 1]
        labels_vis: colored segmentation for visualization
        shadow_intensity: per-pixel estimated shadow intensity
        q_cand_map: per-pixel "good candidate" score
    """
    h, w = img_float.shape[:2]
    img_u8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float64)
    hue = hsv[:, :, 0] / 180.0

    conf_maps = []
    last_labels = None
    last_q_cand = None

    scales = [felz_scale * (0.5 + i) for i in range(n_segmentations)]

    for scale in scales:
        labels = felzenszwalb(img_u8[:, :, ::-1], scale=scale,
                              sigma=felz_sigma, min_size=felz_min_size)
        n_labels = labels.max() + 1

        seg_rgb, seg_bc, seg_hue, seg_count = _per_segment_means(
            img_float, bc_refined, hue, labels, n_labels)

        neighbor_pairs = _find_neighbor_pairs(labels)
        if not neighbor_pairs:
            continue

        pairs_arr = np.array(list(neighbor_pairs))
        a_ids, b_ids = pairs_arr[:, 0], pairs_arr[:, 1]

        # For each pair, determine which is darker (inside) by bright channel
        bc_a = seg_bc[a_ids]
        bc_b = seg_bc[b_ids]
        a_darker = bc_a < bc_b
        in_ids = np.where(a_darker, a_ids, b_ids)
        out_ids = np.where(a_darker, b_ids, a_ids)

        bc_in = seg_bc[in_ids]
        bc_out = seg_bc[out_ids]
        hue_in = seg_hue[in_ids]
        hue_out = seg_hue[out_ids]
        rgb_in = seg_rgb[in_ids]
        rgb_out = seg_rgb[out_ids]

        # Filter by edge ratio threshold
        bc_ratio = np.where(bc_out > 1e-6, bc_in / bc_out, 1.0)
        valid = (bc_ratio > 1.0 / theta_e) & (bc_ratio < theta_e)

        if np.sum(valid) < 5:
            continue

        bc_ratio_v = bc_ratio[valid]
        hue_diff_v = (hue_in - hue_out)[valid]
        rgb_in_v = rgb_in[valid]
        rgb_out_v = rgb_out[valid]
        in_ids_v = in_ids[valid]

        # Eq. 27: q(i) = 1 if all RGB channels darker inside
        q_per_pair = np.all(rgb_in_v < rgb_out_v, axis=1).astype(np.float64)

        # Per-segment q_cand: average q over all pairs involving each segment
        q_cand = np.zeros(n_labels)
        q_count = np.zeros(n_labels)
        np.add.at(q_cand, in_ids_v, q_per_pair)
        np.add.at(q_count, in_ids_v, 1)
        q_count_safe = np.maximum(q_count, 1)
        q_cand = q_cand / q_count_safe

        # Good candidates: pairs from segments with high q_cand
        weights = q_cand[in_ids_v]
        good_mask = weights > 0.3

        if np.sum(good_mask) < 5:
            continue

        # Fit GMMs to good-candidate distributions
        gmm_bc = _fit_gmm_to_histogram(bc_ratio_v[good_mask])
        gmm_hue = _fit_gmm_to_histogram(hue_diff_v[good_mask])

        # Eq. 28: per-segment confidence from GMM
        p_bright = np.zeros(n_labels)
        p_hue = np.zeros(n_labels)

        # Vectorized: score all valid border pairs, then aggregate per segment
        if gmm_bc is not None:
            all_bc_scores = np.exp(gmm_bc.score_samples(bc_ratio_v.reshape(-1, 1)))
        else:
            all_bc_scores = np.zeros(len(bc_ratio_v))

        if gmm_hue is not None:
            all_hue_scores = np.exp(gmm_hue.score_samples(hue_diff_v.reshape(-1, 1)))
        else:
            all_hue_scores = np.zeros(len(hue_diff_v))

        # Take max score per segment
        for idx in range(len(in_ids_v)):
            sid = in_ids_v[idx]
            if all_bc_scores[idx] > p_bright[sid]:
                p_bright[sid] = all_bc_scores[idx]
            if all_hue_scores[idx] > p_hue[sid]:
                p_hue[sid] = all_hue_scores[idx]

        if p_bright.max() > 0:
            p_bright /= p_bright.max()
        if p_hue.max() > 0:
            p_hue /= p_hue.max()

        # Eq. 29
        p_combined = q_cand * (p_bright + p_hue) / 2.0

        conf_maps.append(p_combined[labels])
        last_labels = labels
        last_q_cand = q_cand

    if not conf_maps:
        return (np.zeros((h, w)), np.zeros((h, w)),
                np.zeros((h, w)), np.zeros((h, w)))

    confidence_map = np.mean(conf_maps, axis=0)
    confidence_map = confidence_map / (confidence_map.max() + 1e-6)

    shadow_intensity = (1.0 - bc_refined) * confidence_map
    q_cand_map = last_q_cand[last_labels]

    return confidence_map, last_labels, shadow_intensity, q_cand_map


def colorize_segments(img_float, labels, confidence_map, style='random_tinted'):
    """Color segments with different styles."""
    n_labels = labels.max() + 1
    h, w = labels.shape

    if style == 'mean_color':
        seg_rgb_sum = np.zeros((n_labels, 3), dtype=np.float64)
        seg_count = np.zeros(n_labels, dtype=np.float64)
        flat_labels = labels.ravel()
        for c in range(3):
            np.add.at(seg_rgb_sum[:, c], flat_labels, img_float[:, :, c].ravel())
        np.add.at(seg_count, flat_labels, 1)
        seg_mean = seg_rgb_sum / np.maximum(seg_count, 1)[:, None]
        vis = (np.clip(seg_mean[labels], 0, 1) * 255).astype(np.uint8)
    elif style == 'gray_random':
        rng = np.random.RandomState(42)
        grays = rng.randint(80, 220, size=n_labels).astype(np.uint8)
        g = grays[labels]
        vis = np.stack([g, g, g], axis=2)
    elif style == 'gray_weighted':
        seg_conf_sum = np.zeros(n_labels, dtype=np.float64)
        seg_count = np.zeros(n_labels, dtype=np.float64)
        flat_labels = labels.ravel()
        np.add.at(seg_conf_sum, flat_labels, confidence_map.ravel())
        np.add.at(seg_count, flat_labels, 1)
        seg_conf = seg_conf_sum / np.maximum(seg_count, 1)
        g = (100 + seg_conf * 155).astype(np.uint8)
        g_map = g[labels]
        vis = np.stack([g_map, g_map, g_map], axis=2)
    else:
        rng = np.random.RandomState(42)
        colors = rng.randint(60, 220, size=(n_labels, 3)).astype(np.uint8)
        vis = colors[labels]

        if style == 'random_tinted':
            conf_3ch = confidence_map[:, :, None]
            shadow_tint = np.array([0, 0, 200], dtype=np.float64)
            vis = vis.astype(np.float64) * (1 - conf_3ch * 0.7) + shadow_tint * conf_3ch * 0.7
            vis = np.clip(vis, 0, 255).astype(np.uint8)

    edges = np.zeros((h, w), dtype=bool)
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    vis[edges] = [255, 255, 255]

    return vis


