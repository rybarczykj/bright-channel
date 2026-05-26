import sys
import numpy as np
import cv2
from pathlib import Path
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


def refine_transmission(img_float, t, radius=40, eps=0.001):
    """Refine transmission with guided filter using original image as guide."""
    guide = img_float.astype(np.float32)
    if guide.ndim == 3:
        guide_gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        guide_gray = guide
    t_f32 = t.astype(np.float32)
    t_refined = cv2.ximgproc.guidedFilter(guide_gray, t_f32, radius=radius, eps=eps)
    return np.clip(t_refined.astype(np.float64), 0, 1)


def recover_scene(img_float, A, t, t0=0.1):
    """J(x) = (I(x) - A) / max(t(x), t0) + A"""
    t_clamped = np.maximum(t, t0)[:, :, None]
    J = (img_float - A[None, None, :]) / t_clamped + A[None, None, :]
    return np.clip(J, 0, 1)


def transmission_to_depth(t):
    """d(x) = -log(t(x)), relative depth (unknown beta)."""
    depth = -np.log(np.maximum(t, 1e-6))
    depth = depth / (depth.max() + 1e-6)
    return depth


def dehaze(img_float, kappa=15, omega=0.95, t0=0.1, gf_radius=40, gf_eps=0.001):
    """Full He et al. dehazing pipeline. Returns dehazed image, transmission,
    depth map, and atmospheric light."""
    dc = dark_channel(img_float, kappa)
    A = estimate_atmospheric_light(img_float, dc)
    t_raw = estimate_transmission(img_float, A, kappa, omega)
    t_refined = refine_transmission(img_float, t_raw, gf_radius, gf_eps)
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


def multiscale_bright_channel(img_float, scales=None, beta=0.1, confidence_threshold=0.5):
    """Sec. 3.2: compute bright channel at multiple scales, combine via
    geometric mean of per-scale confidence values (Eq. 10).

    Pixels with low combined confidence get the value from the smallest scale.
    """
    if scales is None:
        scales = [3, 7, 15, 31]

    n_scales = len(scales)
    h, w = img_float.shape[:2]

    bc_per_scale = []
    conf_per_scale = []

    for kappa in scales:
        bc = bright_channel(img_float, kappa)
        bc_norm = normalize_bright_channel(bc, beta)
        bc_ref = erode_bright_channel(bc_norm, kappa)

        conf = compute_confidence(img_float, bc_ref)

        bc_per_scale.append(bc_ref)
        conf_per_scale.append(conf)

    # Eq. 10: geometric mean of confidences across scales
    conf_stack = np.stack(conf_per_scale, axis=0)
    combined_conf = np.prod(conf_stack, axis=0) ** (1.0 / n_scales)

    # Use smallest-scale bright channel as base, override with larger scales
    # where confidence is high
    result = bc_per_scale[0].copy()
    for j in range(1, n_scales):
        mask = conf_per_scale[j] >= confidence_threshold
        result[mask] = bc_per_scale[j][mask]

    # Where combined confidence is low, fall back to smallest scale
    low_conf = combined_conf < confidence_threshold
    result[low_conf] = bc_per_scale[0][low_conf]

    return result, combined_conf


def compute_confidence(img_float, bc_refined, n_segments=500):
    """Sec. 3.1: per-segment shadow confidence via Eq. 9 check.
    Vectorized: precomputes per-label means, finds neighbors via label
    boundary shifts rather than per-segment dilation."""
    h, w = img_float.shape[:2]
    img_u8 = (img_float * 255).astype(np.uint8)

    slic = cv2.ximgproc.createSuperpixelSLIC(img_u8, cv2.ximgproc.SLIC, region_size=max(h, w) // 30)
    slic.iterate(10)
    labels = slic.getLabels()
    n_labels = slic.getNumberOfSuperpixels()

    # Precompute per-label mean bright channel and per-channel means
    label_bc_sum = np.zeros(n_labels)
    label_count = np.zeros(n_labels)
    label_channel_sum = np.zeros((n_labels, 3))

    np.add.at(label_bc_sum, labels.ravel(), bc_refined.ravel())
    np.add.at(label_count, labels.ravel(), 1)
    for c in range(3):
        np.add.at(label_channel_sum[:, c], labels.ravel(), img_float[:, :, c].ravel())

    label_count_safe = np.maximum(label_count, 1)
    label_bc_mean = label_bc_sum / label_count_safe
    label_channel_mean = label_channel_sum / label_count_safe[:, None]

    # Find neighbor pairs via shifted labels (right and down)
    neighbor_set = set()
    if w > 1:
        diff_h = labels[:, :-1] != labels[:, 1:]
        ys, xs = np.where(diff_h)
        for y, x in zip(ys, xs):
            a, b = labels[y, x], labels[y, x + 1]
            if a != b:
                neighbor_set.add((min(a, b), max(a, b)))
    if h > 1:
        diff_v = labels[:-1, :] != labels[1:, :]
        ys, xs = np.where(diff_v)
        for y, x in zip(ys, xs):
            a, b = labels[y, x], labels[y + 1, x]
            if a != b:
                neighbor_set.add((min(a, b), max(a, b)))

    # For each segment, count Eq. 9 violations across all neighbors
    violations = np.zeros(n_labels)
    total_pairs = np.zeros(n_labels)

    for a, b in neighbor_set:
        for c in range(3):
            # If the darker segment (by bc) has a higher channel mean, it's a violation
            if label_bc_mean[a] < label_bc_mean[b]:
                dark, light = a, b
            else:
                dark, light = b, a
            if label_channel_mean[dark, c] > label_channel_mean[light, c]:
                violations[dark] += 1
            total_pairs[dark] += 1
            total_pairs[light] += 1

    total_pairs_safe = np.maximum(total_pairs, 1)
    violation_rate = violations / total_pairs_safe

    # Confidence per label
    darkness = 1.0 - label_bc_mean
    consistency = 1.0 - violation_rate
    label_conf = np.where(
        label_bc_mean > 0.75, 0.0,
        np.where(violation_rate > 0.5, 0.0, darkness * consistency)
    )

    # Map back to pixels
    conf_map = label_conf[labels]
    conf_map = cv2.GaussianBlur(conf_map, (0, 0), sigmaX=3)

    return conf_map


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


def mrf_refine(img_float, bc_input, n_labels=32, n_iterations=5, lambda_smooth=5.0):
    """Sec. 4.3: MRF refinement via ICM (iterative conditional modes).

    Singleton: phi_i(x_i) = (x_i - bc(i))^2  (Eq. 15)
    Pairwise: psi_{i,j}(x_i, x_j) = (x_i - x_j)^2 * min_k(edge_k)^2  (Eq. 16)

    Smooths across shadow boundaries (invariants show no edge) while
    preserving texture boundaries (invariants show edges).
    """
    norm_rgb, c1c2c3, log_chrom = compute_illumination_invariants(img_float)
    invariants = [norm_rgb, c1c2c3, log_chrom]

    h, w = bc_input.shape
    labels = np.round(bc_input * (n_labels - 1)).astype(np.int32)
    label_values = np.linspace(0, 1, n_labels)

    # Precompute minimum edge strength across invariants for each pixel pair
    # For 4-connected neighbors: right and down
    min_edge_right = np.ones((h, w), dtype=np.float64) * 1e6
    min_edge_down = np.ones((h, w), dtype=np.float64) * 1e6

    for inv in invariants:
        gray = np.mean(inv, axis=2) if inv.ndim == 3 else inv
        diff_right = np.abs(gray[:, 1:] - gray[:, :-1])
        diff_down = np.abs(gray[1:, :] - gray[:-1, :])

        min_edge_right[:, :-1] = np.minimum(min_edge_right[:, :-1], diff_right)
        min_edge_down[:-1, :] = np.minimum(min_edge_down[:-1, :], diff_down)

    min_edge_right = np.minimum(min_edge_right, 1.0)
    min_edge_down = np.minimum(min_edge_down, 1.0)

    # ICM: iteratively update each pixel to minimize local energy
    for iteration in range(n_iterations):
        changed = 0
        for y in range(h):
            for x in range(w):
                current = labels[y, x]
                best_label = current
                best_energy = float('inf')

                for l in range(max(0, current - 3), min(n_labels, current + 4)):
                    val = label_values[l]
                    # Singleton (Eq. 15)
                    energy = (val - bc_input[y, x]) ** 2

                    # Pairwise with 4 neighbors (Eq. 16)
                    if x > 0:
                        nb_val = label_values[labels[y, x - 1]]
                        edge = min_edge_right[y, x - 1]
                        energy += lambda_smooth * (val - nb_val) ** 2 * edge ** 2
                    if x < w - 1:
                        nb_val = label_values[labels[y, x + 1]]
                        edge = min_edge_right[y, x]
                        energy += lambda_smooth * (val - nb_val) ** 2 * edge ** 2
                    if y > 0:
                        nb_val = label_values[labels[y - 1, x]]
                        edge = min_edge_down[y - 1, x]
                        energy += lambda_smooth * (val - nb_val) ** 2 * edge ** 2
                    if y < h - 1:
                        nb_val = label_values[labels[y + 1, x]]
                        edge = min_edge_down[y, x]
                        energy += lambda_smooth * (val - nb_val) ** 2 * edge ** 2

                    if energy < best_energy:
                        best_energy = energy
                        best_label = l

                if best_label != current:
                    labels[y, x] = best_label
                    changed += 1

        print(f"  ICM iteration {iteration + 1}/{n_iterations}: {changed} pixels changed")
        if changed == 0:
            break

    result = label_values[labels]
    return result, invariants


def mrf_refine_fast(img_float, bc_input, lambda_smooth=50.0):
    """Fast MRF approximation using guided filtering.

    Uses the minimum edge response across illumination invariants as the
    guide — same intuition as Eq. 16 but runs in milliseconds instead of
    minutes.
    """
    norm_rgb, c1c2c3, log_chrom = compute_illumination_invariants(img_float)

    # Build a guide image from minimum gradient across invariants
    guides = []
    for inv in [norm_rgb, c1c2c3, log_chrom]:
        gray = np.mean(inv, axis=2).astype(np.float32)
        guides.append(gray)

    # Use each invariant as a guide and take the result that preserves
    # the most shadow detail
    bc_f32 = bc_input.astype(np.float32)
    results = []
    for guide in guides:
        filtered = cv2.ximgproc.guidedFilter(guide, bc_f32, radius=8, eps=0.01)
        results.append(filtered)

    # Combine: for each pixel, use the guide that produced the most
    # edge-preserving result (closest to original where edges exist)
    result = np.mean(results, axis=0)

    return result.astype(np.float64), (norm_rgb, c1c2c3, log_chrom)


def process_image(image_path, kappa=15, beta=0.1):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path}")
        sys.exit(1)

    img_float = img.astype(np.float64) / 255.0

    bc = bright_channel(img_float, kappa)
    bc_norm = normalize_bright_channel(bc, beta)
    bc_refined = erode_bright_channel(bc_norm, kappa)

    return img, bc, bc_norm, bc_refined


def process_multiscale(image_path, scales=None, beta=0.1):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path}")
        sys.exit(1)

    img_float = img.astype(np.float64) / 255.0
    result, confidence = multiscale_bright_channel(img_float, scales, beta)
    return img, result, confidence


def to_u8(arr):
    if arr.ndim == 3:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def run_full_pipeline(image_path, kappa=15, beta=0.1, scales=None, use_fast_mrf=True):
    """Run entire pipeline and save all intermediate outputs."""
    if scales is None:
        scales = [3, 7, 15, 31]

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path}")
        sys.exit(1)
    img_float = img.astype(np.float64) / 255.0

    stem = Path(image_path).stem
    out_dir = Path(image_path).parent / f"{stem}_pipeline"
    out_dir.mkdir(exist_ok=True)

    # Step 1: original
    cv2.imwrite(str(out_dir / "1_original.png"), img)
    print("1. Original saved")

    # Step 2: bright channel raw
    bc = bright_channel(img_float, kappa)
    cv2.imwrite(str(out_dir / "2_bright_channel_raw.png"), to_u8(bc))
    print("2. Bright channel (raw) saved")

    # Step 3: normalized
    bc_norm = normalize_bright_channel(bc, beta)
    cv2.imwrite(str(out_dir / "3_bright_channel_normalized.png"), to_u8(bc_norm))
    print("3. Bright channel (normalized) saved")

    # Step 4: refined (eroded)
    bc_refined = erode_bright_channel(bc_norm, kappa)
    cv2.imwrite(str(out_dir / "4_bright_channel_refined.png"), to_u8(bc_refined))
    print("4. Bright channel (refined) saved")

    # Step 5: multi-scale + confidence
    ms_result, confidence = multiscale_bright_channel(img_float, scales, beta)
    cv2.imwrite(str(out_dir / "5_multiscale.png"), to_u8(ms_result))
    cv2.imwrite(str(out_dir / "5_confidence.png"), to_u8(confidence))
    print("5. Multi-scale + confidence saved")

    # Step 6: illumination invariants
    norm_rgb, c1c2c3, log_chrom = compute_illumination_invariants(img_float)
    cv2.imwrite(str(out_dir / "6a_invariant_norm_rgb.png"), to_u8(norm_rgb))
    cv2.imwrite(str(out_dir / "6b_invariant_c1c2c3.png"), to_u8(c1c2c3))
    cv2.imwrite(str(out_dir / "6c_invariant_log_chrom.png"), to_u8(log_chrom))
    print("6. Illumination invariants saved")

    # Step 7: MRF refinement
    if use_fast_mrf:
        print("7. Running MRF (guided filter approximation)...")
        mrf_result, _ = mrf_refine_fast(img_float, bc_refined)
    else:
        print("7. Running MRF (ICM, this may take a while)...")
        mrf_result, _ = mrf_refine(img_float, bc_refined)
    cv2.imwrite(str(out_dir / "7_mrf_refined.png"), to_u8(mrf_result))
    print("7. MRF refined saved")

    print(f"\nAll outputs saved to {out_dir}/")
    return out_dir


def save_results(image_path, img, bc, bc_norm, bc_refined):
    stem = Path(image_path).stem
    out_dir = Path(image_path).parent / f"{stem}_bright_channel"
    out_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(out_dir / "original.png"), img)
    cv2.imwrite(str(out_dir / "bright_channel_raw.png"), to_u8(bc))
    cv2.imwrite(str(out_dir / "bright_channel_normalized.png"), to_u8(bc_norm))
    cv2.imwrite(str(out_dir / "bright_channel_refined.png"), to_u8(bc_refined))

    print(f"Results saved to {out_dir}/")
    return out_dir


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bright_channel.py <image_path> [--multiscale] [--full] [--slow-mrf] [patch_size] [beta]")
        sys.exit(1)

    image_path = sys.argv[1]
    full_pipeline = "--full" in sys.argv
    multiscale = "--multiscale" in sys.argv
    slow_mrf = "--slow-mrf" in sys.argv

    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    beta = float(args[1]) if len(args) > 1 else 0.1

    if full_pipeline:
        scales_str = args[0] if args else "3,7,15,31"
        scales = [int(s) for s in scales_str.split(",")]
        run_full_pipeline(image_path, kappa=scales[2] if len(scales) > 2 else 15,
                          beta=beta, scales=scales, use_fast_mrf=not slow_mrf)
    elif multiscale:
        scales_str = args[0] if args else "3,7,15,31"
        scales = [int(s) for s in scales_str.split(",")]
        print(f"Multi-scale bright channel with scales={scales}, beta={beta}")
        img, result, confidence = process_multiscale(image_path, scales, beta)

        stem = Path(image_path).stem
        out_dir = Path(image_path).parent / f"{stem}_bright_channel"
        out_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(out_dir / "multiscale.png"), to_u8(result))
        cv2.imwrite(str(out_dir / "confidence.png"), to_u8(confidence))
        print(f"Results saved to {out_dir}/")
    else:
        kappa = int(args[0]) if args else 15
        img, bc, bc_norm, bc_refined = process_image(image_path, kappa, beta)
        save_results(image_path, img, bc, bc_norm, bc_refined)
