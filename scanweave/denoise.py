import numpy as np
from .arrays import srgb_decode, srgb_encode, normalize_weights, weighted_rgb, safe_divide, blur, luminance, smoothstep


def sigma_clip(stack, valid, sigma=2, iterations=2):
    accepted = valid[:, :, :, None] > .5
    accepted = np.broadcast_to(accepted, stack.shape).copy()
    for _ in range(iterations):
        count = accepted.sum(axis=0)
        mean = safe_divide(np.sum(stack * accepted, axis=0), count)
        variance = safe_divide(np.sum((stack - mean) ** 2 * accepted, axis=0), count)
        deviation = np.sqrt(variance)
        reject = np.abs(stack - mean) > sigma * np.maximum(deviation, 1e-5)
        candidate = accepted & ~reject
        empty = candidate.sum(axis=0) == 0
        candidate[:, empty] = accepted[:, empty]
        accepted = candidate
    mean = safe_divide(np.sum(stack * accepted, axis=0), accepted.sum(axis=0))
    rejected = 1 - accepted.mean(axis=(0, 3))
    return mean, rejected.astype(np.float32), accepted.mean(axis=3).astype(np.float32)


def detail_restoration(stack, valid, base, settings):
    """Recover frame-consistent fine detail that temporal averaging softened.

    Unlike unsharp masking, the restored signal is taken only from structure
    shared by the enabled frames (thin text edges, fine lines and paper grain),
    never from a single-frame outlier such as a hot pixel or a misaligned area.
    Disabled frames and painted exclusion regions carry a zero validity and so
    never participate in either the estimate or the confidence map.
    """
    amount = float(settings['detail_preserve'])
    if amount <= 0:
        zero = np.zeros(base.shape[:2], dtype=np.float32)
        return base.copy(), zero
    valid3 = np.clip(valid[:, :, :, None], 0, 1)
    coverage = np.clip(valid.sum(axis=0), 0, None)
    coverage3 = coverage[:, :, None]
    lowpass = blur(base, float(settings['detail_radius']))
    base_residual = base - lowpass
    residuals = stack - lowpass[None]
    # Robust per-frame noise scale from the temporal residuals, so the detail
    # threshold tracks the actual noise of this sequence rather than a guess.
    deviations = (residuals - base_residual[None]) * valid3
    scale = np.sqrt(safe_divide((deviations ** 2).sum(axis=0), np.maximum(coverage3 - 1, 1)))
    scale = np.maximum(scale, 1e-4)
    # A frame only supports a detail when its residual agrees with the shared
    # (robust mean) residual; a single-frame spike or a shifted edge fails here.
    agreement = (np.abs(residuals - base_residual[None]) <=
                 float(settings['detail_threshold']) * scale)
    support = (agreement * valid3).sum(axis=0)
    reliability = smoothstep(float(settings['detail_consistency']),
                             min(.95, float(settings['detail_consistency']) + .25),
                             safe_divide(support, coverage3))
    # Temporal signal-to-noise: real fine detail rises above the per-frame noise.
    strength = smoothstep(1., float(settings['detail_threshold']),
                          np.abs(base_residual) / np.maximum(scale, 1e-4))
    # At least two frames must agree before any detail is trusted.
    enough = coverage >= 2
    detail_mask = reliability * strength
    detail_mask *= enough[:, :, None]
    detail_mask = np.clip(detail_mask, 0, 1).astype(np.float32)
    restored = lowpass + base_residual * (1 + amount * detail_mask)
    restored = np.where(coverage3 > 1e-6, restored, base)
    mask2d = luminance(detail_mask).astype(np.float32)
    return np.clip(restored, 0, 1).astype(np.float32), mask2d


def detail_overlay(image, detail_mask):
    """Visualize where detail preservation acts, overlaid on the result image."""
    mask = np.clip(detail_mask, 0, 1)[:, :, None]
    tint = np.broadcast_to(np.array([.95, .45, .12], np.float32), image.shape)
    # Darken the page slightly so the warm detail regions read clearly.
    base = image * .55
    result = base * (1 - mask) + tint * mask
    return np.clip(result, 0, 1).astype(np.float32)


def denoise_stack(prepared, settings):
    source = prepared['images']
    images = [srgb_decode(image) for image in source] if settings['linear'] else source
    stack = np.stack(images)
    valid = np.stack(prepared['validities'])
    weights = valid * np.asarray([f['weight'] for f in prepared['frames']])[:, None, None]
    rejected = np.zeros_like(valid[0])
    method = settings['method']
    if method == 'sigma':
        image, rejected, accepted = sigma_clip(stack, weights, settings['sigma'], settings['iterations'])
        weights = accepted * weights
    elif method == 'median':
        masked = np.ma.array(stack, mask=np.broadcast_to((valid < .5)[:, :, :, None], stack.shape))
        image = np.ma.median(masked, axis=0).filled(0).astype(np.float32)
    elif method == 'trimmed':
        masked = np.where(valid[:, :, :, None] > .5, stack, np.nan)
        lo = np.nanquantile(masked, settings['trim'], axis=0)
        hi = np.nanquantile(masked, 1 - settings['trim'], axis=0)
        accepted = (stack >= lo) & (stack <= hi) & (valid[:, :, :, None] > .5)
        image = safe_divide((stack * accepted).sum(axis=0), accepted.sum(axis=0))
        rejected = 1 - accepted.mean(axis=(0, 3))
        weights = accepted.mean(axis=3) * weights
    else:
        if method == 'mean':
            weights = valid
        image = weighted_rgb(images, normalize_weights(weights))
    image = np.clip(image, 0, 1).astype(np.float32)
    image, detail_mask = detail_restoration(stack, valid, image, settings)
    variance = np.sum((stack - image[None]) ** 2 * valid[:, :, :, None], axis=0)
    variance = safe_divide(variance, valid.sum(axis=0)[:, :, None])
    if settings['linear']:
        image = srgb_encode(image)
    coverage = valid.max(axis=0)
    image[coverage < .001] = 0
    effective = safe_divide(np.sum(weights, axis=0) ** 2, np.sum(weights ** 2, axis=0))
    fields = {'image': np.clip(image, 0, 1), 'valid': coverage, 'variance': variance.mean(axis=2), 'rejection': rejected, 'detail_mask': detail_mask, 'effective_frames': effective / len(images)}
    report = {'mode': 'denoise', 'method': method, 'linear_space': settings['linear'], 'mean_effective_frames': float(effective.mean()), 'ideal_noise_reduction_factor': float(np.sqrt(np.maximum(effective.mean(), 1))), 'mean_rejected': float(rejected.mean()), 'detail_preserve': float(settings['detail_preserve']), 'mean_detail_mask': float(detail_mask[coverage > .001].mean()) if np.any(coverage > .001) else 0., 'detail_note': '细节保留仅回补多帧一致的高频残差（浅字、细线与纸张颗粒），不是整图锐化；单帧坏点、错位、停用帧与排除遮罩区域不参与估计。', 'note': '理论降噪因子假设独立同分布噪声，配准误差、压缩与运动会降低实际收益。'}
    return fields, report
