import numpy as np
from .geometry import canvas_bounds, warp, feather_weight, support_bounds, largest_rectangle, crop_arrays
from .arrays import luminance, normalize_weights, weighted_rgb, safe_divide
from .pyramids import multiband
from .validation import Invalid
from . import illumination


def estimate_gains(images, validities, reference=0):
    count = len(images)
    equations, targets = [], []
    for first in range(count):
        for second in range(first + 1, count):
            overlap = (validities[first] > .99) & (validities[second] > .99)
            a, b = luminance(images[first]), luminance(images[second])
            overlap &= (a > .03) & (a < .95) & (b > .03) & (b < .95)
            if overlap.sum() < 80:
                continue
            ratio = np.median(np.log(a[overlap]) - np.log(b[overlap]))
            equation = np.zeros(count)
            equation[first], equation[second] = 1, -1
            equations.append(equation)
            targets.append(-ratio)
    anchor = np.zeros(count)
    anchor[reference] = 10
    equations.append(anchor)
    targets.append(0)
    for index in range(count):
        regularizer = np.zeros(count)
        regularizer[index] = .01
        equations.append(regularizer)
        targets.append(0)
    logarithms, _, _, _ = np.linalg.lstsq(np.stack(equations), np.asarray(targets), rcond=None)
    return np.clip(np.exp(logarithms), .25, 4).astype(np.float32)


def _blend(fields, masks, settings):
    weights = [feather_weight(mask, settings['feather']) * frame['weight'] for mask, frame in zip(masks, fields['frames'])]
    seam_reports = []
    if settings['blend'] == 'seam':
        from .seams import seam_blend
        image, seam_reports = seam_blend(fields['images'], masks, min(40, settings['feather']))
    elif settings['blend'] == 'overwrite':
        image = np.zeros_like(fields['images'][0])
        for field, mask in zip(fields['images'], masks):
            image[mask > .5] = field[mask > .5]
    elif settings['blend'] == 'multiband':
        image = multiband(fields['images'], weights, settings['levels'], masks)
    else:
        image = weighted_rgb(fields['images'], normalize_weights(weights))
    return image, weights, seam_reports


def stitch(prepared, settings, progress=None, cancelled=None):
    width, height = prepared['size']
    size, matrices, canvas = canvas_bounds(prepared['matrices'], width, height, settings['max_edge'])
    warped, masks, geometric = [], [], []
    for index, (image, valid, support, matrix) in enumerate(zip(prepared['images'], prepared['validities'], prepared['geometric_validities'], matrices)):
        if cancelled and cancelled():
            raise InterruptedError('拼接已取消')
        warped.append(warp(image, matrix, size))
        masks.append(warp(valid, matrix, size))
        geometric.append(warp(support, matrix, size))
        if progress:
            progress(.4 + .2 * (index + 1) / len(matrices), '构建拼接画布')
    gains = estimate_gains(warped, masks, prepared['reference_index']) if settings['gain'] else np.ones(len(warped))
    warped = [np.clip(image * gain, 0, 1) for image, gain in zip(warped, gains)]
    if progress:
        progress(.62, '估计缓变亮度')
    if settings['illumination'] != 'off':
        fields, illumination_report = illumination.estimate(
            warped, masks, geometric, prepared['frames'], prepared['reference_index'],
            settings['illumination'], settings['illumination_strength'], settings['illumination_limit'],
        )
        uncorrected = warped
        corrected = illumination.apply_fields(warped, fields, size, masks)
        illumination_report['detail_preservation'] = illumination.detail_preservation(uncorrected, corrected, masks)
        if cancelled and cancelled():
            raise InterruptedError('拼接已取消')
    else:
        corrected = warped
        uncorrected = warped
        illumination_report = {'enabled': False, 'method': 'off', 'degree': 0, 'applied': False, 'strength': settings['illumination_strength'], 'limit_ev': settings['illumination_limit'], 'observations': 0, 'warnings': [], 'pairs': [], 'frames': [], 'unreliable_frames': [], 'note': '缓变亮度补偿已关闭，只使用整体增益与融合。'}
    context = {'images': corrected, 'frames': prepared['frames']}
    image, weights, seam_reports = _blend(context, masks, settings)
    # The same masks/settings blended without the smooth fields, for the page
    # comparison.  Costs one extra blend pass only when compensation was used.
    before = None
    if illumination_report.get('applied'):
        if progress:
            progress(.76, '生成补偿前对照组')
        context = {'images': uncorrected, 'frames': prepared['frames']}
        before, _, _ = _blend(context, masks, settings)
    valid = np.max(masks, axis=0)
    normalized = normalize_weights(weights)
    contributions = np.argmax(normalized, axis=0).astype(np.float32) / max(len(warped) - 1, 1)
    coverage = np.sum(np.stack(masks) > .99, axis=0).astype(np.float32) / len(masks)
    fields = {'image': np.clip(image, 0, 1), 'valid': valid, 'contribution': contributions, 'coverage': coverage}
    if before is not None:
        fields['before_illumination'] = np.clip(before, 0, 1)
    bounds = [0, 0, size[0], size[1]]
    if settings['crop'] == 'bounds':
        bounds = support_bounds(valid)
    elif settings['crop'] == 'largest_rectangle':
        bounds = largest_rectangle(valid)
    fields = crop_arrays(fields, bounds)
    if 'before_illumination' in fields:
        illumination_report['difference'] = illumination.difference_stops(fields['image'], fields['before_illumination'], fields['valid'])
    report = {'mode': 'panorama', 'canvas': canvas, 'original_canvas': list(size), 'crop': bounds, 'gain_compensation': gains.tolist(), 'blend': settings['blend'], 'illumination': illumination_report, 'note': '采用平面单应/仿射画布，不是球面全景；近景视差和滚动快门需要额外拍摄控制。'}
    report['seams'] = seam_reports
    return fields, report
