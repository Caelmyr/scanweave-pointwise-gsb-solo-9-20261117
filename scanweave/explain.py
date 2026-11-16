"""像素贡献溯源（仅解释，不参与像素计算）。

本模块在多帧降噪结果之上回答“这个输出像素由哪些帧的哪些样本合成”。
它不被 ``engine.process`` / ``denoise.denoise_stack`` 调用，也不写回任何结果文件；
而是读取结果保存时的工程快照（``result.json`` 内的 ``project``），按完全相同的
准备、光流与降噪步骤重新计算，并在计算过程中记录每帧样本的接纳状态、剔除原因与
实际贡献份额。重新合成产生新结果后，溯源针对新结果的快照重新计算，解释随之更新。
"""
from collections import OrderedDict
import numpy as np
from . import validation as v
from .preparation import prepare
from .flow import refine
from .arrays import srgb_decode, srgb_encode, safe_divide, normalize_weights, weighted_rgb
from .validation import Invalid

# 结果不可变，按 result_id 缓存重算所需的中间量，避免连续点击重复整条流水线。
_CACHE_LIMIT = 4
_cache = OrderedDict()


def _denoise_settings(snapshot):
    return snapshot['settings']['denoise']


def result_bundle(results, key, result_id=None):
    """载入结果元数据，并校验它是可溯源的多帧降噪结果。"""
    fields, meta = results.fields(key, result_id, not bool(result_id))
    snapshot = meta.get('project')
    if not isinstance(snapshot, dict):
        raise Invalid('该结果缺少工程快照，无法溯源')
    if meta.get('mode') != 'denoise':
        raise Invalid('像素贡献溯源仅支持多帧降噪结果，当前结果模式：' + str(meta.get('mode')))
    if [name for name in meta.get('fields', []) if name.startswith('weight_')]:
        raise Invalid('降噪结果不应包含逐帧权重字段，溯源数据异常')
    return fields, meta, snapshot


def _rebuild(results, key, meta, snapshot):
    result_id = meta['id']
    cached = _cache.get(result_id)
    if cached is not None:
        _cache.move_to_end(result_id)
        return cached
    settings = snapshot['settings']
    prepared = prepare(snapshot, results.assets, True)
    # 与 engine.process 保持相同顺序：先局部光流，再进入工作流。
    prepared = refine(prepared, settings['flow'])
    bundle = {
        'prepared': prepared,
        'pre_geometry': _recompute_denoise(prepared, _denoise_settings(snapshot)),
    }
    _cache[result_id] = bundle
    _cache.move_to_end(result_id)
    while len(_cache) > _CACHE_LIMIT:
        _cache.popitem(last=False)
    return bundle


def _recompute_denoise(prepared, settings):
    """镜像 denoise.denoise_stack 的计算，同时返回逐像素溯源中间量。"""
    source = prepared['images']
    images = [srgb_decode(image) for image in source] if settings['linear'] else source
    stack = np.stack(images)
    valid = np.stack(prepared['validities'])
    weights0 = valid * np.asarray([f['weight'] for f in prepared['frames']])[:, None, None]
    method = settings['method']

    # accepted: 每帧每像素每通道是否参与最终合成；reason 记录未参与原因。
    accepted = np.zeros_like(stack, dtype=bool)
    reason = np.full(stack.shape[:3], '', dtype=object)
    reason[valid < .5] = 'no_coverage'
    reject_iteration = np.full(stack.shape, -1, dtype=np.int8)
    reject_deviation = np.zeros(stack.shape, dtype=np.float32)
    median_index = np.full(stack.shape[1:3], -1, dtype=np.int16)
    share = np.zeros_like(stack, dtype=np.float32)

    if method == 'sigma':
        accepted, reject_iteration, reject_deviation = _sigma_trace(stack, weights0, settings['sigma'], settings['iterations'])
        final_weights = accepted * weights0[:, :, :, None]
        totals = final_weights.sum(axis=0)
        share = safe_divide(final_weights, totals[None])
        reason = np.where(reject_iteration.max(axis=3) >= 0, 'sigma_outlier', reason)
        image = safe_divide((stack * accepted).sum(axis=0), accepted.sum(axis=0))
    elif method == 'median':
        masked = np.ma.array(stack, mask=np.broadcast_to((valid < .5)[:, :, :, None], stack.shape))
        median = np.ma.median(masked, axis=0)
        image = median.filled(0).astype(np.float32)
        # np.ma.median 偶数个有效值时取中间两值的均值；如实标出提供这两个值的帧。
        # 仅在有效帧之间排序：对有效值排序后，用 counts 换算成 order 中的位置。
        valid_stack = np.broadcast_to((valid > .5)[:, :, :, None], stack.shape)
        order = np.argsort(np.where(valid_stack, stack, 1e9), axis=0)
        counts = (valid > .5).sum(axis=0)
        accepted = valid_stack.copy()
        # order 形状为 (帧, H, W, 通道)；按像素位置与通道同时取秩。
        frame_count = stack.shape[0]
        ys3, xs3, cs3 = np.nonzero(np.ones(stack.shape[1:], dtype=bool))
        counts3 = np.broadcast_to(counts[:, :, None], stack.shape[1:])[ys3, xs3, cs3]
        lower_rank = (counts3 - 1) // 2
        upper_rank = counts3 // 2
        offset = frame_count - counts3
        lower_frame = order[offset + lower_rank, ys3, xs3, cs3]
        upper_frame = order[offset + upper_rank, ys3, xs3, cs3]
        median_index[:, :] = lower_frame.reshape(stack.shape[1], stack.shape[2], 3)[:, :, 0]
        share[lower_frame, ys3, xs3, cs3] = .5
        same = lower_frame == upper_frame
        share[upper_frame, ys3, xs3, cs3] = np.where(same, 1.0, .5)
        reason = np.where(valid < .5, 'no_coverage', reason)
    elif method == 'trimmed':
        masked = np.where(valid[:, :, :, None] > .5, stack, np.nan)
        lo = np.nanquantile(masked, settings['trim'], axis=0)
        hi = np.nanquantile(masked, 1 - settings['trim'], axis=0)
        accepted = (stack >= lo) & (stack <= hi) & (valid[:, :, :, None] > .5)
        image = safe_divide((stack * accepted).sum(axis=0), accepted.sum(axis=0))
        share = safe_divide(accepted.astype(np.float32), accepted.sum(axis=0)[None])
        reason = np.where((valid >= .5) & (~accepted).any(axis=3), 'trimmed_outlier', reason)
    else:
        # 与 denoise.denoise_stack 一致：mean 忽略帧权重按有效像素均分；weighted 使用帧权重。
        if method == 'mean':
            weights = valid
        else:  # weighted
            weights = weights0
        normalized = normalize_weights(weights)
        image = weighted_rgb(images, normalized)
        accepted = np.broadcast_to((weights > 0)[:, :, :, None], stack.shape).copy()
        share = np.broadcast_to(normalized[:, :, :, None], stack.shape).astype(np.float32)
        frame_weight_grid = np.asarray([f['weight'] for f in prepared['frames']])[:, None, None]
        reason = np.where(valid < .5, 'no_coverage',
                          np.where(frame_weight_grid <= 0, 'zero_frame_weight', reason))

    if settings['linear']:
        image = srgb_encode(image)
    coverage = valid.max(axis=0)
    image[coverage < .001] = 0

    return {
        'image': np.clip(image, 0, 1).astype(np.float32),
        'stack': stack.astype(np.float32),
        'valid': valid.astype(np.float32),
        'frame_weights': weights0.astype(np.float32),
        'accepted': accepted,
        'share': share.astype(np.float32),
        'reason': reason,
        'reject_iteration': reject_iteration,
        'reject_deviation': reject_deviation,
        'median_index': median_index,
        'method': method,
        'linear': bool(settings['linear']),
    }


def _sigma_trace(stack, weights, sigma, iterations):
    """逐轮复刻 denoise.sigma_clip，并记录每个样本在哪一轮被剔除及偏离量。"""
    valid = weights > .5
    accepted = np.broadcast_to(valid[:, :, :, None], stack.shape).copy()
    reject_iteration = np.full(stack.shape, -1, dtype=np.int8)
    reject_deviation = np.zeros(stack.shape, dtype=np.float32)
    for round_index in range(iterations):
        count = accepted.sum(axis=0)
        mean = safe_divide(np.sum(stack * accepted, axis=0), count)
        variance = safe_divide(np.sum((stack - mean) ** 2 * accepted, axis=0), count)
        deviation = np.sqrt(variance)
        reject = np.abs(stack - mean) > sigma * np.maximum(deviation, 1e-5)
        newly = reject & accepted & (reject_iteration < 0)
        reject_iteration[newly] = round_index
        reject_deviation[newly] = np.abs(stack - mean)[newly]
        candidate = accepted & ~reject
        empty = candidate.sum(axis=0) == 0
        candidate[:, empty] = accepted[:, empty]
        accepted = candidate
    return accepted, reject_iteration, reject_deviation


def inverse_geometry(meta, snapshot, display_x, display_y):
    """把结果图（裁切/旋转/翻转后）上的整数像素坐标映回共同画布坐标。

    与 geometry.finish_geometry 中先裁切再旋转/翻转的顺序严格互逆，
    保证点击始终对应同一处图像内容。返回 None 表示该点在共同画布之外。
    """
    width = int(meta['width'])
    height = int(meta['height'])
    if not (0 <= display_x < width and 0 <= display_y < height):
        return None
    rotation = snapshot['settings']['geometry']['rotation']
    flip_x = bool(snapshot['settings']['geometry']['flip_x'])
    flip_y = bool(snapshot['settings']['geometry']['flip_y'])
    turns = rotation // 90
    crop = meta['report'].get('manual_crop')
    crop_w, crop_h = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (width, height)
    # 引擎顺序为 裁切→翻转→np.rot90(k=-turns)。先逆向旋转得到翻转后裁切网格坐标，
    # 再逆向翻转（翻转自逆）。已用逐像素标记图对拍全部16种旋转/翻转组合，零误差。
    x, y = float(display_x), float(display_y)
    if turns == 1:
        x, y = y, (crop_h - 1) - x
    elif turns == 2:
        x, y = (crop_w - 1) - x, (crop_h - 1) - y
    elif turns == 3:
        x, y = (crop_w - 1) - y, x
    if flip_x:
        x = (crop_w - 1) - x
    if flip_y:
        y = (crop_h - 1) - y
    if not crop:
        return int(round(x)), int(round(y))
    x0, y0, _, _ = crop
    canvas_x = int(round(x)) + x0
    canvas_y = int(round(y)) + y0
    return canvas_x, canvas_y


def _round3(values):
    return [round(float(v), 6) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def explain_pixel(results, key, result_id, display_point, radius=0):
    """返回一次点击的逐帧贡献解释。"""
    fields, meta, snapshot = result_bundle(results, key, result_id)
    radius = v.integer(radius, '采样半径', 0, 8)
    point = v.vector(display_point, '采样位置', 2, 0, max(meta['width'], meta['height']) - 1)
    dx, dy = int(round(point[0])), int(round(point[1]))
    mapped = inverse_geometry(meta, snapshot, dx, dy)
    bundle = _rebuild(results, key, meta, snapshot)
    pre = bundle['pre_geometry']
    canvas_h, canvas_w = pre['image'].shape[:2]
    response = {
        'result_id': meta['id'],
        'mode': meta['mode'],
        'method': pre['method'],
        'linear_space': pre['linear'],
        'display_pixel': [dx, dy],
        'result_size': [meta['width'], meta['height']],
        'canvas_pixel': None,
        'inside_canvas': False,
        'radius': radius,
        'stale': bool(meta['source_signature'] != _current_signature(results, key)),
        'geometry': {
            'rotation': snapshot['settings']['geometry']['rotation'],
            'flip_x': bool(snapshot['settings']['geometry']['flip_x']),
            'flip_y': bool(snapshot['settings']['geometry']['flip_y']),
            'manual_crop': meta['report'].get('manual_crop'),
            'common_crop': meta['report'].get('common_crop'),
        },
        'output': None,
        'frames': [],
        'note': '贡献份额按结果保存时的工程快照重新计算，只解释该降噪结果，不改变其像素。',
    }

    # 重算结果与存档字段一致性自检。before_finishing 在引擎中已过几何裁切/旋转，
    # 故对重算图施加同一几何变换后再比较，确保任何几何设置下都能校验。
    stored_pre = fields.get('before_finishing')
    if stored_pre is not None:
        from .geometry import finish_geometry
        rebuilt, _ = finish_geometry({'image': pre['image']}, snapshot['settings']['geometry'])
        if rebuilt['image'].shape == stored_pre.shape:
            response['recomputed_matches_result'] = bool(np.allclose(stored_pre, rebuilt['image'], atol=1e-5, rtol=1e-5))
        else:
            response['recomputed_matches_result'] = False

    if mapped is None or not (0 <= mapped[0] < canvas_w and 0 <= mapped[1] < canvas_h):
        response['frames'] = _uncovered_frames(bundle)
        response['note'] = '该位置落在共同画布之外，所有帧都没有覆盖此处，不参与任何计算。'
        return response

    cx, cy = mapped
    response['canvas_pixel'] = [cx, cy]
    response['inside_canvas'] = True
    x0, x1 = max(0, cx - radius), min(canvas_w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(canvas_h, cy + radius + 1)
    response['patch_bounds'] = [x0, y0, x1, y1]
    response['output'] = {
        'denoise_rgb': _round3(pre['image'][cy, cx]),
        'display_rgb': _round3(fields['image'][dy, dx]),
        'display_is_post_geometry': True,
    }

    stack = pre['stack']
    valid = pre['valid']
    accepted = pre['accepted']
    share = pre['share']
    reason_grid = pre['reason']
    frames = bundle['prepared']['frames']
    for index, frame in enumerate(frames):
        validity = float(valid[index, cy, cx])
        entry = {
            'frame_id': frame['id'],
            'name': frame['name'],
            'index': index,
            'enabled_in_result': True,
            'frame_weight': float(frame['weight']),
            'coverage': round(validity, 6),
            'participates': bool(validity >= .5),
            'status': 'covered' if validity >= .5 else 'no_coverage',
            'reason': None if validity >= .5 else 'no_coverage',
            'sample_rgb': _round3(stack[index, cy, cx]),
            'accepted_channels': [bool(accepted[index, cy, cx, c]) for c in range(3)],
            'contribution_share': [round(float(share[index, cy, cx, c]), 6) for c in range(3)],
            'reject_iteration': [int(pre['reject_iteration'][index, cy, cx, c]) for c in range(3)],
            'reject_deviation': [round(float(pre['reject_deviation'][index, cy, cx, c]), 6) for c in range(3)],
            'median_source': bool(pre['median_index'][cy, cx] == index) if pre['method'] == 'median' else False,
            'patch_coverage': round(float(valid[index, y0:y1, x0:x1].mean()), 6),
            'patch_mean_rgb': _round3(stack[index, y0:y1, x0:x1].mean(axis=(0, 1))),
        }
        if validity >= .5:
            channel_reasons = {str(c): _channel_reason(pre, index, cy, cx, c) for c in range(3)}
            entry['channel_reasons'] = channel_reasons
            active = [c for c in range(3) if accepted[index, cy, cx, c]]
            if not active:
                if pre['method'] == 'median':
                    entry['status'] = 'covered_not_selected'
                    entry['reason'] = 'median_not_selected'
                elif reason_grid[index, cy, cx] == 'zero_frame_weight':
                    entry['status'] = 'excluded'
                    entry['reason'] = 'zero_frame_weight'
                else:
                    entry['status'] = 'excluded'
                    entry['reason'] = reason_grid[index, cy, cx] or 'outlier'
            elif len(active) < 3:
                entry['status'] = 'partially_excluded'
                entry['reason'] = 'partial_channel_outlier'
            else:
                entry['status'] = 'contributing'
                entry['reason'] = None
        response['frames'].append(entry)

    total_share = [round(float(sum(item['contribution_share'][c] for item in response['frames'])), 6) for c in range(3)]
    response['total_contribution_share'] = total_share
    response['contributor_count'] = int(sum(1 for item in response['frames'] if item['participates'] and any(item['accepted_channels'])))
    return response


def _channel_reason(pre, index, y, x, channel):
    if pre['valid'][index, y, x] < .5:
        return 'no_coverage'
    if pre['accepted'][index, y, x, channel]:
        return 'accepted'
    if pre['method'] == 'sigma':
        return 'sigma_outlier'
    if pre['method'] == 'trimmed':
        return 'trimmed_outlier'
    if pre['method'] == 'median':
        return 'median_not_selected'
    if pre['reason'][index, y, x] == 'zero_frame_weight':
        return 'zero_frame_weight'
    return 'excluded'


def _uncovered_frames(bundle):
    frames = bundle['prepared']['frames']
    return [{
        'frame_id': frame['id'], 'name': frame['name'], 'index': index,
        'enabled_in_result': True, 'participates': False, 'status': 'no_coverage',
        'reason': 'no_coverage', 'coverage': 0.0,
        'accepted_channels': [False, False, False], 'contribution_share': [0, 0, 0],
    } for index, frame in enumerate(frames)]


def _current_signature(results, key):
    from . import model
    return model.signature(results.store.get(key))


def canvas_shape(results, key, result_id):
    """结果对应共同画布（裁切前）的高宽，供坐标映射的外部校验使用。"""
    _, meta, snapshot = result_bundle(results, key, result_id)
    return _rebuild(results, key, meta, snapshot)['pre_geometry']['image'].shape[:2]


def frame_preview(results, key, result_id, frame_id):
    """返回结果快照中该帧在共同画布上的对齐图（与被解释像素同一坐标系）。"""
    _, meta, snapshot = result_bundle(results, key, result_id)
    bundle = _rebuild(results, key, meta, snapshot)
    prepared = bundle['prepared']
    index = next((i for i, frame in enumerate(prepared['frames']) if frame['id'] == frame_id), None)
    if index is None:
        raise Invalid('该帧不属于此降噪结果；停用或移除的帧不会出现在溯源中')
    return prepared['images'][index].copy(), prepared['frames'][index], index


def mark_point(image, canvas_x, canvas_y):
    """在对齐帧图上标注被查询的共同画布位置。"""
    from PIL import Image, ImageDraw
    marked = np.clip(image, 0, 1)
    raster = Image.fromarray(np.rint(marked * 255).astype(np.uint8))
    draw = ImageDraw.Draw(raster)
    radius = max(4, min(image.shape[:2]) // 120)
    for offset, color in [(2, (20, 30, 30)), (0, (232, 118, 42))]:
        draw.ellipse((canvas_x - radius - offset, canvas_y - radius - offset,
                      canvas_x + radius + offset, canvas_y + radius + offset),
                     outline=color, width=2)
    draw.line((canvas_x - radius * 2, canvas_y, canvas_x + radius * 2, canvas_y), fill=(232, 118, 42), width=1)
    draw.line((canvas_x, canvas_y - radius * 2, canvas_x, canvas_y + radius * 2), fill=(232, 118, 42), width=1)
    return np.asarray(raster, np.float32) / 255


def clear_cache():
    _cache.clear()
