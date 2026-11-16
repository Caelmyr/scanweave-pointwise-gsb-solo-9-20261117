"""Scratch annotation and texture-preserving retouching.

Marks are stored in the editable project document using normalized
coordinates, so saving the project keeps them editable and they participate in
the ordinary revision and undo history.  Repairs are rendered into separate
result records; scan assets and composite results are never overwritten.

Retouching is directional rather than a flat wipe.  For every marked pixel
several rays are cast and the first clean pixel on both sides is sampled; the
ray whose two samples agree best *and* carry visible structure (ink, an edge)
supplies the value.  A thin text stroke or an edge crossing the scratch is
therefore continued along its own direction instead of following the scratch
normal.  Paper grain is carried over from unmarked neighbours, and dark ink
that continues across the band keeps its full strength.  Marks wider than a
thin scratch fall back to a same-texture transfer from an offset that matches
the surrounding border, and only a region with no usable texture at all is
completed smoothly.  Pixels with zero brush coverage are never touched.
"""
import hashlib
import json
import time
from copy import deepcopy
from threading import RLock
import cv2
import numpy as np
from scipy import ndimage as ndi
from . import model
from . import validation as v
from .arrays import luminance

MAX_STROKES = 2000
MAX_POINTS = 4000
DEFAULT_SETTINGS = {'expand': .0015, 'feather': .003, 'search': 32, 'method': 'texture'}
MAX_CROSSING = 48
DIRECTIONS = 16
# Marks wider than this (in pixels) are too broad for two-sided
# interpolation, so they use texture transfer instead.
WIDE_MARK = 12


def default_state():
    return {'strokes': [], 'settings': dict(DEFAULT_SETTINGS), 'confirmed': None}


def settings_spec(body):
    body = v.object_value(body, '修复设置')
    return {
        'expand': v.number(body.get('expand', DEFAULT_SETTINGS['expand']), '修复范围外扩', 0, .05),
        'feather': v.number(body.get('feather', DEFAULT_SETTINGS['feather']), '边缘融合宽度', 0, .02),
        'search': v.integer(body.get('search', DEFAULT_SETTINGS['search']), '纹理取样搜索范围', 8, 160),
        'method': v.choice(body.get('method', DEFAULT_SETTINGS['method']), '修复方式', ['texture', 'smooth']),
    }


def stroke_spec(item):
    item = v.object_value(item, '标记笔画')
    points = [v.vector(point, '标记点', 2, 0, 1) for point in v.list_value(item.get('points'), '标记点', MAX_POINTS)]
    if not points:
        raise v.Invalid('标记笔画至少需要一个点')
    identifier = item.get('id')
    if identifier is None:
        identifier = model.uid()
    else:
        v.identifier(identifier)
    return {
        'id': identifier,
        'mode': v.choice(item.get('mode', 'paint'), '标记方式', ['paint', 'erase']),
        'radius': v.number(item.get('radius', .012), '笔刷半径', .0005, .25),
        'points': points,
    }


def validate_state(state):
    state = v.object_value(state if state is not None else {}, '修复状态')
    strokes = [stroke_spec(item) for item in v.list_value(state.get('strokes', []), '标记笔画', MAX_STROKES)]
    confirmed = state.get('confirmed')
    if confirmed is not None:
        v.identifier(confirmed)
    return {'strokes': strokes, 'settings': settings_spec(state.get('settings', {})), 'confirmed': confirmed}


def state_of(project):
    return validate_state(project.get('restoration'))


def signature(strokes, settings):
    payload = json.dumps({'strokes': strokes, 'settings': settings}, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def coverage(shape, strokes, expand=0):
    """Soft coverage of drawn marks; eraser strokes subtract coverage."""
    height, width = shape[:2]
    mask = np.zeros((height, width), np.float32)
    scale = float(min(height, width))
    for stroke in strokes:
        points = np.rint(np.asarray(stroke['points'], np.float64) * [width - 1, height - 1]).astype(np.int32)
        radius = max(.5, stroke['radius'] * scale)
        layer = np.zeros((height, width), np.uint8)
        if len(points) > 1:
            cv2.polylines(layer, [points], False, 255, max(1, round(radius * 2)), cv2.LINE_AA)
        for point in np.unique(points, axis=0):
            cv2.circle(layer, tuple(point), max(1, round(radius)), 255, -1, cv2.LINE_AA)
        stroke_coverage = layer.astype(np.float32) / 255
        if stroke['mode'] == 'erase':
            mask *= 1 - stroke_coverage
        else:
            mask = np.maximum(mask, stroke_coverage)
    if expand > 0 and mask.any():
        size = 2 * max(1, round(expand * scale)) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    return np.clip(mask, 0, 1)


def _alpha_for(marked, feather):
    """Blend weight that never reaches outside the marked pixels."""
    if not marked.any():
        return np.zeros(marked.shape, np.float32)
    strength = max(feather * min(marked.shape), 1.)
    distance = cv2.distanceTransform(marked.astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(marked, np.clip(distance / strength, 0, 1), 0).astype(np.float32)


def _soften_seam(result, marked, feather):
    """Soften the repair seam without re-mixing any original scratch pixels."""
    if feather <= 0 or not marked.any():
        return result
    sigma = max(.6, feather * min(marked.shape))
    softened = ndi.gaussian_filter(result, (sigma, sigma, 0), mode='reflect').astype(np.float32)
    weight = (1 - _alpha_for(marked, feather))[:, :, None]
    return np.where(marked[:, :, None], result * (1 - weight) + softened * weight, result).astype(np.float32)


def _direction_fill(image, hole, blocked, angle, max_crossing=MAX_CROSSING):
    """Sample unmarked pixels on both sides of a ray at ``angle``.

    ``hole`` is the region to fill, ``blocked`` is every marked pixel (hole
    plus the wide core handled elsewhere).  Samples must be free of both, so a
    wide stroke can never be sampled back into its own repair.
    """
    height, width = hole.shape
    ys, xs = np.nonzero(hole)
    count = len(ys)
    step_y, step_x = np.sin(angle), np.cos(angle)
    forward = np.zeros((count, 3), np.float32)
    backward = np.zeros((count, 3), np.float32)
    forward_distance = np.full(count, np.inf, np.float32)
    backward_distance = np.full(count, np.inf, np.float32)
    missing = np.ones(count, bool)
    for step in range(1, max_crossing + 1):
        if not missing.any():
            break
        for sign, samples, distances in ((1, forward, forward_distance), (-1, backward, backward_distance)):
            sy = np.rint(ys + step_y * step * sign).astype(np.int32)
            sx = np.rint(xs + step_x * step * sign).astype(np.int32)
            inside = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
            if not inside.any():
                continue
            # require the pixel just before the sample to be inside the band,
            # which means this is the first clean pixel beyond the mark
            previous_y = np.rint(ys + step_y * (step - 1) * sign).astype(np.int32)
            previous_x = np.rint(xs + step_x * (step - 1) * sign).astype(np.int32)
            crossed = inside & hole[np.clip(previous_y, 0, height - 1), np.clip(previous_x, 0, width - 1)]
            clean = inside & ~blocked[np.clip(sy, 0, height - 1), np.clip(sx, 0, width - 1)]
            reach = clean & crossed & np.isinf(distances)
            if reach.any():
                index = np.nonzero(reach)[0]
                samples[index] = image[sy[index], sx[index]]
                distances[index] = step
        missing = np.isinf(forward_distance) | np.isinf(backward_distance)
    return forward, backward, forward_distance, backward_distance


def _paper_baseline(image, blocked, sigma=9.):
    """Local paper colour around the marked band (normalized convolution)."""
    known = (~blocked).astype(np.float32)
    weight = ndi.gaussian_filter(known, sigma, mode='nearest')
    total = np.stack([ndi.gaussian_filter(image[:, :, channel] * known, sigma, mode='nearest')
                      for channel in range(3)], axis=2)
    baseline = total / np.maximum(weight, 1e-6)[:, :, None]
    fallback = np.median(image[~blocked], axis=0) if (~blocked).any() else image.reshape(-1, 3).mean(axis=0)
    return np.where(weight[:, :, None] > 1e-3, baseline, fallback.reshape(1, 1, 3)).astype(np.float32)


def _directional_fill(image, hole, blocked, max_crossing=MAX_CROSSING):
    """Reconstruct thin marked bands by agreement between opposite sides.

    Several rays are cast through every pixel of ``hole``.  For each ray the
    first clean pixel on either side is sampled; the ray whose two samples
    agree best and carry the most visible structure (ink, edges) defines the
    local material direction and supplies the fill value.  A thin text stroke
    or an edge crossing the scratch therefore continues along its own
    direction instead of following the scratch normal.
    """
    ys, xs = np.nonzero(hole)
    count = len(ys)
    values = np.zeros((count, 3), np.float32)
    valid = np.zeros(count, bool)
    if not count:
        return {'values': values, 'valid': valid, 'ys': ys, 'xs': xs,
                'first': values.copy(), 'second': values.copy(),
                'first_distance': np.zeros(count, np.float32),
                'second_distance': np.zeros(count, np.float32),
                'normal_y': np.zeros(count, np.float32), 'normal_x': np.zeros(count, np.float32)}
    baseline = _paper_baseline(image, blocked)
    baseline_luma = luminance(baseline)
    paper = baseline_luma[ys, xs]
    best_cost = np.full(count, np.inf, np.float32)
    best_agreement = np.full(count, np.inf, np.float32)
    best_structure = np.zeros(count, np.float32)
    best_first = np.zeros((count, 3), np.float32)
    best_second = np.zeros((count, 3), np.float32)
    first_sample = np.zeros((count, 3), np.float32)
    second_sample = np.zeros((count, 3), np.float32)
    first_distance = np.zeros(count, np.float32)
    second_distance = np.zeros(count, np.float32)
    normal_y = np.zeros(count, np.float32)
    normal_x = np.zeros(count, np.float32)
    for angle in np.arange(DIRECTIONS) * np.pi / DIRECTIONS:
        forward, backward, distance_f, distance_b = _direction_fill(image, hole, blocked, angle, max_crossing)
        complete = np.isfinite(distance_f) & np.isfinite(distance_b) & (distance_f + distance_b > 0)
        total = np.where(complete, distance_f + distance_b, 1.)
        difference = np.abs(forward - backward).max(axis=1)
        # Structure: both sides deviate from the surrounding paper in the same
        # direction (an ink line continues) instead of being bare paper.
        deviation_f = np.abs(luminance(forward[:, None, :])[:, 0] - paper)
        deviation_b = np.abs(luminance(backward[:, None, :])[:, 0] - paper)
        structure = np.minimum(deviation_f, deviation_b)
        cost = np.where(complete, difference + total * .002 - np.minimum(structure, .3) * .8, np.inf)
        better = cost < best_cost
        if not better.any():
            continue
        weight = np.where(complete, distance_b / total, .5)
        blended = backward * weight[:, None] + forward * (1 - weight[:, None])
        # When both sides carry the same ink/edge the blend must reproduce it
        # exactly; a linear ramp would otherwise lighten dark strokes that
        # the scratch crosses.
        agreeing = difference < .035
        blended = np.where(agreeing[:, None], (backward + forward) * .5, blended)
        best_cost[better] = cost[better]
        best_agreement[better] = difference[better]
        best_structure[better] = structure[better]
        best_first[better] = backward[better]
        best_second[better] = forward[better]
        values[better] = blended[better]
        first_sample[better] = backward[better]
        second_sample[better] = forward[better]
        first_distance[better] = distance_b[better]
        second_distance[better] = distance_f[better]
        normal_y[better] = np.sin(angle)
        normal_x[better] = np.cos(angle)
        valid[better] = True
    return {'values': values, 'valid': valid, 'ys': ys, 'xs': xs,
            'first': first_sample, 'second': second_sample,
            'first_distance': first_distance, 'second_distance': second_distance,
            'normal_y': normal_y, 'normal_x': normal_x, 'cost': best_cost,
            'agreement': best_agreement, 'structure': best_structure,
            'ray_first': best_first, 'ray_second': best_second, 'paper': paper}


def _ink_continuation(image, marked, directional):
    """Restore dark ink lines that run across the marked band.

    Two agreeing samples that are both clearly darker than the local paper
    mean the ray runs along a pen stroke.  Such pixels keep the stroke value
    at full strength instead of a softened blend, so a hairline that the
    scratch crosses stays continuous rather than fading out.
    """
    valid = directional['valid']
    if not valid.any():
        return directional['values'], np.zeros(valid.shape, bool)
    first = directional['ray_first'][valid]
    second = directional['ray_second'][valid]
    paper = directional['paper']
    first_luma = luminance(first[:, None, :])[:, 0]
    second_luma = luminance(second[:, None, :])[:, 0]
    dark_first = first_luma < paper * .6
    dark_second = second_luma < paper * .6
    both_dark = dark_first & dark_second
    result = directional['values'][valid].copy()
    result[both_dark] = (first[both_dark] + second[both_dark]) * .5
    flags = np.zeros(valid.shape, bool)
    flags[np.nonzero(valid)[0][both_dark]] = True
    return result, flags


def _grain_transfer(image, marked, directional):
    """Carry paper grain over from unmarked pixels.

    The directional blend is smooth, so a little high-frequency residual is
    copied back from just outside the repair.  It is applied only where the
    two opposite samples look like one material, so ink at a crossing is not
    smeared onto the paper.
    """
    valid = directional['valid']
    if not valid.any():
        return directional['values']
    ys, xs = directional['ys'][valid], directional['xs'][valid]
    low = directional['values'][valid].copy()
    first = directional['first'][valid]
    second = directional['second'][valid]
    consistency = np.abs(first - second).max(axis=1)
    smooth = ndi.gaussian_filter(image, (1.2, 1.2, 0), mode='reflect').astype(np.float32)
    residual = image - smooth
    spread = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    limit = max(3 * spread, .02)
    gate = np.exp(-(consistency / .06) ** 2)
    normal_y = directional['normal_y'][valid]
    normal_x = directional['normal_x'][valid]
    distance, _ = ndi.distance_transform_edt(~marked, return_indices=True)
    room = distance[ys, xs]
    step = np.minimum(room + 2, 6)
    sy = np.rint(ys - normal_y * step).astype(np.int32)
    sx = np.rint(xs - normal_x * step).astype(np.int32)
    height, width = image.shape[:2]
    inside = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
    usable = inside & ~marked[np.clip(sy, 0, height - 1), np.clip(sx, 0, width - 1)]
    sampled = np.zeros_like(low)
    sampled[usable] = residual[sy[usable], sx[usable]]
    gate = gate * (np.abs(sampled).max(axis=1) <= limit)
    return np.clip(low + sampled * gate[:, None], 0, 1)


def min_width(component):
    """Approximate narrowest width of a connected mark, in pixels."""
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    if not component.any():
        return 0.
    return float(distance[component].max()) * 2


def _transfer_offset(image, blocked, component_y, component_x, ring_y, ring_x, search, step=4):
    """Find a texture source offset whose shifted border matches this border.

    The copied area must be completely unmarked.  The border used for matching
    only needs a usable subset, so a mark wider than the search radius can
    still find a source just beyond its own footprint.
    """
    height, width = blocked.shape
    ring = image[ring_y, ring_x]
    span = max(int(component_y.max() - component_y.min()), int(component_x.max() - component_x.min()))
    limit = int(max(search, min(span + search, max(height, width))))
    best = None
    for dy in range(-limit, limit + 1, step):
        for dx in range(-limit, limit + 1, step):
            if dy == 0 and dx == 0:
                continue
            sy, sx = component_y + dy, component_x + dx
            if sy.min() < 0 or sx.min() < 0 or sy.max() >= height or sx.max() >= width:
                continue
            if blocked[sy, sx].any():
                continue
            ry, rx = ring_y + dy, ring_x + dx
            inside = (ry >= 0) & (ry < height) & (rx >= 0) & (rx < width)
            if not inside.any():
                continue
            ry, rx = ry[inside], rx[inside]
            border = ring[inside]
            usable = ~blocked[ry, rx]
            if usable.mean() < .35:
                continue
            difference = image[ry[usable], rx[usable]] - border[usable]
            cost = float(np.mean(difference * difference))
            penalty = (dy * dy + dx * dx) * 1e-8
            score = cost + penalty
            if best is None or score < best[0]:
                best = (score, dy, dx)
    return None if best is None else (best[1], best[2])


def _transfer_fill(image, hole, search, blocked=None):
    """Fill wide painted areas with real texture copied from a matching offset.

    Both the copied area and the border used for matching must lie entirely
    outside ``blocked`` (all marked pixels), so a wide stroke can never copy
    part of the scratch — or another marked region — back into itself.
    """
    blocked = hole if blocked is None else blocked
    labels, count = ndi.label(hole)
    output = image.copy()
    filled = np.zeros(hole.shape, bool)
    fallback = np.zeros(hole.shape, bool)
    ring_kernel = np.ones((7, 7), np.uint8)
    for index in range(1, count + 1):
        component = labels == index
        component_y, component_x = np.nonzero(component)
        ring = cv2.dilate(component.astype(np.uint8), ring_kernel) > 0
        ring &= ~blocked
        ring_y, ring_x = np.nonzero(ring)
        if not len(ring_y):
            fallback[component] = True
            continue
        offset = _transfer_offset(image, blocked, component_y, component_x, ring_y, ring_x, search)
        if offset is None:
            fallback[component] = True
            continue
        dy, dx = offset
        output[component_y, component_x] = image[component_y + dy, component_x + dx]
        filled[component_y, component_x] = True
    return output, filled, fallback


def _smooth_fill(image, marked, radius):
    if not marked.any():
        return image.copy()
    source = np.ascontiguousarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))
    result = cv2.inpaint(source, marked.astype(np.uint8), max(2, int(radius)), cv2.INPAINT_TELEA)
    return np.clip(result.astype(np.float32) / 255, 0, 1)


def repair(image, strokes, settings):
    """Return the repaired image and a report.

    Thin marks keep detail by continuing the real pixels on both sides.
    Broad marks have no meaningful two-sided sample, so they take matching
    texture from a nearby unmarked area.  Anything still unresolved is
    completed smoothly.  Pixels with zero brush coverage never change.
    """
    started = time.monotonic()
    image = np.ascontiguousarray(np.clip(image, 0, 1), np.float32)
    mask = coverage(image.shape, strokes, settings['expand'])
    # Include anti-aliased brush edges so no part of the scratch survives;
    # pixels with zero coverage are the promised untouched region.
    marked = mask > .02
    untouched = mask <= 0
    if not marked.any():
        return image.copy(), {'marked_pixels': 0, 'changed_pixels': 0, 'outside_changed': 0, 'seconds': 0.,
                              'method': 'none', 'strokes': len(strokes), 'note': '没有标记区域，图像保持不变。'}
    report = {'strokes': len(strokes)}
    filled = image.copy()
    handled = np.zeros(marked.shape, bool)
    if settings['method'] == 'texture':
        # Decide the strategy per mark, not per pixel: a thin stroke is
        # continued from both sides, a broad blob takes matching texture.
        labels, components = ndi.label(marked)
        thin_region = np.zeros(marked.shape, bool)
        for index in range(1, components + 1):
            component = labels == index
            if min_width(component) <= WIDE_MARK:
                thin_region |= component
        directional = _directional_fill(image, thin_region, marked)
        valid = directional['valid']
        thin_valid = np.zeros(marked.shape, bool)
        thin_valid[directional['ys'][valid], directional['xs'][valid]] = True
        if thin_valid.any():
            # Ink dashes and fine strokes that the scratch crosses keep their
            # full value; everything else gets the grain-carrying blend.
            ink_values, ink_flags = _ink_continuation(image, marked, directional)
            values = _grain_transfer(image, marked, directional)
            values[ink_flags] = ink_values[ink_flags]
            rows = directional['ys'][valid]
            columns = directional['xs'][valid]
            filled[rows, columns] = values
            handled |= thin_valid
            report['ink_pixels'] = int(ink_flags.sum())
        report['directional_pixels'] = int(thin_valid.sum())
        wide = marked & ~handled
        if wide.any():
            transferred, done, fallback = _transfer_fill(image, wide, settings['search'], marked)
            filled[done] = transferred[done]
            handled |= done
            report['texture_pixels'] = int(done.sum())
        remaining = marked & ~handled
        if remaining.any():
            smooth = _smooth_fill(image, remaining, 4)
            filled[remaining] = smooth[remaining]
            report['smooth_pixels'] = int(remaining.sum())
            handled |= remaining
    else:
        smooth = _smooth_fill(image, marked, 4)
        filled[marked] = smooth[marked]
        report['smooth_pixels'] = int(marked.sum())
    result = np.where(marked[:, :, None], np.clip(filled, 0, 1), image).astype(np.float32)
    result = _soften_seam(result, marked, settings['feather'])
    result = np.where(untouched[:, :, None], image, result).astype(np.float32)
    changed = np.abs(result - image).max(axis=2) > 1 / 1024
    report.update(method=settings['method'], marked_pixels=int(marked.sum()),
                  changed_pixels=int(changed.sum()), outside_changed=int((changed & untouched).sum()),
                  seconds=round(time.monotonic() - started, 3),
                  note='仅标记范围被替换，范围外像素逐像素保持不变；细划痕沿两侧真实像素接续，宽区域移植同类纹理。')
    return result.astype(np.float32), report


def paint_overlay(image, strokes, settings, opacity=.5):
    mask = coverage(image.shape, strokes, settings['expand'])
    tint = np.array([.93, .22, .14], np.float32)
    weight = (mask * opacity)[:, :, None]
    return np.clip(image * (1 - weight) + tint * weight, 0, 1).astype(np.float32)


class Restoration:
    def __init__(self, store, assets, results):
        self.store = store
        self.assets = assets
        self.results = results
        self.lock = RLock()
        self.cache = {}
        self.cache_order = []

    def directory(self, key):
        path = self.store.directory(key) / 'restorations'
        path.mkdir(exist_ok=True)
        return path

    def base(self, key):
        project = self.store.get(key)
        if not project.get('current_result'):
            raise v.Invalid('请先完成合成，再在结果上标记划痕')
        metadata = self.results.metadata(key)
        if metadata['stale']:
            raise v.Invalid('合成结果已过期，请重新合成后再修复划痕')
        fields, metadata = self.results.fields(key, metadata['id'])
        return project, fields['image'].astype(np.float32).copy(), metadata

    def list(self, key):
        project = self.store.get(key)
        state = state_of(project)
        current_signature = signature(state['strokes'], state['settings'])
        current_result = project.get('current_result')
        items = []
        directory = self.store.directory(key) / 'restorations'
        if directory.exists():
            for path in directory.glob('*/restoration.json'):
                record = self.store.read(path)
                if not record:
                    continue
                items.append({
                    'id': record['id'], 'created_at': record['created_at'],
                    'width': record['width'], 'height': record['height'],
                    'base_result_id': record['base_result_id'], 'mask_pixels': record.get('mask_pixels', 0),
                    'strokes': record.get('strokes', 0), 'note': record.get('note', ''),
                    'current': record['id'] == state['confirmed'],
                    'matches_marks': record.get('strokes_signature') == current_signature,
                    'base_stale': bool(current_result) and record['base_result_id'] != current_result,
                    'report': record.get('report', {}),
                })
        return sorted(items, key=lambda row: row['created_at'], reverse=True)

    def state(self, key):
        project = self.store.get(key)
        state = state_of(project)
        payload = {
            'strokes': state['strokes'],
            'settings': state['settings'],
            'confirmed': state['confirmed'],
            'counts': {'strokes': len(state['strokes']),
                       'paint': sum(1 for item in state['strokes'] if item['mode'] == 'paint'),
                       'erase': sum(1 for item in state['strokes'] if item['mode'] == 'erase')},
            'canvas': None, 'confirmed_record': None, 'restorations': self.list(key),
        }
        if project.get('current_result'):
            try:
                metadata = self.results.metadata(key)
                payload['canvas'] = {'width': metadata['width'], 'height': metadata['height'],
                                     'result_id': metadata['id'], 'stale': metadata['stale'],
                                     'frame_ids': metadata.get('frame_ids', [])}
            except v.Invalid:
                pass
        if state['confirmed']:
            payload['confirmed_record'] = self.record(key, state['confirmed'], strict=False)
        return payload

    def add_strokes(self, key, revision, body):
        items = v.list_value(body.get('strokes'), '标记笔画', 64)
        if not items:
            raise v.Invalid('请提供标记笔画')
        with self.store.edit(key, revision, '标记划痕修复区域') as project:
            state = state_of(project)
            if len(state['strokes']) + len(items) > MAX_STROKES:
                raise v.Invalid('标记笔画过多，请先清除部分标记')
            state['strokes'].extend(stroke_spec(item) for item in items)
            project['restoration'] = validate_state(state)
        return self.store.get(key)

    def clear(self, key, revision):
        with self.store.edit(key, revision, '清除划痕标记') as project:
            state = state_of(project)
            state['strokes'] = []
            project['restoration'] = validate_state(state)
        return self.store.get(key)

    def remove_stroke(self, key, revision, stroke_id):
        v.identifier(stroke_id)
        with self.store.edit(key, revision, '撤销一笔划痕标记') as project:
            state = state_of(project)
            remaining = [item for item in state['strokes'] if item['id'] != stroke_id]
            if len(remaining) == len(state['strokes']):
                raise v.Invalid('标记笔画不存在')
            state['strokes'] = remaining
            project['restoration'] = validate_state(state)
        return self.store.get(key)

    def update_settings(self, key, revision, body):
        with self.store.edit(key, revision, '调整划痕修复范围') as project:
            state = state_of(project)
            merged = deepcopy(state['settings'])
            merged.update(v.object_value(body.get('settings', body), '修复设置'))
            state['settings'] = settings_spec(merged)
            project['restoration'] = validate_state(state)
        return self.store.get(key)

    def compute(self, key):
        project, image, metadata = self.base(key)
        state = state_of(project)
        cache_key = (key, metadata['id'], signature(state['strokes'], state['settings']))
        with self.lock:
            cached = self.cache.get(cache_key)
        if cached is not None:
            return image, cached[0], cached[1], state
        repaired, report = repair(image, state['strokes'], state['settings'])
        with self.lock:
            self.cache[cache_key] = (repaired, report)
            self.cache_order.append(cache_key)
            while len(self.cache_order) > 2:
                self.cache.pop(self.cache_order.pop(0), None)
        return image, repaired, report, state

    def preview(self, key, view='repaired', fraction=.5):
        image, repaired, report, state = self.compute(key)
        if view == 'original':
            return image, report
        if view == 'repaired':
            return repaired, report
        if view == 'marked':
            return paint_overlay(image, state['strokes'], state['settings']), report
        if view == 'difference':
            delta = np.clip(np.abs(repaired - image).max(axis=2) * 6, 0, 1)
            return np.repeat(delta[:, :, None], 3, axis=2), report
        position = int(round(np.clip(float(fraction), 0, 1) * (image.shape[1] - 1)))
        result = repaired.copy()
        result[:, :position] = image[:, :position]
        if 0 < position < image.shape[1]:
            result[:, position:position + 1] = [1., .78, .32]
        return result, report

    def record(self, key, restoration_id, strict=True):
        v.identifier(restoration_id)
        path = self.store.directory(key) / 'restorations' / restoration_id
        if not (path / 'restoration.json').exists():
            if strict:
                raise v.Invalid('修复结果不存在')
            return None
        return self.store.read(path / 'restoration.json')

    def image(self, key, restoration_id):
        self.record(key, restoration_id)
        path = self.store.directory(key) / 'restorations' / restoration_id
        with np.load(path / 'image.npz', allow_pickle=False) as data:
            return data['image'].copy()

    def confirm(self, key, revision, body):
        project = self.store.get(key)
        if project['revision'] != revision:
            raise v.Conflict('工程已更新，请刷新后再试')
        state = state_of(project)
        if not state['strokes']:
            raise v.Invalid('请先在图上标记需要修复的划痕')
        image, metadata = self.base(key)[1:]
        repaired, report = repair(image, state['strokes'], state['settings'])
        if report['marked_pixels'] == 0:
            raise v.Invalid('标记范围为空，请先标记划痕')
        restoration_id = model.uid()
        directory = self.directory(key) / restoration_id
        directory.mkdir(parents=True, exist_ok=True)
        from .codecs import png_bytes
        (directory / 'image.png').write_bytes(png_bytes(repaired))
        np.savez_compressed(directory / 'image.npz', image=repaired.astype(np.float32))
        record = {
            'id': restoration_id, 'project_id': key, 'base_result_id': metadata['id'],
            'base_revision': metadata['source_revision'], 'created_at': v.timestamp(),
            'width': int(repaired.shape[1]), 'height': int(repaired.shape[0]),
            'strokes_signature': signature(state['strokes'], state['settings']),
            'strokes': len(state['strokes']), 'settings': deepcopy(state['settings']),
            'mask_pixels': report['marked_pixels'], 'changed_pixels': report['changed_pixels'],
            'report': report, 'note': v.text(body.get('note', ''), '修复备注', 500),
        }
        with self.store.lock:
            self.store.write(directory / 'restoration.json', record)
        with self.store.edit(key, revision, '保存划痕修复结果') as working:
            current = state_of(working)
            current['confirmed'] = restoration_id
            working['restoration'] = validate_state(current)
        return {'project': self.store.get(key), 'restoration': record}

    def remove(self, key, revision, restoration_id):
        self.record(key, restoration_id)
        import shutil
        shutil.rmtree(self.store.directory(key) / 'restorations' / restoration_id)
        with self.store.edit(key, revision, '移除划痕修复记录') as project:
            state = state_of(project)
            if state['confirmed'] == restoration_id:
                state['confirmed'] = None
            project['restoration'] = validate_state(state)
        return self.store.get(key)

    def export(self, key, restoration_id, body):
        from .codecs import png_bytes, jpeg_bytes, tiff_bytes
        body = v.object_value(body, '修复导出')
        image = self.image(key, restoration_id)
        format = v.choice(body.get('format', 'png'), '导出格式', ['png', 'jpeg', 'tiff'])
        bits = v.choice(body.get('bits', 8), '导出位深', [8, 16])
        if format == 'jpeg':
            return jpeg_bytes(image, body.get('quality', 94)), 'scratch-restored.jpg', 'image/jpeg'
        if format == 'tiff':
            return tiff_bytes(image), 'scratch-restored-16bit.tiff', 'image/tiff'
        return png_bytes(image, bits), 'scratch-restored.png', 'image/png'
