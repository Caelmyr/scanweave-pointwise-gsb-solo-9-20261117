"""Slow illumination equalisation for flat panorama stitching.

Each scan segment is multiplied by a different, slowly varying lamp/exposure
field.  A single global gain cannot remove that; what remains is a broad
gradient that becomes visible as a seam once the segments are blended.

This module estimates one smooth multiplicative field per frame in linear
light.  The fields are constrained by robust local observations inside the
overlaps and by a ridge prior, so only low-order (planar/quadratic) variation
can be fitted.  Fine detail - ink density, lines and paper texture - keeps its
original relative contrast because every pixel in a small neighbourhood is
scaled by almost the same factor.

Conservative behaviour is deliberate:

* pixels excluded by a paint mask are never used and a heavily occluded
  overlap is reported instead of silently extrapolated;
* a pair with very little usable overlap contributes no equations;
* a frame that is not connected to the reference by reliable overlaps is left
  untouched (global gain only) and listed as unreliable;
* when the observations cannot support a quadratic surface the fit falls back
  to a plane, and when even that is unsupported no smooth field is applied;
* the fitted field is limited in EV and the applied strength can be reduced.

The caller composes both the corrected and uncorrected stacks so the page can
show a before/after comparison for exactly the same settings.
"""

import numpy as np
import cv2
from .arrays import srgb_decode, srgb_encode, luminance

# Polynomial terms in (u, v), u/v normalised to roughly [-1, 1].
_TERMS = {
    1: ((0, 0), (1, 0), (0, 1)),
    2: ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)),
}
_DEGREES = {'planar': 1, 'quadratic': 2}
_WORK_EDGE = 640
_MIN_PAIR_PIXELS = 160
_MIN_PAIR_CELLS = 3
_MIN_FRAME_CELLS = 5
_MIN_CELL_PIXELS = 30
_MID_LOW = .02
_MID_HIGH = .92
_CELL_EDGE = 48
_MAX_CELLS = 18
_MAX_DEVIATION = .2


def degree_for(mode):
    return _DEGREES.get(mode)


def _basis(x, y, degree):
    u = 2 * x - 1
    v = 2 * y - 1
    return np.stack([(u ** i) * (v ** j) for i, j in _TERMS[degree]], axis=-1)


def _support_bounds(masks):
    """Per-frame support rectangle in canvas pixels (used for normalisation).

    Fitting each frame's polynomial inside its own rectangle keeps the field
    from extrapolating across empty canvas margins or far beyond a segment.
    """
    bounds = []
    for mask in masks:
        yy, xx = np.where(mask > .01)
        if not len(xx):
            bounds.append(None)
            continue
        bounds.append((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
    return bounds


def _normalized(bounds, x, y, shape):
    """Canvas-normalised coordinate -> coordinate inside a frame's rectangle."""
    if bounds is None:
        return x, y
    x0, y0, x1, y1 = bounds
    return (x * shape[1] - x0) / max(x1 - x0, 1), (y * shape[0] - y0) / max(y1 - y0, 1)


def _downsample(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _low_resolution(images, masks, geometric):
    height, width = images[0].shape[:2]
    scale = min(1., _WORK_EDGE / max(height, width))
    size = (max(8, round(width * scale)), max(8, round(height * scale)))
    low_masks, low_geometric, lows, logs = [], [], [], []
    for image, mask, support in zip(images, masks, geometric):
        small = _downsample(image.astype(np.float32), size)
        low_masks.append(_downsample(mask.astype(np.float32), size))
        low_geometric.append(_downsample(support.astype(np.float32), size))
        value = luminance(srgb_decode(small))
        lows.append(value)
        logs.append(np.log(np.maximum(value, 1e-4)))
    return low_masks, low_geometric, lows, logs, size


def _valid_overlap(first, second, masks, lows):
    inside = (masks[first] > .98) & (masks[second] > .98)
    inside &= (lows[first] > _MID_LOW) & (lows[first] < _MID_HIGH) & (lows[second] > _MID_LOW) & (lows[second] < _MID_HIGH)
    return inside


def _pair_observations(first, second, valid, logs):
    """Robust median log ratio of one overlap, grouped into spatial cells.

    Only cells whose pixels agree (small MAD) contribute.  A cell that mixes
    two different pieces of content - an unmasked object, torn paper, a moving
    element - is discarded instead of bending the smooth field.
    """
    yy, xx = np.where(valid)
    if len(xx) < _MIN_PAIR_PIXELS:
        return [], 0, None
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    columns = int(np.clip(round((x1 - x0) / _CELL_EDGE), 3, _MAX_CELLS))
    rows = int(np.clip(round((y1 - y0) / _CELL_EDGE), 3, _MAX_CELLS))
    cell_width, cell_height = max(1, (x1 - x0) // columns), max(1, (y1 - y0) // rows)
    columns, rows = (x1 - x0) // cell_width, (y1 - y0) // cell_height
    if columns < 3 or rows < 3:
        return [], 0, None
    difference = logs[first] - logs[second]
    window = difference[y0:y0 + rows * cell_height, x0:x0 + columns * cell_width]
    support = valid[y0:y0 + rows * cell_height, x0:x0 + columns * cell_width]
    cells = window.reshape(rows, cell_height, columns, cell_width).transpose(0, 2, 1, 3).reshape(rows, columns, -1)
    inside = support.reshape(rows, cell_height, columns, cell_width).transpose(0, 2, 1, 3).reshape(rows, columns, -1)
    values = np.where(inside, cells, np.nan)
    with np.errstate(invalid='ignore'):
        middle = np.nanmedian(values, axis=2)
        deviation = np.nanmedian(np.abs(values - middle[:, :, None]), axis=2)
    counts = inside.sum(axis=2)
    observations = []
    for row in range(rows):
        for column in range(columns):
            count = int(counts[row, column])
            if count < _MIN_CELL_PIXELS:
                continue
            value, spread = float(middle[row, column]), float(deviation[row, column])
            if not np.isfinite(value) or not np.isfinite(spread) or spread > _MAX_DEVIATION:
                continue
            observations.append({
                'x': float(x0 + (column + .5) * cell_width) / difference.shape[1],
                'y': float(y0 + (row + .5) * cell_height) / difference.shape[0],
                'value': value,
                'quality': float(np.clip(.05 / max(spread, .005), 0, 1)),
                'pixels': count,
            })
    return observations, int(valid.sum()), (x0, y0, x1, y1)


def _pair_spread(first, second, valid, logs, fields):
    difference = (logs[first] - logs[second])[valid]
    if not difference.size:
        return None, None
    before = difference - np.median(difference)
    after = before + fields[first][valid] - fields[second][valid]
    median_after = np.median(after)
    return (float(np.percentile(np.abs(before), 95) / np.log(2)),
            float(np.percentile(np.abs(after - median_after), 95) / np.log(2)))


def _reliability(pairs, frames, reference):
    edges = [pair for pair in pairs if len(pair['observations']) >= _MIN_PAIR_CELLS]
    neighbours = {index: set() for index in range(frames)}
    for pair in edges:
        neighbours[pair['first']].add(pair['second'])
        neighbours[pair['second']].add(pair['first'])
    reached = {reference}
    queue = [reference]
    while queue:
        current = queue.pop()
        for neighbour in sorted(neighbours[current]):
            if neighbour not in reached:
                reached.add(neighbour)
                queue.append(neighbour)
    cells = [0] * frames
    samples = [0] * frames
    for pair in edges:
        for index in (pair['first'], pair['second']):
            cells[index] += len(pair['observations'])
            samples[index] += pair['pixels']
    reliable = [index in reached and cells[index] >= _MIN_FRAME_CELLS for index in range(frames)]
    reliable[reference] = True
    return edges, cells, samples, reliable


def _variables(frames, reference, degree, reliable):
    return [(index, term) for index in range(frames) if index != reference and reliable[index] for term in range(len(_TERMS[degree]))]


def _solve(pairs, frames, reference, degree, reliable, bounds, canvas):
    """Least-squares fit of all frame fields against the overlap observations.

    Returns ``[frame, term]`` coefficients (the reference row is zero) and a
    small fit report.  A Huber reweighting loop keeps content boundaries and
    yet-unmasked specks from bending the smooth surface.
    """
    terms = _TERMS[degree]
    variables = _variables(frames, reference, degree, reliable)
    lookup = {variable: position for position, variable in enumerate(variables)}
    coefficients = np.zeros((frames, len(terms)))
    rows, targets, weights = [], [], []
    for pair in pairs:
        first, second = pair['first'], pair['second']
        if not (reliable[first] and reliable[second]):
            continue
        for observation in pair['observations']:
            ux, uy = _normalized(bounds[first], observation['x'], observation['y'], canvas)
            vx, vy = _normalized(bounds[second], observation['x'], observation['y'], canvas)
            row = np.zeros(len(variables))
            for term, value in enumerate(_basis(ux, uy, degree)):
                if (first, term) in lookup:
                    row[lookup[(first, term)]] += value
            for term, value in enumerate(_basis(vx, vy, degree)):
                if (second, term) in lookup:
                    row[lookup[(second, term)]] -= value
            rows.append(row)
            targets.append(-observation['value'])
            weights.append(observation['quality'] * np.sqrt(observation['pixels'] / _MIN_CELL_PIXELS))
    if not variables or not rows:
        return coefficients, {'equations': 0, 'residual_stops': 0.}
    matrix = np.stack(rows)
    target = np.asarray(targets, np.float64)
    weight = np.maximum(np.asarray(weights, np.float64), 1e-3)
    solution = np.zeros(len(variables))
    # Ridge prior: constant/slope are cheap, curvature is more strongly pulled
    # to zero, which keeps unsupported extrapolation near the global gain.
    penalty = np.array([.004] + [.012] * (len(terms) - 1))
    diagonal = np.tile(penalty, len(variables) // len(terms))
    for _ in range(4):
        residual = matrix @ solution - target
        scatter = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        huber = np.clip(max(.02, 2.5 * scatter) / np.maximum(np.abs(residual), 1e-9), 0, 1)
        current = weight * huber
        normal = (matrix * current[:, None]).T @ matrix + np.diag(diagonal * max(float(current.sum()), 1.) * .02)
        solution = np.linalg.solve(normal, (matrix * current[:, None]).T @ target)
    for (index, term), position in lookup.items():
        coefficients[index, term] = solution[position]
    residual = matrix @ solution - target
    return coefficients, {'equations': len(rows), 'residual_stops': float(np.sqrt(np.mean(residual ** 2)) / np.log(2))}


def _pair_entries(reports, pairs, reliable, names):
    constrained = {(names[pair['first']], names[pair['second']]) for pair in pairs if reliable[pair['first']] and reliable[pair['second']]}
    for entry in reports:
        entry['constrained'] = (entry['first'], entry['second']) in constrained
    return reports


def estimate(images, masks, geometric, frames, reference, mode='quadratic', strength=1., limit=1.):
    """Estimate smooth log-gain fields. Returns per-frame fields plus a report."""
    degree = degree_for(mode)
    count = len(images)
    names = [frame['name'] for frame in frames]
    report = {
        'enabled': True, 'method': mode, 'degree': degree or 0,
        'strength': float(strength), 'limit_ev': float(limit), 'applied': False,
        'reference': names[reference] if names else '',
        'observations': 0, 'warnings': [], 'pairs': [], 'frames': [],
        'unreliable_frames': [],
        'note': '仅估计低阶缓慢亮度场；文字、线条与原稿局部深浅比例保持不变。补偿前指整体增益后的结果；参考帧自身的缓慢差异不会被消除。',
    }
    if degree is None:
        report['enabled'] = False
        report['note'] = '缓变亮度补偿已关闭，只使用整体增益与融合。'
        return [np.zeros_like(masks[0], np.float32) for _ in range(count)], report
    zero = [np.zeros_like(masks[0], np.float32) for _ in range(count)]
    if count < 2:
        report['enabled'] = False
        report['degree'] = 0
        report['warnings'].append('只有一帧，无需缓变亮度补偿。')
        return zero, report
    low_masks, low_geometric, lows, logs, size = _low_resolution(images, masks, geometric)
    height, width = size[1], size[0]
    # Bounds are taken from the geometric footprint (not the paint mask) and in
    # the reduced grid, so the polynomial is normalised to each segment itself.
    bounds = _support_bounds(low_geometric)
    pairs, pair_reports = [], []
    for first in range(count):
        for second in range(first + 1, count):
            inside = _valid_overlap(first, second, low_masks, lows)
            overlap = (low_geometric[first] > .98) & (low_geometric[second] > .98)
            observations, total, _ = _pair_observations(first, second, inside, logs)
            coverage = float(inside.sum() / max(int(overlap.sum()), 1)) if overlap.any() else 0.
            entry = {
                'first': names[first], 'second': names[second], 'pixels': total,
                'cells': len(observations), 'coverage': round(coverage, 4),
                'constrained': False, 'before_stops': None, 'after_stops': None, 'note': '',
            }
            pair_reports.append(entry)
            if total < _MIN_PAIR_PIXELS:
                entry['note'] = '有效重叠像素过少，不参与缓变拟合。'
                report['warnings'].append(f'{entry["first"]} / {entry["second"]}：有效重叠太少，保持整体增益。')
            elif coverage < .35 and int(overlap.sum()) > _MIN_PAIR_PIXELS:
                entry['note'] = '重叠区域被遮罩或遮挡较多，只采用其中可靠的局部单元。'
                report['warnings'].append(f'{entry["first"]} / {entry["second"]}：重叠被大块遮挡，采用保守拟合。')
            elif len(observations) < _MIN_PAIR_CELLS:
                entry['note'] = '没有足够一致的局部样本，不参与缓变拟合。'
                report['warnings'].append(f'{entry["first"]} / {entry["second"]}：重叠一致性不足，保持整体增益。')
            if len(observations) >= _MIN_PAIR_CELLS:
                pairs.append({'first': first, 'second': second, 'observations': observations, 'pixels': total, 'report': entry})
    edges, cells, samples, reliable = _reliability(pairs, count, reference)

    # A low-order surface needs more independent overlap samples than unknowns.
    # With too little support, prefer a plane, then no smooth field at all.
    def support(frames_ok, current_degree):
        variables = len(_variables(count, reference, current_degree, frames_ok))
        usable = sum(len(pair['observations']) for pair in pairs if frames_ok[pair['first']] and frames_ok[pair['second']])
        return variables, usable

    variables, usable = support(reliable, degree)
    if degree == 2 and 0 < usable < max(2 * variables, 12):
        degree = 1
        report['degree'] = 1
        report['warnings'].append('可靠重叠观测不足以支撑曲面拟合，已保守改用线性缓变。')
        edges, cells, samples, reliable = _reliability(pairs, count, reference)
        variables, usable = support(reliable, degree)
    if usable <= 0 or usable < max(2 * variables, 6):
        reliable = [index == reference for index in range(count)]
        degree = 1
        report['degree'] = 0
        report['warnings'].append('可靠重叠观测太少，无法估计缓变亮度，本次只保留整体增益。')
    coefficients, solve = _solve(pairs, count, reference, degree, reliable, bounds, (height, width)) if degree else (np.zeros((count, len(_TERMS[1]))), {'equations': 0, 'residual_stops': 0.})
    report['observations'] = int(usable)
    report['solve_residual_stops'] = round(solve['residual_stops'], 4)
    terms = _TERMS[degree]
    yy, xx = np.indices((height, width), dtype=np.float32)
    xg, yg = (xx + .5) / width, (yy + .5) / height
    limit_log = float(limit) * np.log(2)
    raw_fields = []
    for index in range(count):
        value = np.zeros((height, width), np.float32)
        if index != reference and reliable[index]:
            ux, uy = _normalized(bounds[index], xg, yg, (height, width))
            grid = _basis(ux, uy, degree)
            for term in range(len(terms)):
                value += np.float32(coefficients[index, term]) * grid[:, :, term]
            value *= float(strength)
        raw_fields.append(value)
    fields = [np.clip(field, -limit_log, limit_log).astype(np.float32) for field in raw_fields]
    applied = any(float(np.max(np.abs(field))) > 1e-6 for field in fields)
    report['applied'] = bool(applied)
    unreliable = []
    for index in range(count):
        if index == reference:
            note = '参考帧，亮度场固定为零。'
        elif not reliable[index]:
            note = '没有通过可靠重叠连接到参考帧，仅保留整体增益。'
            unreliable.append(names[index])
        else:
            note = '已估计缓变亮度场。'
        field = fields[index]
        # Report and clamp only where the frame actually contributes; unused
        # canvas margins may hold harmless extrapolation.
        support = low_geometric[index] > .5
        if not support.any():
            support = np.ones_like(field, bool)
        clamped = float(np.mean(np.abs(raw_fields[index][support]) > limit_log + 1e-6))
        if clamped > .01:
            report['warnings'].append(names[index] + f'：亮度补偿达到 {float(limit):g} EV 上限，已截断。')
        report['frames'].append({
            'index': index, 'name': names[index],
            'observations': int(cells[index]), 'samples': int(samples[index]),
            'reliable': bool(reliable[index]) or index == reference,
            'applied': bool(applied and (index == reference or reliable[index])),
            'field_min_ev': round(float(field[support].min()) / np.log(2), 4),
            'field_max_ev': round(float(field[support].max()) / np.log(2), 4),
            'clamped_fraction': round(clamped, 4), 'note': note,
        })
    report['unreliable_frames'] = unreliable
    for pair in edges:
        entry = pair['report']
        valid = _valid_overlap(pair['first'], pair['second'], low_masks, lows)
        before, after = _pair_spread(pair['first'], pair['second'], valid, logs, fields)
        entry['before_stops'] = None if before is None else round(before, 4)
        entry['after_stops'] = None if after is None else round(after, 4)
        if after is not None and after > .08:
            report['warnings'].append(f'{entry["first"]} / {entry["second"]}：补偿后仍有 {after:.3f} EV 的局部残差，可能来自内容差异或遮挡。')
    report['pairs'] = _pair_entries(pair_reports, pairs, reliable, names)
    report['warnings'] = list(dict.fromkeys(report['warnings']))
    if not applied:
        if float(strength) <= 0:
            report['warnings'].append('缓变亮度强度为0，本次没有改变图像。')
        else:
            report['warnings'].append('没有可用的缓变亮度补偿，结果等同于整体增益。')
    return fields, report


def comparison(after, before, fraction=.5):
    """Split the same blend with and without the smooth fields.

    The left side shows the uncorrected result, the right side the corrected
    one, both on the identical crop, finishing and export settings.  The
    divider is a thin gold line so paper texture is not misread as a seam.
    """
    if after.shape != before.shape:
        return after.copy()
    result = after.copy()
    position = int(np.clip(round(after.shape[1] * fraction), 1, after.shape[1] - 1))
    result[:, :position] = before[:, :position]
    result[:, position - 1:position + 1] = [.72, .56, .33]
    return result.astype(np.float32)


def difference_stops(after, before, valid=None):
    """Absolute smooth-level difference in EV, without counting local detail."""
    if after.shape != before.shape:
        return None
    selected = valid > .5 if valid is not None else np.ones(after.shape[:2], bool)
    if selected.sum() < 64:
        return None
    a = luminance(srgb_decode(after))
    b = luminance(srgb_decode(before))
    values = np.abs(np.log(np.maximum(a, 1e-4)) - np.log(np.maximum(b, 1e-4)))[selected] / np.log(2)
    return {'mean_ev': round(float(np.mean(values)), 4), 'p95_ev': round(float(np.percentile(values, 95)), 4), 'pixels': int(selected.sum()), 'note': '仅统计已应用亮度补偿的结果；局部细节未被平滑。'}


def apply_fields(images, fields, size, masks=None):
    """Scale images in linear light by the upsampled smooth fields.

    Work is limited to each frame's support rectangle, so a large panorama
    canvas with empty margins is not converted back and forth in full.
    """
    corrected = []
    for index, (image, field) in enumerate(zip(images, fields)):
        if not np.any(field):
            corrected.append(image.copy())
            continue
        gain = np.exp(cv2.resize(field, size, interpolation=cv2.INTER_LINEAR))
        if masks is None:
            linear = srgb_decode(image) * gain[:, :, None]
            corrected.append(np.clip(srgb_encode(np.maximum(linear, 0)), 0, 1).astype(np.float32))
            continue
        yy, xx = np.where(masks[index] > 0)
        if not len(xx):
            corrected.append(image.copy())
            continue
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        result = image.copy()
        local = srgb_decode(image[y0:y1, x0:x1]) * gain[y0:y1, x0:x1, None]
        result[y0:y1, x0:x1] = np.clip(srgb_encode(np.maximum(local, 0)), 0, 1)
        corrected.append(result.astype(np.float32))
    return corrected


def detail_preservation(before, after, valid, edge=_WORK_EDGE):
    """Local-structure check: the smooth field may move levels, not detail.

    Returns the correlation of the high-pass residual inside each valid frame
    (1.0 = ink strokes and lines keep their exact shape).  The check runs on a
    reduced copy so it stays cheap for large canvases.
    """
    values = []
    for first, second, support in zip(before, after, valid):
        h, w = support.shape[:2]
        scale = min(1., edge / max(h, w))
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        mask = cv2.resize(support, size, interpolation=cv2.INTER_AREA) > .5
        # Stay away from the frame border: a hard support edge is not image
        # detail, and scaling it with the smooth field would distort the ratio.
        mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        if mask.sum() < 64:
            values.append(None)
            continue
        a = cv2.resize(luminance(first), size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(luminance(second), size, interpolation=cv2.INTER_AREA)
        # Compare the high-pass structure itself, not its absolute level: a
        # valid correction may move levels, it may not reorder or flatten ink.
        a = a - cv2.GaussianBlur(a, (0, 0), 2)
        b = b - cv2.GaussianBlur(b, (0, 0), 2)
        a, b = a[mask], b[mask]
        spread = float(np.std(a) * np.std(b))
        values.append(round(float(np.mean((a - a.mean()) * (b - b.mean())) / spread), 4) if spread > 1e-9 else None)
    return values
