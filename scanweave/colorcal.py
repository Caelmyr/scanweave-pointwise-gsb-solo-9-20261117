import numpy as np
import cv2
from .arrays import srgb_decode, srgb_encode
from . import validation as v

# 模型设计矩阵列数：少于该数量的色块无法稳定求解
MODEL_COLUMNS = {'matrix': 3, 'affine': 4, 'polynomial': 10}
# 可靠性判定阈值（均在 0-1 编码或线性 RGB 尺度上）
CLIP_FRACTION = .02
NONUNIFORM_STD = .08
MIN_SAMPLED_PIXELS = 16
SPAN_LIMIT_LINEAR = .18
CHANNEL_SPAN_LIMIT = .12
CONDITION_LIMIT = 1e5
OUTLIER_DELTA_E = 12.
OUTLIER_MAD_FACTOR = 3.
OUTLIER_MAD_FLOOR = 1.5
OUTLIER_Z_LIMIT = 3.
DARK_PATCH_LINEAR = .05
POOR_FIT_DELTA_E = 8.


def patch_sample(image, bounds, inset=.15):
    x0, y0, x1, y1 = v.vector(bounds, '色块区域', 4, 0, 1)
    if x1 <= x0 or y1 <= y0:
        raise v.Invalid('色块区域大小无效')
    width, height = x1 - x0, y1 - y0
    x0, x1 = x0 + width * inset, x1 - width * inset
    y0, y1 = y0 + height * inset, y1 - height * inset
    h, w = image.shape[:2]
    left, right = int(x0 * w), max(int(x0 * w) + 1, int(x1 * w))
    top, bottom = int(y0 * h), max(int(y0 * h) + 1, int(y1 * h))
    patch = image[top:bottom, left:right]
    encoded = np.median(patch, axis=(0, 1))
    linear = np.median(srgb_decode(patch), axis=(0, 1))
    clipped = (patch < .004) | (patch > .996)
    return {'rgb': encoded.tolist(), 'linear': linear.tolist(),
            'std': patch.std(axis=(0, 1)).tolist(),
            'pixels': int(patch.shape[0] * patch.shape[1]),
            'clipped_fraction': float(np.mean(clipped)),
            'channel_clipped_fraction': [float(np.mean((patch[:, :, c] < .004) | (patch[:, :, c] > .996))) for c in range(3)]}


def feature_matrix(values, model='matrix'):
    values = np.asarray(values, np.float64)
    if model == 'affine':
        return np.concatenate([values, np.ones((len(values), 1))], axis=1)
    if model == 'polynomial':
        r, g, b = values.T
        return np.stack([r, g, b, r * g, r * b, g * b, r * r, g * g, b * b, np.ones_like(r)], axis=1)
    return values


def fit_color(source, target, model='matrix', regularization=.01):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise v.Invalid('色彩校正需要相同数量的RGB输入与目标色块')
    design = feature_matrix(source, model)
    if len(source) < design.shape[1]:
        raise v.Invalid('色块数量不足以求解当前校正模型')
    identity = np.zeros((design.shape[1], 3))
    identity[:3] = np.eye(3)
    penalty = np.sqrt(regularization)
    augmented = np.concatenate([design, np.eye(design.shape[1]) * penalty])
    targets = np.concatenate([target, identity * penalty])
    coefficients, residuals, rank, singular = np.linalg.lstsq(augmented, targets, rcond=None)
    predicted = design @ coefficients
    singular_values = np.asarray(singular, np.float64)
    condition = float(singular_values[0] / max(singular_values[-1], 1e-12)) if singular_values.size else np.inf
    return {'model': model, 'coefficients': coefficients.tolist(),
            'rmse_linear': float(np.sqrt(np.mean((predicted - target) ** 2))),
            'rank': int(rank), 'design_columns': int(design.shape[1]),
            'condition': condition, 'predicted': predicted.tolist()}


def apply_color(image, calibration):
    linear = srgb_decode(image)
    original_shape = linear.shape
    h, w = original_shape[:2]
    flat = linear.reshape(-1, 3)
    result = np.empty_like(flat)
    coefficients = np.asarray(calibration['coefficients'], np.float64)
    for start in range(0, len(flat), 200000):
        features = feature_matrix(flat[start:start + 200000], calibration['model'])
        result[start:start + 200000] = features @ coefficients
    return np.clip(srgb_encode(result.reshape(original_shape)), 0, 1)


def _lab_points(rgb):
    """换算到标准尺度的CIE LAB：L*∈[0,100]，a*/b*≈[-128,127]。"""
    array = np.rint(np.clip(np.asarray(rgb, np.float32), 0, 1) * 255).astype(np.uint8)
    if array.ndim == 1:
        array = array[None]
    lab = cv2.cvtColor(array[None], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    lab[:, 0] = lab[:, 0] * 100. / 255.
    lab[:, 1:] = lab[:, 1:] - 128.
    return lab


def delta_e(predicted, target):
    lab_a, lab_b = _lab_points(predicted), _lab_points(target)
    return np.linalg.norm(lab_a - lab_b, axis=1)


def color_error(predicted, target):
    values = delta_e(predicted, target)
    return {'delta_e76': values.tolist(), 'mean': float(values.mean()),
            'maximum': float(values.max()),
            'note': 'ΔE76基于sRGB数值换算的CIE LAB，数值依赖目标色块、照明和输入色彩空间，仅用于同批次相对比较，非色度计标定。'}


def _span_check(source_linear, targets_linear):
    """色块是否全部挤在相近颜色：以目标色跨度为准（目标全同则采样再分散也没有约束信息）。"""
    issues = []
    targets_linear = np.asarray(targets_linear, np.float64)
    distances = np.linalg.norm(targets_linear[:, None] - targets_linear[None], axis=2)
    channel_span = (targets_linear.max(axis=0) - targets_linear.min(axis=0)).max()
    if float(distances.max()) < SPAN_LIMIT_LINEAR and float(channel_span) < CHANNEL_SPAN_LIMIT:
        issues.append('目标色块全部集中在相近颜色，无法约束整段色彩响应；请增加暗部、亮部和不同色相的色块。')
    return issues


def _ramp_health(calibration):
    """检查校正对中性灰渐变的映射是否单调、是否大面积越界（外推风险）。"""
    ramp = np.repeat(np.linspace(0, 1, 32, dtype=np.float32)[:, None], 3, axis=1)
    mapped = apply_color(srgb_encode(ramp), calibration)
    warnings = []
    differences = np.diff(mapped, axis=0)
    if np.any(differences < -2 / 255):
        warnings.append('校正曲线在中性灰渐变上出现反转，色块覆盖范围外的颜色可能不可靠。')
    if float(mapped.min()) <= 0 or float(mapped.max()) >= 1:
        warnings.append('部分输入被校正到0或255截断点，亮部/暗部存在外推截断风险。')
    return warnings


def _loo_residuals(source, target, model, regularization):
    """留一交叉验证ΔE：第i个值由其余色块拟合的模型预测，不受其自身污染。"""
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    count = len(source)
    values = np.zeros(count, np.float32)
    for index in range(count):
        keep = [j for j in range(count) if j != index]
        if len(keep) < MODEL_COLUMNS[model]:
            return None
        held = fit_color(source[keep], target[keep], model, regularization)
        design = feature_matrix(source[index:index + 1], model)
        predicted = srgb_encode((design @ np.asarray(held['coefficients'], np.float64)).astype(np.float32))
        values[index] = delta_e(predicted, srgb_encode(target[index:index + 1].astype(np.float32)))[0]
    return values


def analyze_patches(image, patches, model='matrix', regularization=.01):
    """采样色块并拟合，返回逐色块可靠性与全局问题。

    issues 为阻断性问题：存在时不得保存或应用校正；
    warnings 为风险提示：色块仍会被标记 reliable=false 供用户判断。
    """
    model = v.choice(model, '色彩模型', list(MODEL_COLUMNS))
    regularization = v.number(regularization, '正则强度', 0, 10)
    patches = v.list_value(patches, '色块', 48)
    issues, warnings, reports = [], [], []
    source, targets_linear, targets_encoded = [], [], []
    for index, patch in enumerate(patches):
        patch = v.object_value(patch, '色块')
        report = patch_sample(image, patch.get('bounds'))
        target_encoded = np.asarray(v.color(patch.get('target')), np.float32)
        target_linear = srgb_decode(target_encoded[None, None])[0, 0]
        source.append(report['linear'])
        targets_linear.append(target_linear.tolist())
        targets_encoded.append(target_encoded.tolist())
        flags = []
        if report['clipped_fraction'] >= CLIP_FRACTION:
            flags.append('采样区域已有通道截断（过暗或过曝），该色块无法反映真实响应')
        if max(report['std']) >= NONUNIFORM_STD:
            flags.append('色块内颜色不均匀，框选可能包含边缘、文字或污渍')
        if report['pixels'] < MIN_SAMPLED_PIXELS:
            flags.append('有效采样像素过少，请放大框选区域')
        report.update(index=index, name=str(patch.get('name') or '色块 %d' % (index + 1)),
                      target=patch.get('target'),
                      target_rgb=target_encoded.tolist(), flags=flags, reliable=not flags)
        reports.append(report)

    need = MODEL_COLUMNS[model]
    if len(patches) < need:
        issues.append(f'{model}模型至少需要{need}个带已知颜色的色块，当前只有{len(patches)}个。')
    issues.extend(_span_check(source, targets_linear))

    fitted = None
    if len(patches) >= need:
        fitted = fit_color(source, targets_linear, model, regularization)
        if fitted['rank'] < fitted['design_columns']:
            issues.append('设计矩阵秩不足，色块提供的色彩信息线性相关，无法求解稳定校正。')
        if fitted['condition'] > CONDITION_LIMIT:
            issues.append(f"拟合条件数 {fitted['condition']:.3g} 超过 {CONDITION_LIMIT:.0f}，色块分布不足以支撑该模型，建议改用matrix模型或增加分散色块。")
        predicted_encoded = srgb_encode(np.asarray(fitted['predicted'], np.float32))
        before = delta_e(np.asarray([r['rgb'] for r in reports]), np.asarray(targets_encoded))
        after = delta_e(predicted_encoded, np.asarray(targets_encoded))
        # 留一交叉验证：用其余色块预测该色块，坏色块不会污染对自身的检验
        loo = _loo_residuals(source, targets_linear, model, regularization)
        loo_center = float(np.median(loo)) if loo is not None else 0.
        loo_mad = max(float(np.median(np.abs(loo - loo_center))), OUTLIER_MAD_FLOOR) if loo is not None else 0.
        for report, value_before, value_after in zip(reports, before, after):
            report['delta_e_before'] = float(value_before)
            report['delta_e_after'] = float(value_after)
            if loo is not None:
                report['delta_e_cv'] = float(loo[report['index']])
                z_score = (loo[report['index']] - loo_center) / loo_mad
                # 高阶模型下最暗色块缺少更暗样本约束，留一外推天然不稳定，单列为暗部警告
                is_dark = float(np.mean(source[report['index']])) < DARK_PATCH_LINEAR
                if is_dark and z_score > OUTLIER_Z_LIMIT:
                    report['flags'].append('该暗部色块缺少更暗样本约束，交叉验证波动较大，暗部校正精度有限')
                elif z_score > OUTLIER_Z_LIMIT and loo[report['index']] > OUTLIER_DELTA_E:
                    report['reliable'] = False
                    report['flags'].append('留一交叉验证残差异常：该色块的目标颜色或框选位置与其他色块不一致')
        dark_warnings = [r for r in reports if any('暗部色块' in flag for flag in r['flags'])]
        if dark_warnings:
            warnings.append('色块 ' + '、'.join(str(r['index'] + 1) for r in dark_warnings) + ' 处于采样范围暗端，暗部校正主要依赖外推，建议补充更暗的色块。')
        fitted['error'] = color_error(predicted_encoded, np.asarray(targets_encoded))
        fitted['error_before'] = color_error(np.asarray([r['rgb'] for r in reports]), np.asarray(targets_encoded))
        warnings.extend(_ramp_health(fitted))
        if fitted['error']['mean'] >= POOR_FIT_DELTA_E:
            warnings.append(f"拟合后平均ΔE76仍为{fitted['error']['mean']:.1f}（超过{POOR_FIT_DELTA_E:.0f}），整体校正精度有限；可尝试更高阶模型或检查目标颜色。")
        hard_failed = [r for r in reports if any('截断' in flag or '不均匀' in flag or '像素过少' in flag for flag in r['flags'])]
        if hard_failed and len(hard_failed) == len(reports):
            issues.append('所有色块采样都不可靠（截断、不均匀或像素过少），不能给出可信校正。')
    else:
        for report in reports:
            report['delta_e_before'] = float(delta_e([report['rgb']], [report['target_rgb']])[0])
            report['delta_e_after'] = None

    if reports:
        clipped = [str(i + 1) for i, r in enumerate(reports) if r['clipped_fraction'] >= CLIP_FRACTION]
        if clipped:
            warnings.append('色块 ' + '、'.join(clipped) + ' 的采样存在通道截断；截断信息无法通过拟合恢复。')
    reliable_count = sum(1 for r in reports if r['reliable'])
    return {'model': model, 'regularization': regularization, 'fit': fitted,
            'patches': reports, 'issues': issues, 'warnings': warnings,
            'reliable_count': reliable_count, 'patch_count': len(reports),
            'usable': not issues}


def calibrate_patches(image, body):
    result = analyze_patches(image, body.get('patches'),
                             body.get('model', 'matrix'),
                             body.get('regularization', .01))
    if not result['usable']:
        raise v.Invalid('校正方案不可靠：' + '；'.join(result['issues']))
    fitted = result['fit']
    fitted['patches'] = result['patches']
    return fitted


def cube_lut(calibration, size=17):
    size = v.integer(size, 'LUT边长', 5, 33)
    values = np.linspace(0, 1, size, dtype=np.float32)
    blue, green, red = np.meshgrid(values, values, values, indexing='ij')
    source = np.stack([red, green, blue], axis=3).reshape(1, -1, 3)
    mapped = apply_color(source, calibration).reshape(-1, 3)
    lines = ['TITLE "ScanWeave Color Calibration"', 'LUT_3D_SIZE ' + str(size), 'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0']
    for r, g, b in mapped:
        lines.append(f'{r:.7f} {g:.7f} {b:.7f}')
    return ('\n'.join(lines) + '\n').encode()
