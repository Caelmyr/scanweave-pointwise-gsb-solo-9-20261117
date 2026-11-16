"""彩色负片反转：片基色取样、按通道反转与黑白点/反差控制。

设计要点：
- 片基参考区域以“源帧自身像素”的归一化坐标保存，与配准矩阵、共同画布、
  输出裁切无关，因此后续配准或裁剪不会让片基区漂移。
- 片基色不写入工程，每次从不可变原始扫描按参考区域重新测量，保证同批扫描
  共用同一套片基参考，且刷新、撤销、恢复版本后结果一致。
"""
import numpy as np
from . import validation as v
from .arrays import fit_size


# 参考区域有效性阈值
MIN_PIXELS = 64
ZERO_LIMIT = .004
SATURATION_LIMIT = .996
ZERO_FRACTION = .02
SATURATION_FRACTION = .02
UNIFORM_STD = .002
CHANNEL_SPREAD = .25
CONTRAST_GAMMA = 1.7


def negative_settings(body):
    body = v.object_value(body, '负片反转设置')
    source = body.get('source_frame')
    if source is not None:
        v.identifier(source)
    roi = v.vector(body.get('roi', [.02, .02, .18, .18]), '片基参考区域', 4, 0, 1)
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise v.Invalid('片基参考区域需要正的宽度和高度')
    if roi[2] - roi[0] > .98 or roi[3] - roi[1] > .98:
        raise v.Invalid('片基参考区域几乎覆盖整帧，请只框选未成像的片基')
    black = v.number(body.get('black', .02), '反转黑点', 0, .8)
    white = v.number(body.get('white', .98), '反转白点', .2, 1)
    if white <= black + .01:
        raise v.Invalid('反转白点必须高于黑点至少0.01')
    contrast = body.get('contrast', [1., 1., 1.])
    contrast = v.vector(contrast, '分通道反差', 3, .2, 3)
    return {
        'enabled': v.boolean(body.get('enabled', False), '启用负片反转'),
        'source_frame': source,
        'roi': roi,
        'black': black,
        'white': white,
        'contrast': contrast,
    }


def pixel_roi(roi, shape):
    """归一化参考区域转像素坐标（相对源帧自身，不做任何配准变换）。"""
    h, w = shape[:2]
    x0, y0, x1, y1 = roi
    left = min(w - 1, max(0, int(np.floor(x0 * w))))
    top = min(h - 1, max(0, int(np.floor(y0 * h))))
    right = max(left + 1, min(w, int(np.ceil(x1 * w))))
    bottom = max(top + 1, min(h, int(np.ceil(y1 * h))))
    return left, top, right, bottom


def measure_base(image, roi):
    """在原始扫描的参考区域内测量片基色，并给出可理解的适用性提示。"""
    left, top, right, bottom = pixel_roi(roi, image.shape)
    patch = np.asarray(image[top:bottom, left:right], dtype=np.float32)
    pixel_count = patch.shape[0] * patch.shape[1]
    problems, warnings = [], []
    if pixel_count < MIN_PIXELS:
        problems.append('参考区域过小（少于{n}个像素），无法稳定估计片基色'.format(n=MIN_PIXELS))
    finite = np.isfinite(patch).all(axis=2)
    if not bool(finite.all()):
        problems.append('参考区域包含无效像素值，请重新框选')
    zero_mask = np.any(patch <= ZERO_LIMIT, axis=2)
    sat_mask = np.any(patch >= SATURATION_LIMIT, axis=2)
    zero_fraction = float(np.mean(zero_mask))
    sat_fraction = float(np.mean(sat_mask))
    if zero_fraction > ZERO_FRACTION:
        problems.append('参考区域含{:.0%}近零像素，疑似黑边或遮挡，不是均匀片基'.format(zero_fraction))
    if sat_fraction > SATURATION_FRACTION:
        problems.append('参考区域含{:.0%}饱和像素，扫描已溢出，无法据此校正片基色'.format(sat_fraction))
    # 中位数对扫描噪声和零星坏点更稳健
    base = np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)
    base = np.where(np.isfinite(base), base, 0)
    means = patch.mean(axis=(0, 1))
    stds = patch.std(axis=(0, 1))
    if np.any(base <= ZERO_LIMIT):
        problems.append('某通道片基测量值接近零，无法安全做除法，请换一处片基区域')
    if np.all(stds < UNIFORM_STD):
        warnings.append('参考区域几乎没有纹理变化，确认它确实是未成像的均匀片基')
    if float(np.max(stds)) > .06:
        warnings.append('参考区域内亮度变化较大，可能框入了成像内容，建议缩小到纯片基边框')
    spread = float(base.max() - base.min())
    if spread < CHANNEL_SPREAD * float(base.max()):
        warnings.append('该区域各通道颜色接近中性，不像典型橙色片基，请确认框选位置')
    return {
        'base': [float(x) for x in base],
        'pixel_bounds': [int(left), int(top), int(right), int(bottom)],
        'pixel_count': int(pixel_count),
        'zero_fraction': zero_fraction,
        'saturated_fraction': sat_fraction,
        'channel_mean': [float(x) for x in means],
        'channel_std': [float(x) for x in stds],
        'problems': problems,
        'warnings': warnings,
        'usable': not problems,
    }


def _channel_contrast(value, factor):
    """以中灰0.5为支点的单通道反差，gamma映射保证0/1端点不动。"""
    if abs(factor - 1.) < 1e-6:
        return value
    gamma = 1. / (1. + (factor - 1.) * CONTRAST_GAMMA)
    return np.clip(value, 0, 1) ** gamma


def invert_image(image, base, settings):
    """对单张透射扫描执行片基消除 + 反转 + 黑白点/分通道反差。

    image/base 均为0–1浮点sRGB编码值；输出为0–1正像。
    """
    base = np.maximum(np.asarray(base, dtype=np.float32), 1e-6)
    corrected = np.clip(np.asarray(image, dtype=np.float32), 0, None) / base[None, None, :]
    positive = 1. - np.clip(corrected, 0, 1)
    black, white = float(settings['black']), float(settings['white'])
    mapped = np.clip((positive - black) / max(white - black, 1e-4), 0, 1)
    factors = settings['contrast']
    channels = [_channel_contrast(mapped[:, :, c], float(factors[c])) for c in range(3)]
    return np.clip(np.stack(channels, axis=2), 0, 1).astype(np.float32)


def source_frame(project):
    """解析片基参考源帧：优先记录的源帧，其次工程参考帧。"""
    from .model import find_frame
    settings = project['settings']['negative']
    frame_id = settings.get('source_frame') or project.get('reference')
    if frame_id is None:
        raise v.Invalid('负片反转需要先指定参考帧')
    return find_frame(project, frame_id)


def base_for_project(project, assets):
    """从源帧的原始不可变扫描测量片基色（同批扫描共用）。"""
    settings = project['settings']['negative']
    frame = source_frame(project)
    image = assets.field(frame['asset'])
    report = measure_base(image, settings['roi'])
    if not report['usable']:
        raise v.Invalid('片基参考区域不适用：' + '；'.join(report['problems']))
    return np.asarray(report['base'], dtype=np.float32), report


def apply_to_prepared(prepared, project, assets):
    """对配准准备后的共同画布图像统一做负片反转。

    片基色只测量一次并应用到全部启用帧；几何有效性（letterbox黑边等）
    原样保留，补边区域不会被当成反转后的画面。
    """
    settings = project['settings']['negative']
    if not settings.get('enabled'):
        return prepared, None
    base, report = base_for_project(project, assets)
    for index, image in enumerate(prepared['images']):
        valid = prepared['validities'][index] > .001
        inverted = invert_image(image, base, settings)
        prepared['images'][index] = np.where(valid[:, :, None], inverted, image)
    report = dict(report)
    report['applied_frames'] = [frame['id'] for frame in prepared['frames']]
    report['source_frame'] = source_frame(project)['id']
    return prepared, report


def preview_image(project, assets, roi=None, settings_override=None):
    """供页面交互使用：返回源帧上应用当前（或候选）反转设置后的正像与测量报告。"""
    from .arrays import letterbox
    current = project['settings']['negative']
    settings = dict(current)
    if settings_override:
        settings.update({k: v for k, v in settings_override.items() if v is not None})
    if roi is not None:
        settings['roi'] = v.vector(roi, '片基参考区域', 4, 0, 1)
    frame = source_frame(project)
    image = assets.field(frame['asset'])
    width, height = fit_size(image.shape, min(project['settings']['size'], 1024))
    working, valid = letterbox(image, width, height)
    report = measure_base(image, settings['roi'])
    if not report['usable']:
        raise v.Invalid('片基参考区域不适用：' + '；'.join(report['problems']))
    base = np.asarray(report['base'], dtype=np.float32)
    inverted = invert_image(working, base, settings)
    # 与合成引擎一致：letterbox 补边区域不参与反转，保持黑色填充
    inverted = np.where((valid > .001)[:, :, None], inverted, working)
    return inverted.astype(np.float32), report
