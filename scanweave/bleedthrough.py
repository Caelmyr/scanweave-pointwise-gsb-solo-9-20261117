"""双面透印（show-through）抑制。

物理模型：薄纸正面扫描到的淡字来自背面墨迹对透射光的衰减。
在已镜像配准的共同画布上，用背面墨迹的暗度估计透印分量，
仅在几何覆盖充分、局部相关可靠且不属于正面自己墨迹的位置扣除；
没有可靠重叠的区域保留原像素，不照搬补偿。

计算在 sRGB 显示域的“纸张暗度”空间进行：透印淡字位于纸张亮部，
该空间下透印与背面暗度的比例在不同墨深下更稳定，也更贴近可见效果。
"""
import cv2
import numpy as np
from . import validation as v
from .arrays import luminance, blur, smoothstep, finite_rgb
from .validation import Invalid


def settings(body):
    body = v.object_value(body, '双面透印设置')
    return {
        'strength': v.number(body.get('strength', .7), '抑制强度', 0, 1),
        'auto_gain': v.boolean(body.get('auto_gain', True), '自动估计透印系数'),
        'gain': v.number(body.get('gain', .25), '手动透印系数', .02, 1),
        'back_blur': v.number(body.get('back_blur', 2.5), '背面扩散模糊（工作像素）', 0, 20),
        'protect': v.number(body.get('protect', .8), '正面字迹保护强度', 0, 1),
        'protect_threshold': v.number(body.get('protect_threshold', .04), '正面淡字保护阈值（暗度）', .005, .4),
        'reliable_correlation': v.number(body.get('reliable_correlation', .25), '可靠重叠相关阈值', -.95, .95),
        'reliable_window': v.integer(body.get('reliable_window', 33), '可靠判断窗口边长（像素）', 5, 127),
    }


def _paper_white(image, valid):
    white = np.empty(3, np.float32)
    for channel in range(3):
        values = image[:, :, channel][valid > .99]
        white[channel] = float(np.percentile(values, 95)) if values.size else 1.
    return np.maximum(white, .05)


def _local_correlation(a, b, mask, window):
    side = window if window % 2 == 1 else window + 1
    kernel = (side, side)
    support = cv2.boxFilter(mask.astype(np.float32), -1, kernel, normalize=False)
    count = np.maximum(support, 1)
    mean_a = cv2.boxFilter(a * mask, -1, kernel) / count
    mean_b = cv2.boxFilter(b * mask, -1, kernel) / count
    var_a = cv2.boxFilter(a * a * mask, -1, kernel) / count - mean_a ** 2
    var_b = cv2.boxFilter(b * b * mask, -1, kernel) / count - mean_b ** 2
    cov = cv2.boxFilter(a * b * mask, -1, kernel) / count - mean_a * mean_b
    correlation = cov / np.sqrt(np.maximum(var_a, 1e-8) * np.maximum(var_b, 1e-8) + 1e-10)
    enough = support >= side * side * .9
    return np.where(enough, np.clip(correlation, -1, 1), 0).astype(np.float32)


def _reliability(front_dark_luma, back_dark_luma, geometry, coarse_correlation, threshold):
    """几何覆盖 × 局部相关 × 背面确有墨迹，三者同时成立才补偿。

    文字页面上大窗口被纸张留白稀释，因此以笔画级小窗口相关为主判据，
    大窗口仅在明显反向相关时作为否决；内容门控用背面局部暗度的相对分位数。
    """
    fine = _local_correlation(front_dark_luma, back_dark_luma, geometry, 11)
    fine_gate = smoothstep(threshold - .12, threshold + .1, fine)
    # coarse 从 -0.5（反向）到 -0.2 时否决从 1 衰减到 0；正相关不否决
    veto = 1 - smoothstep(-.5, -.2, coarse_correlation)
    aligned = fine_gate * (1 - .85 * veto)
    side = 11
    kernel = (side, side)
    geometry_f = geometry.astype(np.float32)
    support = np.maximum(cv2.boxFilter(geometry_f, -1, kernel, normalize=False), 1)
    local_back = cv2.boxFilter(back_dark_luma * geometry_f, -1, kernel) / support
    positive = local_back[geometry > .99]
    low = float(np.percentile(positive, 55))
    high = float(np.percentile(positive, 90))
    back_presence = smoothstep(low, max(high, low * 1.5 + 1e-4), local_back)
    reliable = geometry * aligned * back_presence
    # 扩展到笔画周边的透印晕染区，仍受几何覆盖约束
    return (blur(reliable.astype(np.float32), 1.6) * geometry).astype(np.float32)


def _estimate_gain(front_dark, back_dark, mask, blank_threshold):
    # 只用正面留白（自身暗度低于保护阈值）且背面确有墨迹的像素估计透印系数。
    # 透印在强墨处趋于饱和，全局回归会被高暗度样本拉低，因此在背面暗度分箱内
    # 计算过原点斜率，取各箱中位数作为稳健增益。
    front_luma = luminance(front_dark)
    back_luma = luminance(back_dark)
    blank = (front_luma < blank_threshold) & (mask > .99)
    if blank.sum() < 400:
        return np.full(3, .25, np.float32), int(blank.sum()), False
    floor_cut = float(np.percentile(back_luma[blank], 25))
    ink_cut = float(np.percentile(back_luma[blank], 70))
    selected = blank & (back_luma > ink_cut)
    gain = np.empty(3, np.float32)
    for channel in range(3):
        floor = float(np.median(front_dark[:, :, channel][blank & (back_luma <= floor_cut)]))
        x = back_dark[:, :, channel][selected]
        y = np.maximum(front_dark[:, :, channel][selected] - floor, 0)
        order = np.argsort(x)
        x, y = x[order], y[order]
        slopes = []
        for start in range(0, len(x), max(400, len(x) // 12)):
            bx, by = x[start:start + 800], y[start:start + 800]
            if len(bx) >= 100 and np.sum(bx * bx) > 1e-10:
                slopes.append(float(np.sum(bx * by) / np.sum(bx * bx)))
        gain[channel] = np.clip(float(np.median(slopes)) if slopes else .25, .02, 1.)
    return gain, int(selected.sum()), True


def suppress_show_through(prepared, config):
    frames = prepared['frames']
    if len(frames) != 2:
        raise Invalid('双面透印流程需要恰好两张启用照片：正面（参考帧）与反面（勾选水平镜像）')
    front_index = prepared['reference_index']
    back_index = 1 - front_index
    front = np.clip(finite_rgb(prepared['images'][front_index]), 0, 1)
    back = np.clip(finite_rgb(prepared['images'][back_index]), 0, 1)
    front_valid = prepared['geometric_validities'][front_index]
    back_valid = prepared['geometric_validities'][back_index]
    geometry = np.clip(np.minimum(front_valid, back_valid), 0, 1)

    paper_front = _paper_white(front, front_valid)
    paper_back = _paper_white(back, back_valid)
    front_dark = np.clip(paper_front[None, None, :] - front, 0, 1)
    back_dark = np.clip(paper_back[None, None, :] - back, 0, 1)
    back_spread = blur(back_dark, max(config['back_blur'], 0))

    if config['auto_gain']:
        gain, gain_pixels, gain_ok = _estimate_gain(front_dark, back_spread, geometry, config['protect_threshold'])
        if not gain_ok:
            gain = np.full(3, config['gain'], np.float32)
    else:
        gain, gain_pixels, gain_ok = np.full(3, config['gain'], np.float32), 0, False

    predicted = gain[None, None, :] * back_spread

    front_dark_luma = luminance(front_dark)
    protect_high = config['protect_threshold'] + max(config['protect_threshold'] * 1.5, .05)
    protection = (config['protect'] * smoothstep(config['protect_threshold'], protect_high, front_dark_luma)).astype(np.float32)

    correlation = _local_correlation(front_dark_luma, luminance(back_spread), geometry, config['reliable_window'])
    reliable = _reliability(front_dark_luma, luminance(back_spread), geometry, correlation, config['reliable_correlation'])

    active = (config['strength'] * (1 - protection) * reliable).astype(np.float32)
    correction = predicted * active[:, :, None]
    # 只向纸张底色方向提亮，不制造比纸基更白的像素，保留纸张底色
    ceiling = paper_front[None, None, :] - front
    correction = np.minimum(correction, np.maximum(ceiling, 0))
    result = np.clip(front + correction, 0, 1)

    bleed_luma = luminance(predicted * geometry[:, :, None])
    bleed_max = max(float(np.percentile(bleed_luma[geometry > .99], 99)) if np.any(geometry > .99) else 0., 1e-6)
    bleed_map = np.clip(bleed_luma / bleed_max, 0, 1).astype(np.float32)
    report = {
        'mode': 'bleedthrough',
        'gain': [float(item) for item in gain],
        'gain_auto': bool(config['auto_gain'] and gain_ok),
        'gain_pixels': int(gain_pixels),
        'reliable_fraction': float(np.mean(reliable > .5)),
        'geometric_overlap_fraction': float(np.mean(geometry > .99)),
        'protection_fraction': float(np.mean(protection > .25)),
        'median_local_correlation': float(np.median(correlation[geometry > .99])) if np.any(geometry > .99) else 0.,
        'mean_correction_luma': float(luminance(correction).mean()),
        'paper_white_front': [float(item) for item in paper_front],
        'mirrored_back': bool(frames[back_index].get('mirrored')),
        'color_space': 'sRGB 纸张暗度空间',
        'algorithm': '背面暗度稳健回归估计透印，笔画级局部归一化相关确定可靠重叠，正面墨迹软保护；不可靠区域不补偿。',
        'note': '抑制仅在两面几何重叠且局部相关达标的像素生效；正面自身淡字、纸张底色与细线由保护阈值与纸基上限保留。',
    }
    fields = {
        'image': result.astype(np.float32),
        'front_source': front.astype(np.float32),
        'back_registered': back.astype(np.float32),
        'valid': front_valid.astype(np.float32),
        'reliable': reliable,
        'protection': protection,
        'bleed_map': bleed_map,
    }
    return fields, report
