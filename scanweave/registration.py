import time
import cv2
import numpy as np
from .arrays import luminance, normalize, gradients
from .geometry import warp, normalized_matrix, overlap_metrics
from . import validation as v


# 可信度阈值：启发式参数，仅用于提示人工复核，不代表配准真值。
BLANK_GRADIENT = .0035
BLANK_VARIANCE = .005
PERIODIC_AUTOCORRELATION = .14
LOW_PHASE_RESPONSE = .12
HARD_PHASE_RESPONSE = .05
LOW_PHASE_PEAK_RATIO = 1.6
HARD_PHASE_PEAK_RATIO = 1.15
LOW_ORB_INLIERS = 12
HARD_ORB_INLIERS = 6
HIGH_REPROJECTION_ERROR = 2.5
LOW_ECC_CORRELATION = .55
HARD_ECC_CORRELATION = .25
HIGH_MTB_MISMATCH = .22
HARD_MTB_MISMATCH = .38
LOW_CORRELATION = .45
HARD_CORRELATION = .1
HIGH_NORMALIZED_MAE = .55
HARD_NORMALIZED_MAE = .9


def grayscale(image):
    return luminance(image).astype(np.float32)


def feature_gray(image):
    gray = normalize(grayscale(image), 1, 99)
    raster = np.rint(gray * 255).astype(np.uint8)
    return cv2.createCLAHE(2., (8, 8)).apply(raster)


def autocorrelation_energy(image):
    """重复图案的周期性强度（0–1）：同时考察空间域自相关峰与频域离散峰，
    细周期图案在自相关中会被中心屏蔽区盖住，因此补充 FFT 判别。"""
    gray = grayscale(image)
    scale = min(1., 256 / max(gray.shape))
    if scale < 1.:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    gray = gray - gray.mean()
    energy = float(np.mean(gray ** 2))
    if energy < 1e-10:
        return 0.
    correlation = cv2.matchTemplate(gray, gray, cv2.TM_CCOEFF_NORMED)
    h, w = correlation.shape
    cy, cx = h // 2, w // 2
    radius = max(4, min(h, w) // 12)
    correlation[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius] = 0
    spatial_peak = float(np.clip(np.nanmax(correlation), 0, 1))

    # FFT：周期性图案的能量集中在少数离散峰上，用 top-k 能量占比衡量；
    # 自然图像频谱连续分散，白噪声几乎均匀，两者该指标都明显更低。
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    sh, sw = spectrum.shape
    cy2, cx2 = sh // 2, sw // 2
    yy, xx = np.indices((sh, sw))
    radial = np.hypot(yy - cy2, xx - cx2)
    dc_mask = radial < max(5, min(sh, sw) // 16)
    spectrum[dc_mask] = 0
    total = float(spectrum.sum())
    if total > 1e-12:
        top_k = np.sort(spectrum.ravel())[-20:].sum() / total
    else:
        top_k = 0.
    spectral_score = float(np.clip((top_k - .15) / .08, 0, 1))
    return float(max(spatial_peak, spectral_score * .95))


def texture_report(image):
    """评估参考帧的可配准纹理：空白背景与重复图案都会让自动配准不可靠。"""
    gray = normalize(grayscale(image), 1, 99)
    dx, dy = gradients(gray)
    gradient_energy = float(np.mean(np.hypot(dx, dy)))
    variance = float(np.var(gray))
    periodic = autocorrelation_energy(image)
    labels = []
    if gradient_energy < BLANK_GRADIENT or variance < BLANK_VARIANCE:
        labels.append('blank')
    if periodic >= PERIODIC_AUTOCORRELATION:
        labels.append('repetitive')
    return {'gradient_energy': gradient_energy, 'variance': variance,
            'autocorrelation_peak': periodic, 'labels': labels,
            'note': 'blank=近空白低纹理，repetitive=重复图案导致位移歧义；均为启发式提示。'}


def phase_peak_ratio(a, b, window, shift):
    """主峰相对于次峰的突出程度；峰越不突出，相位位移越可能落在错误周期上。"""
    ga = np.fft.fft2(a * window)
    gb = np.fft.fft2(b * window)
    cross = ga * np.conj(gb)
    magnitude = np.abs(cross)
    surface = np.abs(np.fft.ifft2(np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-12)))
    h, w = surface.shape
    peak = surface[int(round(shift[1])) % h, int(round(shift[0])) % w]
    px, py = int(round(shift[0])), int(round(shift[1]))
    mask = np.ones((h, w), bool)
    for yy in range(max(0, py - 2), min(h, py + 3)):
        for xx in range(max(0, px - 2), min(w, px + 3)):
            mask[yy, xx] = False
    secondary = float(np.max(surface[mask])) if mask.any() else 0.
    ratio = float(peak / secondary) if secondary > 1e-9 else float('inf')
    return min(ratio, 100.)


def phase_align(reference, moving):
    a, b = grayscale(reference), grayscale(moving)
    a = np.log1p(a * 20)
    b = np.log1p(b * 20)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a, b, window)
    if not np.isfinite(shift).all():
        raise v.Invalid('相位相关无法估计位移')
    peak_ratio = phase_peak_ratio(a, b, window, shift)
    matrix = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]], np.float64)
    return matrix, {'method': 'phase', 'response': float(response),
                    'peak_ratio': float(peak_ratio), 'shift': list(shift)}


def orb_align(reference, moving, settings):
    detector = cv2.ORB_create(nfeatures=settings['features'], fastThreshold=10)
    a, des_a = detector.detectAndCompute(feature_gray(reference), None)
    b, des_b = detector.detectAndCompute(feature_gray(moving), None)
    if des_a is None or des_b is None or len(a) < 8 or len(b) < 8:
        raise v.Invalid('特征不足，建议使用相位相关或手动变换')
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = matcher.knnMatch(des_b, des_a, k=2)
    matches = [first for pair in candidates if len(pair) == 2 for first, second in [pair] if first.distance < settings['ratio'] * second.distance]
    if len(matches) < 6:
        raise v.Invalid('可靠匹配不足6对')
    source = np.float32([b[m.queryIdx].pt for m in matches])
    target = np.float32([a[m.trainIdx].pt for m in matches])
    if settings['motion'] == 'homography':
        matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, settings['ransac'])
    elif settings['motion'] == 'translation':
        differences = target - source
        shift = np.median(differences, axis=0)
        inliers = (np.linalg.norm(differences - shift, axis=1) < settings['ransac']).astype(np.uint8)
        matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], np.float64)
    else:
        estimator = cv2.estimateAffinePartial2D if settings['motion'] == 'euclidean' else cv2.estimateAffine2D
        affine, inliers = estimator(source, target, method=cv2.RANSAC, ransacReprojThreshold=settings['ransac'])
        matrix = np.vstack([affine, [0, 0, 1]]) if affine is not None else None
    if matrix is None or inliers is None or np.sum(inliers) < 5:
        raise v.Invalid('RANSAC无法获得稳定变换')
    projected = cv2.perspectiveTransform(source[None], matrix)[0]
    error = np.linalg.norm(projected - target, axis=1)
    selected = inliers.ravel() > 0
    lines = [{'source': source[i].tolist(), 'target': target[i].tolist(), 'inlier': bool(selected[i])} for i in np.linspace(0, len(matches) - 1, min(80, len(matches))).astype(int)]
    return matrix, {'method': 'orb', 'keypoints': [len(a), len(b)], 'matches': len(matches), 'inliers': int(selected.sum()), 'median_reprojection_error': float(np.median(error[selected])), 'match_lines': lines}


def ecc_align(reference, moving, settings):
    mapping = {'translation': cv2.MOTION_TRANSLATION, 'euclidean': cv2.MOTION_EUCLIDEAN, 'affine': cv2.MOTION_AFFINE, 'homography': cv2.MOTION_HOMOGRAPHY}
    motion = mapping[settings['motion']]
    seed, _ = phase_align(reference, moving)
    inverse = np.linalg.inv(seed).astype(np.float32)
    matrix = inverse if motion == cv2.MOTION_HOMOGRAPHY else inverse[:2]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, settings['iterations'], settings['epsilon'])
    try:
        score, result = cv2.findTransformECC(normalize(grayscale(reference), 1, 99), normalize(grayscale(moving), 1, 99), matrix, motion, criteria, None, 5)
    except cv2.error as error:
        raise v.Invalid('ECC未收敛，请先尝试相位相关或特征配准') from error
    if result.shape == (2, 3):
        result = np.vstack([result, [0, 0, 1]])
    return np.linalg.inv(result), {'method': 'ecc', 'correlation': float(score)}


def mtb_align(reference, moving):
    a = normalize(grayscale(reference), 1, 99)
    b = normalize(grayscale(moving), 1, 99)
    median_a, median_b = np.median(a), np.median(b)
    bitmap_a = (a > median_a).astype(np.float32)
    bitmap_b = (b > median_b).astype(np.float32)
    exclusion_a = (np.abs(a - median_a) > 4 / 255).astype(np.float32)
    exclusion_b = (np.abs(b - median_b) > 4 / 255).astype(np.float32)
    prior, prior_details = phase_align(reference, moving)
    seed_x, seed_y = int(round(prior[0, 2])), int(round(prior[1, 2]))
    h, w = a.shape
    best = None
    for dy in range(seed_y - 3, seed_y + 4):
        for dx in range(seed_x - 3, seed_x + 4):
            matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
            shifted = warp(bitmap_b, matrix, (w, h), True)
            exclusion = warp(exclusion_b, matrix, (w, h), True) * exclusion_a
            count = float(exclusion.sum())
            if count < h * w * .1:
                continue
            mismatch = float(np.sum(np.abs(bitmap_a - shifted) * exclusion) / count)
            tie_break = (dx - prior[0, 2]) ** 2 + (dy - prior[1, 2]) ** 2
            candidate = (mismatch, tie_break, dx, dy)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise v.Invalid('中值阈值位图缺少有效纹理')
    error, _, dx, dy = best
    matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    return matrix, {'method': 'mtb', 'shift': [dx, dy], 'bitmap_mismatch': error,
                    'phase_prior_response': prior_details['response'],
                    'phase_peak_ratio': prior_details['peak_ratio'],
                    'note': '相位初值附近搜索中值阈值位图异或误差，排除中灰不稳定像素。'}


def alignment_quality(reference, moving, matrix):
    h, w = reference.shape[:2]
    aligned = warp(moving, matrix, (w, h))
    support = warp(np.ones((h, w), np.float32), matrix, (w, h)) > .99
    if support.mean() < .01:
        return {'overlap': float(support.mean()), 'correlation': 0., 'normalized_mae': 1.}
    a = grayscale(reference)[support]
    b = grayscale(aligned)[support]
    a = (a - a.mean()) / max(a.std(), 1e-6)
    b = (b - b.mean()) / max(b.std(), 1e-6)
    return {'overlap': float(support.mean()), 'correlation': float(np.mean(a * b)), 'normalized_mae': float(np.mean(np.abs(a - b)))}


def assess_frame(details, quality, texture, minimum_overlap):
    """根据算法指标与纹理条件给出可信 / 存疑 / 失败判断和主要疑点。"""
    reasons = []
    hard_fail = False
    method = details.get('method')
    overlap = quality['overlap']
    if overlap < minimum_overlap:
        reasons.append('覆盖率低于设定下限（%.0f%%）' % (minimum_overlap * 100))
        hard_fail = True
    if 'blank' in texture['labels']:
        reasons.append('参考帧近空白、缺少可配准纹理')
    if 'repetitive' in texture['labels']:
        reasons.append('重复图案可能产生周期错位，建议目视确认')
    if method == 'phase':
        response = details.get('response', 0)
        peak_ratio = details.get('peak_ratio', 99)
        if response < LOW_PHASE_RESPONSE:
            reasons.append('相位相关响应过低（%.3f）' % response)
        if peak_ratio < LOW_PHASE_PEAK_RATIO:
            reasons.append('相位主峰不突出（峰比%.2f），位移可能落在错误周期' % peak_ratio)
        if response < HARD_PHASE_RESPONSE and peak_ratio < HARD_PHASE_PEAK_RATIO:
            hard_fail = True
    elif method == 'orb':
        inliers = details.get('inliers', 0)
        reprojection = details.get('median_reprojection_error', 0)
        if inliers < LOW_ORB_INLIERS:
            reasons.append('RANSAC内点偏少（%d）' % inliers)
        if reprojection > HIGH_REPROJECTION_ERROR:
            reasons.append('重投影误差偏大（%.2f像素）' % reprojection)
        if inliers < HARD_ORB_INLIERS:
            hard_fail = True
    elif method == 'ecc':
        correlation = details.get('correlation', 0)
        if correlation < LOW_ECC_CORRELATION:
            reasons.append('ECC相关系数偏低（%.3f）' % correlation)
        if correlation < HARD_ECC_CORRELATION:
            hard_fail = True
    elif method == 'mtb':
        mismatch = details.get('bitmap_mismatch', 0)
        if mismatch > HIGH_MTB_MISMATCH:
            reasons.append('中值位图失配率偏高（%.3f）' % mismatch)
        if details.get('phase_peak_ratio', 99) < LOW_PHASE_PEAK_RATIO:
            reasons.append('相位初值峰比偏低（%.2f）' % details.get('phase_peak_ratio', 0))
        if mismatch > HARD_MTB_MISMATCH:
            hard_fail = True
    if quality['correlation'] < LOW_CORRELATION:
        reasons.append('配准后帧间相关系数偏低（%.3f）' % quality['correlation'])
    if quality['normalized_mae'] > HIGH_NORMALIZED_MAE:
        reasons.append('配准后归一化平均差异偏大（%.3f）' % quality['normalized_mae'])
    if quality['correlation'] < HARD_CORRELATION or quality['normalized_mae'] > HARD_NORMALIZED_MAE:
        hard_fail = True

    score = 100
    score -= min(45, max(0, (LOW_CORRELATION - min(quality['correlation'], LOW_CORRELATION)) * 120))
    score -= min(25, max(0, (quality['normalized_mae'] - .25)) * 40)
    if method == 'phase':
        score -= min(20, max(0, LOW_PHASE_RESPONSE - details.get('response', 0)) * 120)
        score -= min(15, max(0, LOW_PHASE_PEAK_RATIO - details.get('peak_ratio', 99)) * 6)
    elif method == 'orb':
        score -= min(20, max(0, LOW_ORB_INLIERS - details.get('inliers', 0)) * 1.2)
        score -= min(15, max(0, details.get('median_reprojection_error', 0) - 1.) * 8)
    if 'blank' in texture['labels']:
        score = min(score, 34)
    if 'repetitive' in texture['labels']:
        score = min(score, 49)
    if hard_fail:
        score = min(score, 24)
    score = int(max(1, min(99, round(score))))
    status = 'failed' if hard_fail else ('suspect' if reasons else 'trusted')
    if hard_fail:
        reasons.insert(0, '自动配准结果不可信，已标记失败：该帧保留原变换，请手动处理')
    return status, score, reasons


def align_set(prepared, settings, progress=None, cancelled=None):
    images = prepared['images']
    reference_index = prepared['reference_index']
    reference = images[reference_index]
    width, height = prepared['size']
    texture = texture_report(reference)
    methods = {'phase': phase_align, 'orb': orb_align, 'ecc': ecc_align, 'mtb': mtb_align}
    result = []
    counts = {'trusted': 0, 'suspect': 0, 'failed': 0, 'aligned': 0}
    for index, (image, frame) in enumerate(zip(images, prepared['frames'])):
        if cancelled and cancelled():
            raise InterruptedError('配准任务已取消')
        started = time.monotonic()
        entry = {'frame_id': frame['id'], 'name': frame['name'], 'index': index,
                 'is_reference': index == reference_index, 'matrix': None,
                 'pixel_matrix': None, 'quality': None, 'details': None,
                 'status': 'failed', 'confidence_score': 0, 'reasons': [],
                 'seconds': 0.}
        try:
            if index == reference_index:
                matrix, details = np.eye(3), {'method': 'reference'}
            elif settings['method'] in ('orb', 'ecc'):
                matrix, details = methods[settings['method']](reference, image, settings)
            else:
                matrix, details = methods[settings['method']](reference, image)
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
                raise v.Invalid('配准得到退化变换')
            quality = alignment_quality(reference, image, matrix)
            entry.update(matrix=normalized_matrix(matrix, width, height),
                         pixel_matrix=matrix.tolist(), quality=quality, details=details,
                         seconds=round(time.monotonic() - started, 4))
            if index == reference_index:
                entry.update(status='trusted', confidence_score=100,
                             reasons=['参考帧使用恒等变换，无需配准'])
            else:
                status, score, reasons = assess_frame(details, quality, texture, settings['minimum_overlap'])
                entry.update(status=status, confidence_score=score, reasons=reasons)
                counts['aligned'] += 1
            counts[entry['status']] += 1
        except (v.Invalid, cv2.error) as error:
            entry['reasons'] = ['自动配准失败：' + str(error)]
            counts['failed'] += 1
        result.append(entry)
        if progress:
            progress(.35 + .55 * (index + 1) / len(images), '配准 ' + frame['name'])
    overlap = overlap_metrics(prepared['geometric_validities'])
    return {'frames': result, 'reference': prepared['frames'][reference_index]['id'],
            'size': [width, height], 'settings': settings,
            'reference_texture': texture, 'overlap': overlap,
            'summary': {'total': len(result), 'aligned': counts['aligned'],
                        'trusted': counts['trusted'], 'suspect': counts['suspect'],
                        'failed': counts['failed']},
            'note': '可信/存疑为基于纹理与配准指标的启发式判断；存疑与失败帧保留原变换，请逐帧目视复核后再接受。'}
