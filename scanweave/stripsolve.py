"""条带邻接分析与布局求解。

设计原则：
- 只根据图像内容“提出建议”，不强行给出看似完整的图。无法确定邻接或出现矛盾时，
  生成 issue 让用户确认，而不是静默挑选一条路径。
- 计算在缩小的工作图上进行，位置结果换算回条带原始工作像素坐标。
- 支持条带沿主轴(vertical/horizontal)接续，允许次轴错位（扫描走位）。

两阶段判别：
1. ORB 特征 + RANSAC 估计两条带的相对平移，并用“匹配点须落在相对边缘带内”过滤
   周期性纹理导致的非邻接虚假匹配。
2. 在 ORB 初值附近做局部归一化互相关(NCC)搜索，精修主轴/次轴位移并取得峰值评分，
   评分对亮度差异不敏感，能可靠区分真实邻接与无关条带。
"""
import numpy as np
import cv2
from .arrays import luminance, resize, fit_size
from . import stripmodel

# 分析用工作图的长边，控制 CPU 耗时
ANALYSIS_LONG_EDGE = 768
# 边缘带占条带长边的比例
BAND_FRACTION = 0.45
# 评分阈值：真实邻接在精确对齐后 NCC 接近 1，自相似无关条带通常低于 0.72
SCORE_ACCEPT = 0.75
SCORE_WEAK = 0.40
# 重叠合理范围（占条带长边比例）
OVERLAP_MIN = 0.08
OVERLAP_MAX = 0.92
MARGIN_RATIO = 0.25        # 次轴错位超过条带次轴尺寸的该比例，标记错位疑点
GAP_FRACTION = 0.08        # 推断重叠比例低于此值，提示接缝间隙
ORB_FEATURES = 3000
RATIO = 0.75
RANSAC_THRESHOLD = 4.0
MIN_BAND_INLIERS = 2
MIN_INLIERS = 8


def _gray(image):
    gray = luminance(image).astype(np.float32)
    low, high = np.percentile(gray, (1, 99))
    if high - low < 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip((gray - low) / (high - low), 0, 1).astype(np.float32)


def _prepare(image):
    """归一化到分析工作尺寸，返回 (灰度float, 缩放比=工作/原)。"""
    h, w = image.shape[:2]
    tw, th = fit_size(image.shape, ANALYSIS_LONG_EDGE)
    scaled = resize(image, tw, th)
    return _gray(scaled), tw / w


def _orb_gray(gray):
    return np.rint(gray * 255).astype(np.uint8)


def _ncc(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
    return float((a * b).sum() / den)


def _masked_ncc(a, b, valid):
    """仅在 valid（两窗口都有有效内容）的像素上计算归一化互相关。

    扫描走位产生的边缘空白（恒定零区）不参与统计，避免拉低真实邻接评分。
    """
    if valid.sum() < 64:
        return -2.0
    av = a[valid].astype(np.float32)
    bv = b[valid].astype(np.float32)
    av = av - av.mean()
    bv = bv - bv.mean()
    den = np.sqrt((av * av).sum() * (bv * bv).sum()) + 1e-9
    return float((av * bv).sum() / den)


class PreparedStrip:
    def __init__(self, image):
        self.gray, self.scale = _prepare(image)
        self.h, self.w = self.gray.shape
        # 有效内容掩膜：与局部邻域存在差异即为真实内容，恒定空白区为 False
        content = (self.gray > 0.02).astype(np.float32)
        neighbor = cv2.blur(content, (5, 5))
        self.valid = (neighbor > 0.5).astype(np.float32)
        detector = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=8)
        self.keypoints, self.descriptors = detector.detectAndCompute(_orb_gray(self.gray), None)


def _orb_translation(ps_a, ps_b):
    """估计 a 到 b 的仿射平移（dx,dy，工作像素），返回 (dx,dy,内点数,带内点数) 或 None。"""
    if ps_a.descriptors is None or ps_b.descriptors is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(ps_a.descriptors, ps_b.descriptors, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([ps_a.keypoints[m.queryIdx].pt for m in good])
    dst = np.float32([ps_b.keypoints[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD)
    if matrix is None or inliers is None:
        return None
    inliers = inliers.ravel() > 0
    if int(inliers.sum()) < MIN_INLIERS:
        return None
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    return dx, dy, int(inliers.sum()), src[inliers], dst[inliers]


def _band_inlier_count(src, dst, ps_a, ps_b, direction, band):
    """统计落在两条带相对边缘带内的匹配内点数。

    direction 'ab' 表示 a 在上/左、b 在下/右（a 末端接 b 始端）。
    """
    if direction == 'ab':
        near_end_a = src[:, 1] >= ps_a.h - band
        near_start_b = dst[:, 1] <= band
    else:
        near_end_a = dst[:, 1] >= ps_b.h - band
        near_start_b = src[:, 1] <= band
    return int((near_end_a & near_start_b).sum())


def _band_inlier_count_h(src, dst, ps_a, ps_b, direction, band):
    if direction == 'ab':
        near_end_a = src[:, 0] >= ps_a.w - band
        near_start_b = dst[:, 0] <= band
    else:
        near_end_a = dst[:, 0] >= ps_b.w - band
        near_start_b = src[:, 0] <= band
    return int((near_end_a & near_start_b).sum())


def _aligned_ncc(ps_a, ps_b, direction, overlap, cross, axis):
    """按给定重叠量与次轴错位，取两条带等大重叠窗口计算掩膜 NCC。

    cross 表示 b 相对 a 在次轴上的位移（正=向下/向右）。
    走位产生的空白边缘通过有效内容掩膜剔除，不参与评分。
    """
    ga, gb = ps_a.gray, ps_b.gray
    va, vb = ps_a.valid, ps_b.valid
    overlap = int(round(overlap))
    cross = int(round(cross))
    if overlap < 12:
        return -2.0
    if axis == 'vertical':
        Ha, Wa = ga.shape
        Hb, Wb = gb.shape
        if overlap > min(Ha, Hb):
            return -2.0
        if direction == 'ab':
            wa, ma = ga[Ha - overlap:Ha, :], va[Ha - overlap:Ha, :]
            wb, mb = gb[0:overlap, :], vb[0:overlap, :]
        else:
            wa, ma = ga[0:overlap, :], va[0:overlap, :]
            wb, mb = gb[Hb - overlap:Hb, :], vb[Hb - overlap:Hb, :]
        a_x0 = max(0, -cross)
        b_x0 = max(0, cross)
        cw = min(Wa - a_x0, Wb - b_x0)
        if cw < 24:
            return -2.0
        a, b = wa[:, a_x0:a_x0 + cw], wb[:, b_x0:b_x0 + cw]
        valid = (ma[:, a_x0:a_x0 + cw] > .5) & (mb[:, b_x0:b_x0 + cw] > .5)
    else:
        Ha, Wa = ga.shape
        Hb, Wb = gb.shape
        if overlap > min(Wa, Wb):
            return -2.0
        if direction == 'ab':
            wa, ma = ga[:, Wa - overlap:Wa], va[:, Wa - overlap:Wa]
            wb, mb = gb[:, 0:overlap], vb[:, 0:overlap]
        else:
            wa, ma = ga[:, 0:overlap], va[:, 0:overlap]
            wb, mb = gb[:, Wb - overlap:Wb], vb[:, Wb - overlap:Wb]
        a_y0 = max(0, -cross)
        b_y0 = max(0, cross)
        ch = min(Ha - a_y0, Hb - b_y0)
        if ch < 24:
            return -2.0
        a, b = wa[a_y0:a_y0 + ch, :], wb[b_y0:b_y0 + ch, :]
        valid = (ma[a_y0:a_y0 + ch, :] > .5) & (mb[b_y0:b_y0 + ch, :] > .5)
    if a.shape != b.shape or min(a.shape[:2]) < 12 or a.size == 0:
        return -2.0
    return _masked_ncc(a, b, valid)


def _coarse_to_fine(ps_a, ps_b, direction, axis, cross_hint):
    """方向确定后，主轴在全重叠范围粗到精搜索，并独立做全范围次轴扫描。

    不依赖 ORB 的次轴提示（周期纹理下可能错误）；ORB 仅用于确定接续方向。
    返回 (best_score, best_overlap, best_cross)，均为工作像素。
    """
    ga, gb = ps_a.gray, ps_b.gray
    main_len = min(ga.shape[0], gb.shape[0]) if axis == 'vertical' \
        else min(ga.shape[1], gb.shape[1])
    cross_size = ga.shape[1] if axis == 'vertical' else ga.shape[0]
    lo = int(OVERLAP_MIN * main_len)
    hi = int(OVERLAP_MAX * main_len)
    cross_limit = int(0.4 * cross_size)
    cross_grid = np.arange(-cross_limit, cross_limit + 1, 6)

    def best_at(overlap):
        best = (-2.0, float(cross_hint))
        for cr in cross_grid:
            score = _aligned_ncc(ps_a, ps_b, direction, overlap, cr, axis)
            if score > best[0]:
                best = (score, float(cr))
        return best

    # 1) 主轴粗扫（每个候选独立全范围次轴扫描）
    coarse_ov = np.unique(np.linspace(lo, hi, 16).astype(int))
    best_score, best_ov, best_cross = -2.0, lo, 0.0
    for ov in coarse_ov:
        score, cr = best_at(ov)
        if score > best_score:
            best_score, best_ov, best_cross = score, ov, cr
    # 2) 主轴细化（峰值附近，步长 2）
    span = max(8, int(0.16 * main_len))
    fine_ov = np.unique(np.arange(max(lo, best_ov - span),
                                  min(hi, best_ov + span) + 1, 2).astype(int))
    for ov in fine_ov:
        score, cr = best_at(ov)
        if score > best_score:
            best_score, best_ov, best_cross = score, ov, cr
    # 3) 次轴在最优重叠上细化（步长 2）
    fine_cross = np.arange(max(-cross_limit, best_cross - 6),
                           min(cross_limit, best_cross + 6) + 1, 2)
    for cr in fine_cross:
        score = _aligned_ncc(ps_a, ps_b, direction, best_ov, cr, axis)
        if score > best_score:
            best_score, best_cross = score, float(cr)
    # 4) 主轴单像素收尾
    for ov in range(max(lo, best_ov - 2), min(hi, best_ov + 2) + 1):
        score = _aligned_ncc(ps_a, ps_b, direction, ov, best_cross, axis)
        if score > best_score:
            best_score, best_ov = score, ov
    return best_score, float(best_ov), float(best_cross)


def analyze_pair(ps_a, ps_b, axis):
    """分析两条带，返回最佳有向邻接 dict 或 None。

    两个接续方向都做粗到精 NCC 扫描并取高分者，从而不依赖 ORB 在周期纹理下
    可能判反的方向。ORB 仅提供次轴错位提示与可信度参考。
    方向 'ab' = a 在 b 前（a 末端接 b 始端）；'ba' 反之。
    """
    ga, gb = ps_a.gray, ps_b.gray
    main_len = min(ga.shape[0], gb.shape[0]) if axis == 'vertical' \
        else min(ga.shape[1], gb.shape[1])
    cross_hint_a = 0.0
    inliers = 0
    band_inliers = 0
    orb = _orb_translation(ps_a, ps_b)
    if orb is not None:
        dx, dy, inliers, src, dst = orb
        if axis == 'vertical' and abs(dy) >= 0.12 * main_len:
            cross_hint_a = dx
            orb_dir = 'ab' if dy < 0 else 'ba'
            band = int(round(BAND_FRACTION * main_len))
            band_inliers = (_band_inlier_count if axis == 'vertical'
                            else _band_inlier_count_h)(src, dst, ps_a, ps_b, orb_dir, band)
        elif axis == 'horizontal' and abs(dx) >= 0.12 * main_len:
            cross_hint_a = dy
            orb_dir = 'ab' if dx < 0 else 'ba'
            band = int(round(BAND_FRACTION * main_len))
            band_inliers = _band_inlier_count_h(src, dst, ps_a, ps_b, orb_dir, band)

    candidates = []
    for direction in ('ab', 'ba'):
        # ba 方向时次轴提示取反（b 相对 a 的反向视角）
        hint = cross_hint_a if direction == 'ab' else -cross_hint_a
        score, overlap, cross = _coarse_to_fine(ps_a, ps_b, direction, axis, hint)
        candidates.append((score, direction, overlap, cross))
    score, direction, overlap, cross = max(candidates, key=lambda c: c[0])
    if score < SCORE_WEAK:
        return None
    return {
        'direction': direction, 'score': float(score),
        'cross': float(cross), 'overlap': float(overlap),
        'overlap_fraction': float(np.clip(overlap / max(main_len, 1), 0, 1)),
        'inliers': inliers, 'band_inliers': band_inliers,
        'cross_size': ps_a.w if axis == 'vertical' else ps_a.h,
        'main_size_a': ps_a.h if axis == 'vertical' else ps_a.w,
        'main_size_b': ps_b.h if axis == 'vertical' else ps_b.w,
    }


def build_graph(prepared, strips, axis):
    """两两分析，返回无向条带对的最佳有向邻接列表（含两端条带 id）。"""
    edges = []
    n = len(prepared)
    for i in range(n):
        for j in range(i + 1, n):
            result = analyze_pair(prepared[i], prepared[j], axis)
            if result is None:
                continue
            if result['direction'] == 'ab':
                a_id, b_id = strips[i]['id'], strips[j]['id']
                ps_a, ps_b = prepared[i], prepared[j]
            else:
                a_id, b_id = strips[j]['id'], strips[i]['id']
                ps_a, ps_b = prepared[j], prepared[i]
            result.update({'a': a_id, 'b': b_id,
                           'scale_a': ps_a.scale, 'scale_b': ps_b.scale})
            edges.append(result)
    edges.sort(key=lambda e: -e['score'])
    return edges


def _to_model_edge(edge):
    cross_px = edge['cross'] / edge['scale_b']
    overlap_px = edge['overlap'] / edge['scale_a']
    main_a = edge['main_size_a'] / edge['scale_a']
    return {
        'a': edge['a'], 'b': edge['b'], 'side': 'after',
        'score': round(float(np.clip(edge['score'], 0, 1)), 4),
        'overlap_fraction': round(float(np.clip(overlap_px / max(main_a, 1), 0, 1)), 4),
        'offset': [round(cross_px, 2), 0.0],
        'status': 'candidate', 'source': 'content',
        '_cross_px': round(cross_px, 2),
        '_overlap_px': round(overlap_px, 2),
    }


def solve_layout(strips, images, axis='vertical'):
    """完整求解：建图 -> 排序建议 -> 位置 -> 疑点。

    返回 (model_edges, issues, positions, order)。
    positions: {strip_id: [x, y]} 条带原始工作像素坐标。
    """
    if len(strips) < 2:
        return [], [], {s['id']: list(s['position']) for s in strips}, [s['id'] for s in strips]

    prepared = [PreparedStrip(img) for img in images]
    raw_edges = build_graph(prepared, strips, axis)
    model_edges = [_to_model_edge(e) for e in raw_edges]
    by_id = {s['id']: s for s in strips}
    issues = []

    def issue(kind, severity, message, refs):
        issues.append({'id': stripmodel.uid(), 'kind': kind, 'severity': severity,
                       'message': message, 'strips': refs, 'resolved': False})

    strong = [e for e in raw_edges if e['score'] >= SCORE_ACCEPT]
    weak = [e for e in raw_edges if SCORE_WEAK <= e['score'] < SCORE_ACCEPT]

    # 每个节点只保留其评分最高的一条出边 / 入边，构造线性链图，避免分叉误报
    best_next, best_prev = {}, {}
    for e in sorted(strong, key=lambda x: -x['score']):
        best_next.setdefault(e['a'], e)
        best_prev.setdefault(e['b'], e)
    chain_edges = list(best_next.values())

    # 歧义：在完整强边图中，某节点存在评分接近的多个候选
    outgoing, incoming = {}, {}
    for e in strong:
        outgoing.setdefault(e['a'], []).append(e)
        incoming.setdefault(e['b'], []).append(e)
    for a, candidates in outgoing.items():
        if len(candidates) > 1:
            top = sorted(candidates, key=lambda x: -x['score'])
            if top[0]['score'] - top[1]['score'] < 0.12:
                issue('ambiguous_neighbor', 'warning',
                      '条带存在多个相近的后继邻接，无法唯一确定先后关系，请人工确认。',
                      [a] + [c['b'] for c in top[:3]])
    for b, candidates in incoming.items():
        if len(candidates) > 1:
            top = sorted(candidates, key=lambda x: -x['score'])
            if top[0]['score'] - top[1]['score'] < 0.12:
                issue('ambiguous_neighbor', 'warning',
                      '条带存在多个相近的前驱邻接，拼接顺序存在歧义，请人工确认。',
                      [b] + [c['a'] for c in top[:3]])

    for e in weak:
        issue('weak_match', 'warning',
              '条带间重叠相似度偏低（评分 %.2f），邻接关系不确定，请确认是否相邻。' % e['score'],
              [e['a'], e['b']])

    connected = set()
    for e in chain_edges:
        connected.add(e['a'])
        connected.add(e['b'])
    for s in strips:
        if s['id'] not in connected:
            issue('no_neighbor', 'error',
                  '未找到与该条带可信相邻的其他条带，无法自动确定其位置。',
                  [s['id']])

    order, cycle_pairs = _greedy_order(strips, chain_edges)
    for pair in cycle_pairs:
        issue('cycle', 'error',
              '邻接关系出现相互指向的闭环，先后顺序自相矛盾，已按最低置信边断开，请人工裁决。',
              pair)
    positions = _place_strips(strips, order, raw_edges, axis, by_id)

    # 间隙与错位疑点
    for e in raw_edges:
        if e['score'] >= SCORE_ACCEPT:
            if e['overlap_fraction'] < GAP_FRACTION:
                issue('gap', 'warning',
                      '推断的相邻重叠比例过小（%.1f%%），接缝处可能存在间隙或内容缺失。'
                      % (e['overlap_fraction'] * 100), [e['a'], e['b']])
            cross_ratio = abs(e['cross']) / max(e['cross_size'], 1)
            if cross_ratio > MARGIN_RATIO:
                issue('conflict', 'warning',
                      '相邻条带横向错位约 %.0f%%，请确认对齐或扫描走位。' % (cross_ratio * 100),
                      [e['a'], e['b']])

    for me in model_edges:
        for internal in ('_cross_px', '_overlap_px', '_shift_dx', '_shift_dy'):
            me.pop(internal, None)
    return model_edges, issues, positions, order


def _greedy_order(strips, chain_edges):
    """基于每节点唯一最优出边的链图构造线性序列，并破除闭环。

    链图中每个节点至多一条出边。入度为 0 的节点是链头；若某个连通分量没有
    链头，说明它是闭合环，此时断开该环上评分最低的边，产生一个链头，并把该
    闭环作为疑点对返回（而不是静默选择）。
    返回 (order, cycle_pairs)。
    """
    ids = [s['id'] for s in strips]
    if not chain_edges:
        return ids, []
    best_next, best_prev = {}, {}
    for e in sorted(chain_edges, key=lambda x: -x['score']):
        best_next.setdefault(e['a'], e)
        best_prev.setdefault(e['b'], e)

    cycle_pairs = []
    # 找闭环分量：从任一未被链头覆盖的节点沿出边追踪，回到已在当前路径上的节点即环
    heads = [i for i in ids if i not in best_prev]
    reachable = set()
    for head in heads:
        node = head
        while node is not None and node not in reachable:
            reachable.add(node)
            node = best_next[node]['b'] if node in best_next else None
    # 不在任何链头可达集合内的节点属于闭环分量
    for start in ids:
        if start in reachable:
            continue
        path, index = [], {}
        node = start
        while node not in index:
            index[node] = len(path)
            path.append(node)
            if node not in best_next:
                break
            node = best_next[node]['b']
        if node in index:
            ring = path[index[node]:]
            # 环上评分最低的边断开
            ring_edges = [best_next[n] for n in ring if n in best_next]
            weakest = min(ring_edges, key=lambda e: e['score'])
            cycle_pairs.append([weakest['a'], weakest['b']])
            del best_next[weakest['a']]
            best_prev.pop(weakest['b'], None)
        for n in path:
            reachable.add(n)

    # 基于破除环后的链图，从所有链头顺序遍历
    best_prev = {}
    for e in best_next.values():
        best_prev.setdefault(e['b'], e)
    heads = [i for i in ids if i not in best_prev]
    visited, order = set(), []
    for head in heads:
        node = head
        while node is not None and node not in visited:
            visited.add(node)
            order.append(node)
            edge = best_next.get(node)
            node = edge['b'] if edge else None
    for sid in ids:
        if sid not in visited:
            visited.add(sid)
            order.append(sid)
    return order, cycle_pairs


def _place_strips(strips, order, raw_edges, axis, by_id):
    """沿主轴摆放，次轴应用精修后的错位；重叠量由匹配决定。

    每条带的位置直接由前一条带的位置推导：
        主轴 = 前一条主轴 + 前一条主轴长度 − 两者重叠量
    链头放在原点；与前一条没有可信邻接边的断链条带，按自身长度无重叠堆叠，
    因此不会产生虚假间隙，也不会整体错位一个条带高度。
    """
    measure = {(e['a'], e['b']): e for e in raw_edges if e['score'] >= SCORE_ACCEPT}
    positions = {}
    prev_id = None
    for sid in order:
        strip = by_id[sid]
        w, h = strip['width'], strip['height']
        edge = measure.get((prev_id, sid)) if prev_id else None
        if edge is not None and prev_id in positions:
            overlap_px = edge['overlap'] / edge['scale_a']
            cross_px = edge['cross'] / edge['scale_b']
            prev_pos = positions[prev_id]
            if axis == 'vertical':
                main_pos = prev_pos[1] + by_id[prev_id]['height'] - overlap_px
                positions[sid] = [round(cross_px, 2), round(main_pos, 2)]
            else:
                main_pos = prev_pos[0] + by_id[prev_id]['width'] - overlap_px
                positions[sid] = [round(main_pos, 2), round(cross_px, 2)]
        else:
            # 链头或断链：紧贴当前已布置内容堆叠（无重叠信息时不留间隙）
            if axis == 'vertical':
                base = max((p[1] + by_id[k]['height'] for k, p in positions.items()), default=0.0)
                positions[sid] = [0.0, round(base, 2)]
            else:
                base = max((p[0] + by_id[k]['width'] for k, p in positions.items()), default=0.0)
                positions[sid] = [round(base, 2), 0.0]
        prev_id = sid
    min_x = min(p[0] for p in positions.values())
    min_y = min(p[1] for p in positions.values())
    return {sid: [round(p[0] - min_x, 2), round(p[1] - min_y, 2)] for sid, p in positions.items()}


def canvas_size(strips, positions, axis):
    max_x = max_y = 0.0
    for s in strips:
        x, y = positions[s['id']]
        max_x = max(max_x, x + s['width'])
        max_y = max(max_y, y + s['height'])
    return int(np.ceil(max_x)), int(np.ceil(max_y))
