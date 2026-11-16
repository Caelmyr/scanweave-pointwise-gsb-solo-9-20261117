"""条带工作台补充测试：横向接续、走位错位、手动重排、旋转与非破坏性。"""
import io
import sys
import tempfile
import pathlib

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scanweave.store import Store            # noqa: E402
from scanweave.assets import Assets          # noqa: E402
from scanweave.stripstore import StripStore  # noqa: E402
from scanweave.stripservice import StripService  # noqa: E402
from tests.make_strip_fixture import make_scroll, cut_strips  # noqa: E402
from tests.test_strips import png_bytes      # noqa: E402


def horizontal_scroll(width=1500, height=300, seed=11):
    """横向长幅：转置纵向夹具得到左右接续的原稿。"""
    vertical = make_scroll(width=height, total_height=width, seed=seed)
    return np.transpose(vertical, (1, 0, 2)).copy()


def cut_horizontal(scroll, strip_width=360, overlap=90):
    h, w = scroll.shape[:2]
    out, x, i = [], 0, 0
    while x < w:
        x1 = min(w, x + strip_width)
        out.append(('hstrip_%02d' % i, scroll[:, x:x1, :].copy()))
        if x1 >= w:
            break
        x = x1 - overlap
        i += 1
    return out


def main():
    tmp_root = pathlib.Path(__file__).resolve().parents[1] / '.tmp_tests'
    tmp = tempfile.mkdtemp(prefix='strip_h_', dir=str(tmp_root))
    base = Store(tmp)
    assets = Assets(base)
    sstore = StripStore(tmp)
    svc = StripService(sstore, assets)

    print('== 横向接续 + 走位错位 ==')
    scroll = horizontal_scroll()
    true_strips = cut_horizontal(scroll)
    rng = np.random.default_rng(5)
    # 给每条带加轻微纵向错位（扫描走位）：平移并在边缘留空白，模拟真实走位而非卷绕
    jittered = []
    for name, img in true_strips:
        shift = int(rng.integers(-12, 13))
        moved = np.zeros_like(img)
        if shift >= 0:
            moved[shift:, :, :] = img[:img.shape[0] - shift, :, :]
        else:
            moved[:shift, :, :] = img[-shift:, :, :]
        jittered.append((name, moved))
    order = list(range(len(jittered)))
    rng.shuffle(order)
    mixed = [jittered[i] for i in order]
    ids = [assets.import_image(png_bytes(img), n + '.png')['id'] for n, img in mixed]
    sheet, _ = svc.create({'name': '横向长卷', 'axis': 'horizontal'})
    sid = sheet['id']
    sheet, _ = svc.add_strips(sid, {'revision': sheet['revision'], 'asset_ids': ids})
    sheet = svc.analyze(sid, {'revision': sheet['revision']})
    names = [s['name'] for s in sheet['strips']]
    true_names = [n + '.png' for n, _ in true_strips]
    print('恢复顺序:', names)
    assert names == true_names, '横向顺序恢复失败'
    errors = [i for i in sheet['issues'] if i['severity'] == 'error' and not i['resolved']]
    assert not errors, '横向场景不应有错误级疑点: %s' % errors
    xs = [s['position'][0] for s in sheet['strips']]
    assert all(xs[i + 1] > xs[i] for i in range(len(xs) - 1)), '横向主轴位置未递增'
    print('横向位置:', [round(x, 1) for x in xs])

    print('== 手动重排（修改前后关系）==')
    current = [s['id'] for s in sheet['strips']]
    swapped = [current[1], current[0]] + current[2:]
    sheet = svc.reorder(sid, {'revision': sheet['revision'], 'strip_ids': swapped})
    assert [s['id'] for s in sheet['strips']] == swapped
    manual = [e for e in sheet['edges'] if e['source'] == 'manual']
    assert manual and manual[0]['status'] == 'accepted', '手动重排应生成人工邻接边'
    assert sheet['confirmed'] is False, '重排后应重置确认'
    # 撤销恢复顺序
    sheet = svc.undo(sid, {'revision': sheet['revision']})
    assert [s['id'] for s in sheet['strips']] == current

    print('== 旋转仅影响摆放，不改动原素材 ==')
    target = sheet['strips'][0]['id']
    asset_before = assets.path(sheet['strips'][0]['asset']).read_bytes()
    sheet = svc.update_strip(sid, target, {'revision': sheet['revision'], 'rotation': 90})
    assert next(s for s in sheet['strips'] if s['id'] == target)['rotation'] == 90
    asset_after = assets.path(sheet['strips'][0]['asset']).read_bytes()
    assert asset_before == asset_after, '旋转不得改写原始素材'

    print('== 重置手动调整清除旋转与人工标记 ==')
    sheet = svc.reset_layout(sid, {'revision': sheet['revision']})
    assert all(s['rotation'] == 0 and not s['manual'] for s in sheet['strips'])
    assert all(i['resolved'] is False for i in sheet['issues'])

    print('== 确认与横向导出 ==')
    sheet = svc.confirm(sid, {'revision': sheet['revision']})
    assert sheet['confirmed'] is True
    raw, fname, mime = svc.export(sid, {'format': 'tiff'})
    exported = np.asarray(Image.open(io.BytesIO(raw)))
    print('横向导出:', fname, exported.shape, '原稿:', scroll.shape)
    # 走位错位会让包围盒在次轴/主轴略外扩以完整容纳每条带像素（不裁切），
    # 因此导出尺寸不小于原稿；走位幅度为 ±12，次轴两侧外扩合计约 24px，
    # 再计入匹配 cross 的次轴对齐，给定有界容差。
    assert exported.shape[0] >= scroll.shape[0]
    assert exported.shape[1] >= scroll.shape[1]
    assert exported.shape[0] - scroll.shape[0] <= 96, '次轴外扩超出走位合理范围'
    assert exported.shape[1] - scroll.shape[1] <= 96, '主轴外扩超出走位合理范围'
    # 导出图中央主体区域应与原稿内容一致（忽略边缘走位空白带）
    body = exported[:scroll.shape[0], :scroll.shape[1], :].astype(np.float32) / 255.0
    content = body.mean(axis=2) > 0.02
    if content.sum() > 1000:
        print('有效内容像素占比:', round(float(content.mean()), 3))
        assert content.mean() > 0.85, '拼接主体内容覆盖不足，可能存在间隙'

    print('== 移除条带后引用清理 ==')
    remove_id = sheet['strips'][-1]['id']
    sheet = svc.remove_strips(sid, {'revision': sheet['revision'], 'strip_ids': [remove_id]})
    assert all(remove_id not in (e['a'], e['b']) for e in sheet['edges'])
    assert all(remove_id not in i['strips'] for i in sheet['issues'])
    assert all(s['id'] != remove_id for s in sheet['strips'])

    print('== 旋转后合成包围盒按有效尺寸计算 ==')
    # 新建一个仅两条带的小布局，把第二条旋转 90°，验证导出不报错且尺寸合理
    small_imgs = [np.random.default_rng(1).random((200, 120, 3)).astype(np.float32),
                  np.random.default_rng(2).random((200, 120, 3)).astype(np.float32)]
    sids = [assets.import_image(png_bytes(im), 'rot_%d.png' % k)['id'] for k, im in enumerate(small_imgs)]
    rs, _ = svc.create({'name': '旋转', 'axis': 'vertical'})
    rid = rs['id']
    rs, _ = svc.add_strips(rid, {'revision': rs['revision'], 'asset_ids': sids})
    target = rs['strips'][1]['id']
    rs = svc.update_strip(rid, target, {'revision': rs['revision'], 'rotation': 90})
    # 旋转后手动给个不重叠位置以便确认
    rs = svc.update_strip(rid, rs['strips'][0]['id'],
                          {'revision': rs['revision'], 'position': [0.0, 0.0]})
    rs = svc.update_strip(rid, target,
                          {'revision': rs['revision'], 'position': [0.0, 220.0], 'rotation': 90})
    # 无 error 疑点即可确认（随机图无邻接 -> no_neighbor 是 error，先解决它）
    for issue in rs['issues']:
        if issue['severity'] == 'error' and not issue['resolved']:
            rs = svc.resolve_issue(rid, {'revision': rs['revision'],
                                         'issue_id': issue['id'], 'resolved': True})
    rs = svc.confirm(rid, {'revision': rs['revision']})
    raw, _, _ = svc.export(rid, {'format': 'png'})
    rot_out = np.asarray(Image.open(io.BytesIO(raw)))
    print('旋转布局导出尺寸:', rot_out.shape)
    # 第二条旋转后为 120 高、200 宽，放在 y=220，底沿应到 340；宽度取最大 200
    assert rot_out.shape[0] >= 340 and rot_out.shape[1] >= 200

    print('\n补充测试全部通过 ✔  数据目录:', tmp)


if __name__ == '__main__':
    main()
