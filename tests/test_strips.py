"""条带工作台后端测试：算法、服务、HTTP API 与非破坏性保证。

运行（在仓库根目录，普通模式以便加载 /opt 下的依赖；测试数据使用独立临时目录）：
  python3 tests/test_strips.py
"""
import io
import sys
import tempfile
import pathlib

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scanweave import create_app            # noqa: E402
from scanweave.stripstore import StripStore  # noqa: E402
from scanweave.stripservice import StripService  # noqa: E402
from scanweave.assets import Assets          # noqa: E402
from scanweave import stripsolve             # noqa: E402
from tests.make_strip_fixture import make_scroll, cut_strips, shuffled  # noqa: E402


def png_bytes(image):
    return Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).tobytes() if False else \
        _encode(image)


def _encode(image):
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(buf, 'PNG')
    return buf.getvalue()


def make_assets(assets, strips):
    ids = []
    for name, image in strips:
        meta = assets.import_image(png_bytes(image), name + '.png')
        ids.append(meta['id'])
    return ids


def main():
    # 根分区可能只读/无空间，统一使用仓库所在独立盘下的临时目录
    tmp_root = pathlib.Path(__file__).resolve().parents[1] / '.tmp_tests'
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='strip_test_', dir=str(tmp_root))
    from scanweave.store import Store
    base_store = Store(tmp)
    assets = Assets(base_store)
    strip_store = StripStore(tmp)
    service = StripService(strip_store, assets)

    scroll = make_scroll()
    true_strips = cut_strips(scroll)
    order, mixed = shuffled(true_strips)
    print('真实顺序:', list(range(len(true_strips))), ' 导入乱序:', order)
    asset_ids = make_assets(assets, mixed)

    # 1) 创建布局并加入乱序条带
    sheet, _ = service.create({'name': '测试长卷', 'axis': 'vertical'})
    sid = sheet['id']
    sheet, _ = service.add_strips(sid, {'revision': sheet['revision'], 'asset_ids': asset_ids})
    assert len(sheet['strips']) == len(mixed), '条带数量不符'
    before_assets = {p.name for p in assets.root.glob('*')}

    # 2) 运行邻接分析
    sheet = service.analyze(sid, {'revision': sheet['revision']})
    analyzed_order = [s['name'] for s in sheet['strips']]
    true_names = [n + '.png' for n, _ in true_strips]
    print('分析后顺序:', analyzed_order)
    # 算法应恢复出真实顺序（内容纹理充分）
    assert analyzed_order == true_names, '顺序恢复失败: %s != %s' % (analyzed_order, true_names)
    # 强邻接边应覆盖全部相邻对
    strong = [e for e in sheet['edges'] if e['score'] >= stripsolve.SCORE_ACCEPT]
    print('邻接边数:', len(sheet['edges']), ' 强边:', len(strong))
    assert len(strong) >= len(true_strips) - 1, '强邻接边不足'

    # 3) 位置连续性：相邻条带主轴坐标应递增且有重叠
    positions = [s['position'] for s in sheet['strips']]
    ys = [p[1] for p in positions]
    assert all(ys[i + 1] > ys[i] for i in range(len(ys) - 1)), '主轴位置未递增'
    print('主轴位置:', [round(y, 1) for y in ys])

    # 4) 无 error 级疑点（纹理充分场景）
    errors = [i for i in sheet['issues'] if i['severity'] == 'error' and not i['resolved']]
    print('疑点:', [(i['kind'], i['severity']) for i in sheet['issues']])
    assert not errors, '充分纹理场景不应有错误级疑点: %s' % errors

    # 5) 确认门禁：未确认不能导出
    try:
        service.export(sid, {'format': 'png'})
        raise AssertionError('未确认布局不应允许导出')
    except Exception as exc:
        assert '尚未确认' in str(exc), '导出门禁提示错误: %s' % exc

    # 6) 确认后导出，且画布高度近似原稿长度
    sheet = service.confirm(sid, {'revision': sheet['revision']})
    assert sheet['confirmed'] is True
    raw, name, mime = service.export(sid, {'format': 'png'})
    exported = np.asarray(Image.open(io.BytesIO(raw)))
    print('导出:', name, mime, exported.shape, '原稿:', scroll.shape)
    # 正确重叠拼接后，导出长幅的高/宽应与原稿一致（条带完整覆盖、无间隙）
    assert exported.shape[0] == scroll.shape[0], '导出高度应等于原稿高度'
    assert exported.shape[1] == scroll.shape[1], '宽度应一致'
    # 像素级：重建图与原稿在大部分区域应高度一致（重叠羽化仅影响接缝附近）
    exported_f = exported.astype(np.float32) / 255.0
    diff = np.abs(exported_f - scroll).mean()
    print('导出与原稿平均绝对差:', round(float(diff), 4))
    assert diff < 0.06, '拼接结果与原稿偏差过大，邻接对齐可能错误'

    # 7) 非破坏性：原始素材文件集合不变、内容哈希仍可校验
    after_assets = {p.name for p in assets.root.glob('*')}
    assert before_assets == after_assets, '素材文件被改动或删除'
    for aid in asset_ids:
        report = assets.verify(aid)
        assert report['hash_valid'] and report['finite'], '素材完整性被破坏'

    # 8) 手动拖动修改位置 -> confirmed 复位，需要重新确认
    target = sheet['strips'][1]['id']
    sheet = service.update_strip(sid, target, {'revision': sheet['revision'],
                                               'position': [12.0, 88.0]})
    assert sheet['confirmed'] is False, '几何修改后应重置确认状态'
    moved = next(s for s in sheet['strips'] if s['id'] == target)
    assert moved['position'] == [12.0, 88.0] and moved['manual'] is True

    # 9) 撤销恢复
    sheet = service.undo(sid, {'revision': sheet['revision']})
    moved = next(s for s in sheet['strips'] if s['id'] == target)
    assert moved['manual'] is False, '撤销未恢复手动标记'

    print('\n== 场景 B：无纹理/无邻接与矛盾检测 ==')
    # 两张互不相关、内容平坦的图，应产生 no_neighbor 错误疑点且无法确认
    flat_a = np.full((300, 240, 3), 0.5, np.float32)
    flat_b = np.full((300, 240, 3), 0.62, np.float32)
    # 加一点点噪声但完全不相关
    rng = np.random.default_rng(0)
    flat_a += rng.normal(0, 0.01, flat_a.shape)
    flat_b += rng.normal(0, 0.01, flat_b.shape)
    ids2 = make_assets(assets, [('flat_a', flat_a), ('flat_b', flat_b)])
    s2, _ = service.create({'name': '无邻接', 'axis': 'vertical'})
    s2, _ = service.add_strips(s2['id'], {'revision': s2['revision'], 'asset_ids': ids2})
    s2 = service.analyze(s2['id'], {'revision': s2['revision']})
    kinds = {i['kind'] for i in s2['issues']}
    print('平坦无关图疑点:', kinds, ' 边评分:',
          [(round(e['score'], 3), e['status']) for e in s2['edges']])
    assert 'no_neighbor' in kinds or all(e['score'] < stripsolve.SCORE_ACCEPT for e in s2['edges']), \
        '无邻接场景应报告 no_neighbor 或无强边'
    if any(i['severity'] == 'error' and not i['resolved'] for i in s2['issues']):
        try:
            service.confirm(s2['id'], {'revision': s2['revision']})
            raise AssertionError('存在错误级疑点时不应允许确认')
        except Exception as exc:
            assert '错误级疑点' in str(exc)

    print('\n== 场景 C：HTTP API 冒烟 ==')
    app = create_app(tmp)
    client = app.test_client()
    r = client.get('/api/strip-sheets')
    assert r.status_code == 200 and len(r.get_json()) >= 2
    r = client.post('/api/strip-sheets', json={'name': 'API布局', 'axis': 'horizontal'})
    assert r.status_code == 201
    new_id = r.get_json()['id']
    r = client.get('/api/strip-sheets/' + new_id)
    assert r.status_code == 200
    r = client.get('/strip-workbench')
    assert r.status_code == 200 and '条带'.encode() in r.data
    r = client.get('/api/strip-sheets/' + sid + '/preview')
    assert r.status_code == 200 and r.mimetype == 'image/png'
    # 修订冲突
    r = client.patch('/api/strip-sheets/' + new_id, json={'revision': 999, 'name': 'x'})
    assert r.status_code == 409
    print('HTTP API 全部通过')

    print('\n全部后端测试通过 ✔  数据目录:', tmp)


if __name__ == '__main__':
    main()
