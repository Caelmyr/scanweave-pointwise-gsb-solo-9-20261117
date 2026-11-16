#!/usr/bin/env python3
"""端到端验证像素贡献溯源：结果像素不变、贡献可解释、几何坐标往返一致。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scanweave import create_app
from scanweave.store import Store
from scanweave.assets import Assets
from scanweave.results import Results
from scanweave import model
from scanweave.explain import explain_pixel, inverse_geometry, clear_cache


def build_project(store, assets, name, frames, mode_settings=None, geometry=None, weights=None):
    project = store.create(name)
    rng = np.random.default_rng(7)
    with store.edit(project['id'], 0, '构建验证序列') as p:
        p['settings']['mode'] = 'denoise'
        p['settings']['size'] = 384
        if mode_settings:
            p['settings']['denoise'].update(mode_settings)
        if geometry:
            p['settings']['geometry'].update(geometry)
        for i, image in enumerate(frames):
            asset = assets.put(image.astype(np.float32), f'frame-{i}.png', {'synthetic': True})
            frame = model.frame(asset['id'], f'frame-{i}')
            if weights:
                frame['weight'] = weights[i]
            p['frames'].append(frame)
        p['reference'] = p['frames'][0]['id']
    return store.get(project['id'])


def synthetic_frames(count=6, h=260, w=340):
    rng = np.random.default_rng(11)
    base = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[:h, :w]
    base[..., 0] = .25 + .4 * xx / w
    base[..., 1] = .2 + .4 * yy / h
    base[..., 2] = .35
    base[60:120, 80:200] = [.7, .2, .15]
    frames = []
    for i in range(count):
        shifted = np.roll(base, shift=(i - 2) * 3, axis=1)
        noisy = np.clip(shifted + rng.normal(0, .01, shifted.shape), 0, 1).astype(np.float32)
        if i == count - 1:  # 异常帧：局部亮斑，应被 sigma/trimmed 剔除
            noisy[100:140, 150:190] = .98
        frames.append(noisy)
    return frames


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('  ok -', message)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(tmp)
        assets = Assets(store)
        results = Results(store, assets)
        frames = synthetic_frames()

        print('[1] sigma 降噪结果与重算一致性')
        project = build_project(store, assets, 'sigma 工程', frames, {'method': 'sigma', 'sigma': 2, 'iterations': 2, 'linear': True})
        meta = results.run(project)
        fields, meta = results.fields(project['id'], meta['id'], False)
        clear_cache()
        report = explain_pixel(results, project['id'], meta['id'], [170, 110])
        check(report['recomputed_matches_result'] is True, '重算降噪像素与存档 before_finishing 完全一致')
        check(len(report['frames']) == len(frames), '解释包含全部启用帧')
        for total in report['total_contribution_share']:
            check(abs(total - 1) < 1e-4, f'各通道贡献份额之和为 1（{total}）')
        check(all(0 <= s <= 1 for f in report['frames'] for s in f['contribution_share']), '所有份额位于 [0,1]')
        statuses = {f['status'] for f in report['frames']}
        check({'contributing', 'excluded', 'partially_excluded'} & statuses, '存在贡献/剔除状态区分: ' + str(statuses))
        excluded = [f for f in report['frames'] if f['status'] in ('excluded', 'partially_excluded')]
        if excluded:
            sample = excluded[0]
            check(sample['reason'] in ('sigma_outlier', 'partial_channel_outlier'), '剔除原因明确: ' + sample['reason'])
            check(any(it >= 0 for it in sample['reject_iteration']), '记录 sigma 剔除轮次')
            check(any(d > 0 for d in sample['reject_deviation']), '记录剔除偏离量')
            check(sum(sample['contribution_share']) == 0, '被剔除帧贡献份额为零')

        print('[2] median / trimmed / mean / weighted 方法可解释且份额自洽')
        for method in ('median', 'trimmed', 'mean', 'weighted'):
            p2 = build_project(store, assets, method + ' 工程', frames, {'method': method})
            m2 = results.run(p2)
            clear_cache()
            r2 = explain_pixel(results, p2['id'], m2['id'], [170, 110])
            check(r2['recomputed_matches_result'] is True, method + ' 重算像素一致')
            for total in r2['total_contribution_share']:
                check(abs(total - 1) < 1e-4, method + ' 贡献份额之和为 1')
            if method == 'median':
                selected = [f for f in report['frames']]  # 仅检查结构存在
                check(any('median_source' in f for f in r2['frames']), 'median 标出中值来源帧')
                check(sum(f['median_source'] for f in r2['frames']) >= 1, '至少一帧提供中值')

        print('[3] 无覆盖帧不显示为参与过计算')
        p3 = build_project(store, assets, '无覆盖工程', frames, {'method': 'mean'})
        with store.edit(p3['id'], p3['revision'], '停用一帧') as p:
            p['frames'][1]['enabled'] = False
        m3 = results.run(store.get(p3['id']))
        clear_cache()
        r3 = explain_pixel(results, p3['id'], m3['id'], [100, 100])
        result_frame_ids = {f['frame_id'] for f in r3['frames']}
        disabled_id = store.get(p3['id'])['frames'][1]['id']
        check(disabled_id not in result_frame_ids, '停用帧不出现在结果溯源帧列表中')
        check(len(r3['frames']) == len(frames) - 1, '溯源帧数等于结果实际启用帧数')

        print('[4] 停用异常帧并重新合成，解释随新结果更新')
        with store.edit(project['id'], project['revision'], '停用异常帧') as p:
            p['frames'][-1]['enabled'] = False
        project_new = store.get(project['id'])
        check(project_new['current_result'] is None, '签名变化后当前结果自动失效')
        meta_new = results.run(project_new)
        clear_cache()
        r_old = explain_pixel(results, project['id'], meta['id'], [170, 110])
        r_new = explain_pixel(results, project['id'], meta_new['id'], [170, 110])
        check(len(r_old['frames']) == len(frames) and len(r_new['frames']) == len(frames) - 1,
              '新结果解释只包含剩余启用帧')
        check(r_new['result_id'] != r_old['result_id'], '解释绑定各自的结果')
        check(r_new['stale'] is False, '当前结果的解释不标记过期')
        check(r_old['stale'] is True, '旧结果相对当前工程标记为过期但仍可解释')

        print('[5] 裁切与旋转/翻转后的点击映射回同一图像内容（以引擎几何链为基准）')
        from scanweave.explain import canvas_shape
        for rotation in (0, 90, 180, 270):
            for flip_x in (False, True):
                for flip_y in (False, True):
                    pg = build_project(store, assets, f'geo{rotation}{flip_x}{flip_y}', frames, {'method': 'mean'},
                                       geometry={'crop': [.1, .12, .9, .88], 'rotation': rotation,
                                                 'flip_x': flip_x, 'flip_y': flip_y, 'auto_crop': False})
                    mg = results.run(store.get(pg['id']))
                    _, mg = results.fields(pg['id'], mg['id'], False)
                    canvas_h, canvas_w = canvas_shape(results, pg['id'], mg['id'])
                    x0, y0, x1, y1 = mg['report']['manual_crop']
                    # 基准：在完整共同画布上放置唯一像素编号，再严格复刻引擎的 裁切→翻转→旋转。
                    marker = np.arange(canvas_h * canvas_w).reshape(canvas_h, canvas_w)
                    transformed = marker[y0:y1, x0:x1].copy()
                    if flip_x:
                        transformed = np.flip(transformed, 1)
                    if flip_y:
                        transformed = np.flip(transformed, 0)
                    transformed = np.rot90(transformed, k=-(rotation // 90))
                    gh, gw = transformed.shape
                    bad = 0
                    seen = set()
                    for dyy in range(gh):
                        for dxx in range(gw):
                            ground_y, ground_x = divmod(int(transformed[dyy, dxx]), canvas_w)
                            mapped = inverse_geometry(mg, mg['project'], dxx, dyy)
                            seen.add(mapped)
                            if mapped != (ground_x, ground_y):
                                bad += 1
                    check(bad == 0, f'rotation={rotation} flip_x={flip_x} flip_y={flip_y} 逆映射与引擎逐像素一致')
                    check(len(seen) == gw * gh, f'rotation={rotation} 逆映射一一对应无碰撞')

        # 用最后一个几何结果验证 explain_pixel 使用的坐标与 inverse_geometry 完全一致。
        r5 = explain_pixel(results, pg['id'], mg['id'], [gw // 2, gh // 2])
        check(r5['inside_canvas'] is True, '结果中心可映射到共同画布')
        check(r5['canvas_pixel'] == list(inverse_geometry(mg, mg['project'], gw // 2, gh // 2)), '解释使用相同逆映射坐标')

        print('[5b] 自动共同裁切（无手动裁切）时逆映射仍一一对应')
        pa = build_project(store, assets, '自动裁切', frames, {'method': 'mean'},
                           geometry={'rotation': 90, 'auto_crop': True})
        ma = results.run(store.get(pa['id']))
        _, ma = results.fields(pa['id'], ma['id'], False)
        auto_h, auto_w = canvas_shape(results, pa['id'], ma['id'])
        check(ma['report']['manual_crop'] == [0, 0, auto_w, auto_h], '全幅时手动裁切框为整幅画布')
        seen = set()
        for dyy in range(ma['height']):
            for dxx in range(ma['width']):
                seen.add(inverse_geometry(ma, ma['project'], dxx, dyy))
        check(len(seen) == ma['width'] * ma['height'], '自动裁切结果逆映射一一对应')

        print('[6] 结果像素未被解释流程修改')
        before = (store.directory(project['id']) / 'results' / meta_new['id'] / 'fields.npz').read_bytes()
        explain_pixel(results, project['id'], meta_new['id'], [50, 50], radius=2)
        after = (store.directory(project['id']) / 'results' / meta_new['id'] / 'fields.npz').read_bytes()
        check(before == after, '溯源调用前后 fields.npz 字节完全相同')

        print('[7] HTTP 层端到端')
        app = create_app(tmp)
        client = app.test_client()
        resp = client.post(f'/api/projects/{pa["id"]}/pixel-explain', json={'result_id': ma['id'], 'point': [.5, .5]})
        check(resp.status_code == 200, 'pixel-explain 返回 200')
        payload = resp.get_json()
        check(payload['result_id'] == ma['id'] and len(payload['frames']) == len(frames), 'HTTP 贡献帧完整')
        frame_id = payload['frames'][0]['frame_id']
        resp2 = client.get(f'/api/projects/{pa["id"]}/explain-frame/{frame_id}?result_id={ma["id"]}&marked=true&point=.5,.5')
        check(resp2.status_code == 200 and resp2.mimetype == 'image/png', '对齐帧标记图返回 PNG')
        resp3 = client.post(f'/api/projects/{pa["id"]}/pixel-explain', json={'result_id': ma['id'], 'point': [.5, .5], 'radius': 2})
        check(resp3.status_code == 200 and resp3.get_json()['radius'] == 2, '采样半径参数生效')
        # 非降噪结果应得到明确的 400，而不是静默错误数据。
        pf = build_project(store, assets, '融合工程', frames)
        with store.edit(pf['id'], pf['revision'], '切换为融合模式') as p:
            p['settings']['mode'] = 'fusion'
        mf = results.run(store.get(pf['id']))
        resp4 = client.post(f'/api/projects/{pf["id"]}/pixel-explain', json={'result_id': mf['id'], 'point': [.5, .5]})
        check(resp4.status_code == 400, '非降噪结果拒绝溯源并返回 400')

    print('\n全部验证通过 ✔')


if __name__ == '__main__':
    main()
