"""批次色彩校正的回归测试，可直接 `python3 tests/test_color_calibration.py` 运行。

覆盖：色块不足/颜色聚集/截断等阻断提示、留一交叉验证的坏色块识别、
方案持久化与应用互斥、合成与导出使用校正、原始素材不变、工程包往返。
"""
import io
import shutil
import sys
import time
import zipfile
import json as js
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanweave import create_app
from scanweave.scanner import install, CHART_PALETTE
from scanweave.colorcal import analyze_patches
import numpy as np


def wait_job(client, job_id):
    for _ in range(100):
        job = client.get('/api/jobs/' + job_id).json
        if job['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
            return job
        time.sleep(.3)
    raise AssertionError('任务未结束')


def main():
    data_dir = Path('/workspace/.test-data/color-cal-%d' % int(time.time()))
    app = create_app(str(data_dir))
    install(app)
    client = app.test_client()
    project = client.post('/api/scanner/sample', json={}).json
    key = project['id']
    palette = client.get('/api/scanner/chart-palette').json
    assert len(palette) == 12 and palette[0]['bounds'], '样张色卡应自带位置'
    patches = [{'name': q['name'], 'bounds': q['bounds'], 'target': q['target']} for q in palette]

    def analyze(extra=None, model='matrix'):
        body = {'frame_id': project['reference'], 'model': model, 'patches': patches}
        if extra:
            body.update(extra)
        return client.post('/api/projects/%s/color-plans/analyze' % key, json=body).json

    # 1. 正常拟合：平均色差显著下降且全部色块可靠
    result = analyze()
    assert result['usable'], result['issues']
    assert result['reliable_count'] == 12
    assert result['fit']['error']['mean'] < result['fit']['error_before']['mean'] * .4
    print('1. 正常拟合 ΔE %.2f -> %.2f' % (result['fit']['error_before']['mean'], result['fit']['error']['mean']))

    # 2. 色块不足：阻断且不允许保存
    weak = client.post('/api/projects/%s/color-plans/analyze' % key,
                       json={'frame_id': project['reference'], 'patches': patches[:2]}).json
    assert not weak['usable'] and any('至少需要' in issue for issue in weak['issues'])
    refused = client.post('/api/projects/%s/color-plans' % key,
                          json={'revision': project['revision'], 'name': '坏方案',
                                'frame_id': project['reference'], 'patches': patches[:2]})
    assert refused.status_code == 400
    print('2. 色块不足被阻断：', weak['issues'][0])

    # 3. 颜色全部相近：阻断
    clustered = [{'name': 'c', 'bounds': palette[i % 12]['bounds'], 'target': '#808080'} for i in range(4)]
    crowded = client.post('/api/projects/%s/color-plans/analyze' % key,
                          json={'frame_id': project['reference'], 'patches': clustered}).json
    assert not crowded['usable'] and any('集中在相近颜色' in issue for issue in crowded['issues'])
    print('3. 颜色聚集被阻断：', crowded['issues'][0])

    # 4. 采样截断：色块被标记不可靠
    blown = np.full((80, 80, 3), 1.0, np.float32)
    blown[40:] = .5
    clip_result = analyze_patches(blown, [
        {'name': 'blown', 'bounds': [.05, .05, .45, .45], 'target': '#f0f0f0'},
        {'name': 'a', 'bounds': [.55, .55, .7, .7], 'target': '#808080'},
        {'name': 'b', 'bounds': [.72, .55, .88, .7], 'target': '#906060'}], 'matrix')
    assert not clip_result['patches'][0]['reliable']
    assert any('截断' in flag for flag in clip_result['patches'][0]['flags'])
    print('4. 截断色块不可靠：', clip_result['patches'][0]['flags'][0])

    # 5. 留一交叉验证：错误目标颜色被识别，正常方案无误报
    bad_patches = [dict(q, target='#ff00ff' if i == 5 else q['target']) for i, q in enumerate(patches)]
    bad = client.post('/api/projects/%s/color-plans/analyze' % key,
                      json={'frame_id': project['reference'], 'patches': bad_patches}).json
    flagged = [q['index'] for q in bad['patches'] if not q['reliable']]
    assert flagged == [5], flagged
    assert all(q['reliable'] for q in result['patches'])
    print('5. 坏色块仅第6个被标记，正常方案无误报')

    # 6. 保存、应用互斥、持久化
    saved = client.post('/api/projects/%s/color-plans' % key,
                        json={'revision': project['revision'], 'name': '方案A',
                              'frame_id': project['reference'], 'model': 'matrix',
                              'patches': patches}).json
    plan_a = saved['plan']
    applied = client.post('/api/projects/%s/color-plans/%s/apply' % (key, plan_a['id']),
                          json={'revision': saved['project']['revision']}).json
    assert applied['plan']['applied']
    revision = applied['project']['revision']
    saved_b = client.post('/api/projects/%s/color-plans' % key,
                          json={'revision': revision, 'name': '方案B',
                                'frame_id': project['reference'], 'model': 'affine',
                                'patches': patches}).json
    applied_b = client.post('/api/projects/%s/color-plans/%s/apply' % (key, saved_b['plan']['id']),
                            json={'revision': saved_b['project']['revision']}).json
    states = {p['name']: p['applied'] for p in applied_b['project']['settings']['color_plans']}
    assert states == {'方案A': False, '方案B': True}
    reopened = client.get('/api/projects/%s' % key).json
    assert len(reopened['settings']['color_plans']) == 2
    assert reopened['settings']['color_plans'][1]['patches'][0]['target'].startswith('#')
    print('6. 方案持久化且同时只有一个生效')

    # 7. 合成报告记录校正，导出可用，原始素材哈希不变
    job = client.post('/api/projects/%s/jobs' % key,
                      json={'revision': reopened['revision'], 'kind': 'process'}).json
    finished = wait_job(client, job['id'])
    assert finished['status'] == 'completed', finished.get('error')
    meta = client.get('/api/projects/%s/current' % key).json
    assert meta['report']['calibration'][0]['color_plan']['plan_id'] == saved_b['plan']['id']
    export = client.post('/api/projects/%s/export-image' % key, json={'format': 'png'})
    assert export.status_code == 200
    manifest = client.get('/api/scanner/%s/manifest' % key).json
    for entry in manifest['frames']:
        verify = client.get('/api/assets/%s/verify' % entry['asset_sha256']).json
        assert verify['hash_valid']
    print('7. 合成/导出走校正，%d 帧原始素材哈希不变' % len(manifest['frames']))

    # 8. 工程包包含方案并可重新导入
    bundle_job = client.post('/api/projects/%s/jobs' % key,
                             json={'revision': reopened['revision'], 'kind': 'project'}).json
    finished = wait_job(client, bundle_job['id'])
    assert finished['status'] == 'completed'
    raw = client.get('/api/jobs/%s/download' % bundle_job['id']).data
    bundled = js.loads(zipfile.ZipFile(io.BytesIO(raw)).read('project.json'))
    assert len(bundled['settings']['color_plans']) == 2
    imported = client.post('/api/import-project',
                           data={'file': (io.BytesIO(raw), 'project.zip')},
                           content_type='multipart/form-data').json
    assert len(imported['settings']['color_plans']) == 2
    print('8. 工程包导出/导入保留全部校正方案')
    shutil.rmtree(data_dir, ignore_errors=True)
    print('全部通过')


if __name__ == '__main__':
    main()
