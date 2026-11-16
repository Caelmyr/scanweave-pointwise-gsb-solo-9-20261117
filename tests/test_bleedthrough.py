"""双面透印（bleedthrough）流程的后端测试。

运行：python -m pytest tests/  或  python tests/test_bleedthrough.py
"""
import io
import time
import numpy as np

from scanweave import create_app
from scanweave.codecs import decode_image
from scanweave.bleedthrough import suppress_show_through, settings as bleed_settings


def make_client(tmp_path):
    app = create_app(str(tmp_path))
    return app, app.test_client()


def demo_project(client):
    project = client.post('/api/demos/bleedthrough').get_json()
    return project['id'], project


def revision(client, key):
    return client.get(f'/api/projects/{key}').get_json()['revision']


def wait_job(client, job_id, tries=100):
    for _ in range(tries):
        job = client.get('/api/jobs/' + job_id).get_json()
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.2)
    raise AssertionError('任务未结束')


def test_workflow_registered():
    app, client = make_client('/tmp/sw_pytest_1')
    modes = [w['key'] for w in client.get('/api/workflows').get_json()]
    assert 'bleedthrough' in modes
    fresh = client.post('/api/projects', json={'name': 't'}).get_json()
    assert fresh['settings']['bleedthrough']['strength'] == .7


def test_demo_has_mirrored_back(tmp_path='/tmp/sw_pytest_2'):
    app, client = make_client(tmp_path)
    _, project = demo_project(client)
    assert len(project['frames']) == 2
    assert project['frames'][1]['mirrored'] is True
    assert project['reference'] == project['frames'][0]['id']
    assert project['settings']['mode'] == 'bleedthrough'


def test_assign_roles_and_validation(tmp_path='/tmp/sw_pytest_3'):
    app, client = make_client(tmp_path)
    key, project = demo_project(client)
    front, back = project['frames'][0]['id'], project['frames'][1]['id']
    response = client.post(f'/api/projects/{key}/bleedthrough/assign',
                           json={'revision': revision(client, key), 'front': back, 'back': front})
    assert response.status_code == 200
    updated = response.get_json()
    assert updated['reference'] == back
    assert [frame['mirrored'] for frame in updated['frames']] == [True, False]
    assert all(frame['enabled'] for frame in updated['frames'])
    # 同一帧不能同时充当正反面
    response = client.post(f'/api/projects/{key}/bleedthrough/assign',
                           json={'revision': revision(client, key), 'front': front, 'back': front})
    assert response.status_code == 400
    # 修订号冲突
    response = client.post(f'/api/projects/{key}/bleedthrough/assign',
                           json={'revision': 0, 'front': front, 'back': back})
    assert response.status_code == 409


def test_preview_views_and_metrics(tmp_path='/tmp/sw_pytest_4'):
    app, client = make_client(tmp_path)
    key, _ = demo_project(client)
    for view in ['result', 'before', 'overlay', 'split', 'checker',
                 'bleed_map', 'reliable', 'protection', 'back']:
        response = client.post(f'/api/projects/{key}/bleedthrough/preview', json={'view': view})
        assert response.status_code == 200 and response.data[:4] == b'\x89PNG'
    metrics = client.post(f'/api/projects/{key}/bleedthrough/preview',
                          json={'view': 'metrics'}).get_json()
    assert 0 < metrics['geometric_overlap_fraction'] <= 1
    assert metrics['gain_auto'] is True
    assert all(.02 <= g <= 1 for g in metrics['gain'])
    assert 'psnr_db' in metrics and 'ssim' in metrics
    # 未知视图被拒
    assert client.post(f'/api/projects/{key}/bleedthrough/preview',
                       json={'view': 'nope'}).status_code == 400


def test_strength_zero_is_identity(tmp_path='/tmp/sw_pytest_5'):
    app, client = make_client(tmp_path)
    key, _ = demo_project(client)
    before = decode_image(client.post(f'/api/projects/{key}/bleedthrough/preview',
                                      json={'view': 'before'}).data)[0]
    suppressed = decode_image(client.post(
        f'/api/projects/{key}/bleedthrough/preview',
        json={'view': 'result', 'settings': {'strength': 0}}).data)[0]
    assert np.abs(suppressed - before).max() == 0


def test_suppression_targets_showthrough_only(tmp_path='/tmp/sw_pytest_6'):
    app, client = make_client(tmp_path)
    key, _ = demo_project(client)
    def render(view, settings=None):
        return decode_image(client.post(
            f'/api/projects/{key}/bleedthrough/preview',
            json={'view': view, 'settings': settings or {}}).data)[0]
    before, after, back = render('before'), render('result'), render('back')
    reliable = render('reliable')[:, :, 0] > .5
    protection = render('protection')[:, :, 0] > .25
    back_luma, front_luma = back.mean(axis=2), before.mean(axis=2)
    showthrough = reliable & ~protection & (back_luma < np.percentile(back_luma[reliable], 30))
    untouched = ~reliable & (front_luma > np.percentile(front_luma, 80))
    # 透印区被向纸基提亮
    assert (after.mean(axis=2)[showthrough] - front_luma[showthrough]).mean() > .004
    # 无可靠重叠的留白区像素完全不变
    assert np.abs(after[untouched] - before[untouched]).max() == 0
    # 输出不会比原始正面更暗，也不会超过纸基变白
    assert (after >= before - 1e-6).all()


def test_requires_exactly_two_frames(tmp_path='/tmp/sw_pytest_7'):
    app, client = make_client(tmp_path)
    key, project = demo_project(client)
    back = project['frames'][1]['id']
    client.patch(f'/api/projects/{key}/frames/{back}',
                 json={'revision': revision(client, key), 'enabled': False})
    response = client.post(f'/api/projects/{key}/bleedthrough/preview', json={})
    assert response.status_code == 400 and '恰好两张' in response.get_json()['error']


def test_full_job_export_and_auxiliaries(tmp_path='/tmp/sw_pytest_8'):
    app, client = make_client(tmp_path)
    key, _ = demo_project(client)
    job = client.post(f'/api/projects/{key}/jobs',
                      json={'kind': 'process', 'revision': revision(client, key)}).get_json()
    assert wait_job(client, job['id'])['status'] == 'completed'
    meta = client.get(f'/api/projects/{key}/current').get_json()
    assert {'image', 'front_source', 'back_registered', 'reliable',
            'protection', 'bleed_map', 'valid'} <= set(meta['fields'])
    assert client.get(f'/api/projects/{key}/integrity').get_json()['valid']
    diagnostic = client.post(f'/api/projects/{key}/diagnostic-image',
                             json={'kind': 'bleed_before_after'})
    assert diagnostic.status_code == 200 and diagnostic.data[:4] == b'\x89PNG'
    export = client.post(f'/api/projects/{key}/export-image',
                         json={'format': 'png', 'bits': 16})
    assert export.status_code == 200 and len(export.data) > 1000
    # 原始正反面素材仍然保留
    for frame in client.get(f'/api/projects/{key}').get_json()['frames']:
        assert client.get(f'/api/assets/{frame["asset"]}/image').status_code == 200


def test_project_bundle_roundtrip_keeps_mirror(tmp_path='/tmp/sw_pytest_9'):
    app, client = make_client(tmp_path)
    key, _ = demo_project(client)
    job = client.post(f'/api/projects/{key}/jobs',
                      json={'kind': 'project', 'revision': revision(client, key)}).get_json()
    assert wait_job(client, job['id'])['status'] == 'completed'
    bundle = client.get('/api/jobs/' + job['id'] + '/download').data
    imported = client.post('/api/import-project',
                           data={'file': (io.BytesIO(bundle), 'p.zip')},
                           content_type='multipart/form-data')
    assert imported.status_code == 201
    project = imported.get_json()
    assert [frame['mirrored'] for frame in project['frames']] == [False, True]
    assert project['settings']['mode'] == 'bleedthrough'


def test_settings_validation(tmp_path='/tmp/sw_pytest_10'):
    app, client = make_client(tmp_path)
    key, _ = demo_project(client)
    for bad in [{'reliable_window': 4}, {'unknown': 1}, {'auto_gain': 'yes'},
                {'strength': 2}, {'back_blur': -1}, {'reliable_correlation': 2}]:
        assert client.patch(f'/api/projects/{key}',
                            json={'revision': revision(client, key),
                                  'settings': {'bleedthrough': bad}}).status_code == 400
    ok = client.patch(f'/api/projects/{key}',
                      json={'revision': revision(client, key),
                            'settings': {'bleedthrough': {'strength': .33}}})
    assert ok.status_code == 200
    assert abs(ok.get_json()['settings']['bleedthrough']['strength'] - .33) < 1e-9


def test_manual_gain_path(tmp_path='/tmp/sw_pytest_11'):
    app, client = make_client(tmp_path)
    key, _ = demo_project(client)
    metrics = client.post(f'/api/projects/{key}/bleedthrough/preview',
                          json={'view': 'metrics',
                                'settings': {'auto_gain': False, 'gain': .4}}).get_json()
    assert metrics['gain_auto'] is False
    assert all(abs(g - .4) < 1e-6 for g in metrics['gain'])


def test_algorithm_contract_on_synthetic_pair():
    # 直接构造一对：背面墨迹只出现在正面留白处，验证可靠区被补偿、正面墨迹不动
    h, w = 240, 320
    rng = np.random.default_rng(7)
    paper = np.array([.88, .86, .8], np.float32)
    front = np.broadcast_to(paper, (h, w, 3)).copy()
    back = np.broadcast_to(paper, (h, w, 3)).copy()
    front[40:70, 30:140] = paper * .35          # 正面自己的深字
    back[120:150, 180:290] = paper * .25        # 背面深字
    front[120:150, 180:290] = paper * .92       # 透印淡字（正面留白处）
    front = np.clip(front + rng.normal(0, .002, front.shape), 0, 1).astype(np.float32)
    back = np.clip(back + rng.normal(0, .002, back.shape), 0, 1).astype(np.float32)
    valid = np.ones((h, w), np.float32)
    prepared = {
        'frames': [{'id': 'a' * 32, 'asset': '', 'mirrored': True},
                   {'id': 'b' * 32, 'asset': '', 'mirrored': False}],
        'images': [front, back],
        'validities': [valid, valid],
        'geometric_validities': [valid, valid],
        'reference_index': 0,
    }
    fields, report = suppress_show_through(prepared, bleed_settings({}))
    assert fields['image'].shape == front.shape
    assert np.isfinite(fields['image']).all() and fields['image'].min() >= 0 and fields['image'].max() <= 1
    # 正面自己的深字区域几乎不变（保护生效）
    own_ink = slice(40, 70), slice(30, 140)
    assert np.abs(fields['image'][own_ink] - front[own_ink]).mean() < .005
    # 透印区被提亮
    bleed = slice(120, 150), slice(180, 290)
    assert fields['image'][bleed].mean() > front[bleed].mean() + .003
    # 仅允许噪声级（1%）越过纸基均值，不存在成片过曝
    assert np.mean(fields['image'] > paper.max() + .01) < .005
    assert report['geometric_overlap_fraction'] == 1.


if __name__ == '__main__':
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            params = inspect.signature(fn).parameters
            try:
                if 'tmp_path' in params:
                    fn('/tmp/' + name)
                else:
                    fn()
                print('PASS', name)
            except Exception as error:
                failures += 1
                print('FAIL', name, error)
    raise SystemExit(1 if failures else 0)
