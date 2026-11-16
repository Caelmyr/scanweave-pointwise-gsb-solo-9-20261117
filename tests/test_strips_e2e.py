"""通过 Flask 测试客户端做完整 HTTP 端到端冒烟（含文件上传与原始素材完整性）。"""
import io
import sys
import tempfile
import pathlib

import numpy as np
from PIL import Image

sys.path.insert(0, '/workspace')
from scanweave import create_app  # noqa: E402
from tests.make_strip_fixture import make_scroll, cut_strips, shuffled  # noqa: E402


def encode(image):
    buf = io.BytesIO()
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(buf, 'PNG')
    return buf.getvalue()


def main():
    tmp = tempfile.mkdtemp(prefix='e2e_', dir='/workspace/.tmp_tests')
    client = create_app(tmp).test_client()

    scroll = make_scroll()
    true = cut_strips(scroll)
    order, mixed = shuffled(true)
    print('导入乱序:', order)

    # 上传为不可变素材
    asset_ids, hashes_before = [], {}
    for name, img in mixed:
        data = {'file': (io.BytesIO(encode(img)), name + '.png')}
        r = client.post('/api/assets', data=data, content_type='multipart/form-data')
        assert r.status_code == 201, r.data
        meta = r.get_json()
        asset_ids.append(meta['id'])
        hashes_before[meta['id']] = client.get('/api/assets/%s/verify' % meta['id']).get_json()

    # 建布局、加条带
    r = client.post('/api/strip-sheets', json={'name': '端到端长卷', 'axis': 'vertical'})
    assert r.status_code == 201
    sid = r.get_json()['id']
    r = client.post('/api/strip-sheets/%s/strips' % sid, json={'revision': 0, 'asset_ids': asset_ids})
    assert r.status_code == 201
    sheet = r.get_json()

    # 分析
    r = client.post('/api/strip-sheets/%s/analyze' % sid, json={'revision': sheet['revision']})
    assert r.status_code == 200
    sheet = r.get_json()
    names = [t['name'] for t in sheet['strips']]
    print('恢复顺序:', names)
    assert names == [n + '.png' for n, _ in true]

    # 未确认导出应被门禁拒绝
    r = client.post('/api/strip-sheets/%s/export' % sid, json={'format': 'png'})
    assert r.status_code == 400 and '尚未确认' in r.get_json()['error']

    # 解决全部未决疑点
    for issue in sheet['issues']:
        if not issue['resolved']:
            r = client.post('/api/strip-sheets/%s/issues/%s/resolve' % (sid, issue['id']),
                            json={'revision': sheet['revision'], 'resolved': True})
            assert r.status_code == 200
            sheet = r.get_json()

    # 接受第一条内容邻接边（人工裁决）
    edge = next(e for e in sheet['edges'] if e['source'] == 'content')
    r = client.post('/api/strip-sheets/%s/edges' % sid,
                    json={'revision': sheet['revision'], 'a': edge['a'], 'b': edge['b'], 'status': 'accepted'})
    assert r.status_code == 200
    sheet = r.get_json()

    # 确认并导出三种格式
    r = client.post('/api/strip-sheets/%s/confirm' % sid, json={'revision': sheet['revision']})
    assert r.status_code == 200 and r.get_json()['confirmed']
    for fmt, ctype in [('png', 'image/png'), ('jpeg', 'image/jpeg'), ('tiff', 'image/tiff')]:
        r = client.post('/api/strip-sheets/%s/export' % sid, json={'format': fmt})
        assert r.status_code == 200 and r.mimetype == ctype, (fmt, r.data[:120])
    r = client.post('/api/strip-sheets/%s/export' % sid, json={'format': 'png'})
    exported = np.asarray(Image.open(io.BytesIO(r.data)))
    print('导出尺寸:', exported.shape, '原稿:', scroll.shape)
    assert exported.shape[:2] == scroll.shape[:2]

    # 预览与缩略图
    assert client.get('/api/strip-sheets/%s/preview' % sid).status_code == 200
    tid = sheet['strips'][0]['id']
    assert client.get('/api/strip-sheets/%s/strips/%s/thumbnail' % (sid, tid)).status_code == 200
    assert client.get('/api/strip-sheets/%s/manifest' % sid).status_code == 200

    # 原始扫描完整性：哈希在全部操作后保持有效，像素未被重写
    for aid in asset_ids:
        after = client.get('/api/assets/%s/verify' % aid).get_json()
        assert after['hash_valid'] and after['finite']
        assert after['shape'] == hashes_before[aid]['shape']
    print('原始素材哈希全部保持有效，未被裁切或重写')

    # 撤销/重做（每次用服务端返回的最新修订号）
    latest = client.get('/api/strip-sheets/%s' % sid).get_json()
    r = client.post('/api/strip-sheets/%s/undo' % sid, json={'revision': latest['revision']})
    assert r.status_code == 200
    r = client.post('/api/strip-sheets/%s/redo' % sid,
                    json={'revision': r.get_json()['revision']})
    assert r.status_code == 200
    current_rev = r.get_json()['revision']

    # 复制与删除
    r = client.post('/api/strip-sheets/%s/duplicate' % sid, json={'revision': current_rev})
    assert r.status_code == 201
    dup_id = r.get_json()['id']
    r = client.delete('/api/strip-sheets/%s' % dup_id)
    assert r.status_code == 200
    # 删除布局不影响素材
    for aid in asset_ids:
        assert client.get('/api/assets/%s/verify' % aid).status_code == 200
    print('删除布局后素材仍保留')

    print('\n端到端 HTTP 工作流全部通过 ✔')


if __name__ == '__main__':
    main()
