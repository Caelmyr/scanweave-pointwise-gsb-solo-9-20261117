"""Scanning intake and provenance, independent of pixel processing workflows."""
import hashlib
import io
import json
import numpy as np
from PIL import Image, ImageDraw
from flask import request, jsonify
from . import model
from .validation import Invalid


# 样张内置色卡：目标为绘制时的 sRGB 已知值，前端可直接取用，避免人工抄录出错。
CHART_PALETTE = [
    {'name': '深灰', 'target': '#3a3a3a'},
    {'name': '中灰', 'target': '#7d7d7d'},
    {'name': '浅灰', 'target': '#c2c2c2'},
    {'name': '近白', 'target': '#eeeeee'},
    {'name': '砖红', 'target': '#b54832'},
    {'name': '赭黄', 'target': '#c9973f'},
    {'name': '橄榄绿', 'target': '#6f8f4e'},
    {'name': '青绿', 'target': '#3f9c8f'},
    {'name': '靛蓝', 'target': '#3d5a96'},
    {'name': '紫罗兰', 'target': '#7d4f9c'},
    {'name': '玫红', 'target': '#bd5f86'},
    {'name': '橙棕', 'target': '#c06f3a'},
]
CHART_COLUMNS = 6
CHART_TOP = 34
CHART_HEIGHT = 72
SAMPLE_WIDTH = 720
SAMPLE_HEIGHT = 480


def _chart_bounds(width, height):
    rows = 2
    margin = 40
    gap = 8
    cell_w = (width - margin * 2 - gap * (CHART_COLUMNS - 1)) // CHART_COLUMNS
    cell_h = (CHART_HEIGHT - gap) // rows
    result = []
    for index in range(len(CHART_PALETTE)):
        row, column = divmod(index, CHART_COLUMNS)
        left = margin + column * (cell_w + gap)
        top = CHART_TOP + row * (cell_h + gap)
        right, bottom = left + cell_w, top + cell_h
        result.append([left / width, top / height, (right + 1) / width, (bottom + 1) / height])
    return result


for _index, _bounds in enumerate(_chart_bounds(SAMPLE_WIDTH, SAMPLE_HEIGHT)):
    CHART_PALETTE[_index]['bounds'] = _bounds


def _draw_chart(raster, width):
    """在样张片基内侧顶部绘制两排已知颜色的色卡，返回归一化框 [x0,y0,x1,y1]。"""
    draw = ImageDraw.Draw(raster)
    height = raster.height
    bounds = _chart_bounds(width, height)
    for index, item in enumerate(CHART_PALETTE):
        x0, y0, x1, y1 = bounds[index]
        left, top = round(x0 * width), round(y0 * height)
        right, bottom = round(x1 * width) - 1, round(y1 * height) - 1
        draw.rectangle((left, top, right, bottom), fill=item['target'])
        draw.rectangle((left, top, right, bottom), outline='#111111', width=1)
    return bounds


def scanning_sample(width=720, height=480):
    """Synthetic transmission specimen: neutral border and colored detail."""
    y, x = np.indices((height, width), dtype=np.float32)
    positive = np.empty((height, width, 3), np.float32)
    positive[:, :, 0] = .12 + .75 * x / width
    positive[:, :, 1] = .1 + .75 * y / height
    positive[:, :, 2] = .3 + .2 * np.sin(x / 31)
    raster = Image.fromarray(np.uint8(np.clip(positive, 0, 1) * 255))
    draw = ImageDraw.Draw(raster)
    for i in range(9):
        left = 95 + i * 58
        draw.rectangle((left, 170, left + 42, 310), fill=(i * 27, i * 27, i * 27))
    draw.text((100, 125), 'SCANWEAVE / SYNTHETIC TRANSMISSION', fill='white')
    _draw_chart(raster, width)
    positive = np.asarray(raster, np.float32) / 255
    # 模拟同一批扫描一致的扫描仪色偏：每通道独立的非线性响应 + 轻微通道串扰。
    response = np.empty_like(positive)
    gains = np.array([1.06, .92, .80], np.float32)
    response[:, :, 0] = np.power(positive[:, :, 0], .92)
    response[:, :, 1] = np.power(positive[:, :, 1], 1.05)
    response[:, :, 2] = np.power(positive[:, :, 2], 1.12)
    cross_talk = np.array([[.94, .05, .01], [.04, .93, .03], [.02, .07, .91]], np.float32)
    scanned = np.clip(response * gains @ cross_talk.T, 0, 1)
    # 均匀橙色片基（透射扫描边框），不受色卡影响。
    base = np.array([.85, .52, .28], np.float32)
    scanned[:32] = base
    scanned[-32:] = base
    scanned[:, :32] = base
    scanned[:, -32:] = base
    return scanned.astype(np.float32)


def manifest(store, assets, key):
    project = store.get(key)
    entries = []
    for frame in project['frames']:
        meta = assets.metadata(frame['asset'])
        entries.append({'frame_id': frame['id'], 'name': frame['name'],
                        'asset_sha256': frame['asset'], 'width': meta['width'],
                        'height': meta['height'], 'enabled': frame['enabled'],
                        'synthetic': meta.get('synthetic', False)})
    return {'schema': 'scanweave.intake.v1', 'project_id': key,
            'revision': project['revision'], 'name': project['name'],
            'notes': project['notes'], 'tags': project['tags'], 'frames': entries,
            'chart_palette': CHART_PALETTE,
            'settings': project['settings'],
            'source_policy': 'Immutable imported samples; edits produce separate results.'}


def install(app):
    store, assets = app.extensions['store'], app.extensions['assets']

    @app.post('/api/scanner/sample')
    def sample():
        current = store.create('扫描整理 · 合成透射样张')
        image = scanning_sample()
        rng = np.random.default_rng(15)
        with store.edit(current['id'], 0, '导入合成扫描重复采样') as project:
            project['settings']['mode'] = 'denoise'
            project['settings']['size'] = 768
            project['notes'] = '合成素材，非真实胶片；四周32像素为均匀片基，顶部含12格已知色卡，内部为灰阶和彩色图案。'
            project['tags'] = ['synthetic', 'transmission', '1200dpi']
            for i in range(3):
                captured = np.clip(image + rng.normal(0, .001, image.shape), 0, 1).astype(np.float32)
                item = assets.put(captured, f'scan-repeat-{i+1}.png',
                                  {'synthetic': True, 'scanner': 'synthetic-v1', 'dpi': 1200})
                project['frames'].append(model.frame(item['id'], item['name']))
            project['reference'] = project['frames'][0]['id']
        return jsonify(store.get(current['id'])), 201

    @app.get('/api/scanner/<key>/manifest')
    def get_manifest(key):
        return jsonify(manifest(store, assets, key))

    @app.get('/api/scanner/<key>/manifest.json')
    def download_manifest(key):
        data = json.dumps(manifest(store, assets, key), ensure_ascii=False, indent=2).encode()
        from flask import Response
        return Response(data, mimetype='application/json',
                        headers={'Content-Disposition': 'attachment; filename="scan-manifest.json"'})

    @app.get('/api/scanner/<key>/integrity')
    def integrity(key):
        current = store.get(key)
        checks = [assets.verify(f['asset']) for f in current['frames']]
        return jsonify({'project_id': key, 'assets': checks,
                        'all_valid': all(c['hash_valid'] for c in checks)})
