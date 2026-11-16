"""Scanning intake and provenance, independent of pixel processing workflows."""
import hashlib
import io
import json
import numpy as np
from PIL import Image, ImageDraw
from flask import request, jsonify
from . import model
from .validation import Invalid


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
    positive = np.asarray(raster, np.float32) / 255
    density = .2 + 1.8 * (1 - positive)
    transmission = np.power(10., -density) * np.array([.85, .52, .28])
    transmission[:32] = [.85, .52, .28]
    transmission[-32:] = [.85, .52, .28]
    transmission[:, :32] = [.85, .52, .28]
    transmission[:, -32:] = [.85, .52, .28]
    return transmission.astype(np.float32)


def difficult_sample(width=720, height=480, rng=None):
    """Hard registration set: textured reference, near-blank frame,
    repetitive-pattern frame and an off-content frame that cannot align."""
    rng = rng or np.random.default_rng(7)
    reference = scanning_sample(width, height)

    def shift(image, dx, dy):
        moved = np.roll(image, (dy, dx), axis=(0, 1))
        if dy > 0:
            moved[:dy] = image[:1]
        if dx > 0:
            moved[:, :dx] = image[:, :1]
        return np.clip(moved, 0, 1).astype(np.float32)

    # 近空白：仅均匀片基 + 极弱噪声，自动配准几乎没有可用纹理。
    blank = np.ones_like(reference) * np.array([.85, .52, .28])
    blank += rng.normal(0, .0008, blank.shape)
    # 重复图案：参考内容上叠加周期格栅，结构可配准但存在周期性位移歧义。
    yy, xx = np.indices((height, width), dtype=np.float32)
    grid = (.5 + .5 * np.sin(xx / 9) * np.sin(yy / 9))[..., None]
    periodic = reference * .45 + np.repeat(grid, 3, axis=2) * np.array([.9, .6, .35]) * .55
    periodic[:32] = periodic[-32:] = [.85, .52, .28]
    periodic[:, :32] = periodic[:, -32:] = [.85, .52, .28]
    # 异内容：完全不同的画面，与参考帧无共同结构，应判失败而非污染其他帧。
    unrelated = np.zeros_like(reference)
    unrelated[..., 0] = .25 + .65 * (yy / height)
    unrelated[..., 1] = .55 * np.abs(np.sin(xx / 23))
    unrelated[..., 2] = .4
    frames = [
        reference,
        np.clip(shift(blank, 6, 4) + rng.normal(0, .001, reference.shape), 0, 1).astype(np.float32),
        np.clip(shift(periodic.astype(np.float32), 7, 0) + rng.normal(0, .001, reference.shape), 0, 1).astype(np.float32),
        np.clip(unrelated + rng.normal(0, .002, reference.shape), 0, 1).astype(np.float32),
    ]
    return [np.clip(item, 0, 1).astype(np.float32) for item in frames]


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
            project['notes'] = '合成素材，非真实胶片；四周32像素为均匀片基，内部为灰阶和彩色图案。'
            project['tags'] = ['synthetic', 'transmission', '1200dpi']
            for i in range(3):
                captured = np.clip(image + rng.normal(0, .001, image.shape), 0, 1).astype(np.float32)
                item = assets.put(captured, f'scan-repeat-{i+1}.png',
                                  {'synthetic': True, 'scanner': 'synthetic-v1', 'dpi': 1200})
                project['frames'].append(model.frame(item['id'], item['name']))
            project['reference'] = project['frames'][0]['id']
        return jsonify(store.get(current['id'])), 201

    @app.post('/api/scanner/difficult-sample')
    def difficult():
        current = store.create('扫描整理 · 疑难配准样张')
        images = difficult_sample()
        names = ['scan-reference.png', 'scan-near-blank.png',
                 'scan-repetitive.png', 'scan-off-content.png']
        rng = np.random.default_rng(31)
        with store.edit(current['id'], 0, '导入疑难配准合成采样') as project:
            project['settings']['mode'] = 'denoise'
            project['settings']['size'] = 768
            project['notes'] = '合成疑难素材：参考帧纹理充分，另含近空白、重复图案和异内容帧，用于逐帧配准复核。'
            project['tags'] = ['synthetic', 'transmission', 'registration-review']
            for image, name in zip(images, names):
                captured = np.clip(image + rng.normal(0, .0006, image.shape), 0, 1).astype(np.float32)
                item = assets.put(captured, name,
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
