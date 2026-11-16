"""Scanning intake and provenance, independent of pixel processing workflows."""
import hashlib
import io
import json
import numpy as np
from PIL import Image, ImageDraw
from flask import request, jsonify
from . import model
from .validation import Invalid


def scratch_lines(width=720, height=480, seed=11):
    """Thin bright scan scratches crossing text and detailed areas.

    They are generated once for the batch and copied into every repeat, which
    mirrors a real dirty scanner: temporal denoising cannot remove them, but
    an annotated retouch can.  Returns an HxW coverage field of each defect.
    """
    import cv2
    rng = np.random.default_rng(seed)
    spec = [
        # (start, end, thickness) in pixel coordinates
        ((300, 452), (352, 300), 1),
        ((330, 470), (418, 330), 1),
        ((196, 112), (352, 146), 1),
    ]
    layers = []
    for start, end, thickness in spec:
        raster = np.zeros((height, width), np.uint8)
        wobble = np.cumsum(rng.normal(0, .45, 24))
        points = []
        for index in range(24):
            t = index / 23
            points.append((int(round(start[0] + (end[0] - start[0]) * t + wobble[index] * .6)),
                           int(round(start[1] + (end[1] - start[1]) * t + wobble[index]))))
        cv2.polylines(raster, [np.asarray(points, np.int32)], False, 255, thickness, cv2.LINE_AA)
        layers.append(raster.astype(np.float32) / 255)
    return layers


def apply_scratches(image, layers=None):
    layers = scratch_lines(image.shape[1], image.shape[0]) if layers is None else layers
    result = image.copy()
    for coverage in layers:
        # Pale dust streak over the transmission: bright but not pure white.
        result = result * (1 - coverage[:, :, None] * .55) + coverage[:, :, None] * np.array([.74, .72, .66], np.float32)
    return np.clip(result, 0, 1).astype(np.float32)


def scanning_sample(width=720, height=480, scratched=False):
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
    if scratched:
        transmission = apply_scratches(transmission)
    return transmission.astype(np.float32)


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
        clean = scanning_sample()
        layers = scratch_lines(clean.shape[1], clean.shape[0])
        rng = np.random.default_rng(15)
        with store.edit(current['id'], 0, '导入合成扫描重复采样') as project:
            project['settings']['mode'] = 'denoise'
            project['settings']['size'] = 768
            project['notes'] = '合成素材，非真实胶片；四周32像素为均匀片基，内部为灰阶和彩色图案。三条细长划痕在多帧中位置一致（模拟扫描仪脏污），可在“划痕修复”中涂出并修复。'
            project['tags'] = ['synthetic', 'transmission', '1200dpi', 'scratched']
            for i in range(3):
                captured = np.clip(apply_scratches(clean, layers) + rng.normal(0, .001, clean.shape), 0, 1).astype(np.float32)
                item = assets.put(captured, f'scan-repeat-{i+1}.png',
                                  {'synthetic': True, 'scanner': 'synthetic-v1', 'dpi': 1200, 'scratches': 3})
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
