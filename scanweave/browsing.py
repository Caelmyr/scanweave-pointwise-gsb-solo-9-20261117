"""Fast browsing pyramid for the viewer.

Only the browser uses this module: previews and detail tiles are display copies
of the already stored frame or result. Quality checks and exports keep reading
the exact float fields and PNG output, so browsing never changes delivered data.
"""
import os
import threading
from collections import OrderedDict

import cv2
import numpy as np

from . import model
from . import validation as v

TILE_SIZE = 256
MAX_LEVELS = 9
PREVIEW_MAX_EDGE = 2048


def level_shapes(width, height):
    shapes = []
    width, height = int(width), int(height)
    while True:
        shapes.append((width, height))
        if len(shapes) >= MAX_LEVELS or (width <= 32 and height <= 32):
            break
        width, height = max(1, width // 2), max(1, height // 2)
    return shapes


def to_uint8(image):
    return np.ascontiguousarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))


def jpeg_uint8(rgb, quality=88):
    ok, encoded = cv2.imencode('.jpg', rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise v.Invalid('预览编码失败')
    return encoded.tobytes()


def png_uint8(rgb):
    ok, encoded = cv2.imencode('.png', rgb[:, :, ::-1])
    if not ok:
        raise v.Invalid('明细切片编码失败')
    return encoded.tobytes()


class Browsing:
    """Thread-safe LRU of decoded display levels shared by all viewers.

    Previews and detail levels are regenerated from the stored PNG/asset on a
    miss, so eviction never changes what quality checks or exports see.
    """

    def __init__(self, store, assets, results):
        self.store = store
        self.assets = assets
        self.results = results
        self.tile_size = TILE_SIZE
        budget = float(os.environ.get('SCANWEAVE_BROWSE_CACHE_MB', '96'))
        # One 4096px scan level is roughly 36MB, so keep room for at least one.
        self.budget = max(64. * 1024 * 1024, budget * 1024 * 1024)
        self.cache = OrderedDict()
        self.bytes = 0
        self.lock = threading.RLock()
        self.stripes = [threading.RLock() for _ in range(64)]

    # -- bounded level cache ------------------------------------------------
    def _get(self, key):
        with self.lock:
            item = self.cache.get(key)
            if item is None:
                return None
            self.cache.move_to_end(key)
            return item[0]

    def _put(self, key, value):
        size = int(value.nbytes)
        with self.lock:
            old = self.cache.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            if size > self.budget:
                return value
            self.cache[key] = (value, size)
            self.bytes += size
            while self.bytes > self.budget and self.cache:
                _, (_, evicted) = self.cache.popitem(last=False)
                self.bytes -= evicted
        return value

    def _token_lock(self, token):
        return self.stripes[hash(token) % len(self.stripes)]

    def cache_report(self):
        with self.lock:
            return {'entries': len(self.cache), 'bytes': self.bytes, 'budget_bytes': int(self.budget)}

    # -- sources ------------------------------------------------------------
    def _result_source(self, key, result_id):
        metadata = self.results.metadata(key, result_id)
        return 'r1.' + metadata['id'], (lambda: self._load_result(key, metadata['id'])), metadata

    def _load_result(self, key, result_id):
        directory = self.results.directory(key, result_id)
        path = directory / 'image.png'
        if path.exists():
            decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None and decoded.ndim == 3:
                return np.ascontiguousarray(decoded[:, :, ::-1])
        fields, _ = self.results.fields(key, result_id, False)
        return to_uint8(fields['image'])

    def _frame_source(self, project, frame_id):
        frame = model.find_frame(project, frame_id)
        metadata = self.assets.metadata(frame['asset'])
        width, height = int(metadata['width']), int(metadata['height'])
        token = 'f1.' + frame['asset']
        return token, (lambda: self._load_frame(frame['asset'])), {'width': width, 'height': height, 'name': frame['name'], 'asset': frame['asset']}

    def _load_frame(self, asset_id):
        # Native pixels: zoomed detail of the source stays exact. The LRU bound
        # keeps this at one image at a time, and a level larger than the budget
        # is still returned (just not cached) so it always renders.
        return to_uint8(self.assets.field(asset_id))

    def describe(self, key, view='result', frame_id=None, result_id=None):
        project = self.store.get(key)
        if view == 'frame':
            if not frame_id:
                return {'available': False, 'reason': '选择一张照片查看原始扫描', 'view': 'frame'}
            token, loader, extra = self._frame_source(project, frame_id)
            width, height = extra['width'], extra['height']
            info = {'available': True, 'view': 'frame', 'token': token, 'width': width, 'height': height,
                    'frame_id': frame_id, 'title': extra['name'], 'stale': False, 'frame_count': 1,
                    'thumb_url': '/api/assets/' + extra['asset'] + '/thumbnail'}
        else:
            result_id = result_id or project['current_result']
            if not result_id:
                return {'available': False, 'reason': '尚未合成结果，显示当前选中照片', 'view': 'result'}
            token, loader, metadata = self._result_source(key, result_id)
            width, height = int(metadata['width']), int(metadata['height'])
            info = {'available': True, 'view': 'result', 'token': token, 'width': width, 'height': height,
                    'result_id': metadata['id'], 'title': metadata['name'], 'mode': metadata['mode'],
                    'stale': bool(metadata.get('stale')), 'frame_count': len(metadata.get('frame_ids', [])),
                    'thumb_url': '/api/projects/' + key + '/result-image/thumb?result_id=' + metadata['id']}
        info['tile_size'] = self.tile_size
        info['levels'] = [{'level': index, 'width': w, 'height': h} for index, (w, h) in enumerate(level_shapes(width, height))]
        info['_loader'] = loader
        return info

    @staticmethod
    def public(info):
        return {name: value for name, value in info.items() if not name.startswith('_')}

    # -- levels and tiles ---------------------------------------------------
    def level_array(self, info, level):
        token = info['token']
        level = max(0, min(len(info['levels']) - 1, int(level)))
        with self._token_lock(token):
            cached = self._get((token, level))
            if cached is not None:
                return cached
            current = self._get((token, 0))
            if current is None:
                current = self._put((token, 0), info['_loader']())
            for index in range(1, level + 1):
                nxt = self._get((token, index))
                if nxt is None:
                    size = (max(1, current.shape[1] // 2), max(1, current.shape[0] // 2))
                    nxt = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
                    self._put((token, index), nxt)
                current = nxt
            return current

    def preview(self, key, view='result', frame_id=None, result_id=None, width=None, height=None, token=None):
        info = self.describe(key, view, frame_id, result_id)
        if not info['available']:
            raise v.Invalid(info['reason'])
        if token and token != info['token']:
            raise v.Conflict('浏览内容已更新')
        target_width = int(info['width'])
        target_height = int(info['height'])
        if width or height:
            try:
                limit_w = min(PREVIEW_MAX_EDGE, max(32, int(width))) if width else PREVIEW_MAX_EDGE
                limit_h = min(PREVIEW_MAX_EDGE, max(32, int(height))) if height else PREVIEW_MAX_EDGE
            except (TypeError, ValueError):
                raise v.Invalid('预览尺寸无效')
            scale = min(limit_w / info['width'], limit_h / info['height'], 1.)
            target_width = max(1, round(info['width'] * scale))
            target_height = max(1, round(info['height'] * scale))
        else:
            scale = min(1., 512. / max(info['width'], info['height']))
            target_width = max(1, round(info['width'] * scale))
            target_height = max(1, round(info['height'] * scale))
        level = 0
        for index in range(len(info['levels']) - 1, -1, -1):
            shape = info['levels'][index]
            if shape['width'] >= target_width and shape['height'] >= target_height:
                level = index
                break
        data = self.level_array(info, level)
        if (data.shape[1], data.shape[0]) != (target_width, target_height):
            interpolation = cv2.INTER_AREA if target_width <= data.shape[1] else cv2.INTER_LINEAR
            data = cv2.resize(data, (target_width, target_height), interpolation=interpolation)
        return jpeg_uint8(data), {'X-Browse-Token': info['token'], 'X-Image-Width': str(info['width']),
                                  'X-Image-Height': str(info['height']), 'X-Preview-Level': str(level),
                                  'X-Preview-Width': str(target_width), 'X-Preview-Height': str(target_height)}

    def tile(self, key, view='result', frame_id=None, result_id=None, token=None, level=0, x=0, y=0, size=TILE_SIZE):
        info = self.describe(key, view, frame_id, result_id)
        if not info['available']:
            raise v.Invalid(info['reason'])
        if not token:
            raise v.Invalid('缺少浏览标识')
        if token != info['token']:
            raise v.Conflict('浏览内容已更新')
        level = v.integer(level, '切片层级', 0, len(info['levels']) - 1)
        size = v.integer(size, '切片边长', 64, 512)
        x = v.integer(x, '切片横坐标', 0, info['levels'][level]['width'] - 1)
        y = v.integer(y, '切片纵坐标', 0, info['levels'][level]['height'] - 1)
        data = self.level_array(info, level)
        crop = data[y:y + size, x:x + size]
        if not crop.size:
            raise v.Invalid('切片超出图像范围')
        return png_uint8(crop), {'X-Browse-Token': info['token'], 'X-Tile-Level': str(level),
                                 'X-Tile-X': str(x), 'X-Tile-Y': str(y),
                                 'X-Tile-Width': str(crop.shape[1]), 'X-Tile-Height': str(crop.shape[0])}
