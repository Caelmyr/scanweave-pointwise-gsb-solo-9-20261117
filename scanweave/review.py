"""Read-only region correspondence review across stored result versions.

The review takes a region drawn on the original reference scan and locates the
same physical content inside two historical results.  Every geometric stage of
the recorded pipeline (lens correction, letterbox, frame matrix, panorama /
drizzle canvas, automatic + manual crop, rotation and flips) is inverted only
virtually: stored results and the currently selected result are never written,
activated or modified.
"""
import base64
import html as html_module
import json
import numpy as np
import cv2
from . import model
from . import validation as v
from .arrays import fit_size, checkerboard, luminance
from .geometry import pixel_matrix
from .lens import camera_matrix
from .codecs import png_bytes

MAX_PATCH = 512
CHANGE_THRESHOLD = 2 / 255
PATCH_CELL = 10


class Correspondence(Exception):
    """Raised when a result version cannot be tied to the selected content."""

    def __init__(self, status, message, warnings=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.warnings = list(warnings or [])


def data_uri(image):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes(image)).decode()


def _check(value, label):
    if not (isinstance(value, (list, tuple)) and len(value) == 4):
        raise Correspondence('unreadable', f'{label}记录缺失，无法还原输出几何。')
    return [float(item) for item in value]


def _grid(bounds, width, height):
    x0 = min(width - 1, max(0, int(np.floor(bounds[0] * width))))
    y0 = min(height - 1, max(0, int(np.floor(bounds[1] * height))))
    x1 = max(x0 + 1, min(width, int(np.ceil(bounds[2] * width))))
    y1 = max(y0 + 1, min(height, int(np.ceil(bounds[3] * height))))
    patch_w, patch_h = x1 - x0, y1 - y0
    scale = min(1., MAX_PATCH / max(patch_w, patch_h))
    columns = max(1, int(round(patch_w * scale)))
    rows = max(1, int(round(patch_h * scale)))
    xs = x0 + (np.arange(columns) + .5) * patch_w / columns - .5
    ys = y0 + (np.arange(rows) + .5) * patch_h / rows - .5
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    grid = np.stack(np.meshgrid(xs, ys, indexing='xy'), axis=-1).reshape(-1, 2)
    return {'grid': grid, 'columns': columns, 'rows': rows, 'pixel_bounds': [x0, y0, x1, y1], 'xs': xs, 'ys': ys}


def _map_arrays(coords, rows, columns):
    return (coords[:, 0].reshape(rows, columns).astype(np.float32),
            coords[:, 1].reshape(rows, columns).astype(np.float32))


def sample_patch(image, coords, valid, rows, columns):
    height, width = image.shape[:2]
    map_x, map_y = _map_arrays(coords, rows, columns)
    inside = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    clipped_x = np.clip(map_x, 0, width - 1)
    clipped_y = np.clip(map_y, 0, height - 1)
    patch = cv2.remap(image, clipped_x, clipped_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = inside
    if valid is not None:
        support = cv2.remap(valid, clipped_x, clipped_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = mask & (support >= .5)
    return patch.astype(np.float32), mask.astype(bool)


def masked_patch(patch, mask):
    background = checkerboard(patch.shape, PATCH_CELL)
    return np.where(mask[:, :, None], patch, background).astype(np.float32)


def patch_metrics(a, b, mask, gain=4.):
    if not mask.any():
        return None
    difference = np.abs(a - b)
    magnitude = np.mean(difference, axis=2)
    values = difference[mask]
    gray_a = luminance(a)[mask]
    gray_b = luminance(b)[mask]
    normalized_mae = None
    correlation = None
    if gray_a.size > 16 and gray_a.std() > 1e-6 and gray_b.std() > 1e-6:
        normalized_a = (gray_a - gray_a.mean()) / gray_a.std()
        normalized_b = (gray_b - gray_b.mean()) / gray_b.std()
        normalized_mae = float(np.mean(np.abs(normalized_a - normalized_b)))
        correlation = float(np.mean(normalized_a * normalized_b))
    return {
        'pixels': int(mask.sum()),
        'overlap_fraction': float(mask.mean()),
        'mae': float(values.mean()),
        'rmse': float(np.sqrt(np.mean((a - b)[mask] ** 2))),
        'p95_abs': float(np.percentile(magnitude[mask], 95)),
        'maximum_abs': float(values.max()),
        'changed_fraction': float(np.mean(magnitude[mask] > CHANGE_THRESHOLD)),
        'normalized_luminance_mae': normalized_mae,
        'luminance_correlation': correlation,
        'difference_scale': gain,
        'note': '差异在显示sRGB图像上计算；整体曝光、色彩或锐化参数不同会造成全局差异，不表示内容位置改变。',
    }


class ResultGeometry:
    """Virtual forward mapping from a source scan to one stored result."""

    def __init__(self, assets, metadata, source_asset, source_frame_id):
        self.assets = assets
        self.metadata = metadata
        self.project = metadata.get('project') or {}
        self.settings = self.project.get('settings') or {}
        self.report = metadata.get('report') or {}
        self.mode = metadata.get('mode') or self.settings.get('mode')
        self.warnings = []
        self.approximate = False
        self.frame = self._resolve_frame(source_asset, source_frame_id)
        self.source_meta = assets.metadata(source_asset)
        self.sw, self.sh = int(self.source_meta['width']), int(self.source_meta['height'])
        self._prepare()

    def _resolve_frame(self, source_asset, source_frame_id):
        frames = [f for f in self.project.get('frames', []) if f.get('asset') == source_asset]
        if not frames:
            raise Correspondence('missing_frame', '该版本工程中没有所选参考扫描素材，无法建立位置对应。')
        enabled = [f for f in frames if f.get('enabled')]
        if not enabled:
            raise Correspondence('disabled_frame', '该版本中对应扫描帧已停用，未参与合成，无法建立可靠对应。')
        exact = [f for f in enabled if f.get('id') == source_frame_id]
        if exact:
            return exact[0]
        if len(enabled) == 1:
            self.warnings.append('同一扫描在该版本中以不同帧编号存在，已按素材内容定位。')
            return enabled[0]
        shapes = {json.dumps(f.get('matrix'), sort_keys=True) for f in enabled}
        if len(shapes) == 1:
            self.warnings.append('同一扫描在该版本中出现多次但变换一致，已按素材内容定位。')
            return enabled[0]
        raise Correspondence('ambiguous_frame', '该版本中同一扫描素材对应多个不同变换的帧，无法确定唯一位置。')

    def _reference_size(self):
        reference = self.project.get('reference')
        frames = self.project.get('frames', [])
        current = next((f for f in frames if f.get('id') == reference), None)
        if current is None:
            raise Correspondence('unreadable', '该版本缺少可用的参考帧记录，无法还原工作画布。')
        meta = self.assets.metadata(current['asset'])
        size = self.settings.get('size', 768)
        return fit_size((meta['height'], meta['width']), size)

    def _prepare(self):
        if not self.settings or 'geometry' not in self.settings:
            raise Correspondence('unreadable', '该版本缺少处理设置记录，无法还原输出几何。')
        self.width, self.height = self._reference_size()
        self.matrix = pixel_matrix(self.frame.get('matrix', model.identity()), self.width, self.height)
        self.lens = self.settings.get('lens') or {'enabled': False}
        geometry = self.settings['geometry']
        self.rotation = int(geometry.get('rotation', 0))
        self.flip_x = bool(geometry.get('flip_x', False))
        self.flip_y = bool(geometry.get('flip_y', False))
        self.manual = _check(self.report.get('manual_crop'), '手动裁切')
        self.common = self.report.get('common_crop')
        if self.common is not None:
            self.common = _check(self.common, '共同覆盖裁切')
        self.panorama_shift = None
        self.panorama_crop = None
        self.drizzle_scale = 1.
        if self.mode == 'panorama':
            canvas = self.report.get('canvas')
            if not isinstance(canvas, dict) or 'scale' not in canvas or 'minimum' not in canvas:
                raise Correspondence('unreadable', '该版本缺少拼接画布记录，无法还原拼接位置。')
            scale = float(canvas['scale'])
            minimum = [float(item) for item in canvas['minimum']]
            self.panorama_shift = np.array([[scale, 0, -minimum[0] * scale], [0, scale, -minimum[1] * scale], [0, 0, 1.]], np.float64)
            self.panorama_crop = _check(self.report.get('crop'), '拼接裁切')
        elif self.mode == 'drizzle':
            self.drizzle_scale = float(self.report.get('scale', 1))
        if self.settings.get('flow', {}).get('enabled') and float(self.settings['flow'].get('strength', 0)) > 0:
            self.approximate = True
            self.warnings.append('该版本启用了局部光流修正，逐像素位移未保存在结果中；对应位置为近似值，不参与差异统计。')
        if self.lens.get('enabled') and (abs(float(self.lens.get('red_scale', 1)) - 1) > .001 or abs(float(self.lens.get('blue_scale', 1)) - 1) > .001):
            self.warnings.append('该版本镜头校正含红/蓝通道径向缩放；对应位置按绿通道几何计算。')
        manual_w = self.manual[2] - self.manual[0]
        manual_h = self.manual[3] - self.manual[1]
        if manual_w <= 0 or manual_h <= 0:
            raise Correspondence('unreadable', '该版本裁切记录无效，无法建立对应。')
        final_w, final_h = (manual_h, manual_w) if self.rotation in (90, 270) else (manual_w, manual_h)
        if (int(round(final_w)), int(round(final_h))) != (int(self.metadata.get('width', 0)), int(self.metadata.get('height', 0))):
            raise Correspondence('geometry_mismatch', '该版本记录的输出尺寸与还原几何不一致，无法确认对应位置。')

    def _lens_forward(self, points):
        if not self.lens.get('enabled'):
            return points
        camera = camera_matrix(self.sw, self.sh, self.lens)
        distortion = np.asarray([self.lens.get('k1', 0), self.lens.get('k2', 0),
                                 self.lens.get('p1', 0), self.lens.get('p2', 0), 0], np.float64)
        destination = cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64),
                                          camera, distortion, None, camera).reshape(-1, 2)
        if not np.isfinite(destination).all():
            raise Correspondence('geometry_mismatch', '镜头校正几何无法反演，无法建立对应位置。')
        return destination

    def _letterbox(self, points):
        ratio = min(self.width / self.sw, self.height / self.sh)
        inner_w = max(1, round(self.sw * ratio))
        inner_h = max(1, round(self.sh * ratio))
        offset_x = (self.width - inner_w) // 2
        offset_y = (self.height - inner_h) // 2
        x = (points[:, 0] + .5) * inner_w / self.sw - .5 + offset_x
        y = (points[:, 1] + .5) * inner_h / self.sh - .5 + offset_y
        return np.stack([x, y], axis=1)

    @staticmethod
    def _apply(matrix, points):
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        projected = homogeneous @ np.asarray(matrix, np.float64).T
        denominator = projected[:, 2]
        if np.any(np.abs(denominator) < 1e-9):
            raise Correspondence('geometry_mismatch', '该版本变换矩阵退化，无法建立对应位置。')
        return projected[:, :2] / denominator[:, None]

    def _orientation(self, points):
        manual_w = self.manual[2] - self.manual[0]
        manual_h = self.manual[3] - self.manual[1]
        x, y = points[:, 0].copy(), points[:, 1].copy()
        if self.rotation == 90:
            x, y = (manual_h - 1) - y, x
            output_w, output_h = manual_h, manual_w
        elif self.rotation == 180:
            x, y = (manual_w - 1) - x, (manual_h - 1) - y
            output_w, output_h = manual_w, manual_h
        elif self.rotation == 270:
            x, y = y, (manual_w - 1) - x
            output_w, output_h = manual_h, manual_w
        else:
            output_w, output_h = manual_w, manual_h
        if self.flip_x:
            x = (output_w - 1) - x
        if self.flip_y:
            y = (output_h - 1) - y
        return np.stack([x, y], axis=1)

    def map_points(self, coords):
        points = np.asarray(coords, dtype=np.float64)
        points = self._lens_forward(points)
        points = self._letterbox(points)
        points = self._apply(self.matrix, points)
        if self.mode == 'panorama':
            points = self._apply(self.panorama_shift, points)
            points = points - np.array(self.panorama_crop[:2], np.float64)
        elif self.mode == 'drizzle':
            points = points * self.drizzle_scale + (self.drizzle_scale - 1) / 2
        elif self.common is not None:
            points = points - np.array(self.common[:2], np.float64)
        points = points - np.array(self.manual[:2], np.float64)
        points = self._orientation(points)
        if not np.isfinite(points).all():
            raise Correspondence('geometry_mismatch', '该版本对应位置计算产生无效坐标，无法建立对应。')
        return points

    def describe(self):
        frame = self.frame
        return {
            'frame_id': frame.get('id'),
            'frame_name': frame.get('name'),
            'asset_id': frame.get('asset'),
            'working_size': [self.width, self.height],
            'mode': self.mode,
            'auto_crop': self.common,
            'manual_crop': self.manual,
            'panorama_crop': self.panorama_crop,
            'drizzle_scale': self.drizzle_scale,
            'rotation': self.rotation,
            'flip_x': self.flip_x,
            'flip_y': self.flip_y,
            'lens_correction': bool(self.lens.get('enabled')),
            'local_flow': bool(self.settings.get('flow', {}).get('enabled')),
        }


def _read_fields(results, key, result_id):
    metadata = results.metadata(key, result_id)
    directory = results.directory(key, metadata['id'])
    with np.load(directory / 'fields.npz', allow_pickle=False) as archive:
        if 'image' not in archive.files or 'valid' not in archive.files:
            raise Correspondence('unreadable', '该版本结果缺少显示图或有效覆盖图，无法提取对应内容。')
        image = archive['image'].copy()
        valid = archive['valid'].copy()
    if image.shape[:2] != (metadata.get('height'), metadata.get('width')) or valid.shape[:2] != image.shape[:2]:
        raise Correspondence('unreadable', '该版本结果字段尺寸与记录不一致，无法确认对应位置。')
    return metadata, image.astype(np.float32), valid.astype(np.float32)


def _settings_summary(settings):
    settings = settings or {}
    geometry = settings.get('geometry', {}) or {}
    lens = settings.get('lens', {}) or {}
    flow = settings.get('flow', {}) or {}
    mode = settings.get('mode')
    mode_settings = settings.get(mode) if isinstance(settings.get(mode), dict) else None
    return {
        'size': settings.get('size'),
        'mode': mode,
        'alignment_method': (settings.get('alignment') or {}).get('method'),
        'crop': geometry.get('crop'),
        'rotation': geometry.get('rotation'),
        'flip_x': geometry.get('flip_x'),
        'flip_y': geometry.get('flip_y'),
        'auto_crop': geometry.get('auto_crop'),
        'lens_enabled': lens.get('enabled'),
        'flow_enabled': flow.get('enabled'),
        'local_layers': len(settings.get('local', []) or []),
        'mode_settings': mode_settings,
        'tone': settings.get('tone') if mode == 'hdr' else None,
        'finishing': settings.get('finishing'),
    }


def _result_entry(metadata, project, include_settings=False):
    snapshot = metadata.get('project') or {}
    settings = snapshot.get('settings')
    return {
        'id': metadata['id'],
        'name': metadata.get('name'),
        'mode': metadata.get('mode'),
        'source_revision': metadata.get('source_revision'),
        'source_signature': metadata.get('source_signature'),
        'created_at': metadata.get('created_at'),
        'width': metadata.get('width'),
        'height': metadata.get('height'),
        'current': project.get('current_result') == metadata['id'],
        'stale': bool(metadata.get('stale')),
        'status': 'unreadable',
        'message': '',
        'available': False,
        'correspondence': None,
        'warnings': [],
        'patch': None,
        'coverage': 0.,
        'settings_summary': _settings_summary(settings),
        'settings': settings if include_settings else None,
        'project_snapshot': {'id': snapshot.get('id'), 'name': snapshot.get('name'),
                             'notes': snapshot.get('notes', ''), 'tags': snapshot.get('tags', [])},
    }


def review(store, assets, results, key, body, include_settings=False):
    data = v.object_value(body, '复核参数')
    project = store.get(key)
    frame_id = data.get('frame_id') or project.get('reference')
    source_frame = model.find_frame(project, frame_id)
    if not source_frame.get('enabled'):
        raise v.Invalid('参考扫描帧需处于启用状态')
    bounds = v.vector(data.get('bounds'), '复核区域', 4, 0, 1)
    if bounds[2] - bounds[0] < .002 or bounds[3] - bounds[1] < .002:
        raise v.Invalid('复核区域过小')
    result_ids = v.list_value(data.get('result_ids'), '历史结果', 2)
    if len(result_ids) != 2:
        raise v.Invalid('请选择两份历史结果')
    for result_id in result_ids:
        v.identifier(result_id)
    if result_ids[0] == result_ids[1]:
        raise v.Invalid('请选择两份不同的历史结果')
    gain = v.number(data.get('gain', 4), '差异增益', 1, 16)

    source_meta = assets.metadata(source_frame['asset'])
    source_image = assets.field(source_frame['asset'])
    width, height = int(source_meta['width']), int(source_meta['height'])
    if source_image.shape[:2] != (height, width):
        raise v.Invalid('参考扫描素材尺寸与档案不一致')
    grid = _grid(bounds, width, height)
    map_x, map_y = _map_arrays(grid['grid'], grid['rows'], grid['columns'])
    reference_patch = cv2.remap(source_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    entries, patches, masks = [], [], []
    for result_id in result_ids:
        try:
            metadata = results.metadata(key, result_id)
        except v.Invalid as error:
            entry = {'id': result_id, 'name': None, 'mode': None, 'source_revision': None, 'source_signature': None,
                     'created_at': None, 'width': None, 'height': None, 'current': False, 'stale': False,
                     'status': 'unreadable', 'message': str(error), 'available': False, 'correspondence': None,
                     'warnings': [], 'patch': None, 'coverage': 0., 'settings_summary': None, 'settings': None, 'project_snapshot': {}}
            entries.append(entry)
            patches.append(None)
            masks.append(None)
            continue
        entry = _result_entry(metadata, project, include_settings)
        patch = mask = None
        try:
            geometry = ResultGeometry(assets, metadata, source_frame['asset'], source_frame['id'])
            _, image, valid = _read_fields(results, key, metadata['id'])
            coords = geometry.map_points(grid['grid'])
            patch, mask = sample_patch(image, coords, valid, grid['rows'], grid['columns'])
            coverage = float(mask.mean())
            entry.update(available=bool(mask.any()), coverage=coverage, patch=data_uri(masked_patch(patch, mask)),
                         warnings=list(geometry.warnings), correspondence=geometry.describe(),
                         approximate=geometry.approximate, patch_size=[grid['columns'], grid['rows']])
            if not mask.any():
                outside = np.all((coords[:, 0] < 0) | (coords[:, 0] > image.shape[1] - 1) |
                                 (coords[:, 1] < 0) | (coords[:, 1] > image.shape[0] - 1))
                if outside:
                    entry.update(status='outside', message='该版本输出画布不含所选区域（已被裁切或旋转到画布之外），不计算差异。')
                else:
                    entry.update(status='no_valid', message='该版本在对应位置没有有效像素（可能已裁掉，或落在填充/排除区域），不计算差异。')
            elif coverage < .995:
                entry.update(status='partial', message='该版本只覆盖所选区域的一部分（' + format(coverage * 100, '.1f') + '%），差异仅在双方有效像素上计算。')
            else:
                entry.update(status='ok', message='已在该版本中定位到所选内容。')
            if mask.any():
                entry['vs_reference'] = patch_metrics(reference_patch, patch, mask, gain)
            if geometry.approximate:
                entry['used_for_difference'] = False
                entry['status'] = 'approximate'
                entry['message'] += ' 该版本的局部位移未保存在结果中，对应位置是近似值，因此不参与差异计算。'
        except Correspondence as error:
            patch = mask = None
            entry.update(status=error.status, message=error.message, warnings=error.warnings + entry.get('warnings', []), available=False, coverage=0.)
        except (OSError, ValueError, KeyError) as error:
            patch = mask = None
            entry.update(status='unreadable', message='该版本结果无法读取：' + str(error), available=False, coverage=0.)
        entries.append(entry)
        patches.append(patch if entry['available'] and entry.get('used_for_difference', True) else None)
        masks.append(mask if entry['available'] and entry.get('used_for_difference', True) else None)

    difference = {'status': 'skipped', 'message': '', 'image': None, 'metrics': None, 'gain': gain,
                  'result_ids': result_ids, 'note': '仅在两份结果都有效覆盖的像素上计算，不把裁切黑边或填充区域计入差异。'}
    if patches[0] is None or patches[1] is None:
        excluded = [entry['id'] for entry in entries if entry['available'] and not entry.get('used_for_difference', True)]
        if excluded:
            difference['message'] = '至少一份版本无法精确还原逐像素对应（例如启用了局部光流），差异未计算，以免把不同位置当作同一内容。'
        else:
            difference['message'] = '至少一份版本无法提供所选区域，差异未计算（不把缺失或填充区域当作差异像素）。'
    else:
        common_mask = masks[0] & masks[1]
        if not common_mask.any():
            difference.update(status='no_overlap', message='两份版本的有效区域没有重叠，无法比较差异。')
        else:
            first, second = patches[0], patches[1]
            metrics = patch_metrics(first, second, common_mask, gain)
            magnitude = np.clip(np.mean(np.abs(first - second), axis=2) * gain, 0, 1)
            heat = cv2.applyColorMap(np.rint(magnitude * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1].astype(np.float32) / 255
            image = np.where(common_mask[:, :, None], heat, checkerboard(heat.shape, PATCH_CELL)).astype(np.float32)
            difference.update(status='ok', message='差异已按双方有效像素计算。', image=data_uri(image), metrics=metrics,
                              coverage=float(common_mask.mean()))

    return {
        'mode': 'region-review',
        'created_at': v.timestamp(),
        'project': {'id': project['id'], 'name': project['name'], 'revision': project['revision'],
                    'current_result': project.get('current_result'), 'notes': project.get('notes', '')},
        'source': {'frame_id': source_frame['id'], 'frame_name': source_frame['name'],
                   'asset_id': source_frame['asset'],
                   'source_sha256': source_meta.get('source_sha256'),
                   'synthetic': bool(source_meta.get('synthetic')),
                   'width': width, 'height': height,
                   'bounds': bounds, 'pixel_bounds': grid['pixel_bounds'],
                   'patch_size': [grid['columns'], grid['rows']],
                   'patch': data_uri(reference_patch)},
        'results': entries,
        'difference': difference,
        'read_only': True,
        'note': '复核只读取已保存的结果与工程快照，不会切换当前结果或改写历史文件。差异在显示sRGB空间计算；各版本整体处理参数不同会影响数值，不代表内容位置变化。',
    }
def _esc(value):
    return html_module.escape(str(value))


STATUS_LABELS = {
    'ok': '已定位', 'partial': '部分覆盖', 'approximate': '近似对应（不参与差异）',
    'outside': '区域已被裁掉', 'no_valid': '无有效像素', 'no_overlap': '无重叠',
    'missing_frame': '缺少对应素材', 'disabled_frame': '对应帧已停用',
    'ambiguous_frame': '对应不唯一', 'geometry_mismatch': '几何无法对齐',
    'unreadable': '无法读取', 'skipped': '未计算',
}


def _status_label(status):
    return STATUS_LABELS.get(status, str(status))


def _result_version_rows(result):
    rows = [
        ('结果编号', result['id']),
        ('来源修订', 'REV ' + str(result['source_revision'])),
        ('来源签名', result['source_signature']),
        ('生成时间', result['created_at']),
        ('处理模式', result['mode']),
        ('输出尺寸', str(result['width']) + ' × ' + str(result['height'])),
        ('当前选中', '是' if result['current'] else '否'),
        ('设置已变更', '是' if result['stale'] else '否'),
    ]
    return ''.join('<tr><th>' + _esc(label) + '</th><td>' + _esc(value) + '</td></tr>' for label, value in rows)


def _settings_rows(settings, result):
    summary = _settings_summary(settings)
    rows = [
        ('工作长边', summary['size']),
        ('合成模式', summary['mode']),
        ('配准方式', summary['alignment_method']),
        ('裁切框', summary['crop']),
        ('旋转', summary['rotation']),
        ('水平翻转', summary['flip_x']),
        ('垂直翻转', summary['flip_y']),
        ('自动共同覆盖裁切', summary['auto_crop']),
        ('镜头校正', summary['lens_enabled']),
        ('局部光流', summary['flow_enabled']),
        ('局部调整层数', summary['local_layers']),
    ]
    for label, name in [('模式参数', 'mode_settings'), ('色调映射', 'tone'), ('全局后期', 'finishing')]:
        if summary[name] is not None:
            rows.append((label, summary[name]))
    rows.append(('对应帧', (result.get('correspondence') or {}).get('frame_name')))
    rows.append(('对应帧编号', (result.get('correspondence') or {}).get('frame_id')))
    return ''.join('<tr><th>' + _esc(label) + '</th><td>' + _esc(value) + '</td></tr>' for label, value in rows)


def html_report(payload):
    title = '区域复核 · ' + payload['project']['name']
    source = payload['source']
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + _esc(title) + '</title>',
        '<style>body{max-width:1180px;margin:32px auto;padding:0 22px;font:14px/1.65 "Noto Sans CJK SC",sans-serif;color:#23343e}'
        'h1{font-weight:500;margin:8px 0}h2{font-weight:500;margin:28px 0 10px;font-size:17px}'
        'table{border-collapse:collapse;width:100%;margin:8px 0 16px}th,td{border-bottom:1px solid #dde3d8;padding:7px 9px;text-align:left;font-size:12px;vertical-align:top}'
        'th{width:190px;color:#66795f;font-weight:500}img{max-width:100%;border:1px solid #d6ded1;background:#f4f6f0}'
        '.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{border:1px solid #dce3d4;border-radius:6px;padding:14px}'
        '.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;margin-left:6px}'
        '.ok{background:#e3efe3;color:#2f6b3a}.warn{background:#f3ead8;color:#8a6a2c}.bad{background:#f6dfd8;color:#97452f}'
        'pre{white-space:pre-wrap;word-break:break-word;font-size:11px;background:#f0f3e9;padding:12px;border-radius:4px;max-height:360px;overflow:auto}'
        '.muted{color:#77866f;font-size:12px}@media print{button{display:none}.card{break-inside:avoid}}</style>',
        '<button onclick="window.print()">打印 / 另存PDF</button>',
        '<h1>' + _esc(title) + '</h1>',
        '<p class="muted">ScanWeave 区域复核 · ' + _esc(payload['created_at']) + ' · 工程修订 ' + _esc(payload['project']['revision']) + '</p>',
        '<p>本报告只读取历史结果与工程快照，未改变当前选中的结果或任何历史文件。</p>',
        '<h2>复核区域与来源</h2><table>',
        '<tr><th>参考扫描帧</th><td>' + _esc(source['frame_name']) + ' / ' + _esc(source['frame_id']) + '</td></tr>',
        '<tr><th>素材摘要</th><td>' + _esc(source['asset_id']) + '</td></tr>',
        '<tr><th>原始文件摘要</th><td>' + _esc(source.get('source_sha256') or '未记录（程序合成或未嵌入）') + '</td></tr>',
        '<tr><th>归一化区域</th><td>' + _esc(source['bounds']) + '</td></tr>',
        '<tr><th>素材像素范围</th><td>' + _esc(source['pixel_bounds']) + '</td></tr>',
        '<tr><th>对照图尺寸</th><td>' + _esc(source['patch_size']) + '</td></tr>',
        '</table>',
        '<h2>并排对照</h2><div class="grid">',
        '<div class="card"><b>原始参考扫描</b><img alt="原始参考扫描区域" src="' + source['patch'] + '"></div>',
    ]
    for result in payload['results']:
        badge = 'ok' if result['status'] == 'ok' else ('warn' if result.get('available') else 'bad')
        parts.append('<div class="card"><b>版本 ' + _esc(result['id'][:12]) + '</b><span class="badge ' + badge + '">' + _esc(_status_label(result['status'])) + '</span>')
        parts.append('<p class="muted">' + _esc(result['message']) + '</p>')
        if result.get('patch'):
            parts.append('<img alt="版本对应区域" src="' + result['patch'] + '">')
        parts.append('</div>')
    difference = payload['difference']
    parts.append('<div class="card"><b>差异（A ↔ B）</b><span class="badge ' + ('ok' if difference['status'] == 'ok' else 'bad') + '">' + _esc(_status_label(difference['status'])) + '</span>')
    parts.append('<p class="muted">' + _esc(difference['message']) + '</p>')
    if difference.get('image'):
        parts.append('<img alt="差异对照图" src="' + difference['image'] + '">')
    parts.append('</div></div>')

    if difference.get('metrics'):
        metrics = difference['metrics']
        labels = [('比较像素', 'pixels'), ('有效重叠比例', 'overlap_fraction'), ('平均绝对差异', 'mae'), ('均方根差异', 'rmse'),
                  ('差异P95', 'p95_abs'), ('最大差异', 'maximum_abs'), ('变化像素比例', 'changed_fraction'),
                  ('归一化亮度MAE', 'normalized_luminance_mae'), ('亮度相关系数', 'luminance_correlation')]
        parts.append('<h2>差异指标</h2><table>' + ''.join('<tr><th>' + _esc(label) + '</th><td>' + _esc(metrics.get(key)) + '</td></tr>' for label, key in labels) + '</table>')
        parts.append('<p class="muted">' + _esc(metrics.get('note', '')) + '</p>')

    parts.append('<h2>版本来源与处理设置</h2>')
    for result in payload['results']:
        parts.append('<div class="card"><h2>' + _esc(result['id']) + '</h2>')
        parts.append('<p><b>状态：</b>' + _esc(_status_label(result['status'])) + ' — ' + _esc(result['message']) + '</p>')
        if result.get('warnings'):
            parts.append('<p class="warn">' + _esc('；'.join(result['warnings'])) + '</p>')
        parts.append('<table>' + _result_version_rows(result) + '</table>')
        parts.append('<table>' + _settings_rows(result.get('settings'), result) + '</table>')
        if result.get('vs_reference') and result['vs_reference'].get('mae') is not None:
            parts.append('<p class="muted">与原始扫描的显示差异（仅供参考）：MAE ' + _esc(round(result['vs_reference']['mae'], 6)) + ' / 重叠 ' + _esc(round(result['vs_reference']['overlap_fraction'], 4)) + '</p>')
        if result.get('correspondence'):
            parts.append('<details><summary>还原的输出几何</summary><pre>' + _esc(json.dumps(result['correspondence'], ensure_ascii=False, indent=2)) + '</pre></details>')
        parts.append('<details><summary>完整处理设置（该结果保存的工程快照）</summary><pre>' + _esc(json.dumps(result.get('settings') or {}, ensure_ascii=False, indent=2)) + '</pre></details>')
        parts.append('</div>')
    parts.append('<h2>说明</h2><p class="muted">' + _esc(payload['note']) + '</p>')
    parts.append('<p class="muted">对照图中棋盘格表示该版本没有有效像素的区域；这些像素不参与差异统计。</p></html>')
    return ''.join(parts)
