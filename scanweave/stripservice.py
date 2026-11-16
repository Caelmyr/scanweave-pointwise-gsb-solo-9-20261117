"""条带工作台服务：编排素材、布局文档、邻接求解与非破坏性导出。"""
import numpy as np
import cv2
from . import stripmodel
from . import stripsolve
from . import validation as v
from .codecs import png_bytes, jpeg_bytes, tiff_bytes
from .arrays import resize, fit_size


class StripService:
    def __init__(self, store, assets):
        self.store = store          # StripStore
        self.assets = assets        # 复用不可变素材库

    # ---------- 布局文档 ----------
    def list_sheets(self, query=''):
        return self.store.list(query)

    def get(self, sheet_id):
        return self.store.get(sheet_id)

    def create(self, body):
        data = v.object_value(body, '请求体')
        axis = v.choice(data.get('axis', 'vertical'), '拼接方向', ['vertical', 'horizontal'])
        return self.store.create(data.get('name', '未命名长幅布局'), axis), 201

    def update(self, sheet_id, body):
        data = v.object_value(body)
        revision = v.revision(data)
        name = data.get('name')
        notes = data.get('notes')
        axis = data.get('axis')

        def mutate(sheet):
            if name is not None:
                sheet['name'] = v.text(name, '布局名称', 120, False)
            if notes is not None:
                sheet['notes'] = v.text(notes, '布局说明', 2000)
            if axis is not None:
                sheet['axis'] = v.choice(axis, '拼接方向', ['vertical', 'horizontal'])

        return self.store.save_raw(sheet_id, revision, '修改布局属性', mutate)

    def duplicate(self, sheet_id):
        return self.store.duplicate(sheet_id), 201

    def delete(self, sheet_id):
        path = self.store._path(sheet_id)
        path.unlink()
        # 历史文件随布局删除；原始素材因可能被其他布局引用而保留
        for history in self.store.history_dir.glob(sheet_id + '_*.json'):
            history.unlink()
        return {'deleted': sheet_id}

    # ---------- 条带增删 ----------
    def add_strips(self, sheet_id, body):
        data = v.object_value(body)
        revision = v.revision(data)
        asset_ids = v.list_value(data.get('asset_ids', []), '素材', stripmodel.MAX_STRIPS)

        def mutate(sheet):
            existing = {s['asset'] for s in sheet['strips']}
            for asset_id in asset_ids:
                v.sha(asset_id)
                if asset_id in existing:
                    raise v.Invalid('该素材已在布局中，请勿重复添加')
                if len(sheet['strips']) >= stripmodel.MAX_STRIPS:
                    raise v.Invalid('单份布局最多 %d 条带' % stripmodel.MAX_STRIPS)
                meta = self.assets.metadata(asset_id)
                image = self.assets.field(asset_id)
                index = len(sheet['strips'])
                record = stripmodel.strip_record(asset_id, meta.get('name', asset_id[:8]),
                                                  index, meta['width'], meta['height'])
                # 初始沿主轴顺序堆叠摆放
                if sheet['axis'] == 'vertical':
                    record['position'] = [0.0, float(sum(s['height'] for s in sheet['strips']))]
                else:
                    record['position'] = [float(sum(s['width'] for s in sheet['strips'])), 0.0]
                sheet['strips'].append(record)
                existing.add(asset_id)
            # 新增条带后旧的邻接分析失效，清空
            sheet['edges'] = []
            sheet['issues'] = []

        return self.store.save_raw(sheet_id, revision, '添加条带', mutate), 201

    def remove_strips(self, sheet_id, body):
        data = v.object_value(body)
        revision = v.revision(data)
        ids = set(v.list_value(data.get('strip_ids', []), '条带', stripmodel.MAX_STRIPS))
        for sid in ids:
            v.identifier(sid)

        def mutate(sheet):
            present = {s['id'] for s in sheet['strips']}
            missing = ids - present
            if missing:
                raise v.Invalid('部分条带不存在')
            sheet['strips'] = [s for s in sheet['strips'] if s['id'] not in ids]
            # 清理引用
            sheet['edges'] = [e for e in sheet['edges']
                              if e['a'] not in ids and e['b'] not in ids]
            sheet['issues'] = [i for i in sheet['issues']
                               if not any(ref in ids for ref in i['strips'])]

        return self.store.save_raw(sheet_id, revision, '移除条带', mutate)

    def reorder(self, sheet_id, body):
        """按给定 strip_ids 重排人工序列（修改前后关系）。"""
        data = v.object_value(body)
        revision = v.revision(data)
        ordered = v.list_value(data.get('strip_ids'), '条带序列', stripmodel.MAX_STRIPS)

        def mutate(sheet):
            by_id = {s['id']: s for s in sheet['strips']}
            if set(ordered) != set(by_id):
                raise v.Invalid('序列必须包含且仅包含当前全部条带')
            sheet['strips'] = [by_id[sid] for sid in ordered]
            # 人工改变顺序后记录人工邻接，内容边保留供对照，同一对仅保留一条
            self._rebuild_manual_edges(sheet)
            sheet['edges'] = self._dedupe_edges(sheet['edges'])

        return self.store.save_raw(sheet_id, revision, '调整条带先后关系', mutate)

    def _rebuild_manual_edges(self, sheet):
        """根据当前 strips 顺序生成人工邻接边，保留已有的人工裁决状态。"""
        manual = []
        for first, second in zip(sheet['strips'], sheet['strips'][1:]):
            manual.append({'a': first['id'], 'b': second['id'], 'side': 'after',
                           'score': 1.0, 'overlap_fraction': 0.0, 'offset': [0, 0],
                           'status': 'accepted', 'source': 'manual'})
        # 内容边保留（供对照），人工边追加
        content = [e for e in sheet['edges'] if e['source'] == 'content']
        sheet['edges'] = content + manual

    def update_strip(self, sheet_id, strip_id, body):
        """手动拖动修改某条带位置，或旋转/启用状态。"""
        data = v.object_value(body)
        revision = v.revision(data)
        v.identifier(strip_id)

        def mutate(sheet):
            strip = next((s for s in sheet['strips'] if s['id'] == strip_id), None)
            if strip is None:
                raise v.Invalid('条带不存在')
            if 'position' in data:
                pos = v.vector(data['position'], '条带位置', 2, -40000, 40000)
                strip['position'] = [round(pos[0], 2), round(pos[1], 2)]
                strip['manual'] = True
            if 'rotation' in data:
                strip['rotation'] = v.choice(data['rotation'], '旋转', [0, 90, 180, 270])
                strip['manual'] = True
            if 'enabled' in data:
                strip['enabled'] = v.boolean(data['enabled'], '启用状态')

        return self.store.save_raw(sheet_id, revision, '手动调整条带', mutate)

    def reset_layout(self, sheet_id, body):
        """清除全部手动位置/旋转标记，恢复为未手动编辑状态（不删除条带与分析）。"""
        data = v.object_value(body)
        revision = v.revision(data)

        def mutate(sheet):
            for strip in sheet['strips']:
                strip['manual'] = False
                strip['rotation'] = 0
            # 人工裁决边回退为内容候选状态
            for edge in sheet['edges']:
                if edge.get('source') == 'manual':
                    edge['source'] = 'content'
                    edge['status'] = 'candidate'
            for issue in sheet['issues']:
                issue['resolved'] = False

        return self.store.save_raw(sheet_id, revision, '重置手动调整', mutate)

    # ---------- 邻接分析 ----------
    def analyze(self, sheet_id, body):
        data = v.object_value(body) if body is not None else {}
        revision = v.revision(data)
        sheet = self.store.get(sheet_id)
        axis = sheet['axis']
        strips = [s for s in sheet['strips']]
        if len(strips) < 2:
            raise v.Invalid('至少需要两条条带才能分析邻接关系')
        images = [self.assets.field(s['asset']) for s in strips]

        edges, issues, positions, order = stripsolve.solve_layout(strips, images, axis)

        def mutate(working):
            # 分析只产出“内容边”，不伪造人工邻接；无邻接时 edges 为空并以疑点呈现
            working['edges'] = edges
            working['issues'] = issues
            # 应用建议位置（保留已手动调整条带的位置）
            for strip in working['strips']:
                if not strip.get('manual') and strip['id'] in positions:
                    strip['position'] = positions[strip['id']]
            # 按建议顺序重排（用户之后可再手动调整）
            by_id = {s['id']: s for s in working['strips']}
            working['strips'] = [by_id[sid] for sid in order if sid in by_id]

        saved = self.store.save_raw(sheet_id, revision, '分析条带邻接', mutate)
        return saved

    @staticmethod
    def _dedupe_edges(edges):
        chosen = {}
        priority = {'accepted': 3, 'rejected': 2, 'candidate': 1}
        for e in edges:
            key = tuple(sorted((e['a'], e['b'])))
            if key not in chosen or priority.get(e['status'], 0) > priority.get(chosen[key]['status'], 0):
                chosen[key] = e
        return list(chosen.values())

    def set_edge(self, sheet_id, body):
        """人工接受/拒绝某条邻接关系。"""
        data = v.object_value(body)
        revision = v.revision(data)
        a, b = v.identifier(data.get('a')), v.identifier(data.get('b'))
        status = v.choice(data.get('status'), '邻接裁决', ['accepted', 'rejected', 'candidate'])

        def mutate(sheet):
            edge = next((e for e in sheet['edges']
                         if tuple(sorted((e['a'], e['b']))) == tuple(sorted((a, b)))), None)
            if edge is None:
                raise v.Invalid('邻接关系不存在，请先运行分析')
            edge['status'] = status
            edge['source'] = 'manual' if status != 'candidate' else edge.get('source', 'content')
            # 人工裁决后，关联的疑点标记为已解决
            for issue in sheet['issues']:
                if a in issue['strips'] and b in issue['strips']:
                    issue['resolved'] = True

        return self.store.save_raw(sheet_id, revision, '人工裁决邻接关系', mutate)

    def resolve_issue(self, sheet_id, body):
        data = v.object_value(body)
        revision = v.revision(data)
        issue_id = v.identifier(data.get('issue_id'))
        resolved = v.boolean(data.get('resolved', True), '疑点状态')

        def mutate(sheet):
            issue = next((i for i in sheet['issues'] if i['id'] == issue_id), None)
            if issue is None:
                raise v.Invalid('疑点不存在')
            issue['resolved'] = resolved

        return self.store.save_raw(sheet_id, revision, '处理疑点', mutate)

    def confirm(self, sheet_id, body):
        data = v.object_value(body)
        revision = v.revision(data)

        def mutate(sheet):
            if not sheet['strips']:
                raise v.Invalid('布局为空，无法确认')
            blocking = [i for i in sheet['issues'] if not i['resolved'] and i['severity'] == 'error']
            if blocking:
                raise v.Invalid('仍有 %d 个错误级疑点未处理，请先确认或解决后再确认布局' % len(blocking))
            sheet['confirmed'] = True

        return self.store.save_raw(sheet_id, revision, '确认布局', mutate, keep_confirmed=True)

    def undo(self, sheet_id, body, redo=False):
        return self.store.undo(sheet_id, v.revision(v.object_value(body)), redo)

    # ---------- 可视化与导出 ----------
    def _composite(self, sheet, background=False):
        """按当前布局把条带绘制到统一画布。

        重叠区域按覆盖次数做等权平均，避免硬接缝；条带像素只做几何摆放，
        不会被裁切或重写（落在画布外的部分仅不参与本次渲染，素材本身不变）。
        background=True 时未覆盖区域铺棋盘底以便观察缺口。
        """
        strips = [s for s in sheet['strips'] if s['enabled']]
        if not strips:
            raise v.Invalid('没有可导出的启用条带')
        positions = {s['id']: s['position'] for s in sheet['strips']}

        # 旋转 90/270 后宽高互换，按旋转后的有效尺寸计算包围盒
        rendered = []
        max_x = max_y = 0.0
        for strip in strips:
            image = self.assets.field(strip['asset'])
            if strip['rotation']:
                image = self._rotate(image, strip['rotation'])
            h, w = image.shape[:2]
            x, y = positions[strip['id']]
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            rendered.append((image, x, y, w, h))
        width, height = max(1, int(np.ceil(max_x))), max(1, int(np.ceil(max_y)))

        accum = np.zeros((height, width, 3), np.float32)
        coverage = np.zeros((height, width), np.float32)
        for image, x, y, w, h in rendered:
            x0, y0 = int(round(x)), int(round(y))
            # 源/目标裁剪区间：画布外部分不渲染，但不丢弃素材像素
            sx0, sy0 = max(0, -x0), max(0, -y0)
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(width, x0 + w), min(height, y0 + h)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            patch = image[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
            accum[dy0:dy1, dx0:dx1] += patch
            coverage[dy0:dy1, dx0:dx1] += 1.0
        canvas = accum / np.maximum(coverage, 1.0)[..., None]
        if background:
            from .arrays import checkerboard
            board = checkerboard(canvas.shape)
            alpha = np.clip(coverage, 0, 1)[..., None]
            canvas = canvas * alpha + board * (1 - alpha)
        return np.clip(canvas, 0, 1), coverage, (width, height)

    @staticmethod
    def _rotate(image, rotation):
        flag = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotation)
        if flag is None:
            return image
        return cv2.rotate(image, flag)

    def preview_png(self, sheet_id):
        sheet = self.store.get(sheet_id)
        image, coverage, size = self._composite(sheet, background=True)
        return png_bytes(image)

    def strip_thumbnail(self, sheet_id, strip_id):
        sheet = self.store.get(sheet_id)
        strip = next((s for s in sheet['strips'] if s['id'] == strip_id), None)
        if strip is None:
            raise v.Invalid('条带不存在')
        image = self.assets.field(strip['asset'])
        if strip['rotation']:
            image = self._rotate(image, strip['rotation'])
        w, h = fit_size(image.shape, 320)
        return png_bytes(resize(image, w, h))

    def export(self, sheet_id, body):
        data = v.object_value(body)
        sheet = self.store.get(sheet_id)
        if not sheet['confirmed']:
            raise v.Invalid('布局尚未确认。请处理疑点并确认布局后再导出，避免导出错误拼接的整图。')
        fmt = v.choice(data.get('format', 'png'), '导出格式', ['png', 'jpeg', 'tiff'])
        image, coverage, size = self._composite(sheet, background=False)
        name = 'strip-layout.' + ('jpg' if fmt == 'jpeg' else fmt)
        if fmt == 'png':
            return png_bytes(image, 16), name, 'image/png'
        if fmt == 'jpeg':
            return jpeg_bytes(image, data.get('quality', 92)), name, 'image/jpeg'
        return tiff_bytes(image, 16), name, 'image/tiff'

    def layout_manifest(self, sheet_id):
        """导出与图像一致的布局说明 JSON（不含像素，仅几何、哈希、疑点）。"""
        sheet = self.store.get(sheet_id)
        return {
            'format': 'scanweave-strip-layout-v1',
            'sheet_id': sheet['id'],
            'name': sheet['name'],
            'axis': sheet['axis'],
            'confirmed': sheet['confirmed'],
            'revision': sheet['revision'],
            'strips': [{k: s[k] for k in ('id', 'asset', 'name', 'width', 'height',
                                          'position', 'rotation', 'manual', 'enabled')}
                       for s in sheet['strips']],
            'edges': sheet['edges'],
            'issues': sheet['issues'],
            'note': '原始扫描以内容哈希引用，未被裁切或重写；导出图像严格按此布局渲染。',
        }
