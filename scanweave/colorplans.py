"""批次色彩校正方案的持久化、验证与应用。

方案保存在工程 settings.color_plans 中，随工程 JSON 一起读写、撤销和导入导出；
只引用归一化坐标与已知颜色，不复制像素，原始扫描素材保持不变。
"""
import re
from copy import deepcopy
from . import model
from . import validation as v
from .colorcal import MODEL_COLUMNS, analyze_patches, apply_color

MAX_PLANS = 8
MAX_PATCHES = 48
HEX_COLOR = re.compile(r'#[0-9a-fA-F]{6}')
UID = re.compile(r'[0-9a-f]{32}')


def _is_uid(value):
    return isinstance(value, str) and bool(UID.fullmatch(value))


def validate_color(value):
    if not isinstance(value, str) or not HEX_COLOR.fullmatch(value):
        raise v.Invalid('颜色必须为#RRGGBB')
    return value.lower()


def validate_bounds(value):
    bounds = v.vector(value, '色块区域', 4, 0, 1)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise v.Invalid('色块区域需要正的宽度和高度')
    return bounds


def validate_patch(patch):
    patch = v.object_value(patch, '色块')
    return {
        'id': v.identifier(patch.get('id')),
        'name': v.text(patch.get('name', '色块'), '色块名称', 40),
        'bounds': validate_bounds(patch.get('bounds')),
        'target': validate_color(patch.get('target')),
    }


def validate_plan(plan):
    plan = v.object_value(plan, '色彩校正方案')
    v.identifier(plan.get('id'))
    v.text(plan.get('name'), '方案名称', 120, False)
    model_name = v.choice(plan.get('model', 'matrix'), '色彩模型', list(MODEL_COLUMNS))
    patches = [validate_patch(item) for item in v.list_value(plan.get('patches'), '色块', MAX_PATCHES)]
    ids = [item['id'] for item in patches]
    if len(set(ids)) != len(ids):
        raise v.Invalid('色块编号重复')
    frame_id = plan.get('frame_id')
    if frame_id is not None:
        v.identifier(frame_id)
    result = {
        'id': plan['id'], 'name': v.text(plan.get('name'), '方案名称', 120, False),
        'frame_id': frame_id,
        'model': model_name,
        'regularization': v.number(plan.get('regularization', .01), '正则强度', 0, 10),
        'patches': patches,
        'applied': v.boolean(plan.get('applied', False), '应用状态'),
        'coefficients': None, 'fit_report': None, 'created_at': v.timestamp(),
        'updated_at': v.timestamp(),
    }
    if plan.get('coefficients') is not None:
        coefficients = v.list_value(plan.get('coefficients'), '校正系数', MODEL_COLUMNS[model_name])
        if len(coefficients) != MODEL_COLUMNS[model_name]:
            raise v.Invalid('校正系数与模型不匹配')
        result['coefficients'] = [v.vector(row, '系数行', 3, -100, 100) for row in coefficients]
    fit_report = plan.get('fit_report')
    if fit_report is not None:
        v.object_value(fit_report, '拟合报告')
        try:
            import json
            json.dumps(fit_report, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise v.Invalid('拟合报告需要可保存的有限JSON数据') from error
        result['fit_report'] = deepcopy(fit_report)
    if plan.get('created_at'):
        result['created_at'] = plan['created_at']
    if plan.get('updated_at'):
        result['updated_at'] = plan['updated_at']
    return result


def color_plans(value):
    items = v.list_value(value or [], '色彩校正方案', MAX_PLANS)
    plans = [validate_plan(item) for item in items]
    if sum(1 for plan in plans if plan['applied']) > 1:
        raise v.Invalid('同一时间只能应用一个色彩校正方案')
    return plans


def calibration_of(plan):
    if not plan.get('applied') or not plan.get('coefficients'):
        return None
    return {'model': plan['model'], 'coefficients': plan['coefficients']}


def active_plan(settings):
    for plan in settings.get('color_plans', []):
        if plan.get('applied') and plan.get('coefficients'):
            return plan
    return None


def apply_to_image(image, settings):
    """对准备阶段的单帧图像应用已启用方案；无方案时原样返回。"""
    plan = active_plan(settings)
    if plan is None:
        return image, None
    corrected = apply_color(image, calibration_of(plan))
    return corrected, {'plan_id': plan['id'], 'plan_name': plan['name'],
                      'model': plan['model'], 'frame_id': plan['frame_id'],
                      'note': '批次色彩校正作用于线性RGB，原始扫描素材未被修改。'}


class ColorPlans:
    def __init__(self, store, assets):
        self.store = store
        self.assets = assets

    def _find(self, project, plan_id):
        v.identifier(plan_id)
        plan = next((item for item in project['settings'].get('color_plans', []) if item['id'] == plan_id), None)
        if plan is None:
            raise v.Invalid('色彩校正方案不存在')
        return plan

    def _frame_image(self, project, frame_id):
        frame = model.find_frame(project, frame_id)
        image = self.assets.field(frame['asset'])
        from .arrays import fit_size, resize
        w, h = fit_size(image.shape, min(project['settings']['size'], 1024))
        return frame, resize(image, w, h)

    def analyze(self, key, body):
        """不写入工程的试算：采样、拟合、给出阻断问题与逐色块可靠性。"""
        project = self.store.get(key)
        frame_id = v.identifier(body.get('frame_id') or project.get('reference') or '')
        _, image = self._frame_image(project, frame_id)
        patches = [{'name': v.text(v.object_value(patch, '色块').get('name', '色块'), '色块名称', 40),
                    'bounds': validate_bounds(patch.get('bounds')),
                    'target': validate_color(patch.get('target'))}
                   for patch in v.list_value(body.get('patches'), '色块', MAX_PATCHES)]
        result = analyze_patches(image, patches,
                                 body.get('model', 'matrix'),
                                 body.get('regularization', .01))
        result['frame_id'] = frame_id
        return result

    def save(self, key, revision, body):
        frame_id = v.identifier(body.get('frame_id') or self.store.get(key).get('reference') or '')
        raw_patches = v.list_value(body.get('patches'), '色块', MAX_PATCHES)
        model_name = v.choice(body.get('model', 'matrix'), '色彩模型', list(MODEL_COLUMNS))
        regularization = v.number(body.get('regularization', .01), '正则强度', 0, 10)
        project = self.store.get(key)
        frame, image = self._frame_image(project, frame_id)
        # 先规整色块，保证试算与落盘使用同一份已验证数据
        patches = []
        for index, patch in enumerate(raw_patches):
            patch = v.object_value(patch, '色块')
            patches.append({'id': patch.get('id') if _is_uid(patch.get('id')) else model.uid(),
                            'name': v.text(patch.get('name', '色块 %d' % (index + 1)), '色块名称', 40),
                            'bounds': validate_bounds(patch.get('bounds')),
                            'target': validate_color(patch.get('target'))})
        analysis = analyze_patches(image, patches, model_name, regularization)
        if not analysis['usable']:
            raise v.Invalid('校正方案不可靠，未保存：' + '；'.join(analysis['issues']))
        plan_id = body.get('plan_id')
        if plan_id is not None:
            v.identifier(plan_id)
        payload = {
            'id': plan_id or model.uid(),
            'name': v.text(body.get('name'), '方案名称', 120, False),
            'frame_id': frame_id, 'model': model_name,
            'regularization': regularization,
            'patches': patches,
            'applied': v.boolean(body.get('applied', False), '应用状态'),
            'coefficients': analysis['fit']['coefficients'],
            'fit_report': self._report(analysis),
        }
        with self.store.edit(key, revision, '保存色彩校正方案') as current:
            plans = current['settings'].setdefault('color_plans', [])
            if len(plans) >= MAX_PLANS and not any(item['id'] == payload['id'] for item in plans):
                raise v.Invalid('最多保存%d个色彩校正方案' % MAX_PLANS)
            existing = next((item for item in plans if item['id'] == payload['id']), None)
            if existing is not None:
                payload['created_at'] = existing.get('created_at', v.timestamp())
                payload['applied'] = existing.get('applied', False) and payload['applied']
                index = plans.index(existing)
                plans[index] = validate_plan(payload)
                saved = plans[index]
            else:
                saved = validate_plan(payload)
                plans.append(saved)
            if saved['applied']:
                for item in plans:
                    if item['id'] != saved['id']:
                        item['applied'] = False
        return {'project': self.store.get(key), 'plan': self.public(self._find(self.store.get(key), payload['id'])),
                'analysis': self._without_coefficients(analysis)}

    def set_applied(self, key, revision, plan_id, applied):
        with self.store.edit(key, revision, '应用色彩校正方案' if applied else '停用色彩校正方案') as project:
            target = self._find(project, plan_id)
            if applied:
                if not target.get('coefficients'):
                    raise v.Invalid('方案尚未完成可靠拟合，不能应用')
                frame_ids = {frame['id'] for frame in project['frames'] if frame.get('enabled')}
                if target.get('frame_id') and target['frame_id'] not in frame_ids:
                    raise v.Invalid('色卡采样帧未启用，无法把该校正应用到当前批次；请先启用该扫描或重新基于启用帧拟合。')
            for plan in project['settings']['color_plans']:
                plan['applied'] = applied and plan['id'] == plan_id
        current = self.store.get(key)
        return {'project': current, 'plan': self.public(self._find(current, plan_id))}

    def remove(self, key, revision, plan_id):
        with self.store.edit(key, revision, '删除色彩校正方案') as project:
            target = self._find(project, plan_id)
            project['settings']['color_plans'].remove(target)
        return self.store.get(key)

    def preview(self, key, body):
        """返回校正前后分屏预览，不修改工程。"""
        project = self.store.get(key)
        frame_id = body.get('frame_id') or project.get('reference')
        frame, image = self._frame_image(project, frame_id)
        if body.get('plan_id'):
            plan = self._find(project, body['plan_id'])
            calibration = calibration_of(plan)
            if calibration is None:
                raise v.Invalid('该方案尚未拟合或已停用')
        else:
            patches = [{'name': v.text(v.object_value(patch, '色块').get('name', '色块'), '色块名称', 40),
                        'bounds': validate_bounds(patch.get('bounds')),
                        'target': validate_color(patch.get('target'))}
                       for patch in v.list_value(body.get('patches'), '色块', MAX_PATCHES)]
            analysis = analyze_patches(image, patches,
                                       body.get('model', 'matrix'),
                                       body.get('regularization', .01))
            if not analysis['usable']:
                raise v.Invalid('校正方案不可靠：' + '；'.join(analysis['issues']))
            calibration = {'model': analysis['model'], 'coefficients': analysis['fit']['coefficients']}
        corrected = apply_color(image, calibration)
        fraction = v.number(body.get('fraction', .5), '分屏位置', 0, 1)
        from .comparison import split_view
        return split_view(image, corrected, fraction)

    def _report(self, analysis):
        fit = analysis['fit']
        return {
            'model': fit['model'], 'rank': fit['rank'], 'condition': fit['condition'],
            'rmse_linear': fit['rmse_linear'],
            'delta_e_mean_before': fit['error_before']['mean'],
            'delta_e_max_before': fit['error_before']['maximum'],
            'delta_e_mean_after': fit['error']['mean'],
            'delta_e_max_after': fit['error']['maximum'],
            'reliable_count': analysis['reliable_count'],
            'patch_count': analysis['patch_count'],
            'issues': analysis['issues'], 'warnings': analysis['warnings'],
            'patches': [{'index': item['index'], 'name': item.get('name'),
                         'target': item['target'], 'reliable': item['reliable'],
                         'flags': item['flags'],
                         'delta_e_before': item['delta_e_before'],
                         'delta_e_after': item['delta_e_after'],
                         'delta_e_cv': item.get('delta_e_cv'),
                         'clipped_fraction': item['clipped_fraction'],
                         'std': item['std']}
                        for item in analysis['patches']],
            'note': fit['error']['note'],
        }

    def _without_coefficients(self, analysis):
        result = deepcopy(analysis)
        if result.get('fit'):
            result['fit'].pop('coefficients', None)
            result['fit'].pop('predicted', None)
        return result

    def public(self, plan):
        return {key: deepcopy(plan[key]) for key in
                ['id', 'name', 'frame_id', 'model', 'regularization', 'patches',
                 'applied', 'coefficients', 'fit_report', 'created_at', 'updated_at']}
