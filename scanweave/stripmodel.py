"""长幅条带拼接布局的数据模型与校验。

一份“条带布局”(sheet) 描述若干条带在统一画布上的相对位置，以及由图像内容
推断出的候选邻接关系(edges)和需要人工确认的疑点(issues)。布局只保存几何与
关系，不重写任何原始扫描像素；原始扫描以不可变素材(asset)的哈希引用。
"""
import uuid
from . import validation as v

# 单份布局允许的条带上限，控制两两配准的计算量
MAX_STRIPS = 16
# 邻接候选与疑点上限，避免无纹理素材产生海量记录
MAX_EDGES = 120
MAX_ISSUES = 80


def uid():
    return uuid.uuid4().hex


def blank_sheet(name):
    now = v.timestamp()
    return {
        'schema': 1,
        'id': uid(),
        'name': name,
        'revision': 0,
        'created_at': now,
        'updated_at': now,
        'axis': 'vertical',          # vertical=条带沿上下接续；horizontal=左右接续
        'strips': [],                # 条带，顺序即当前人工序列
        'edges': [],                 # 内容分析得到的候选邻接关系
        'issues': [],                # 待人工确认的疑点
        'confirmed': False,          # 用户确认布局无未解疑点后才允许整图导出
        'notes': '',
        'undo': [],
        'redo': [],
    }


def strip_record(asset, name, index, width, height):
    """新建条带记录。position 为条带左上角在统一画布的工作像素坐标。"""
    return {
        'id': uid(),
        'asset': asset,
        'name': name,
        'width': int(width),
        'height': int(height),
        'position': [0, 0],          # [x, y] 工作像素，可被拖动修改
        'rotation': 0,               # 0/90/180/270，仅用于摆放，不改原素材
        'manual': False,             # 位置是否经过人工调整
        'enabled': True,
    }


def asset_name(asset):
    return asset[:8]


def check_strip(strip, ids):
    v.object_value(strip, '条带')
    v.identifier(strip.get('id'))
    if strip['id'] in ids:
        raise v.Invalid('条带编号重复')
    ids.add(strip['id'])
    v.sha(strip.get('asset'))
    v.text(strip.get('name'), '条带名称', 180, False)
    v.integer(strip.get('width'), '条带宽度', 1, 8192)
    v.integer(strip.get('height'), '条带高度', 1, 8192)
    pos = v.vector(strip.get('position', [0, 0]), '条带位置', 2, -40000, 40000)
    strip['position'] = [round(pos[0], 3), round(pos[1], 3)]
    rot = v.choice(strip.get('rotation', 0), '条带旋转', [0, 90, 180, 270])
    strip['rotation'] = rot
    v.boolean(strip.get('manual', False), '人工调整标记')
    v.boolean(strip.get('enabled', True), '启用条带')
    return strip


def check_edge(edge, strip_ids):
    v.object_value(edge, '邻接关系')
    v.identifier(edge.get('a'))
    v.identifier(edge.get('b'))
    if edge['a'] not in strip_ids or edge['b'] not in strip_ids:
        raise v.Invalid('邻接关系引用了不存在的条带')
    if edge['a'] == edge['b']:
        raise v.Invalid('邻接关系不能指向自身')
    v.choice(edge.get('side'), '邻接方向', ['before', 'after'])
    v.number(edge.get('score'), '邻接评分', 0, 1)
    v.number(edge.get('overlap_fraction'), '重叠比例', 0, 1)
    offset = v.vector(edge.get('offset', [0, 0]), '邻接位移', 2, -40000, 40000)
    edge['offset'] = [round(offset[0], 3), round(offset[1], 3)]
    v.choice(edge.get('status'), '邻接状态', ['candidate', 'accepted', 'rejected'])
    v.choice(edge.get('source'), '邻接来源', ['content', 'manual'])
    return edge


def check_issue(issue, strip_ids):
    v.object_value(issue, '疑点')
    v.identifier(issue.get('id'))
    v.choice(issue.get('kind'), '疑点类型',
             ['no_neighbor', 'ambiguous_neighbor', 'conflict', 'weak_match', 'cycle', 'gap'])
    v.choice(issue.get('severity'), '疑点级别', ['warning', 'error'])
    v.text(issue.get('message'), '疑点说明', 300, False)
    refs = v.list_value(issue.get('strips', []), '疑点关联条带', MAX_STRIPS)
    for ref in refs:
        v.identifier(ref)
        if ref not in strip_ids:
            raise v.Invalid('疑点引用了不存在的条带')
    issue['strips'] = refs
    v.boolean(issue.get('resolved', False), '疑点已解决')
    return issue


def validate(sheet):
    if sheet.get('schema') != 1:
        raise v.Invalid('不支持的条带布局格式')
    v.identifier(sheet.get('id'))
    v.text(sheet.get('name'), '布局名称', 120, False)
    v.choice(sheet.get('axis', 'vertical'), '拼接方向', ['vertical', 'horizontal'])
    ids = set()
    strips = v.list_value(sheet.get('strips'), '条带', MAX_STRIPS)
    for strip in strips:
        check_strip(strip, ids)
    # 素材不允许在同一份布局里重复
    assets = [s['asset'] for s in strips]
    if len(set(assets)) != len(assets):
        raise v.Invalid('同一份布局中不能重复添加同一扫描素材')
    edges = v.list_value(sheet.get('edges', []), '邻接关系', MAX_EDGES)
    for edge in edges:
        check_edge(edge, ids)
    issues = v.list_value(sheet.get('issues', []), '疑点', MAX_ISSUES)
    issue_ids = set()
    for issue in issues:
        if issue['id'] in issue_ids:
            raise v.Invalid('疑点编号重复')
        issue_ids.add(issue['id'])
        check_issue(issue, ids)
    v.boolean(sheet.get('confirmed', False), '布局确认状态')
    v.text(sheet.get('notes', ''), '布局说明', 2000)
    return sheet


def public(sheet):
    """返回给前端的视图（内部历史字段不暴露）。"""
    return {k: sheet[k] for k in (
        'schema', 'id', 'name', 'revision', 'created_at', 'updated_at', 'axis',
        'strips', 'edges', 'issues', 'confirmed', 'notes', 'undo', 'redo')}
