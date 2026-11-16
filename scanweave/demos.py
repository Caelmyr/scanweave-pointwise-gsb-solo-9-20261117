import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from . import model
from . import validation as v
from .arrays import srgb_encode, srgb_decode, blur
from .geometry import normalized_matrix


DEMOS = [
    {'id': 'exposure', 'name': '窗边静物 · 曝光包围', 'mode': 'fusion', 'frames': 5, 'description': '室内暗部与明亮窗景的5档曝光，练习曝光融合与HDR。'},
    {'id': 'focus', 'name': '微距字卡 · 景深序列', 'mode': 'focus', 'frames': 5, 'description': '不同水平区域清晰的5张序列，检查焦点索引与融合边界。'},
    {'id': 'night', 'name': '蓝调城市 · 降噪序列', 'mode': 'denoise', 'frames': 8, 'description': '独立噪声与偶发坏点，练习离群像素剔除。'},
    {'id': 'panorama', 'name': '建筑长卷 · 平面拼接', 'mode': 'panorama', 'frames': 3, 'description': '同一平面场景的三块重叠视野，带已知示范变换。'},
    {'id': 'motion', 'name': '移动物体 · 去鬼影', 'mode': 'fusion', 'frames': 5, 'description': '曝光变化叠加移动物体，比较运动抑制与手绘排除遮罩。'},
    {'id': 'trails', 'name': '夜色流光 · 轨迹合成', 'mode': 'trails', 'frames': 8, 'description': '固定背景中的移动亮点，练习亮色合成与渐隐。'},
    {'id': 'bleedthrough', 'name': '薄纸档案 · 双面透印', 'mode': 'bleedthrough', 'frames': 2, 'description': '正反两面扫描：正面浅字与细线需保留，背面深字透过薄纸形成淡字，练习镜像对齐与透印抑制。'},
]


def scene(width=900, height=600, night=False):
    yy, xx = np.indices((height, width), dtype=np.float32)
    background = np.zeros((height, width, 3), np.float32)
    if night:
        background[:] = [.035, .065, .13]
        background += (1 - yy / height)[:, :, None] * np.array([.04, .06, .1])
    else:
        background[:] = [.29, .33, .31]
        background += (1 - xx / width)[:, :, None] * np.array([.22, .18, .12])
    raster = Image.fromarray(np.rint(np.clip(background, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(raster)
    sx, sy = width / 900, height / 600
    def box(bounds, color, outline=None, width_line=1):
        draw.rectangle([round(bounds[0] * sx), round(bounds[1] * sy), round(bounds[2] * sx), round(bounds[3] * sy)], fill=color, outline=outline, width=width_line)
    box((0, 430, 900, 600), '#4e4639' if not night else '#141e30')
    box((360, 45, 830, 410), '#263b42', '#aaa18a', 5)
    box((375, 60, 815, 395), '#a9c8bd' if not night else '#162c4a')
    for index in range(11):
        left = 380 + index * 40
        top = 210 - (index % 4) * 23
        box((left, top, left + 33, 390), '#718d89' if not night else '#213449')
        for row in range(5):
            for column in range(3):
                color = '#d2cfaa' if not night else ('#e4b66d' if (index + row + column) % 3 else '#40556c')
                box((left + 5 + column * 9, top + 15 + row * 25, left + 9 + column * 9, top + 24 + row * 25), color)
    for x in [375, 590, 815]:
        box((x, 55, x + 8, 400), '#394c4c')
    box((375, 223, 815, 233), '#455750')
    draw.ellipse((int(95 * sx), int(445 * sy), int(340 * sx), int(495 * sy)), fill='#2b3029')
    draw.ellipse((int(120 * sx), int(330 * sy), int(285 * sx), int(480 * sy)), fill='#bf9b65', outline='#dec89a', width=3)
    draw.ellipse((int(120 * sx), int(325 * sy), int(285 * sx), int(355 * sy)), fill='#d0af76')
    draw.ellipse((int(140 * sx), int(332 * sy), int(265 * sx), int(348 * sy)), fill='#67533b')
    for index in range(12):
        angle = index / 12 * np.pi * 2
        px, py = 210 + np.cos(angle) * (50 + index % 3 * 12), 290 + np.sin(angle) * 60
        draw.line((int(200 * sx), int(342 * sy), int(px * sx), int((py - 70) * sy)), fill='#345247', width=max(2, int(3 * sx)))
        draw.ellipse((int((px - 26) * sx), int((py - 100) * sy), int((px + 22) * sx), int((py - 53) * sy)), fill=['#4d7460', '#769571', '#2e594c'][index % 3])
    box((390, 465, 700, 520), '#b69a6c', '#ddcc9b', 2)
    box((405, 435, 680, 465), '#2f555a', '#8baaa4', 2)
    box((435, 410, 655, 435), '#bf8061')
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(10, int(17 * sx)))
    except OSError:
        font = ImageFont.load_default()
    draw.text((int(420 * sx), int(442 * sy)), 'LIGHT / FORM / TIME', fill='#dcd0aa', font=font)
    draw.text((int(425 * sx), int(482 * sy)), 'A STUDY IN PHOTOGRAPHY', fill='#524f40', font=font)
    draw.line((0, int(550 * sy), width, int(550 * sy)), fill='#74654c', width=2)
    return np.asarray(raster, np.float32) / 255


def exposures(image, count=5, moving=False):
    linear = srgb_decode(image)
    yy, xx = np.indices(image.shape[:2], dtype=np.float32)
    window = (xx > image.shape[1] * .42) & (xx < image.shape[1] * .9) & (yy < image.shape[0] * .65)
    linear *= np.where(window, 4., .55)[:, :, None]
    result = []
    for index in range(count):
        exposure = 2. ** (index - count // 2)
        captured = np.clip(srgb_encode(linear * exposure), 0, 1)
        if moving:
            left = round(image.shape[1] * (.3 + index * .09))
            captured[round(image.shape[0] * .58):round(image.shape[0] * .73), left:left + 45] = [.75, .28, .12]
        result.append((captured, exposure))
    return result


def focus_sequence(image, count=5):
    h, w = image.shape[:2]
    blurred = blur(image, 4)
    x = np.arange(w, dtype=np.float32) / w
    result = []
    for index in range(count):
        center = (index + .5) / count
        sharp = np.exp(-((x - center) / (.5 / count)) ** 8)
        field = image * sharp[None, :, None] + blurred * (1 - sharp[None, :, None])
        result.append((field.astype(np.float32), 1.))
    return result


def noisy_sequence(image, count=8, seed=42):
    rng = np.random.default_rng(seed)
    linear = srgb_decode(image)
    result = []
    for index in range(count):
        noise = rng.normal(0, .008, linear.shape).astype(np.float32)
        shot = rng.normal(0, np.sqrt(np.maximum(linear, 0)) * .025).astype(np.float32)
        field = np.clip(srgb_encode(np.maximum(linear + noise + shot, 0)), 0, 1)
        if index % 2 == 0:
            x, y = rng.integers(10, image.shape[1] - 10), rng.integers(10, image.shape[0] - 10)
            field[y:y + 2, x:x + 2] = 1
        result.append((field, 1.))
    return result


def _ink_sheet(width, height, blocks, seed):
    """生成文字与线条的线性暗度图（0=纸基，1=浓墨）。"""
    rng = np.random.default_rng(seed)
    raster = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(raster)
    try:
        font_big = ImageFont.truetype('DejaVuSans.ttf', 26)
        font_small = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font_big = font_small = ImageFont.load_default()
    for index, (top, left, text, faint) in enumerate(blocks):
        font = font_small if faint else font_big
        draw.text((left, top), text, fill=95 if faint else 215, font=font)
        if index % 3 == 0:
            draw.line((left, top + 34, left + 250, top + 34), fill=70 if faint else 175, width=1)
        if index % 4 == 1:
            draw.rectangle((left, top + 42, left + 90, top + 56), outline=80 if faint else 165, width=1)
    density = np.asarray(raster, np.float32) / 255
    return blur(density, .7)


def two_sided_scan(width=760, height=560, seed=23):
    """按透射物理合成薄纸正反两面扫描，返回正面与反面 sRGB 图。"""
    rng = np.random.default_rng(seed)
    front_blocks = [(60 + i * 62, 70 + (i % 2) * 26, f'ARCHIVE LINE {i + 1:02d}  FRONT', i % 2 == 1) for i in range(7)]
    back_blocks = [(70 + i * 58, 90 + ((i + 1) % 2) * 22, f'REVERSE ENTRY {i + 1:02d}', False) for i in range(8)]
    front_ink = _ink_sheet(width, height, front_blocks, seed)
    back_ink = _ink_sheet(width, height, back_blocks, seed + 1)
    back_mirror = np.ascontiguousarray(np.flip(back_ink, axis=1))
    front_mirror = np.ascontiguousarray(np.flip(front_ink, axis=1))
    # 透印经过纸张扩散，且比自身墨迹淡
    front_bleed = blur(back_mirror, 2.2) * .55
    back_bleed = blur(front_mirror, 2.2) * .45
    paper = np.array([.87, .85, .79], np.float32)
    grain = rng.normal(0, .004, (height, width, 1)).astype(np.float32)
    vignette = 1 - .04 * (((np.indices((height, width))[1] / width - .5) ** 2 + (np.indices((height, width))[0] / height - .5) ** 2))[:, :, None]

    def capture(own, bleed, own_strength):
        linear = paper[None, None, :] * vignette
        linear *= np.exp(-(own[:, :, None] * own_strength + bleed[:, :, None] * .9))
        linear += grain
        return np.clip(srgb_encode(np.clip(linear, 0, 1)), 0, 1).astype(np.float32)

    front = capture(front_ink, front_bleed, .85)
    back = capture(back_ink, back_bleed, .95)
    return front, back


def create_demo(store, assets, kind):
    spec = next((entry for entry in DEMOS if entry['id'] == kind), None)
    if spec is None:
        raise v.Invalid('示范序列不存在')
    current = store.create(spec['name'])
    if kind == 'bleedthrough':
        front, back = two_sided_scan()
        with store.edit(current['id'], 0, '载入双面透印示范') as project:
            front_asset = assets.put(front, 'front-side.png', {'synthetic': True, 'scanner': 'synthetic-thin-paper'})
            back_asset = assets.put(back, 'back-side.png', {'synthetic': True, 'scanner': 'synthetic-thin-paper'})
            front_frame = model.frame(front_asset['id'], '正面扫描 front-side')
            back_frame = model.frame(back_asset['id'], '反面扫描 back-side')
            back_frame['mirrored'] = True
            back_frame['matrix'][0][2] = .012
            project['frames'].extend([front_frame, back_frame])
            project['reference'] = front_frame['id']
            project['settings']['mode'] = 'bleedthrough'
            project['settings']['size'] = 768
            project['settings']['bleedthrough']['strength'] = .8
            project['notes'] = spec['description'] + ' 合成薄纸透射模型：背面深字经扩散后透到正面；原始正反面素材均不可变。'
        return store.get(current['id'])
    base = scene(night=kind in ('night', 'trails'))
    if kind in ('exposure', 'motion'):
        sequence = exposures(base, moving=kind == 'motion')
    elif kind == 'focus':
        sequence = focus_sequence(base)
    elif kind == 'night':
        sequence = noisy_sequence(base)
    elif kind == 'trails':
        sequence = []
        for index in range(8):
            field = base.copy() * .6
            for dot in range(16):
                x = int(40 + dot * 48 + index * 3) % 900
                y = int(90 + np.sin(dot * .5 + index * .1) * 60)
                cv2.circle(field, (x, y), 2, (1., .85, .6), -1)
            sequence.append((field, 1.))
    else:
        wide = scene(1500, 600)
        sequence = [(wide[:, x:x + 900].copy(), 1.) for x in [0, 300, 600]]
    with store.edit(current['id'], 0, '载入合成摄影示范') as project:
        project['settings']['mode'] = spec['mode']
        project['settings']['size'] = 768
        project['notes'] = spec['description'] + ' 所有示范图由本地程序生成，用于复现算法，不冒充实拍照片。'
        for index, (field, exposure) in enumerate(sequence):
            asset = assets.put(field.astype(np.float32), f'{kind}-{index + 1:02d}.png', {'synthetic': True, 'exif': {'ExposureTime': exposure}})
            frame = model.frame(asset['id'], f'{kind}-{index + 1:02d}', exposure)
            if kind == 'panorama':
                frame['matrix'][0][2] = index / 3
            project['frames'].append(frame)
        project['reference'] = project['frames'][len(sequence) // 2]['id'] if kind != 'panorama' else project['frames'][0]['id']
        if kind == 'motion':
            project['settings']['fusion']['ghost'] = .85
    return store.get(current['id'])
