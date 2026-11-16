"""合成一张长幅原稿，再切成若干重叠条带并打乱顺序，供条带工作台测试。

这是测试夹具生成器（开发者自有脚本），输出到独立的临时数据目录，
不接触 /workspace 下的任何应用代码目录。
"""
import numpy as np
from PIL import Image, ImageDraw


def make_scroll(width=360, total_height=1400, seed=7):
    """生成带丰富纵向纹理的长幅图：渐变底 + 网格 + 随机色块 + 文字行。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:total_height, 0:width]
    base = np.zeros((total_height, width, 3), np.float32)
    base[..., 0] = 0.85 - 0.35 * (yy / total_height)
    base[..., 1] = 0.80 - 0.20 * (xx / width)
    base[..., 2] = 0.70 - 0.25 * (yy / total_height)
    raster = Image.fromarray(np.rint(np.clip(base, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(raster)
    # 横向网格线，提供强纵向定位特征
    for y in range(0, total_height, 46):
        draw.line([(0, y), (width, y)], fill=(40, 60, 70), width=2)
    # 纵向线
    for x in range(0, width, 72):
        draw.line([(x, 0), (x, total_height)], fill=(120, 90, 60), width=1)
    # 随机色块，每条带都会包含若个，保证纹理独特
    palette = [(180, 70, 60), (60, 120, 170), (90, 150, 90), (200, 160, 60), (120, 80, 150)]
    for _ in range(60):
        x0 = int(rng.integers(0, width - 40))
        y0 = int(rng.integers(0, total_height - 40))
        x1 = min(width, x0 + int(rng.integers(20, 70)))
        y1 = min(total_height, y0 + int(rng.integers(20, 70)))
        draw.rectangle([x0, y0, x1, y1], fill=palette[int(rng.integers(0, len(palette)))])
    # 序号文字，方便人眼核对顺序
    for y in range(30, total_height, 120):
        draw.text((10, y), 'ROW-%04d' % y, fill=(20, 20, 20))
    # 唯一非周期标记：每行随机位置的独特色块，打破网格自相似
    rng2 = np.random.default_rng(seed + 100)
    unique_colors = [(int(rng2.integers(30, 230)), int(rng2.integers(30, 230)),
                      int(rng2.integers(30, 230))) for _ in range(40)]
    for k, y in enumerate(range(60, total_height - 30, 90)):
        x = int(rng2.integers(width // 2, width - 60))
        draw.ellipse((x, y, x + 26, y + 26), fill=unique_colors[k % len(unique_colors)],
                     outline=(15, 15, 15), width=2)
        draw.text((x + 30, y + 6), 'MK%02d' % k, fill=(10, 10, 10))
    return np.asarray(raster, dtype=np.float32) / 255.0


def cut_strips(scroll, strip_height=360, overlap=90, jitter_x=0, rng=None):
    """沿纵向切成重叠条带。返回 [(name, image), ...]，保持真实顺序。"""
    rng = rng or np.random.default_rng(1)
    h, w = scroll.shape[:2]
    strips = []
    y = 0
    index = 0
    while y < h:
        y1 = min(h, y + strip_height)
        image = scroll[y:y1, :, :].copy()
        if jitter_x and image.shape[1] > jitter_x * 2:
            shift = int(rng.integers(-jitter_x, jitter_x + 1))
            if shift != 0:
                image = np.roll(image, shift, axis=1)
        strips.append(('strip_%02d' % index, image))
        if y1 >= h:
            break
        y = y1 - overlap
        index += 1
    return strips


def shuffled(strips, seed=3):
    rng = np.random.default_rng(seed)
    order = list(range(len(strips)))
    rng.shuffle(order)
    return order, [strips[i] for i in order]


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/striptest_data'
    scroll = make_scroll()
    strips = cut_strips(scroll)
    Image.fromarray(np.rint(scroll * 255).astype(np.uint8)).save(out + '/_original_scroll.png')
    order, mixed = shuffled(strips)
    print('true order:', list(range(len(strips))))
    print('shuffled import order:', order)
    for name, image in mixed:
        Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(
            out + '/' + name + '.png')
    print('wrote', len(mixed), 'strips to', out)
