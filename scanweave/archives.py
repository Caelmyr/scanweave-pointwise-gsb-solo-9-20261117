import io
import zipfile
import json
from pathlib import PurePosixPath
import numpy as np
from . import model
from . import validation as v


def open_archive(raw):
    if len(raw) > 96 * 1024 * 1024:
        raise v.Invalid('工程包不得超过96MB')
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as error:
        raise v.Invalid('无效的ZIP工程包') from error
    names = set()
    total = 0
    try:
        if len(archive.infolist()) > 200:
            raise v.Invalid('工程包文件过多')
        for entry in archive.infolist():
            name = entry.filename
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or name in names or ':' in name:
                raise v.Invalid('工程包路径无效或重复')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise v.Invalid('工程包不能含符号链接')
            if entry.flag_bits & 1:
                raise v.Invalid('不支持加密工程包')
            total += entry.file_size
            if entry.file_size > 160 * 1024 * 1024 or total > 512 * 1024 * 1024:
                raise v.Invalid('工程包解压尺寸超限')
            if entry.file_size / max(entry.compress_size, 1) > 3000:
                raise v.Invalid('工程包压缩率异常')
            names.add(name)
        if 'project.json' not in names:
            raise v.Invalid('工程包缺少project.json')
        return archive
    except Exception:
        archive.close()
        raise


def read_json(archive, name):
    try:
        raw = archive.read(name)
        if len(raw) > 4 * 1024 * 1024:
            raise v.Invalid('工程元数据过大')
        return json.loads(raw.decode('utf8'))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise v.Invalid('工程元数据损坏') from error


def read_array(archive, name):
    try:
        payload = archive.read(name)
        with zipfile.ZipFile(io.BytesIO(payload)) as inner:
            entries = inner.infolist()
            if len(entries) != 1 or entries[0].filename != 'image.npy':
                raise v.Invalid('素材NPZ结构错误')
            if entries[0].file_size > 145000000:
                raise v.Invalid('素材浮点数组超限')
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            image = data['image'].copy()
    except (KeyError, ValueError, zipfile.BadZipFile) as error:
        raise v.Invalid('素材浮点数组无法读取') from error
    if image.dtype != np.float32 or image.ndim not in (2, 3):
        raise v.Invalid('素材必须是float32二维或三通道数组')
    if max(image.shape[:2]) > 4096 or min(image.shape[:2]) < 1:
        raise v.Invalid('素材尺寸无效')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise v.Invalid('素材数值无效')
    return image


def import_restorations(store, archive, created_key):
    """Copy scratch-restoration records carried inside a project archive.

    Returns a mapping from archived record ids to the newly stored ids.
    """
    directory = store.directory(created_key) / 'restorations'
    directory.mkdir(exist_ok=True)
    mapping = {}
    from . import model
    for name in archive.namelist():
        parts = PurePosixPath(name).parts
        if len(parts) != 3 or parts[0] != 'restorations' or parts[2] != 'restoration.json':
            continue
        record = read_json(archive, name)
        try:
            v.identifier(record.get('id'))
        except v.Invalid:
            continue
        payload = archive.read('restorations/' + record['id'] + '/image.npz')
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            image = data['image'].copy()
        if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
            continue
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            continue
        new_id = model.uid()
        target = directory / new_id
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / 'image.npz', image=image)
        record['id'] = new_id
        record['project_id'] = created_key
        store.write(target / 'restoration.json', record)
        mapping[parts[1]] = new_id
    return mapping


def import_project(store, assets, raw):
    with open_archive(raw) as archive:
        imported = read_json(archive, 'project.json')
        model.validate(imported)
        references = {frame['asset'] for frame in imported['frames']}
        references.update(frame['mask'] for frame in imported['frames'] if frame['mask'])
        calibration = imported['settings']['calibration']
        references.update(asset for asset in [calibration['dark'], calibration['flat']] if asset)
        decoded = {}
        for asset_id in references:
            v.sha(asset_id)
            metadata = read_json(archive, 'assets/' + asset_id + '.json')
            kind = v.choice(metadata.get('kind'), '素材类型', ['mask', 'image'])
            field = read_array(archive, 'assets/' + asset_id + '.npz')
            if (kind == 'image' and (field.ndim != 3 or field.shape[2] != 3)) or (kind == 'mask' and field.ndim != 2):
                raise v.Invalid('素材形状与类型不符')
            decoded[asset_id] = (field, metadata)
        mapping = {}
        for asset_id, (field, metadata) in decoded.items():
            saved = assets.put(field, metadata.get('name', '导入素材'), {'exif': metadata.get('exif', {})}, metadata['kind'])
            mapping[asset_id] = saved['id']
        created = store.create(imported['name'][:105] + ' · 导入')
        try:
            with store.edit(created['id'], 0, '导入可编辑工程') as current:
                for key, value in imported.items():
                    if key not in ('id', 'name', 'created_at', 'updated_at', 'revision', 'undo', 'redo', 'current_result'):
                        current[key] = value
                for frame in current['frames']:
                    frame['asset'] = mapping[frame['asset']]
                    if frame['mask']:
                        frame['mask'] = mapping[frame['mask']]
                for key in ['dark', 'flat']:
                    if current['settings']['calibration'][key]:
                        current['settings']['calibration'][key] = mapping[current['settings']['calibration'][key]]
            mapping_done = import_restorations(store, archive, created['id'])
            confirmed = (imported.get('restoration') or {}).get('confirmed')
            if confirmed:
                with store.edit(created['id'], store.get(created['id'])['revision'], '关联导入的修复记录') as current:
                    # A record that could not be restored must not stay
                    # referenced, otherwise the project points at nothing.
                    current['restoration']['confirmed'] = mapping_done.get(confirmed)
        except Exception:
            import shutil
            shutil.rmtree(store.directory(created['id']))
            raise
    return store.get(created['id'])
