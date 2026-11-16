"""条带布局的持久化：独立于摄影工程的 sheets 目录，带修订锁与撤销历史。"""
import os
from pathlib import Path
from threading import RLock
from contextlib import contextmanager
from copy import deepcopy
from . import validation as v
from . import stripmodel


class StripStore:
    def __init__(self, root):
        self.root = Path(root)
        self.sheets = self.root / 'sheets'
        self.history_dir = self.root / 'strip_history'
        self.lock = RLock()
        for path in (self.root, self.sheets, self.history_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _path(self, sheet_id):
        v.identifier(sheet_id)
        path = self.sheets / (sheet_id + '.json')
        if not path.exists():
            raise v.Invalid('条带布局不存在')
        return path

    def _read(self, path):
        return self._load(path)

    def _load(self, path):
        import json
        try:
            return json.loads(path.read_text(encoding='utf8'))
        except (OSError, json.JSONDecodeError) as error:
            raise v.Invalid('布局文件读取失败：' + path.name) from error

    def _write(self, path, data):
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + stripmodel.uid() + '.tmp')
        try:
            with temporary.open('w', encoding='utf8') as stream:
                json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def list(self, query=''):
        with self.lock:
            result = []
            for path in self.sheets.glob('*.json'):
                sheet = self._load(path)
                if query.casefold() not in (sheet['name'] + ' ' + sheet.get('notes', '')).casefold():
                    continue
                result.append(self.summary(sheet))
        return sorted(result, key=lambda item: item['updated_at'], reverse=True)

    @staticmethod
    def summary(sheet):
        open_issues = sum(1 for issue in sheet['issues'] if not issue['resolved'])
        return {
            'id': sheet['id'], 'name': sheet['name'], 'revision': sheet['revision'],
            'created_at': sheet['created_at'], 'updated_at': sheet['updated_at'],
            'axis': sheet['axis'], 'strip_count': len(sheet['strips']),
            'confirmed': sheet['confirmed'], 'open_issues': open_issues,
        }

    def get(self, sheet_id):
        with self.lock:
            return self._load(self._path(sheet_id))

    def create(self, name, axis='vertical'):
        name = v.text(name, '布局名称', 120, False)
        sheet = stripmodel.blank_sheet(name)
        sheet['axis'] = v.choice(axis, '拼接方向', ['vertical', 'horizontal'])
        with self.lock:
            self._write(self.sheets / (sheet['id'] + '.json'), sheet)
        return sheet

    @contextmanager
    def edit(self, sheet_id, revision, operation, keep_confirmed=False):
        with self.lock:
            path = self._path(sheet_id)
            original = self._load(path)
            if original['revision'] != revision:
                raise v.Conflict('布局已更新，请刷新后再试')
            working = deepcopy(original)
            yield working
            stripmodel.validate(working)
            entry_id = stripmodel.uid()
            entry = {
                'id': entry_id, 'operation': operation, 'time': v.timestamp(),
                'before': deepcopy(original), 'after': deepcopy(working),
            }
            self._write(self.history_dir / (sheet_id + '_' + entry_id + '.json'), entry)
            working['revision'] = revision + 1
            working['updated_at'] = v.timestamp()
            working['undo'] = (original['undo'] + [entry_id])[-100:]
            working['redo'] = []
            # 几何或关系一旦改变需要重新确认；显式确认操作除外
            if not keep_confirmed:
                working['confirmed'] = False
            self._write(path, working)

    def save_raw(self, sheet_id, revision, operation, mutator, keep_confirmed=False):
        """供分析流程整体替换 edges/issues/positions 的便捷封装。"""
        with self.edit(sheet_id, revision, operation, keep_confirmed) as working:
            mutator(working)
        return self.get(sheet_id)

    def undo(self, sheet_id, revision, redo=False):
        with self.lock:
            path = self._path(sheet_id)
            original = self._load(path)
            if original['revision'] != revision:
                raise v.Conflict('修订已改变')
            source = original['redo'] if redo else original['undo']
            if not source:
                raise v.Invalid('没有可重做记录' if redo else '没有可撤销记录')
            entry = self._load(self.history_dir / (sheet_id + '_' + source[-1] + '.json'))
            restored = deepcopy(entry['after' if redo else 'before'])
            restored['undo'] = original['undo'] + [source[-1]] if redo else original['undo'][:-1]
            restored['redo'] = original['redo'][:-1] if redo else original['redo'] + [source[-1]]
            restored['revision'] = revision + 1
            restored['updated_at'] = v.timestamp()
            stripmodel.validate(restored)
            self._write(path, restored)
        return self.get(sheet_id)

    def history(self, sheet_id):
        sheet = self.get(sheet_id)
        result = []
        for entry_id in reversed(sheet['undo']):
            entry = self._load(self.history_dir / (sheet_id + '_' + entry_id + '.json'))
            result.append({k: entry[k] for k in ('id', 'operation', 'time')})
        return result

    def duplicate(self, sheet_id):
        source = self.get(sheet_id)
        created = self.create(source['name'][:100] + ' · 副本', source['axis'])
        with self.lock:
            copied = deepcopy(source)
            copied['id'] = created['id']
            copied['name'] = created['name']
            copied['created_at'] = created['created_at']
            copied.update(revision=0, updated_at=v.timestamp(), undo=[], redo=[],
                          confirmed=False)
            # 重新生成条带与疑点的内部 id，避免与源布局混淆
            id_map = {}
            for strip in copied['strips']:
                new_id = stripmodel.uid()
                id_map[strip['id']] = new_id
                strip['id'] = new_id
            for edge in copied['edges']:
                edge['a'], edge['b'] = id_map[edge['a']], id_map[edge['b']]
            for issue in copied['issues']:
                issue['id'] = stripmodel.uid()
                issue['strips'] = [id_map[s] for s in issue['strips']]
            stripmodel.validate(copied)
            self._write(self.sheets / (copied['id'] + '.json'), copied)
        return self.get(copied['id'])
