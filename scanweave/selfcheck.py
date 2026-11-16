"""Self-checks for scratch annotation and texture-preserving restoration.

Run with:  python3 -m scanweave.selfcheck
"""
import io
import shutil
import tempfile
import time
import json
import numpy as np
from . import create_app
from .scanner import install
from .exporting import project_bundle
from . import restoration


def _texture_case():
    """A thin scratch over paper grain, dashes and an edge must stay textured."""
    height, width = 240, 340
    rng = np.random.default_rng(4)
    truth = np.zeros((height, width, 3), np.float32) + [.86, .83, .74]
    truth += rng.normal(0, .025, (height, width, 1)).astype(np.float32)
    for line in range(5):
        y = 50 + line * 26
        for x in range(50, 290, 8):
            truth[y:y + 2, x:x + 4] = [.16, .15, .14]
    truth[150:, 220:] = [.34, .42, .40]
    scratched = truth.copy()
    for step in range(150):
        y, x = 30 + step, 60 + int(step * 1.1)
        scratched[y:y + 2, x:x + 2] = [.96, .95, .91]
    stroke = {'id': 'a' * 32, 'mode': 'paint', 'radius': .01,
              'points': [[60 / width, 30 / height], [225 / width, 180 / height]]}
    result, report = restoration.repair(scratched, [stroke], restoration.DEFAULT_SETTINGS)
    mask = restoration.coverage((height, width), [stroke], restoration.DEFAULT_SETTINGS['expand'])
    assert report['outside_changed'] == 0, report
    assert not ((np.abs(result - scratched).max(axis=2) > 1 / 1024) & (mask <= 0)).any()
    inside = result[mask > .02]
    assert float(inside.std()) > .02, 'paper grain must survive inside the repair'
    # Dashes and the paper/edge boundary must not collapse to one flat colour.
    assert float(np.abs(result[mask > .5] - truth[mask > .5]).mean()) < .06
    # A broad mark must be filled with real texture, not left as scratch.
    broad = {'id': 'b' * 32, 'mode': 'paint', 'radius': .06,
             'points': [[110 / width, 200 / height], [240 / width, 200 / height]]}
    broad_mask = restoration.coverage((height, width), [broad], .002) > .5
    band = scratched.copy()
    band[broad_mask] = [.93, .92, .86]
    filled, report = restoration.repair(band, [broad], dict(restoration.DEFAULT_SETTINGS, expand=.002))
    assert report['outside_changed'] == 0, report
    residual = np.abs(filled[broad_mask] - [.93, .92, .86]).max(axis=1)
    assert residual.mean() > .2, 'a broad mark must not keep the scratch colour'
    assert report['texture_pixels'] > 0, 'a broad mark should transfer real texture'
    # And the copied texture must resemble the original material.
    assert float(np.abs(filled[broad_mask] - truth[broad_mask]).mean()) < .08, report


def _wait(c, job_id, limit=600):
    for _ in range(limit):
        job = c.get('/api/jobs/' + job_id).get_json()
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.05)
    raise AssertionError('job did not finish')


def _sample(c):
    project = c.post('/api/scanner/sample').get_json()
    job = c.post('/api/projects/' + project['id'] + '/jobs',
                 json={'kind': 'process', 'revision': project['revision']}).get_json()
    finished = _wait(c, job['id'])
    assert finished['status'] == 'completed', finished.get('error')
    return project['id']


def _image(c, url):
    from PIL import Image
    response = c.get(url)
    assert response.status_code == 200, response.status_code
    return np.asarray(Image.open(io.BytesIO(response.data)).convert('RGB'), np.float32) / 255


def main():
    _texture_case()
    root = tempfile.mkdtemp(prefix='scanweave-selfcheck-')
    try:
        app = create_app(root)
        install(app)
        services = app.extensions
        c = app.test_client()
        key = _sample(c)

        # --- marks are saved in the editable project document -------------
        project = c.get('/api/projects/' + key).get_json()
        before = c.get('/api/projects/' + key + '/result-image/image').data
        marks = {'strokes': [
            {'mode': 'paint', 'radius': .008, 'points': [[.15, .55], [.45, .62], [.82, .45]]},
            {'mode': 'paint', 'radius': .01, 'points': [[.5, .2], [.52, .35]]},
        ]}
        response = c.post('/api/projects/' + key + '/scratch-marks',
                          json={'revision': project['revision'], **marks})
        assert response.status_code == 200, response.get_json()
        project = response.get_json()
        assert len(project['restoration']['strokes']) == 2
        assert c.get('/api/projects/' + key + '/result-image/image').data == before, 'composite must not change'

        # --- preview keeps everything outside the marks untouched ---------
        original = _image(c, '/api/projects/' + key + '/result-image/image')
        repaired_response = c.get('/api/projects/' + key + '/scratch-preview?view=repaired')
        report = json.loads(repaired_response.headers['X-Scratch-Report'])
        assert report['method'] == 'texture', report
        assert report['marked_pixels'] > 0
        assert report['outside_changed'] == 0, report
        from PIL import Image
        repaired = np.asarray(Image.open(io.BytesIO(repaired_response.data)).convert('RGB'), np.float32) / 255
        changed = np.abs(repaired - original).max(axis=2) > 1.5 / 255
        assert changed.any(), 'repair must change something'
        mask = __import__('scanweave.restoration', fromlist=['x']).coverage(
            original.shape, project['restoration']['strokes'], project['restoration']['settings']['expand'])
        assert not (changed & (mask <= 0)).any(), 'pixels outside the marks changed'

        # --- comparison views render --------------------------------------
        for view in ('original', 'marked', 'difference', 'split'):
            assert c.get('/api/projects/' + key + '/scratch-preview?view=' + view).status_code == 200, view

        # --- erase removes coverage ---------------------------------------
        eraser = c.post('/api/projects/' + key + '/scratch-marks',
                        json={'revision': project['revision'],
                              'strokes': [{'mode': 'erase', 'radius': .05, 'points': [[.45, .62]]}]})
        assert eraser.status_code == 200
        project = eraser.get_json()
        assert project['restoration']['strokes'][-1]['mode'] == 'erase'

        # --- undo a single stroke ------------------------------------------
        stroke_id = project['restoration']['strokes'][-1]['id']
        undone = c.post('/api/projects/' + key + '/scratch-marks/' + stroke_id + '/remove',
                        json={'revision': project['revision']})
        assert undone.status_code == 200
        project = undone.get_json()
        assert len(project['restoration']['strokes']) == 2

        # --- settings adjust the repair range ------------------------------
        adjusted = c.patch('/api/projects/' + key + '/scratch-settings',
                           json={'revision': project['revision'], 'expand': .004, 'feather': .002})
        assert adjusted.status_code == 200, adjusted.get_json()
        project = adjusted.get_json()
        assert abs(project['restoration']['settings']['expand'] - .004) < 1e-9

        # --- confirm creates a separate record -----------------------------
        confirmed = c.post('/api/projects/' + key + '/scratch-restore',
                           json={'revision': project['revision'], 'note': '自测'})
        assert confirmed.status_code == 201, confirmed.get_json()
        output = confirmed.get_json()
        record = output['restoration']
        project = output['project']
        assert project['restoration']['confirmed'] == record['id']
        assert c.get('/api/projects/' + key + '/result-image/image').data == before
        image = c.get('/api/projects/' + key + '/scratch-restorations/' + record['id'] + '/image')
        assert image.status_code == 200

        # --- export only the confirmed repair ------------------------------
        exported = c.post('/api/projects/' + key + '/scratch-restorations/' + record['id'] + '/export',
                          json={'format': 'png', 'bits': 8})
        assert exported.status_code == 200 and exported.data[:8] == b'\x89PNG\r\n\x1a\n'

        # --- marks survive save/reload and undo/redo -----------------------
        c2 = app.test_client()
        reloaded = c2.get('/api/projects/' + key).get_json()
        assert len(reloaded['restoration']['strokes']) == 2
        undo = c2.post('/api/projects/' + key + '/undo', json={'revision': reloaded['revision']})
        assert undo.status_code == 200
        state = undo.get_json()
        assert state['restoration']['confirmed'] is None
        # Undoing an annotation-only change must keep the composite usable.
        assert state['current_result'], 'undoing marks must not invalidate the composite'
        assert c2.get('/api/projects/' + key + '/scratch-preview?view=repaired').status_code == 200
        redo = c2.post('/api/projects/' + key + '/redo', json={'revision': state['revision']})
        assert redo.status_code == 200
        assert redo.get_json()['restoration']['confirmed'] == record['id']

        # --- project bundle carries marks and repair records ---------------
        store, assets = services['store'], services['assets']
        raw = project_bundle(store.get(key), assets, store)
        imported = c.post('/api/import-project', data={'file': (io.BytesIO(raw), 'p.zip')},
                          content_type='multipart/form-data')
        assert imported.status_code == 201, imported.get_json()
        copy = imported.get_json()
        assert len(copy['restoration']['strokes']) == 2
        assert copy['restoration']['confirmed'] is not None
        records = c.get('/api/projects/' + copy['id'] + '/scratch-restorations').get_json()
        assert records and records[0]['current']
        copy_image = c.get('/api/projects/' + copy['id'] + '/scratch-restorations/'
                           + copy['restoration']['confirmed'] + '/image')
        assert copy_image.status_code == 200
        print('scanweave scratch restoration self-check passed')
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    main()
