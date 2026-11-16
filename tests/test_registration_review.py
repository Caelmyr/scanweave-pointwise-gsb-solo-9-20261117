"""Per-frame registration review: resilience, confidence, selective apply,
and staleness enforcement. Run with: python -I -m unittest discover -s tests."""
import tempfile
import time
import unittest

from scanweave import create_app
from scanweave.scanner import install, difficult_sample, scanning_sample
from scanweave.registration import texture_report, autocorrelation_energy
import numpy as np


def wait_for(client, job_id, attempts=80):
    for _ in range(attempts):
        job = client.get('/api/jobs/' + job_id).json
        if job['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
            return job
        time.sleep(.2)
    raise AssertionError('job did not finish')


class RegistrationReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app(self.tmp.name)
        install(self.app)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def start_align(self, route):
        project = self.client.post(route).json
        job = self.client.post(
            '/api/projects/%s/jobs' % project['id'],
            json={'kind': 'align', 'revision': project['revision']}).json
        return project, wait_for(self.client, job['id'])

    def test_one_failing_frame_keeps_other_results(self):
        project, job = self.start_align('/api/scanner/difficult-sample')
        # 整个任务仍完成，而不是因单帧失败整体失败。
        self.assertEqual(job['status'], 'completed')
        statuses = {f['name']: f['status'] for f in job['details']['frames']}
        self.assertEqual(statuses['scan-reference.png'], 'trusted')
        self.assertEqual(statuses['scan-repetitive.png'], 'suspect')
        self.assertEqual(statuses['scan-near-blank.png'], 'failed')
        self.assertEqual(statuses['scan-off-content.png'], 'failed')
        summary = job['details']['summary']
        self.assertEqual(summary['total'], 4)
        self.assertEqual(summary['failed'], 2)
        self.assertGreaterEqual(summary['trusted'], 1)

    def test_normal_texture_scan_is_fully_trusted(self):
        _, job = self.start_align('/api/scanner/sample')
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['details']['summary']['suspect'], 0)
        self.assertEqual(job['details']['summary']['failed'], 0)
        self.assertEqual(job['details']['summary']['trusted'], 3)

    def test_selective_apply_preserves_other_matrices(self):
        project, job = self.start_align('/api/scanner/difficult-sample')
        pid = project['id']
        original = {f['id']: f['matrix'] for f in project['frames']}
        entries = {f['frame_id']: f for f in job['details']['frames']}
        failed_ids = [fid for fid, e in entries.items() if e['status'] == 'failed']
        suspect_ids = [fid for fid, e in entries.items() if e['status'] == 'suspect']

        # 失败帧不能被接受。
        refused = self.client.post(
            '/api/jobs/%s/apply-alignment' % job['id'],
            json={'revision': project['revision'], 'frame_ids': failed_ids[:1]})
        self.assertEqual(refused.status_code, 400)

        # 只接受存疑帧，其余帧保留原变换。
        response = self.client.post(
            '/api/jobs/%s/apply-alignment' % job['id'],
            json={'revision': project['revision'], 'frame_ids': suspect_ids})
        self.assertEqual(response.status_code, 200)
        updated = {f['id']: f['matrix'] for f in response.json['project']['frames']}
        self.assertNotEqual(updated[suspect_ids[0]], original[suspect_ids[0]])
        for fid in failed_ids:
            self.assertEqual(updated[fid], original[fid])
        self.assertEqual(sorted(response.json['applied_frame_ids']), sorted(suspect_ids))

    def test_default_apply_skips_failed_frames(self):
        project, job = self.start_align('/api/scanner/difficult-sample')
        original = {f['id']: f['matrix'] for f in project['frames']}
        response = self.client.post(
            '/api/jobs/%s/apply-alignment' % job['id'],
            json={'revision': project['revision']})
        self.assertEqual(response.status_code, 200)
        entries = {f['frame_id']: f for f in job['details']['frames']}
        failed_ids = [fid for fid, e in entries.items() if e['status'] == 'failed']
        updated = {f['id']: f['matrix'] for f in response.json['project']['frames']}
        for fid in failed_ids:
            self.assertEqual(updated[fid], original[fid])
        self.assertEqual(set(response.json['skipped_frame_ids']), set(failed_ids))

    def test_reference_change_marks_job_stale_and_blocks_reuse(self):
        project, job = self.start_align('/api/scanner/difficult-sample')
        pid = project['id']
        frames = project['frames']
        other = next(f['id'] for f in frames if f['id'] != project['reference'])
        changed = self.client.patch(
            '/api/projects/%s' % pid,
            json={'revision': project['revision'], 'reference': other}).json
        stale = self.client.get('/api/jobs/' + job['id']).json
        self.assertTrue(stale['stale'])
        self.assertIn('重新配准', stale['stale_reason'])
        blocked = self.client.post(
            '/api/jobs/%s/apply-alignment' % job['id'],
            json={'revision': changed['revision'], 'frame_ids': [frames[0]['id']]})
        self.assertEqual(blocked.status_code, 409)
        preview = self.client.get(
            '/api/jobs/%s/review/%s' % (job['id'], frames[0]['id']))
        self.assertEqual(preview.status_code, 409)

    def test_frame_toggle_marks_job_stale(self):
        project, job = self.start_align('/api/scanner/difficult-sample')
        target = project['frames'][1]['id']
        self.client.patch(
            '/api/projects/%s/frames/%s' % (project['id'], target),
            json={'revision': project['revision'], 'enabled': False})
        self.assertTrue(self.client.get('/api/jobs/' + job['id']).json['stale'])

    def test_own_apply_keeps_remaining_frames_actionable(self):
        project, job = self.start_align('/api/scanner/difficult-sample')
        entries = job['details']['frames']
        trusted = [e['frame_id'] for e in entries if e['status'] == 'trusted']
        first = self.client.post(
            '/api/jobs/%s/apply-alignment' % job['id'],
            json={'revision': project['revision'], 'frame_ids': trusted})
        self.assertEqual(first.status_code, 200)
        # 应用自身结果不会让任务过期，剩余帧可继续分批接受。
        self.assertFalse(self.client.get('/api/jobs/' + job['id']).json['stale'])
        suspect = [e['frame_id'] for e in entries if e['status'] == 'suspect']
        second = self.client.post(
            '/api/jobs/%s/apply-alignment' % job['id'],
            json={'revision': first.json['project']['revision'], 'frame_ids': suspect})
        self.assertEqual(second.status_code, 200)

    def test_review_preview_modes(self):
        project, job = self.start_align('/api/scanner/difficult-sample')
        entry = job['details']['frames'][1]
        for kind in ('overlay', 'difference', 'split', 'checker'):
            response = self.client.get(
                '/api/jobs/%s/review/%s?kind=%s' % (job['id'], entry['frame_id'], kind))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'image/png')

    def test_texture_diagnostics(self):
        reference = scanning_sample()
        report = texture_report(reference)
        self.assertNotIn('blank', report['labels'])
        blank = np.ones_like(reference) * .6
        self.assertIn('blank', texture_report(blank)['labels'])
        y, x = np.indices(reference.shape[:2], dtype=np.float32)
        grid = (.5 + .5 * np.sin(x / 9) * np.sin(y / 9))[..., None]
        repetitive = np.repeat(grid, 3, axis=2).astype(np.float32)
        self.assertGreater(autocorrelation_energy(repetitive), .14)
        # 普通纹理样张不应被误判为重复图案。
        self.assertLess(autocorrelation_energy(scanning_sample()), .14)

    def test_alignment_setting_change_marks_job_stale(self):
        project, job = self.start_align('/api/scanner/sample')
        self.assertFalse(job['stale'])
        changed = self.client.patch(
            '/api/projects/%s' % project['id'],
            json={'revision': project['revision'],
                  'settings': {'alignment': {'method': 'orb'}}}).json
        stale = self.client.get('/api/jobs/' + job['id']).json
        self.assertTrue(stale['stale'])
        blocked = self.client.post(
            '/api/jobs/%s/apply-alignment' % job['id'],
            json={'revision': changed['revision']})
        self.assertEqual(blocked.status_code, 409)

    def test_unfinished_job_cannot_be_applied(self):
        project = self.client.post('/api/scanner/difficult-sample').json
        job = self.client.post(
            '/api/projects/%s/jobs' % project['id'],
            json={'kind': 'align', 'revision': project['revision']}).json
        self.client.post('/api/jobs/%s/cancel' % job['id'])
        finished = wait_for(self.client, job['id'])
        self.assertIn(finished['status'], ('cancelled', 'completed'))
        if finished['status'] != 'completed':
            response = self.client.post(
                '/api/jobs/%s/apply-alignment' % job['id'],
                json={'revision': project['revision']})
            self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
