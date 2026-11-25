import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class CompressedFixtureDriftTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'test_data').mkdir()
        self.checker = self.root / 'scripts/check_fixture_drift.py'
        shutil.copyfile(Path(__file__).resolve().parents[1] / self.checker.name, self.checker)
        self.fixture = self.root / 'test_data/scenarios.json.gz'
        self.write_fixture(0.75)
        subprocess.run(['git', 'init', '-q'], cwd=self.root, check=True)
        subprocess.run(['git', 'add', 'test_data'], cwd=self.root, check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                        '-c', 'commit.gpgSign=false', 'commit', '-qm', 'baseline'], cwd=self.root, check=True)

    def write_fixture(self, score, mtime=0):
        self.fixture.write_bytes(gzip.compress(json.dumps({'scores': [score]}).encode(), mtime=mtime))

    def check(self):
        return subprocess.run([sys.executable, self.checker], cwd=self.root, capture_output=True, text=True)

    def test_compares_data_with_float_tolerance(self):
        self.write_fixture(0.75000001, mtime=12345)
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.write_fixture(0.8)
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn('scenarios.json.gz.scores[0]', result.stdout)

    def test_rejects_damaged_archive_with_filename(self):
        self.fixture.write_bytes(b'incomplete')
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn('scenarios.json.gz: invalid fixture', result.stdout)

    def test_detects_missing_and_untracked_gzip_fixtures(self):
        self.fixture.rename(self.fixture.with_name('new.json.gz'))
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn('scenarios.json.gz: committed fixture was not regenerated', result.stdout)
        self.assertIn('new.json.gz: generated but not committed', result.stdout)


if __name__ == '__main__':
    unittest.main()
