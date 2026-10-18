import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('git'), 'Git is needed to verify publication exclusions')
class PublicationTests(unittest.TestCase):
    def test_private_runtime_files_and_archived_prototypes_are_excluded(self):
        private = {
            'src/aiconfig.json', '.env', '.env.local', 'data/resume.pdf',
            'data/resume.docx', 'data/resume.txt', 'data/nested/resume.txt',
            '.local/conversations.sqlite3', '.local/conversations.sqlite3-wal',
            '.local/private.key', '.cache/results.json', 'src/models/model.safetensors',
            'concepts/data.json', 'concepts/unified/index.html',
            'concepts/reports/report.md', 'concepts/unified/assets/before-report.png',
        }
        public = {'README.md', 'src/aiconfig.example.json', 'data/.gitkeep',
                  'tests/test_publication.py', 'frontend/package-lock.json', 'agent-browser.json'}
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(['git', 'init', '--quiet', folder], check=True, capture_output=True)
            Path(folder, '.gitignore').write_bytes((ROOT / '.gitignore').read_bytes())
            result = subprocess.run(
                ['git', '-C', folder, '-c', 'core.excludesFile=NUL',
                 'check-ignore', '--no-index', '--stdin', '-z'],
                input='\0'.join(sorted(private | public)) + '\0',
                capture_output=True, text=True, check=True,
            )
            excluded = set(result.stdout.rstrip('\0').split('\0'))
            self.assertEqual(excluded, private)


if __name__ == '__main__':
    unittest.main()
