import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAFE_ARG = '--disable-features=AutofillAiWalletPrivatePasses'


class BrowserLaunchTests(unittest.TestCase):
    def test_playwright_launches_disable_windows_password_probe(self):
        launches = 0
        for folder in ('extended-e2e', 'real-resumes'):
            for path in (ROOT / '.cache' / folder).glob('*.py'):
                tree = ast.parse(path.read_text(encoding='utf-8'))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                        continue
                    if node.func.attr not in ('launch', 'launch_persistent_context'):
                        continue
                    launches += 1
                    args = next((item.value for item in node.keywords if item.arg == 'args'), None)
                    with self.subTest(path=str(path), line=node.lineno):
                        self.assertIsNotNone(args, 'Browser launch must disable the Windows password probe')
                        self.assertIn(SAFE_ARG, ast.literal_eval(args))
        if not launches:
            self.skipTest('No local Playwright scripts present')

    def test_agent_browser_defaults_disable_windows_password_probe(self):
        config = json.loads((ROOT / 'agent-browser.json').read_text(encoding='utf-8'))
        self.assertEqual(config['args'], SAFE_ARG)

    def test_agent_browser_scripts_pass_the_safe_flag_explicitly(self):
        paths = [*(ROOT / '.cache' / 'bulk220').glob('*-ui.py'),
                 *(ROOT / '.cache' / 'extended-e2e').glob('visual_suite.py')]
        for path in paths:
            with self.subTest(path=str(path)):
                source = path.read_text(encoding='utf-8')
                self.assertTrue("'--args', '" + SAFE_ARG + "'" in source,
                                'CLI scripts must pass the safe flag even outside the project directory')


if __name__ == '__main__':
    unittest.main()
