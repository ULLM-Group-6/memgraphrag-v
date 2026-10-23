import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('single', ROOT / 'scripts/webqa_test_single.py')
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)


class SingleWorker(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(config='base.yaml', run_id='tuned', output_run_id='tuned-test-single')
        self.env = {'MGRV_TEXT_PYTHON': 'text', 'MGRV_VISION_PYTHON': 'vision',
                    'MGRV_READER_PYTHON': 'reader', 'MGRV_READER_MODEL': 'reader-model'}
        self.commands = single.commands(self.args, Path('selected.yaml'), .75, Path('/art'), Path('/scripts'), self.env)

    def test_single_full_shard_for_every_phase(self):
        self.assertEqual([name for name, _ in self.commands], ['queries', 'retrieve', 'read'])
        for _, command in self.commands:
            for key, value in (('--shard-count', '1'), ('--shard-index', '0'),
                               ('--split', 'test'), ('--beta', '0.75'),
                               ('--config', 'selected.yaml'), ('--run-id', 'tuned-test-single')):
                self.assertEqual(command[command.index(key) + 1], value)
            self.assertNotIn('--limit', command)

    def test_environments_and_cached_retrieval_entry(self):
        self.assertEqual([cmd[0] for _, cmd in self.commands], ['vision', 'text', 'reader'])
        retrieval = self.commands[1][1]
        self.assertTrue(retrieval[1].endswith('webqa_test_single.py'))
        self.assertEqual(retrieval[retrieval.index('--base-run-id') + 1], 'tuned')
        self.assertEqual(retrieval[retrieval.index('--base-config') + 1], 'base.yaml')
        reader = self.commands[2][1]
        self.assertEqual(reader[reader.index('--reader-model') + 1], 'reader-model')


if __name__ == '__main__':
    unittest.main()
