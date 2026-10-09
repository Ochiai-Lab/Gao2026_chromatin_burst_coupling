"""Publication labels must not change sample membership or random seeds."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import recalculate_source_data_ci as ci


class PublicReplicateTests(unittest.TestCase):
    def test_sora_manifest_is_one_public_experiment(self):
        path = ROOT / 'analyses/08_SoRa/scripts/sora_fixed_publication_manifest.json'
        manifest = json.loads(path.read_text())
        self.assertEqual(len(manifest['records']), 8)
        self.assertEqual({r['biological_replicate'] for r in manifest['records']}, {'rep1'})
        self.assertEqual(manifest['biological_replicate_group_count'], 4)
        self.assertEqual(len({r['analysis_id'] for r in manifest['records']}), 8)
        self.assertTrue(all(isinstance(r['random_seed'], int) for r in manifest['records']))

    def test_sora_ci_seeds_are_explicit_and_complete(self):
        self.assertEqual(len(ci.SORA_BOOTSTRAP_SEEDS), 12)
        self.assertTrue(all('_rep1/' in key for key in ci.SORA_BOOTSTRAP_SEEDS))
        self.assertTrue(all(isinstance(value, int) for value in ci.SORA_BOOTSTRAP_SEEDS.values()))


if __name__ == '__main__':
    unittest.main()
