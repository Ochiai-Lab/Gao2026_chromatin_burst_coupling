"""State-boundary tests: labels change, measurements and algorithm inputs do not."""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gao2026_state_labels import export_transcription_states, read_legacy_state_csv


class StateLabelTests(unittest.TestCase):
    def setUp(self):
        self.original = pd.DataFrame({
            'state': ['ON', 'OFF', 'Random', None],
            'raw_state': ['OFF', 'ON', 'Unknown', None],
            'kind': ['ON', 'OFF', 'not_a_state', None],
            'on_off': [1, 0, 1, 0],
            'measurement': [1.25, -0.03, 2.1e-17, 5.0],
            'cell_uid': ['a', 'b', 'c', 'd'],
        })

    def test_export_is_nonmutating_and_preserves_other_columns(self):
        before = self.original.copy(deep=True)
        exported = export_transcription_states(self.original)
        pd.testing.assert_frame_equal(self.original, before)
        self.assertEqual(exported.state.iloc[:2].tolist(), ['Active', 'Inactive'])
        pd.testing.assert_frame_equal(exported[['on_off', 'measurement', 'cell_uid']],
                                      before[['on_off', 'measurement', 'cell_uid']])
        self.assertEqual(exported.state.iloc[2], 'Random')

    def test_old_and_new_csv_produce_identical_algorithm_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'old.csv'
            new = Path(directory) / 'new.csv'
            self.original.to_csv(old, index=False)
            export_transcription_states(self.original).to_csv(new, index=False)
            pd.testing.assert_frame_equal(read_legacy_state_csv(old), read_legacy_state_csv(new))

    def test_numeric_states_are_not_relabelled(self):
        frame = pd.DataFrame({'state': [0, 1], 'kind': [1, 0]})
        pd.testing.assert_frame_equal(export_transcription_states(frame), frame)


if __name__ == '__main__':
    unittest.main()
