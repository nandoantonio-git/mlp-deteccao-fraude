"""Regression tests for duplicate leakage and conflicting target labels."""
import unittest

import pandas as pd

from src.preprocessing import FEATURE_COLUMNS, deduplicate_dataset
from src.train import split_data


class DatasetSplitTests(unittest.TestCase):
    def data(self):
        frame = pd.DataFrame([{col: float(i) for col in FEATURE_COLUMNS} for i in range(100)])
        frame["Class"] = [i % 2 for i in range(100)]
        return frame

    def test_duplicates_removed_before_split_and_no_cross_overlap(self):
        unique = self.data()
        raw = pd.concat([unique, unique.iloc[:30]], ignore_index=True)
        train, val, test, yt, yv, ys = split_data(raw)
        self.assertEqual(len(train) + len(val) + len(test), 100)
        hashes = [set(pd.util.hash_pandas_object(x, index=False)) for x in (train, val, test)]
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
        self.assertEqual(int(yt.sum() + yv.sum() + ys.sum()), 50)

    def test_conflicting_labels_rejected(self):
        frame = self.data()
        conflict = frame.iloc[[0]].copy()
        conflict["Class"] = 1
        with self.assertRaisesRegex(ValueError, "conflicting"):
            deduplicate_dataset(pd.concat([frame, conflict], ignore_index=True))

    def test_cleaning_is_idempotent_and_does_not_mutate_input(self):
        raw = pd.concat([self.data(), self.data()], ignore_index=True)
        cleaned = deduplicate_dataset(raw)
        self.assertEqual(len(raw), 200)
        pd.testing.assert_frame_equal(cleaned, deduplicate_dataset(cleaned))
