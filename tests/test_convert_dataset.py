import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_dataset.py"
SPEC = importlib.util.spec_from_file_location("convert_dataset", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.source = Path(self.folder.name) / "jets.h5"
        self.output = Path(self.folder.name) / "jets.mat"
        self.values = np.arange(6 * 800, dtype=np.float32).reshape(6, 800)
        self.frame = pd.DataFrame(self.values, columns=converter.PARTICLE_COLUMNS)
        self.frame["is_signal_new"] = [0, 1, 0, 1, 0, 1]

    def write_frame(self):
        self.frame.to_hdf(self.source, key="table", mode="w", format="table")

    def test_column_order_limit_shape_and_provenance(self):
        self.frame = self.frame[self.frame.columns[::-1]]
        self.write_frame()
        converter.convert_dataset(self.source, self.output, max_jets=4)
        data = loadmat(self.output)
        np.testing.assert_array_equal(data["particleData"], self.values[:4])
        np.testing.assert_array_equal(data["labels"], [[0], [1], [0], [1]])
        self.assertEqual(data["particleData"].dtype, np.float32)
        info = json.loads(data["provenance_json"][0])
        self.assertEqual(info["converted_jets"], 4)
        self.assertEqual(len(info["source_sha256"]), 64)

    def test_rejects_missing_column(self):
        self.frame = self.frame.drop(columns="PX_2")
        self.write_frame()
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            converter.convert_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_nonfinite_particles(self):
        self.frame.loc[0, "PX_0"] = np.nan
        self.write_frame()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            converter.convert_dataset(self.source, self.output)

    def test_rejects_invalid_or_single_class_labels(self):
        for labels in ([0, 1, 2, 0, 1, 0], [1] * 6):
            self.frame["is_signal_new"] = labels
            self.write_frame()
            with self.assertRaisesRegex(ValueError, "both classes"):
                converter.convert_dataset(self.source, self.output)

    def test_existing_output_requires_force(self):
        self.write_frame()
        self.output.write_bytes(b"existing data")
        with self.assertRaises(FileExistsError):
            converter.convert_dataset(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing data")


if __name__ == "__main__":
    unittest.main()
