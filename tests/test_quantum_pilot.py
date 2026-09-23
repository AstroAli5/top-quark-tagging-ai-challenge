import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from quantum_pilot import balanced_rows, fidelity_kernel, independent_auc, jet_features


class QuantumPilotTests(unittest.TestCase):
    def test_collinear_massless_jet_and_scale_invariance(self):
        particles = np.zeros((1, 200, 4))
        particles[0, :3, :2] = 1.
        expected = [[0., 0., 1/np.sqrt(3), np.log(4)]]
        np.testing.assert_allclose(jet_features(particles), expected, atol=1e-12)
        np.testing.assert_allclose(jet_features(10*particles), expected, atol=1e-12)
        with self.assertRaises(ValueError): jet_features(np.zeros((1, 800)))

    def test_sampling_preserves_class_balance_and_row_identity(self):
        labels = np.r_[np.zeros(80), np.ones(20)]
        selected = balanced_rows(labels, 20, 17)
        self.assertEqual(len(set(selected)), 20)
        self.assertEqual(labels[selected].sum(), 10)
        np.testing.assert_array_equal(selected, balanced_rows(labels, 20, 17))

    def test_kernel_is_phase_invariant_and_auc_handles_ties(self):
        states = np.array([[1, 0], [0, 1], [1, 1]], complex)
        states /= np.linalg.norm(states, axis=1)[:, None]
        k = fidelity_kernel(states, states*np.exp(.7j))
        np.testing.assert_allclose(k, [[1, 0, .5], [0, 1, .5], [.5, .5, 1]], atol=1e-12)
        self.assertAlmostEqual(independent_auc([0, 1, 0, 1], [.2, .2, .8, .9]), .625)

    @unittest.skipUnless(importlib.util.find_spec('qiskit'), 'Optional Qiskit environment')
    def test_qiskit_encoding_has_normalized_states_and_unit_diagonal(self):
        from quantum_pilot import encode
        states = encode(np.array([[0., .1, .2, .3], [.4, .5, .6, .7]]))
        self.assertEqual(states.shape, (2, 16))
        np.testing.assert_allclose(np.diag(fidelity_kernel(states, states)), 1., atol=1e-12)


if __name__ == '__main__': unittest.main()
