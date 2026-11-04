import unittest
from types import SimpleNamespace

from spvtp.vision_traj.data.dataset import resolve_split_seed


class ResolveSplitSeedTest(unittest.TestCase):
    def test_split_seed_is_independent_of_training_seed(self):
        cfg = SimpleNamespace(
            data={"split_seed": 42},
            train=SimpleNamespace(seed=7),
        )
        self.assertEqual(resolve_split_seed(cfg), 42)

    def test_legacy_config_falls_back_to_training_seed(self):
        cfg = SimpleNamespace(data={}, train=SimpleNamespace(seed=7))
        self.assertEqual(resolve_split_seed(cfg), 7)


if __name__ == "__main__":
    unittest.main()
