import unittest

from spvtp.vision_traj.analysis.pick_place_analysis import _assignments


class PickPlaceAssignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.boundary = {
            "pick_raw_frame": 80,
            "split_raw_frame": 130,
            "place_raw_frame": 180,
        }

    def test_task_half_assignment(self) -> None:
        self.assertIn(("task_half", "pick"), _assignments(100, self.boundary, 15))
        self.assertIn(("task_half", "place"), _assignments(150, self.boundary, 15))

    def test_contact_assignment(self) -> None:
        self.assertIn(("contact_window", "pick"), _assignments(85, self.boundary, 15))
        self.assertIn(("contact_window", "place"), _assignments(170, self.boundary, 15))
        self.assertEqual(len(_assignments(130, self.boundary, 15)), 1)


if __name__ == "__main__":
    unittest.main()
