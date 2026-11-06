import unittest
import numpy as np
from lane_center_estimation import SpatioTemporalLaneEstimator


class LaneCenterEstimatorTests(unittest.TestCase):
    def test_two_visible_markings_and_memory(self):
        est = SpatioTemporalLaneEstimator(history_size=8, min_pixels=5,
                                          initial_lane_width_px=40, lane_width_cm=20)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[72:93, 25:28] = 3
        mask[72:93, 72:75] = 2
        first = est.estimate(mask)
        self.assertEqual(first.source, "near")
        self.assertAlmostEqual(first.center_px, 49.5, delta=2)
        missing = est.estimate(np.zeros_like(mask))
        self.assertEqual(missing.source, "memory")
        self.assertAlmostEqual(first.center_px, missing.center_px)

    def test_far_field_when_near_empty(self):
        est = SpatioTemporalLaneEstimator(min_pixels=5, initial_lane_width_px=40)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[48:55, 25:28] = 3
        mask[48:55, 72:75] = 2
        self.assertEqual(est.estimate(mask).source, "far")

    def test_no_lane_and_no_history_returns_unavailable(self):
        est = SpatioTemporalLaneEstimator()
        self.assertEqual(est.estimate(np.zeros((80, 120), dtype=np.uint8)).source, "unavailable")

    def test_memory_is_bounded_to_eight_observations(self):
        est = SpatioTemporalLaneEstimator(history_size=8, min_pixels=5)
        for shift in range(12):
            mask = np.zeros((100, 140), dtype=np.uint8)
            mask[72:93, 12+shift:16+shift] = 3
            mask[72:93, 110+shift:114+shift] = 2
            est.estimate(mask)
        self.assertEqual(len(est.history), 8)


if __name__ == "__main__":
    unittest.main()
