import cv2
import os
import sys
import unittest

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.foundation


class FoundationTest(unittest.TestCase):
    def test_non_max_suppression(self):
        self.assertEqual(
            leapvision.foundation.non_max_suppression([
                {'score': 0.5, 'box': (0, 0, 320, 480)}
            ]),
            [
                {'score': 0.5, 'box': (0, 0, 320, 480)}
            ]
        )
        self.assertEqual(
            leapvision.foundation.non_max_suppression([
                {'score': 0.5, 'box': (0, 0, 320, 480)},
                {'score': 0.6, 'box': (10, 0, 320, 480)},
                {'score': 0.7, 'box': (10, 20, 320, 480)}
            ]),
            [
                {'score': 0.7, 'box': (10, 20, 320, 480)}
            ]
        )

    def test_box_opposite_corners(self):
        self.assertEqual(
            leapvision.foundation.box_opposite_corners((0, 0, 320, 480)),
            (0, 0, 320, 480)
        )
        self.assertEqual(
            leapvision.foundation.box_opposite_corners((10, 20, 320, 480)),
            (10, 20, 330, 500)
        )
        self.assertEqual(
            leapvision.foundation.box_opposite_corners((-10, -20, 320, 480)),
            (-10, -20, 310, 460)
        )

    def test_box_intersection_area(self):
        # a contains b
        self.assertEqual(
            leapvision.foundation.box_intersection_area(
                (0, 0, 320, 480),
                (50, 50, 70, 70)
            ),
            400
        )
        # no intersection
        self.assertEqual(
            leapvision.foundation.box_intersection_area(
                (0, 0, 320, 480),
                (321, 481, 10, 20)
            ),
            0
        )

    def test_box_union_area(self):
        # a contains b
        self.assertEqual(
            leapvision.foundation.box_union_area(
                (0, 0, 320, 480),
                (50, 50, 70, 70)
            ),
            (320*480)
        )
        # a overlaps with b
        self.assertEqual(
            leapvision.foundation.box_union_area(
                (0, 0, 320, 480),
                (300, 400, 1024, 1024)
            ),
            1048576.0
        )
        # no intersection in x
        self.assertEqual(
            leapvision.foundation.box_union_area(
                (0, 0, 320, 480),
                (321, 0, 10, 20)
            ),
            0
        )
        # no intersection in y
        self.assertEqual(
            leapvision.foundation.box_union_area(
                (0, 0, 320, 480),
                (0, 481, 10, 20)
            ),
            0
        )

    def test_box_intersection_over_union(self):
        # a contains b
        self.assertAlmostEqual(
            leapvision.foundation.box_intersection_over_union(
                (0, 0, 320, 480),
                (50, 50, 70, 70)
            ),
            0.0026,
            delta=0.001
        )
        # a overlaps with b
        self.assertAlmostEqual(
            leapvision.foundation.box_intersection_over_union(
                (0, 0, 320, 480),
                (300, 400, 1024, 1024)
            ),
            0.0015,
            delta=0.001
        )
        # no intersection in x
        self.assertEqual(
            leapvision.foundation.box_intersection_over_union(
                (0, 0, 320, 480),
                (321, 0, 10, 20)
            ),
            0
        )
        # no intersection in y
        self.assertEqual(
            leapvision.foundation.box_intersection_over_union(
                (0, 0, 320, 480),
                (0, 481, 10, 20)
            ),
            0
        )


if __name__ == '__main__':
    unittest.main()
