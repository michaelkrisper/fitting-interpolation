import unittest

from quadratic import fit


class TestFitFunction(unittest.TestCase):

    def test_simple_parabola(self):
        # Test with a standard upward-opening parabola: y = x^2 + 2x + 3
        points = [(-10, 83), (5, 38), (2, 11)]
        a, b, c, xopt, yzero = fit(points[0], points[1], points[2])
        self.assertAlmostEqual(a, 1.0)
        self.assertAlmostEqual(b, 2.0)
        self.assertAlmostEqual(c, 3.0)
        self.assertAlmostEqual(xopt, -1.0)
        self.assertIsNone(yzero)

    def test_downward_parabola_with_roots(self):
        # Test with a downward-opening parabola with integer roots: y = -x^2 + 4
        points = [(-2, 0), (0, 4), (2, 0)]
        a, b, c, xopt, yzero = fit(points[0], points[1], points[2])
        self.assertAlmostEqual(a, -1.0)
        self.assertAlmostEqual(b, 0.0)
        self.assertAlmostEqual(c, 4.0)
        self.assertAlmostEqual(xopt, 0.0)
        self.assertIsNotNone(yzero)
        # Sort roots for consistent comparison
        roots = sorted(yzero)
        self.assertAlmostEqual(roots[0], -2.0)
        self.assertAlmostEqual(roots[1], 2.0)

    def test_linear_case(self):
        # Test with a set of collinear points (a line): y = 3x + 5
        points = [(-1, 2), (0, 5), (2, 11)]
        a, b, c, xopt, yzero = fit(points[0], points[1], points[2])
        self.assertAlmostEqual(a, 0.0)
        self.assertAlmostEqual(b, 3.0)
        self.assertAlmostEqual(c, 5.0)
        self.assertEqual(xopt, float('inf'))
        self.assertIsNotNone(yzero)
        self.assertAlmostEqual(yzero[0], -5.0/3.0)

    def test_collinear_horizontal(self):
        # Test with a horizontal line: y = 5
        points = [(-1, 5), (0, 5), (2, 5)]
        a, b, c, xopt, yzero = fit(points[0], points[1], points[2])
        self.assertAlmostEqual(a, 0.0)
        self.assertAlmostEqual(b, 0.0)
        self.assertAlmostEqual(c, 5.0)
        self.assertIsNone(yzero) # No roots if c!=0

    def test_duplicate_x_error(self):
        # Test that a ValueError is raised for non-unique x-coordinates
        points = [(-1, 2), (-1, 5), (2, 11)]
        with self.assertRaisesRegex(ValueError, "unique x-coordinates"):
            fit(points[0], points[1], points[2])

    def test_exactly_collinear_points_do_not_divide_by_zero(self):
        # y = x: the fitted a is exactly 0.0
        a, b, c, xopt, yzero = fit((0, 0), (1, 1), (2, 2))
        self.assertEqual(a, 0.0)
        self.assertEqual(xopt, float('inf'))
        self.assertEqual(len(yzero), 1)
        self.assertAlmostEqual(yzero[0], 0.0)

    def test_points_in_any_order(self):
        # y = 2x^2 - 3x + 1, points given unsorted
        a, b, c, xopt, yzero = fit((2, 3), (0, 1), (1, 0))
        self.assertAlmostEqual(a, 2.0)
        self.assertAlmostEqual(b, -3.0)
        self.assertAlmostEqual(c, 1.0)
        self.assertAlmostEqual(xopt, 0.75)
        self.assertEqual(sorted(round(r, 9) for r in yzero), [0.5, 1.0])

    def test_single_touching_root(self):
        # y = (x - 1)^2 touches zero at x = 1
        a, b, c, xopt, yzero = fit((0, 1), (1, 0), (3, 4))
        self.assertAlmostEqual(xopt, 1.0)
        self.assertAlmostEqual(yzero[0], 1.0)
        self.assertAlmostEqual(yzero[1], 1.0)
