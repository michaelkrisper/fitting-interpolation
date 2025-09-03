from math import sqrt, isclose
import unittest

def fit(p1, p2, p3):
    """
    Calculates the coefficients of a quadratic equation y = ax^2 + bx + c
    that passes through three given points. It is optimized for performance
    and readability.
    """
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]

    if x1 == x2 or x1 == x3 or x2 == x3:
        raise ValueError("Input points must have unique x-coordinates.")

    a = (y1/((x1-x2)*(x1-x3)) + y2/((x2-x1)*(x2-x3)) + y3/((x3-x1)*(x3-x2)))
    b = (y1 - y2 - a * (x1**2 - x2**2)) / (x1 - x2)
    c = y1 - a * x1**2 - b * x1
    xopt = -b / (2 * a) if not isclose(a, 0, abs_tol=1e-9) else float('inf')
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        yzero = None
    else:
        sqrt_d = sqrt(discriminant)
        if isclose(a, 0, abs_tol=1e-9): # Check if the function is essentially linear
            yzero = (-c / b, -c / b) if not isclose(b, 0, abs_tol=1e-9) else None
        else:
            yzero = ((-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a))

    return a, b, c, xopt, yzero

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

if __name__ == '__main__':
    unittest.main(verbosity=2)
    print(fit(*[(x, 1*x**2 + 2*x + 3) for x in [-10, 5, 2]]))
