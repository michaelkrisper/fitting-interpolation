import unittest

from interpolation import (
    linear_regression,
    logistic_regression,
    orthogonal_regression,
    polynomial_regression,
    quadratic_regression,
)


class TestInterpolationMethods(unittest.TestCase):

    def test_linear_regression_basic(self):
        """Test linear regression with perfect line"""
        # y = 2x + 3
        points = [(0, 3), (1, 5), (2, 7), (3, 9)]
        slope, intercept, r_squared = linear_regression(points)

        self.assertAlmostEqual(slope, 2.0, places=6)
        self.assertAlmostEqual(intercept, 3.0, places=6)
        self.assertAlmostEqual(r_squared, 1.0, places=6)

    def test_linear_regression_insufficient_points(self):
        """Test linear regression error handling"""
        with self.assertRaises(ValueError):
            linear_regression([(1, 1)])

    def test_linear_regression_with_noise(self):
        """Test linear regression with noisy data"""
        # Approximately y = x + 1 with some noise
        points = [(0, 1.1), (1, 1.9), (2, 3.1), (3, 3.9), (4, 5.1)]
        slope, intercept, r_squared = linear_regression(points)

        self.assertAlmostEqual(slope, 1.0, places=1)
        self.assertAlmostEqual(intercept, 1.0, places=1)
        self.assertGreater(r_squared, 0.9)

    def test_quadratic_regression_basic(self):
        """Test quadratic regression with perfect parabola"""
        # y = x^2 + 2x + 1
        points = [(0, 1), (1, 4), (2, 9), (-1, 0)]
        a, b, c, r_squared = quadratic_regression(points)

        self.assertAlmostEqual(a, 1.0, places=6)
        self.assertAlmostEqual(b, 2.0, places=6)
        self.assertAlmostEqual(c, 1.0, places=6)
        self.assertAlmostEqual(r_squared, 1.0, places=6)

    def test_quadratic_regression_insufficient_points(self):
        """Test quadratic regression error handling"""
        with self.assertRaises(ValueError):
            quadratic_regression([(0, 1), (1, 2)])

    def test_polynomial_regression_basic(self):
        """Test polynomial regression"""
        # y = x^3 + x^2 + x + 1
        points = [(0, 1), (1, 4), (2, 15), (-1, 0), (3, 40)]
        coeffs, r_squared = polynomial_regression(points, 3)

        # Coefficients should be [1, 1, 1, 1] (from highest to lowest degree)
        self.assertAlmostEqual(coeffs[0], 1.0, places=6)  # x^3
        self.assertAlmostEqual(coeffs[1], 1.0, places=6)  # x^2
        self.assertAlmostEqual(coeffs[2], 1.0, places=6)  # x
        self.assertAlmostEqual(coeffs[3], 1.0, places=6)  # constant
        self.assertAlmostEqual(r_squared, 1.0, places=6)

    def test_polynomial_regression_linear(self):
        """Test polynomial regression of degree 1 (should match linear regression)"""
        points = [(0, 3), (1, 5), (2, 7), (3, 9)]
        coeffs, r_squared = polynomial_regression(points, 1)
        slope, intercept, r_squared_linear = linear_regression(points)

        self.assertAlmostEqual(coeffs[0], slope, places=6)
        self.assertAlmostEqual(coeffs[1], intercept, places=6)
        self.assertAlmostEqual(r_squared, r_squared_linear, places=6)

    def test_logistic_regression_basic(self):
        """Test logistic regression with sigmoid-like data"""
        # Create points that roughly follow a sigmoid
        points = [(-2, 0.1), (-1, 0.25), (0, 0.5), (1, 0.75), (2, 0.9)]
        a, b, r_squared = logistic_regression(points)

        # Should have positive slope (a > 0) and reasonable fit
        self.assertGreater(a, 0)
        self.assertGreater(r_squared, 0.5)

    def test_logistic_regression_invalid_y_values(self):
        """Test logistic regression with invalid y values"""
        with self.assertRaises(ValueError):
            logistic_regression([(-1, -0.1), (0, 0.5), (1, 1.1)])

    def test_orthogonal_regression_basic(self):
        """Test orthogonal regression with perfect line"""
        # y = x + 1
        points = [(0, 1), (1, 2), (2, 3), (3, 4)]
        slope, intercept, r_squared = orthogonal_regression(points)

        self.assertAlmostEqual(slope, 1.0, places=6)
        self.assertAlmostEqual(intercept, 1.0, places=6)
        self.assertAlmostEqual(r_squared, 1.0, places=6)

    def test_orthogonal_regression_vertical_line(self):
        """Test orthogonal regression with vertical line"""
        points = [(5, 0), (5, 1), (5, 2), (5, 3)]
        slope, intercept, r_squared = orthogonal_regression(points)

        self.assertEqual(slope, float('inf'))
        self.assertAlmostEqual(intercept, 5.0, places=6)
        self.assertGreater(r_squared, 0.5)

    def test_linear_regression_identical_x_raises(self):
        with self.assertRaises(ValueError):
            linear_regression([(1, 1), (1, 2), (1, 3)])

    def test_linear_regression_constant_y(self):
        slope, intercept, r_squared = linear_regression([(0, 4), (1, 4), (2, 4)])
        self.assertAlmostEqual(slope, 0.0)
        self.assertAlmostEqual(intercept, 4.0)
        self.assertEqual(r_squared, 1.0)

    def test_quadratic_regression_matches_three_point_fit(self):
        # With exactly three points, least squares interpolates them
        points = [(-10, 83), (5, 38), (2, 11)]
        a, b, c, r_squared = quadratic_regression(points)
        self.assertAlmostEqual(a, 1.0, places=6)
        self.assertAlmostEqual(b, 2.0, places=6)
        self.assertAlmostEqual(c, 3.0, places=6)
        self.assertAlmostEqual(r_squared, 1.0, places=6)

    def test_quadratic_regression_identical_x_raises(self):
        with self.assertRaises(ValueError):
            quadratic_regression([(1, 1), (1, 2), (1, 3)])

    def test_quadratic_regression_least_squares(self):
        # Residuals of the best fit are orthogonal to x^2, x and 1
        points = [(-2, 4.3), (-1, 0.8), (0, 0.1), (1, 1.2), (2, 3.9), (3, 9.2)]
        a, b, c, _ = quadratic_regression(points)
        residuals = [(x, y - (a * x**2 + b * x + c)) for x, y in points]
        for power in range(3):
            self.assertAlmostEqual(sum(r * x**power for x, r in residuals), 0.0, places=6)

    def test_polynomial_regression_invalid_degree(self):
        with self.assertRaises(ValueError):
            polynomial_regression([(0, 0), (1, 1)], 0)

    def test_polynomial_regression_insufficient_points(self):
        with self.assertRaises(ValueError):
            polynomial_regression([(0, 0), (1, 1), (2, 4)], 3)

    def test_polynomial_regression_degree_2_matches_quadratic(self):
        points = [(-2, 4.3), (-1, 0.8), (0, 0.1), (1, 1.2), (2, 3.9), (3, 9.2)]
        coeffs, r2_poly = polynomial_regression(points, 2)
        a, b, c, r2_quad = quadratic_regression(points)
        for got, expected in zip(coeffs, (a, b, c)):
            self.assertAlmostEqual(got, expected, places=6)
        self.assertAlmostEqual(r2_poly, r2_quad, places=6)

    def test_logistic_regression_symmetric_data(self):
        # Data symmetric around x = 0 gives b == 0 and an increasing curve
        points = [(-3, 0.05), (-1, 0.27), (0, 0.5), (1, 0.73), (3, 0.95)]
        a, b, r_squared = logistic_regression(points)
        self.assertGreater(a, 0.5)
        self.assertAlmostEqual(b, 0.0, places=6)
        self.assertGreater(r_squared, 0.99)

    def test_logistic_regression_decreasing(self):
        points = [(-2, 0.9), (-1, 0.75), (0, 0.5), (1, 0.25), (2, 0.1)]
        a, _, _ = logistic_regression(points)
        self.assertLess(a, 0)

    def test_logistic_regression_accepts_binary_labels(self):
        a, _, _ = logistic_regression([(-2, 0), (-1, 0), (1, 1), (2, 1)])
        self.assertGreater(a, 0)

    def test_logistic_regression_insufficient_points(self):
        with self.assertRaises(ValueError):
            logistic_regression([(0, 0.5)])

    def test_orthogonal_regression_horizontal_line(self):
        slope, intercept, r_squared = orthogonal_regression([(0, 2), (1, 2), (2, 2), (3, 2)])
        self.assertAlmostEqual(slope, 0.0, places=6)
        self.assertAlmostEqual(intercept, 2.0, places=6)
        self.assertAlmostEqual(r_squared, 1.0, places=6)

    def test_orthogonal_regression_is_symmetric_in_x_and_y(self):
        # Swapping x and y must give the inverse slope (unlike ordinary least squares)
        points = [(0, 0.3), (1, 1.6), (2, 4.1), (3, 5.8), (4, 8.4)]
        slope, _, _ = orthogonal_regression(points)
        swapped_slope, _, _ = orthogonal_regression([(y, x) for x, y in points])
        self.assertAlmostEqual(slope * swapped_slope, 1.0, places=6)

    def test_orthogonal_regression_insufficient_points(self):
        with self.assertRaises(ValueError):
            orthogonal_regression([(1, 1)])
