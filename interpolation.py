from math import exp, log, sqrt
import unittest
from typing import List, Tuple, Optional, Union


def linear_regression(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """
    Performs linear regression (least squares fitting) on a set of points.
    
    Args:
        points: List of (x, y) coordinate tuples
        
    Returns:
        Tuple containing (slope, intercept, r_squared)
        
    Raises:
        ValueError: If fewer than 2 points are provided
    """
    if len(points) < 2:
        raise ValueError("Linear regression requires at least 2 points")
    
    # Extract x and y values
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    n = len(points)
    
    # Calculate means
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    
    # Calculate slope and intercept using least squares formula
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)
    
    if abs(denominator) < 1e-10:
        raise ValueError("Cannot perform linear regression on points with identical x-coordinates")
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calculate R-squared
    y_pred = [slope * x + intercept for x in x_vals]
    ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_vals, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return slope, intercept, r_squared


def _solve_linear_system_3x3(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve a 3x3 linear system Ax = b using Gaussian elimination.
    """
    # Create augmented matrix
    n = len(A)
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        # Check for singular matrix
        if abs(aug[i][i]) < 1e-10:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            factor = aug[k][i] / aug[i][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]
    
    return x


def quadratic_regression(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    Performs quadratic regression (least squares fitting) on a set of points.
    
    Args:
        points: List of (x, y) coordinate tuples
        
    Returns:
        Tuple containing (a, b, c, r_squared) for equation y = ax^2 + bx + c
        
    Raises:
        ValueError: If fewer than 3 points are provided
    """
    if len(points) < 3:
        raise ValueError("Quadratic regression requires at least 3 points")
    
    # Extract x and y values
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    n = len(points)
    
    # Set up normal equations: A^T A * coeffs = A^T * y
    # Where A is the design matrix [x^2, x, 1]
    
    # Calculate A^T A
    sum_x4 = sum(x**4 for x in x_vals)
    sum_x3 = sum(x**3 for x in x_vals)
    sum_x2 = sum(x**2 for x in x_vals)
    sum_x = sum(x_vals)
    
    # Calculate A^T y
    sum_x2y = sum(x**2 * y for x, y in zip(x_vals, y_vals))
    sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
    sum_y = sum(y_vals)
    
    # Normal equations matrix
    A_matrix = [
        [sum_x4, sum_x3, sum_x2],
        [sum_x3, sum_x2, sum_x],
        [sum_x2, sum_x, n]
    ]
    
    b_vector = [sum_x2y, sum_xy, sum_y]
    
    # Solve for coefficients
    try:
        coeffs = _solve_linear_system_3x3(A_matrix, b_vector)
        a, b, c = coeffs
    except:
        raise ValueError("Unable to solve quadratic regression system")
    
    # Calculate R-squared
    y_mean = sum(y_vals) / n
    y_pred = [a * x**2 + b * x + c for x in x_vals]
    ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_vals, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return a, b, c, r_squared


def _gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
    """
    n = len(A)
    # Create augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        
        # Swap rows
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        # Check for singular matrix
        if abs(aug[i][i]) < 1e-10:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Eliminate column
        for k in range(i + 1, n):
            factor = aug[k][i] / aug[i][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]
    
    return x


def polynomial_regression(points: List[Tuple[float, float]], degree: int) -> Tuple[List[float], float]:
    """
    Performs polynomial regression of specified degree on a set of points.
    
    Args:
        points: List of (x, y) coordinate tuples
        degree: Degree of the polynomial (must be >= 1)
        
    Returns:
        Tuple containing (coefficients, r_squared) where coefficients are ordered
        from highest degree to constant term
        
    Raises:
        ValueError: If degree < 1 or insufficient points provided
    """
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1")
    
    if len(points) <= degree:
        raise ValueError(f"Polynomial regression of degree {degree} requires at least {degree + 1} points")
    
    # Extract x and y values
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    n = len(points)
    
    # Build normal equations matrix (A^T A) and vector (A^T y)
    # For polynomial of degree d, we need (d+1) x (d+1) matrix
    matrix_size = degree + 1
    A_matrix = [[0.0] * matrix_size for _ in range(matrix_size)]
    b_vector = [0.0] * matrix_size
    
    # Fill the normal equations
    for i in range(matrix_size):
        for j in range(matrix_size):
            # A_matrix[i][j] = sum(x^(i+j) for x in x_vals)
            power = i + j
            A_matrix[i][j] = sum(x**power for x in x_vals)
        
        # b_vector[i] = sum(x^i * y for x, y in zip(x_vals, y_vals))
        b_vector[i] = sum(x**i * y for x, y in zip(x_vals, y_vals))
    
    # Solve for coefficients (lowest to highest degree)
    try:
        coeffs_low_to_high = _gaussian_elimination(A_matrix, b_vector)
    except ValueError:
        raise ValueError("Unable to solve polynomial regression system")
    
    # Convert to highest to lowest degree (to match numpy.polyfit format)
    coeffs = coeffs_low_to_high[::-1]
    
    # Calculate R-squared
    y_mean = sum(y_vals) / n
    y_pred = []
    for x in x_vals:
        y_p = sum(coeff * (x ** (degree - i)) for i, coeff in enumerate(coeffs))
        y_pred.append(y_p)
    
    ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_vals, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return coeffs, r_squared


def logistic_regression(points: List[Tuple[float, float]], max_iterations: int = 1000, 
                       tolerance: float = 1e-6) -> Tuple[float, float, float]:
    """
    Performs logistic regression fitting for binary classification.
    Fits a sigmoid function: y = 1 / (1 + exp(-(a*x + b)))
    
    Args:
        points: List of (x, y) coordinate tuples where y should be between 0 and 1
        max_iterations: Maximum number of iterations for gradient descent
        tolerance: Convergence tolerance
        
    Returns:
        Tuple containing (a, b, r_squared) for sigmoid parameters
        
    Raises:
        ValueError: If y values are not in [0, 1] range or insufficient points
    """
    if len(points) < 2:
        raise ValueError("Logistic regression requires at least 2 points")
    
    # Extract x and y values
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    # Check that y values are in valid range [0, 1]
    for y in y_vals:
        if y < 0 or y > 1:
            raise ValueError("Logistic regression requires y values to be between 0 and 1")
    
    # Avoid log(0) by clipping extreme values
    y_vals = [max(1e-15, min(1 - 1e-15, y)) for y in y_vals]
    
    # Initialize parameters
    a, b = 0.0, 0.0
    learning_rate = 0.01
    
    n = len(points)
    
    for iteration in range(max_iterations):
        # Forward pass: compute predictions
        predictions = []
        for x in x_vals:
            linear_combination = a * x + b
            # Prevent overflow in exp function
            if linear_combination > 500:
                pred = 1.0
            elif linear_combination < -500:
                pred = 0.0
            else:
                pred = 1 / (1 + exp(-linear_combination))
            predictions.append(pred)
        
        # Compute gradients
        error_sum = sum(pred - y for pred, y in zip(predictions, y_vals))
        da = sum((pred - y) * x for pred, y, x in zip(predictions, y_vals, x_vals)) / n
        db = error_sum / n
        
        # Update parameters
        new_a = a - learning_rate * da
        new_b = b - learning_rate * db
        
        # Check for convergence
        if abs(new_a - a) < tolerance and abs(new_b - b) < tolerance:
            break
            
        a, b = new_a, new_b
    
    # Calculate R-squared (pseudo R-squared for logistic regression)
    y_pred = []
    for x in x_vals:
        linear_combination = a * x + b
        if linear_combination > 500:
            pred = 1.0
        elif linear_combination < -500:
            pred = 0.0
        else:
            pred = 1 / (1 + exp(-linear_combination))
        y_pred.append(pred)
    
    y_mean = sum(y_vals) / len(y_vals)
    ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_vals, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return a, b, r_squared


def orthogonal_regression(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """
    Performs orthogonal regression (total least squares) on a set of points.
    Minimizes the sum of squared perpendicular distances to the line.
    
    Args:
        points: List of (x, y) coordinate tuples
        
    Returns:
        Tuple containing (slope, intercept, r_squared)
        
    Raises:
        ValueError: If fewer than 2 points are provided
    """
    if len(points) < 2:
        raise ValueError("Orthogonal regression requires at least 2 points")
    
    # Extract x and y values
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    n = len(points)
    
    # Center the data
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    x_centered = [x - x_mean for x in x_vals]
    y_centered = [y - y_mean for y in y_vals]
    
    # Calculate covariance matrix elements
    cov_xx = sum(x * x for x in x_centered) / (n - 1)
    cov_yy = sum(y * y for y in y_centered) / (n - 1)
    cov_xy = sum(x * y for x, y in zip(x_centered, y_centered)) / (n - 1)
    
    # For 2x2 matrix [[cov_xx, cov_xy], [cov_xy, cov_yy]], 
    # eigenvalues are roots of: λ² - (cov_xx + cov_yy)λ + (cov_xx*cov_yy - cov_xy²) = 0
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    
    # Calculate eigenvalues using quadratic formula
    discriminant = trace * trace - 4 * det
    if discriminant < 0:
        discriminant = 0  # Handle numerical errors
    
    lambda1 = (trace + sqrt(discriminant)) / 2
    lambda2 = (trace - sqrt(discriminant)) / 2
    
    # The first principal component corresponds to the larger eigenvalue
    larger_eigenvalue = max(lambda1, lambda2)
    
    # Calculate the corresponding eigenvector
    if abs(cov_xy) > 1e-10:
        # Eigenvector is [cov_xy, larger_eigenvalue - cov_xx]
        v1 = cov_xy
        v2 = larger_eigenvalue - cov_xx
    elif abs(cov_xx - larger_eigenvalue) > 1e-10:
        # Use the other formulation
        v1 = larger_eigenvalue - cov_yy
        v2 = cov_xy
    else:
        # Special case: if cov_xy ≈ 0 and cov_xx ≈ larger_eigenvalue
        if cov_xx > cov_yy:
            v1, v2 = 1.0, 0.0  # Horizontal line
        else:
            v1, v2 = 0.0, 1.0  # Vertical line
    
    # Normalize the eigenvector
    norm = sqrt(v1 * v1 + v2 * v2)
    if norm > 1e-10:
        v1 /= norm
        v2 /= norm
    
    # Calculate slope (handle vertical lines)
    if abs(v1) < 1e-10:
        # Nearly vertical line
        slope = float('inf')
        intercept = x_mean
        # For vertical lines, R² is based on x-variance
        total_var = sum((x - x_mean)**2 + (y - y_mean)**2 for x, y in zip(x_vals, y_vals))
        explained_var = sum((y - y_mean)**2 for y in y_vals)
    else:
        slope = v2 / v1
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared as the proportion of variance explained by first principal component
        total_var = sum((x - x_mean)**2 + (y - y_mean)**2 for x, y in zip(x_vals, y_vals))
        explained_var = larger_eigenvalue * (n - 1)
    
    r_squared = explained_var / total_var if total_var > 0 else 1.0
    
    return slope, intercept, r_squared


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


def demo():
    """Demonstrate all interpolation methods with example data"""
    print("=== Interpolation Methods Demo ===\n")
    
    # Linear regression example
    print("1. Linear Regression:")
    linear_points = [(1, 2.1), (2, 4.0), (3, 5.9), (4, 8.1), (5, 9.9)]
    slope, intercept, r2 = linear_regression(linear_points)
    print(f"   Data: {linear_points}")
    print(f"   Equation: y = {slope:.3f}x + {intercept:.3f}")
    print(f"   R-squared: {r2:.4f}\n")
    
    # Quadratic regression example
    print("2. Quadratic Regression:")
    quad_points = [(0, 1), (1, 2.8), (2, 8.9), (3, 19.1), (-1, -2.1)]
    a, b, c, r2 = quadratic_regression(quad_points)
    print(f"   Data: {quad_points}")
    print(f"   Equation: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    print(f"   R-squared: {r2:.4f}\n")
    
    # Polynomial regression example
    print("3. Polynomial Regression (degree 3):")
    poly_points = [(0, 1), (1, 2), (2, 9), (3, 28), (-1, -4)]
    coeffs, r2 = polynomial_regression(poly_points, 3)
    print(f"   Data: {poly_points}")
    print(f"   Equation: y = {coeffs[0]:.3f}x³ + {coeffs[1]:.3f}x² + {coeffs[2]:.3f}x + {coeffs[3]:.3f}")
    print(f"   R-squared: {r2:.4f}\n")
    
    # Logistic regression example
    print("4. Logistic Regression:")
    logistic_points = [(-3, 0.05), (-1, 0.27), (0, 0.5), (1, 0.73), (3, 0.95)]
    a, b, r2 = logistic_regression(logistic_points)
    print(f"   Data: {logistic_points}")
    print(f"   Sigmoid: y = 1/(1 + exp(-({a:.3f}x + {b:.3f})))")
    print(f"   R-squared: {r2:.4f}\n")
    
    # Orthogonal regression example
    print("5. Orthogonal Regression:")
    ortho_points = [(1, 1.9), (2, 3.1), (3, 4.9), (4, 6.1)]
    slope, intercept, r2 = orthogonal_regression(ortho_points)
    print(f"   Data: {ortho_points}")
    if slope == float('inf'):
        print(f"   Vertical line: x = {intercept:.3f}")
    else:
        print(f"   Equation: y = {slope:.3f}x + {intercept:.3f}")
    print(f"   R-squared: {r2:.4f}\n")


if __name__ == '__main__':
    # Run demo first, then tests
    demo()
    print("Running unit tests...\n")
    unittest.main(verbosity=2)