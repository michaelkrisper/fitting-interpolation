#!/usr/bin/env python3
"""
Examples demonstrating all interpolation methods in the fitting-interpolation package.
"""

from quadratic import fit
from interpolation import (
    linear_regression,
    quadratic_regression, 
    polynomial_regression,
    logistic_regression,
    orthogonal_regression
)


def example_original_quadratic():
    """Demonstrate the original quadratic fitting for 3 points."""
    print("=== Original Quadratic Fitting (3 points) ===")
    
    # Define three points that lie on y = 2x² - 3x + 1
    p1 = (0, 1)
    p2 = (1, 0)  # 2(1)² - 3(1) + 1 = 0
    p3 = (2, 3)  # 2(2)² - 3(2) + 1 = 3
    
    a, b, c, xopt, yzero = fit(p1, p2, p3)
    
    print(f"Points: {p1}, {p2}, {p3}")
    print(f"Quadratic equation: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    print(f"Vertex x-coordinate: {xopt:.3f}")
    if yzero:
        print(f"Roots: x = {yzero[0]:.3f}, x = {yzero[1]:.3f}")
    else:
        print("No real roots")
    print()


def example_linear_regression():
    """Demonstrate linear regression."""
    print("=== Linear Regression ===")
    
    # Data with some noise around y = 2x + 1
    points = [(0, 1.1), (1, 2.9), (2, 5.1), (3, 6.9), (4, 9.1)]
    
    slope, intercept, r_squared = linear_regression(points)
    
    print(f"Data points: {points}")
    print(f"Linear equation: y = {slope:.3f}x + {intercept:.3f}")
    print(f"R-squared: {r_squared:.4f}")
    print()


def example_quadratic_regression():
    """Demonstrate quadratic regression with multiple points."""
    print("=== Quadratic Regression (Multiple Points) ===")
    
    # Data around y = x² - 2x + 3 with noise
    points = [(0, 3.1), (1, 2.0), (2, 2.9), (3, 6.1), (4, 11.0), (-1, 5.9)]
    
    a, b, c, r_squared = quadratic_regression(points)
    
    print(f"Data points: {points}")
    print(f"Quadratic equation: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    print(f"R-squared: {r_squared:.4f}")
    print()


def example_polynomial_regression():
    """Demonstrate polynomial regression."""
    print("=== Polynomial Regression (Degree 3) ===")
    
    # Data around y = x³ - x² + 2x + 1
    points = [(0, 1), (1, 3), (2, 9), (3, 25), (-1, -3), (-2, -15)]
    
    coeffs, r_squared = polynomial_regression(points, degree=3)
    
    print(f"Data points: {points}")
    print(f"Cubic equation: y = {coeffs[0]:.3f}x³ + {coeffs[1]:.3f}x² + {coeffs[2]:.3f}x + {coeffs[3]:.3f}")
    print(f"R-squared: {r_squared:.4f}")
    print()


def example_logistic_regression():
    """Demonstrate logistic regression."""
    print("=== Logistic Regression ===")
    
    # Sigmoid-like data
    points = [(-3, 0.05), (-2, 0.12), (-1, 0.27), (0, 0.5), (1, 0.73), (2, 0.88), (3, 0.95)]
    
    a, b, r_squared = logistic_regression(points)
    
    print(f"Data points: {points}")
    print(f"Sigmoid equation: y = 1 / (1 + exp(-({a:.3f}x + {b:.3f})))")
    print(f"R-squared: {r_squared:.4f}")
    print()


def example_orthogonal_regression():
    """Demonstrate orthogonal regression."""
    print("=== Orthogonal Regression (Total Least Squares) ===")
    
    # Data with errors in both x and y directions
    points = [(1.1, 2.9), (1.9, 4.1), (3.1, 5.9), (3.9, 8.1), (5.1, 9.9)]
    
    slope, intercept, r_squared = orthogonal_regression(points)
    
    print(f"Data points: {points}")
    if slope == float('inf'):
        print(f"Vertical line: x = {intercept:.3f}")
    else:
        print(f"Line equation: y = {slope:.3f}x + {intercept:.3f}")
    print(f"R-squared: {r_squared:.4f}")
    print()


def compare_methods():
    """Compare different regression methods on the same dataset."""
    print("=== Method Comparison ===")
    
    # Data that's roughly linear but can be fit with different methods
    points = [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9), (5, 11)]
    
    print(f"Data points: {points}")
    print()
    
    # Linear regression
    slope, intercept, r2_linear = linear_regression(points)
    print(f"Linear:     y = {slope:.3f}x + {intercept:.3f}, R² = {r2_linear:.4f}")
    
    # Quadratic regression
    a, b, c, r2_quad = quadratic_regression(points)
    print(f"Quadratic:  y = {a:.3f}x² + {b:.3f}x + {c:.3f}, R² = {r2_quad:.4f}")
    
    # Polynomial degree 3
    coeffs, r2_poly = polynomial_regression(points, degree=3)
    print(f"Cubic:      y = {coeffs[0]:.3f}x³ + {coeffs[1]:.3f}x² + {coeffs[2]:.3f}x + {coeffs[3]:.3f}, R² = {r2_poly:.4f}")
    
    # Orthogonal regression
    slope_orth, intercept_orth, r2_orth = orthogonal_regression(points)
    print(f"Orthogonal: y = {slope_orth:.3f}x + {intercept_orth:.3f}, R² = {r2_orth:.4f}")
    print()


if __name__ == "__main__":
    print("Fitting and Interpolation Examples\n")
    
    example_original_quadratic()
    example_linear_regression()
    example_quadratic_regression()
    example_polynomial_regression()
    example_logistic_regression()
    example_orthogonal_regression()
    compare_methods()
    
    print("All examples completed successfully!")