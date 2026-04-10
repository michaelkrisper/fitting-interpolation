# Fitting and Interpolation

This project provides Python functions for various fitting and interpolation methods. It includes the original quadratic fitting for three points, plus multiple regression methods for datasets of any size.

## Features

### Original Quadratic Fitting (`quadratic.py`)
The `fit(p1, p2, p3)` function calculates the coefficients of a quadratic equation that passes through three given points:
- `a`, `b`, `c`: The coefficients of the quadratic equation.
- `xopt`: The x-coordinate of the vertex of the parabola.
- `yzero`: A tuple containing the two roots (x-intercepts) of the equation, or `None` if there are no real roots.

### New Interpolation Methods (`interpolation.py`)
The interpolation module provides five additional regression methods:

1. **Linear Regression** - Least squares fitting of a straight line
2. **Quadratic Regression** - Least squares fitting of a parabola (for multiple points)
3. **Polynomial Regression** - Least squares fitting of polynomials of any degree
4. **Logistic Regression** - Sigmoid curve fitting for binary classification
5. **Orthogonal Regression** - Total least squares (minimizes perpendicular distances)

All methods return R-squared values to measure goodness of fit.

## Usage

### Original Quadratic Fitting

To use the `fit` function, import it from the `quadratic.py` file and pass three points as arguments. Each point should be a tuple of `(x, y)` coordinates.

```python
from quadratic import fit

# Define three points
p1 = (-10, 83)
p2 = (5, 38)
p3 = (2, 11)

# Calculate the quadratic fit
a, b, c, xopt, yzero = fit(p1, p2, p3)

print(f"The equation is y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}")
print(f"The vertex is at x = {xopt:.2f}")
if yzero:
    print(f"The roots are at x = {yzero[0]:.2f} and x = {yzero[1]:.2f}")
else:
    print("There are no real roots.")
```

### New Interpolation Methods

Import the desired functions from `interpolation.py`:

```python
from interpolation import (
    linear_regression, 
    quadratic_regression, 
    polynomial_regression,
    logistic_regression,
    orthogonal_regression
)

# Example data points
points = [(1, 2.1), (2, 4.0), (3, 5.9), (4, 8.1), (5, 9.9)]

# Linear regression
slope, intercept, r_squared = linear_regression(points)
print(f"Linear: y = {slope:.3f}x + {intercept:.3f}, R² = {r_squared:.4f}")

# Quadratic regression  
a, b, c, r_squared = quadratic_regression(points)
print(f"Quadratic: y = {a:.3f}x² + {b:.3f}x + {c:.3f}, R² = {r_squared:.4f}")

# Polynomial regression (degree 3)
coeffs, r_squared = polynomial_regression(points, degree=3)
print(f"Polynomial: coefficients = {coeffs}, R² = {r_squared:.4f}")

# Logistic regression (for y values between 0 and 1)
sigmoid_points = [(-2, 0.1), (-1, 0.3), (0, 0.5), (1, 0.7), (2, 0.9)]
a, b, r_squared = logistic_regression(sigmoid_points)
print(f"Logistic: y = 1/(1 + exp(-({a:.3f}x + {b:.3f}))), R² = {r_squared:.4f}")

# Orthogonal regression (total least squares)
slope, intercept, r_squared = orthogonal_regression(points)
print(f"Orthogonal: y = {slope:.3f}x + {intercept:.3f}, R² = {r_squared:.4f}")
```

## Testing

### Original Quadratic Fitting Tests

The `quadratic.py` file includes a suite of unit tests to ensure the `fit` function is working correctly. To run the tests, execute the script from your terminal:

```bash
python quadratic.py
```

### New Interpolation Methods Tests

The `interpolation.py` file includes comprehensive tests for all new methods. To run tests and see a demonstration:

```bash
python interpolation.py
```

This will first show a demo of all interpolation methods with example data, then run the full unit test suite.

## Method Details

- **Linear Regression**: Uses least squares to fit y = mx + b
- **Quadratic Regression**: Uses least squares to fit y = ax² + bx + c (handles multiple points)
- **Polynomial Regression**: Fits polynomials of any degree using least squares
- **Logistic Regression**: Fits sigmoid curves y = 1/(1 + e^(-(ax + b))) using gradient descent
- **Orthogonal Regression**: Minimizes perpendicular distances to the line (total least squares)

All implementations use only Python standard library (no external dependencies like NumPy).
