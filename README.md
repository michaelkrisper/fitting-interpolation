# fitting-interpolation

Curve fitting and regression in pure Python - no NumPy, no SciPy, standard library only.

[![CI](https://github.com/michaelkrisper/fitting-interpolation/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelkrisper/fitting-interpolation/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/tag/michaelkrisper/fitting-interpolation?sort=semver&label=version)](https://github.com/michaelkrisper/fitting-interpolation/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

## Features

- **Three-point parabola** (`quadratic.fit`): exact quadratic through three points, with vertex and real roots
- **Linear regression**: least squares line `y = mx + b`
- **Quadratic regression**: least squares parabola `y = ax^2 + bx + c` for any number of points
- **Polynomial regression**: least squares polynomial of any degree (coefficients ordered like `numpy.polyfit`)
- **Logistic regression**: sigmoid `y = 1 / (1 + e^-(ax + b))` fitted by gradient descent
- **Orthogonal regression**: total least squares, minimizing perpendicular distances (handles vertical lines)

Every regression also returns the coefficient of determination (R-squared).

## Installation

There is nothing to install besides Python 3.10 or newer. Clone the repository and import the modules:

```bash
git clone https://github.com/michaelkrisper/fitting-interpolation.git
cd fitting-interpolation
python examples.py
```

## Usage

### Parabola through three points

```python
from quadratic import fit

a, b, c, xopt, roots = fit((-2, 0), (0, 4), (2, 0))
# a = -1.0, b = 0.0, c = 4.0, xopt = 0.0, roots = (-2.0, 2.0)
```

`fit` returns the coefficients `a`, `b`, `c`, the x coordinate of the vertex `xopt` and the real roots:
a tuple with two roots, a tuple with one root if the points are collinear (then `a == 0` and
`xopt == inf`), or `None` if the curve never crosses zero. Points with duplicate x coordinates
raise `ValueError`.

### Regression

```python
from interpolation import (
    linear_regression,
    logistic_regression,
    orthogonal_regression,
    polynomial_regression,
    quadratic_regression,
)

points = [(1, 2.1), (2, 4.0), (3, 5.9), (4, 8.1), (5, 9.9)]

slope, intercept, r2 = linear_regression(points)
a, b, c, r2 = quadratic_regression(points)
coeffs, r2 = polynomial_regression(points, degree=3)  # highest degree first
slope, intercept, r2 = orthogonal_regression(points)  # slope == inf for a vertical line, intercept is then x

# y values must lie in [0, 1]
a, b, r2 = logistic_regression([(-2, 0.1), (-1, 0.3), (0, 0.5), (1, 0.7), (2, 0.9)])
```

All functions raise `ValueError` for too few points or degenerate input.

Runnable demos:

```bash
python examples.py        # walk-through of every method
python interpolation.py   # short demo of the regression methods
```

## Development

```bash
pip install pytest ruff
pytest      # run the test suite
ruff check  # lint
```

CI runs both on every push and pull request against Python 3.10 to 3.14.
Every merge to `main` is tagged with the next patch version; bundled releases
with release notes come from the Release workflow.

## Project structure

| File | Contents |
| --- | --- |
| `quadratic.py` | `fit()` - exact parabola through three points |
| `interpolation.py` | regression methods and a small demo |
| `examples.py` | examples for every method |
| `test_*.py` | unit tests |

## Contributing

Changes start with an issue and go through a pull request; see
[CONTRIBUTING](CONTRIBUTING.md).

## License

MIT - see [LICENSE](LICENSE).
