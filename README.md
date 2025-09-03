# Quadratic Fitting and Interpolation

This project provides a Python function to calculate the coefficients of a quadratic equation `y = ax^2 + bx + c` that passes through three given points. It also calculates the vertex of the parabola and its roots (y-intercepts).

## Features

The `fit(p1, p2, p3)` function in `quadratic.py` returns a tuple containing:
- `a`, `b`, `c`: The coefficients of the quadratic equation.
- `xopt`: The x-coordinate of the vertex of the parabola.
- `yzero`: A tuple containing the two roots (x-intercepts) of the equation, or `None` if there are no real roots.

## Usage

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

## Testing

The `quadratic.py` file includes a suite of unit tests to ensure the `fit` function is working correctly. To run the tests, execute the script from your terminal:

```bash
python quadratic.py
```
