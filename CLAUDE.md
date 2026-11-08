# CLAUDE.md

Curve fitting and regression in pure Python, no NumPy/SciPy: an exact parabola through
three points plus linear, quadratic, polynomial, logistic and orthogonal regression,
each with R-squared. Python 3.10+, standard library only. Plain modules, no packaging,
nothing deployed.

## Commands

```bash
python examples.py         # walk-through of every method
python interpolation.py    # short regression demo
pip install pytest ruff
pytest                     # test_quadratic.py, test_interpolation.py, < 1 s
ruff check                 # ruff.toml: E, F, W, I, UP; 120 cols
```

CI (`.github/workflows/ci.yml`) runs ruff on 3.14 and pytest on 3.10 to 3.14.

## Architecture

- `quadratic.py`: `fit(p1, p2, p3)` -> `(a, b, c, xopt, roots)`. Collinear points give
  `a == 0`, `xopt == inf` and a one-element `roots` tuple (or `None` for a horizontal
  line); duplicate x raises `ValueError`.
- `interpolation.py`: `linear_regression`, `quadratic_regression` (3x3 solve),
  `polynomial_regression` (normal equations + Gaussian elimination, coefficients
  highest degree first like `numpy.polyfit`), `logistic_regression` (plain gradient
  descent, y in [0, 1]), `orthogonal_regression` (total least squares; a vertical line
  returns `slope == inf` and the x position as intercept). `demo()` backs `__main__`.
- `examples.py`: printed examples for every method; keep it runnable.

## Rules and pitfalls

- Public functions return plain tuples whose shape is documented in the README; keep
  the order and the degenerate-case conventions (`inf`, `None`) stable.
- Every function raises `ValueError` for too few points or degenerate input; never
  let `ZeroDivisionError` escape (see the collinear fix in `fit()`). Compare floats
  with `math.isclose`/tolerances, not `==`.
- No third-party dependencies, not even in tests (`unittest`-style classes run under
  pytest).
- When changing a signature or return value, update README usage, `examples.py` and
  the tests together.

## Workflow

- Every change starts with a GitHub issue. Branch `<issue>-<slug>` from `main`,
  open a PR with `Closes #<issue>`. Never push to `main` directly.
- PRs auto-merge (squash) once CI and the "Linked ticket" check pass; drafts are not merged.
- Versions exist only as git tags `vX.Y.Z`. Every merge to `main` tags the next patch;
  never edit version numbers in files. Minor/major releases: Actions -> Release.
- Nothing is deployed; a merge to `main` only creates the patch tag.
- Details: CONTRIBUTING.md.
