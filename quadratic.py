from math import isclose, sqrt


def fit(p1, p2, p3):
    """ Calculates the coefficients of a quadratic equation y = ax^2 + bx + c that passes through three given points.
    return: a, b, c: the coefficients
            xoptimum: the x coordinate of the optimum (either min or max, depending on the curvature)
            xroots: tuple with the x coordinates of the roots (two for a parabola, one for a sloped line),
                    or None if the curve doesn't intersect zero."""
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    if x1 == x2 or x1 == x3 or x2 == x3:
        raise ValueError("Input points must have unique x-coordinates.")
    a = (y1/((x1-x2)*(x1-x3)) + y2/((x2-x1)*(x2-x3)) + y3/((x3-x1)*(x3-x2)))
    b = (y1 - y2 - a * (x1**2 - x2**2)) / (x1 - x2)
    c = y1 - a * x1**2 - b * x1
    if isclose(a, 0, abs_tol=1e-9):
        # The points are collinear: y = bx + c has no optimum and at most one root.
        a = 0.0
        xroots = None if isclose(b, 0, abs_tol=1e-9) else (-c / b,)
        return a, b, c, float('inf'), xroots
    discriminant = b**2 - 4 * a * c
    xoptimum = -b / (2 * a)
    xroots = None if discriminant < 0 else ((sqrt(discriminant)-b) / (2 * a), (-sqrt(discriminant)-b) / (2 * a))
    return a, b, c, xoptimum, xroots


if __name__ == '__main__':
    print(fit(*[(x, 1*x**2 + 2*x + 3) for x in [-10, 5, 2]]))
