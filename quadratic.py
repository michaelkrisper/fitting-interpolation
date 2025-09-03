from math import sqrt

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
    xopt = -b / (2 * a) if a != 0 else float('inf')

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        yzero = None
    else:
        sqrt_d = sqrt(discriminant)
        if a == 0:
            yzero = (-c / b, -c / b) if b != 0 else None
        else:
            yzero = ((-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a))

    return a, b, c, xopt, yzero

print(fit(*[(x, 1*x**2 + 2*x + 3) for x in [-10, 5, 2]]))
