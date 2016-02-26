from math import sqrt

def fitquadratic(p1, p2, p3):
    x1, x2, x3 = p1[0], p2[0], p3[0]
    y1, y2, y3 = p1[1], p2[1], p3[1]

    a = (x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/((x1-x2)*(x1-x3)*(x3-x2))
    b = (x1**2*(y2-y3)+x2**2*(y3-y1)+x3**2*(y1-y2))/((x1-x2)*(x1-x3)*(x2-x3))
    c = (x1**2*(x2*y3-x3*y2)+x1*(x3**2*y2-x2**2*y3)+x2*x3*y1*(x2-x3))/((x1-x2)*(x1-x3)*(x2-x3))
    xopt = (x2**2*(y3-y1)-x1**2*(y3-y2)-x3**2*(y2-y1))/(2*(x2*(y3-y1)-x1*(y3-y2)-x3*(y2-y1)))

    if (b**2 - 4*a*c) >= 0:
        yzero = ((-b+sqrt(b**2 - 4*a*c))/(2*a), (-b-sqrt(b**2 - 4*a*c))/(2*a))
    else:
        yzero = None
    return a,b,c, xopt, yzero

print(fitquadratic(*[(x, 1*x**2 + 2*x + 3) for x in [-10, 5, 2]]))
