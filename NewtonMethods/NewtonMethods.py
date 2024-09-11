import numpy
import autograd.numpy 
from autograd import jacobian

def NewtonMethod(F: callable, 
                 x0: autograd.numpy.ndarray, 
                 tol: float, 
                 maxit: int
) -> autograd.numpy.ndarray:
    jac = jacobian(F)

    cond = numpy.linalg.norm(F(x0)) <= tol
    s = numpy.zeros(len(x0))
    i = 0

    while (not cond) and i < maxit:
        xi = x0 + s

        s = numpy.linalg.solve(a = jac(xi),
                               b = -F(xi))

        i += 1
        cond = numpy.linalg.norm(F(x0)) <= tol
    
    return xi

