import autograd.numpy as np
from autograd import jacobian

def NewtonMethod(F: callable, 
                 x0: np.ndarray, 
                 tol: float, 
                 maxit: int
) -> np.ndarray:
    
    jac = jacobian(F)
    xi = x0
    Fe = F(xi)
    cond = np.linalg.norm(F(x0)) > tol
    i = 0

    while cond and i < maxit:
          
        s = np.linalg.solve(a = jac(xi),
                               b = -Fe)
        xi = xi + s
        i += 1
        Fe = F(xi)
        cond = np.linalg.norm(Fe) > tol
    
    return xi

def StationaryNewtonMethod(F: callable, 
                           x0: np.ndarray, 
                           tol: float, 
                           maxit: int
) -> np.ndarray:
    
    J = jacobian(F)(x0)
    xi = x0
    Fe = F(xi)
    cond = np.linalg.norm(Fe) > tol
    i = 0

    while cond and i < maxit:
          
        s = np.linalg.solve(a = J,
                            b = -Fe)
        xi = xi + s
        i += 1
        Fe = F(xi)
        cond = np.linalg.norm(Fe) > tol
    
    return xi

def mStationaryNewtonMethod(F: callable, 
                            x0: np.ndarray, 
                            tol: float, 
                            maxit: int,
                            m: int
) -> np.ndarray:
    
    jac = jacobian(F)
    xi = x0
    Fe = F(xi)
    cond = np.linalg.norm(Fe) > tol
    i = 0

    while cond and i < maxit:
        if i % m == 0:
            J = jac(xi)
        s = np.linalg.solve(a = J,
                            b = -Fe)
        xi = xi + s
        i += 1
        Fe = F(xi)
        cond = np.linalg.norm(Fe) > tol
    
    return xi

def BroydenMethod(F: callable, 
                 x0: np.ndarray, 
                 tol: float, 
                 maxit: int
) -> np.ndarray:
    
    J = jacobian(F)(x0)
    xi = x0
    Fe = F(xi)
    cond = np.linalg.norm(Fe) > tol
    i = 0

    while cond and i < maxit:

        s = np.linalg.solve(a = J,
                            b = -Fe)
        yi = F(xi + s) - Fe
        xi = xi + s
        Fe = F(xi)
        i += 1
        cond = np.linalg.norm(Fe) > tol
        J += np.inner((yi - J @ s), s) / np.inner(s,s)
    
    return xi