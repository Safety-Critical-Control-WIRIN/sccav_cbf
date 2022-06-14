#!/usr/bin/env python

import numpy as np
import cvxpy as cp
import euclid as euc
import scipy as sci

import matplotlib.pyplot as plt

class PolynomialLaneCurve():
    
    def __init__(self, coefficients: np.ndarray):
        
        self.update_coeffs(coefficients)
        
    def update_coeffs(self, coefficients: np.ndarray):
        """Update the equation coefficients. Note that this may end
        of unintentionally changing the order of the equation if the
        size of the coefficient array is not properly maintained.

        Parameters:
            coefficients (np.ndarray): Ordered array of coefficients
            
            >>> coefficients = [a0, a1, a2, ...]
            >>> f(x) = a0 + (a1 * x) + (a2 * x^2) + ... 
        """
        self.coeffs = np.asarray(coefficients)
        self._order = int(coefficients.size - 1)
        self._order_fac = np.math.factorial(self._order)
        
    def eval(self, x:np.ndarray):
        """Evaluate the value of the polynomial f(x) at one or more
        points.

        Parameters:
            x (np.ndarray): Array of data points to be evaluated

        Returns:
            np.ndarray : Array of f(x) values
        """
        f = np.zeros_like(x)
        i = 0
        for a in self.coeffs:
            f += a * np.power(x, i)
            i += 1
        return f
    
    def df(self, x:np.ndarray, m:int = 1):
        """Calculate the value of the m order derivate of f(x)
        at one or more points.

        Parameters:
            x (np.ndarray, float): Array of data points to be evaluated at
            m (int, optional): Order of differentiation. Defaults to 1
        """
        m = int(m)
        df = np.zeros_like(x)
        n = self._order
        
        if m > n:
            return 0
        
        for i in range(m, n+1):
            df += self._nPr(i, m) * self.coeffs[i] * np.power(x, i - m)
        return df
    
    def shortest_distance(self, p: euc.Point2, x0: np.ndarray, options: dict ={'xtol': 1e-8, 'disp': False}):
        """Calculates the shortest distance point on the curve from a given point
        p = (x, y) in 2D. The problem is posed as an unconstrained optimization
        and solved using Newton Conjugate Descent from scipy.optimize.

        Parameters:
        ----------
            p (euc.Vector2): Point to calculate shortest distance from
            x0 (np.ndarray): Optimization starting point
            options (dict, optional): Dictionary of scipy.optimize options.
            Defaults to {'xtol': 1e-8, 'disp': False}

        Returns:
        -------
            (np.ndarray): Solution to the euclidean distance minimization
        """
        
        x0 = np.asarray(x0)
        
        def g(x):
            f = self.eval(x)[0]
            return (x - p.x)**2 + (f - p.y)**2
        
        def dg(x):
            f = self.eval(x)[0]
            df = self.df(x, 1)[0]
            return 2*(x - p.x) + 2*(f - p.y)*df
        
        def ddg(x):
            f = self.eval(x)[0]
            df = self.df(x, 1)[0]
            ddf = self.df(x, 2)[0]
            return 2*(1 + df**2 + f * ddf - p.y * df)
        
        res = sci.optimize.minimize(g, x0, method='Newton-CG',
                                    jac = dg, hess = ddg,
                                    options = options)
        
        return res.x
    
    # classmethods
    @classmethod
    def lsq_curve(cls, x_points: np.ndarray, y_points: np.ndarray, n: int = 3):
        """Fits a cubic polynomial to the given set of 2D data points
        using least squares.

        Args:
            x_points (np.ndarray): X coordinate set of data points
            y_points (np.ndarray): Y coordinate set of data points
            n     (int, optional): Order of the curve to be fit

        Returns:
            CVXPY Variable Solution Value: The coefficients of the cubic
            polynomial in the order `a0`, `a1`, `a2`, `a3`, `a4`; for the equation::
            
            >>> a0 + a1 * x + a2 * x^2 + a3 * x^3
        """
        
        x_points = np.asarray(x_points)
        y_points = np.asarray(y_points)
        
        A = np.ones((x_points.size, n+1))
        
        for i in range(1, n+1):
            A[:, i] = np.power(x_points.flatten(), i)

        coeffs = cp.Variable(n+1)
        cost = cp.sum_squares(A @ coeffs - y_points.reshape(-1))
        lsq = cp.Problem(cp.Minimize(cost))
        lsq.solve()
        
        return cls(coeffs.value)
    
    # Utility functions for the class.
    
    def _nPr(self, n:int, r:int):
        """Calculate the value nPr = n!/(n-r)!

        Parameters:
            n (int)
            r (int)
        
        Returns:
            (int) : nPr
        """
        if n != self._order:
            self._nfac = np.math.factorial(n)
        else:
            self._nfac = self._order_fac
            
        return np.floor(self._nfac/np.math.factorial(n - r))
            
        

def fit_cubic_curve(x_points: np.ndarray, y_points: np.ndarray):
    """Fits a cubic polynomial to the given set of 2D data points
    using least squares.

    Args:
        x_points (np.ndarray): X coordinate set of data points
        y_points (np.ndarray): Y coordinate set of data points

    Returns:
        CVXPY Variable Solution Value: The coefficients of the cubic
        polynomial in the order `a0`, `a1`, `a2`, `a3`, `a4`; for the equation::
        
        >>> a0 + a1 * x + a2 * x^2 + a3 * x^3
    """
    
    x_points = np.array(x_points)
    x_points = x_points.reshape(x_points.size, 1)
    y_points = np.array(y_points)
    
    x_points_sq = np.power(x_points, 2)
    x_points_cu = np.power(x_points, 3)
    
    A = np.ones((x_points.size, 4))

    A[:, 1] = x_points[:, 0]
    A[:, 2] = x_points_sq[:, 0]
    A[:, 3] = x_points_cu[:, 0]

    coeffs = cp.Variable(4)
    cost = cp.sum_squares(A @ coeffs - y_points.reshape(-1))
    lsq = cp.Problem(cp.Minimize(cost))
    lsq.solve()
    
    residual_norm = cp.norm(A @ coeffs - y_points.reshape(-1), p=2).value
    
    return coeffs, lsq, residual_norm

def test():
    x_test_points = np.array([[-1.371, -0.75, 0, 0.333]])
    y_test_points = np.array([[0, 6.938, 3, 1.852]])
    
    coeffs, *_ = fit_cubic_curve(x_test_points, y_test_points)
    del(_)
    a = coeffs.value
    
    # Plotting the results
    res = 0.1
    x_vec = np.linspace
    x_min = -5
    x_max = 5
    fit_curve_x = np.linspace(x_min, x_max, int((x_max - x_min)/res))
    fit_curve_y = a[0] + a[1] * fit_curve_x + a[2] * np.power(fit_curve_x, 2) + a[3] * np.power(fit_curve_x, 3)
    
    fig, ax = plt.subplots()
    ax.plot(x_test_points, y_test_points, 'go')
    ax.plot(fit_curve_x, fit_curve_y, linestyle = "-", color = "black")
    ax.set_ylim(bottom = -10.0, top = 10.0)
    plt.show()
    pass

if __name__ == "__main__":
    test()