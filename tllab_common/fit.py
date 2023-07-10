import numpy as np
from functools import cached_property
from scipy.optimize import minimize, OptimizeResult
from scipy import stats
from abc import ABCMeta, abstractmethod


class Fit(metaclass=ABCMeta):
    def __init__(self, x, y, w=None, log_scale=False):
        x = np.asarray(x)
        y = np.asarray(y)
        if w is None:
            w = np.ones_like(x)
        else:
            w = np.asarray(w)
        self.x, self.y, self.w = nonnan(x, y, w)
        self.log_scale = log_scale
        self.n = len(x)
        self.p_ci95 = None
        self.r_squared = None
        self.chi_squared = None
        self.r_squared_adjusted = None

    @property
    @abstractmethod
    def n_p(self):
        pass

    @property
    def bounds(self):
        return None

    @property
    @abstractmethod
    def p0(self):
        pass

    @staticmethod
    @abstractmethod
    def fun(p, x):
        pass

    def evaluate(self, x=None):
        if x is None:
            x = np.linspace(np.nanmin(self.x), np.nanmax(self.x))
        else:
            x = np.asarray(x)
        return x.real, self.fun(self.p, x)

    def get_cost_fun(self):
        if self.log_scale:
            def cost(p):
                return np.sum(np.abs(self.w * np.log(self.y / self.fun(p, self.x)) ** 2))
        else:
            def cost(p):
                return np.sum(np.abs(self.w * (self.y - self.fun(p, self.x)) ** 2))
        return cost

    @cached_property
    def r(self):
        if len(self.x):
            r = minimize(self.get_cost_fun(), self.p0, method='Nelder-Mead', bounds=self.bounds)
        else:
            r = OptimizeResult(fun=np.nan, message='Empty data', nfev=0, nit=0, status=1, success=False,
                               x=np.full(self.n_p, np.nan))
        if self.log_scale:
            self.chi_squared, self.p_ci95, self.r_squared = fminerr(lambda p, x: np.log(self.fun(p, x)),
                                                                    r.x, np.log(self.y), (self.x,), self.w)
        else:
            self.chi_squared, self.p_ci95, self.r_squared = fminerr(self.fun, r.x, self.y, (self.x,), self.w)
        self.r_squared_adjusted = 1 - (1 - self.r_squared) * (self.n - 1) / (len(r.x) - 1)
        return r

    @property
    def p(self):
        return np.full(self.n_p, np.nan) if self.r is None else self.r.x

    @property
    def log_likelihood(self):
        return -self.n * np.log(2 * np.pi * self.r.fun / (self.n - 1)) / 2 - (self.n - 1) / 2

    @property
    def bic(self):
        """ Bayesian Information Criterion: the fit with the smallest bic should be the best fit """
        return self.n_p * np.log(self.n) - 2 * self.log_likelihood

    def ftest(self, fit2):
        """ returns the p-value for the hypothesis that fit2 is the better fit,
            assuming fit2 is the fit with more free parameters
            if the fits are swapped the p-value will be negative """
        if not np.all(self.x == fit2.x):
            raise ValueError('Only two fits on the same data can be compared.')
        if self.n_p == fit2.n_p:
            raise ValueError('The two fits cannot have the same number of parameters.')
        rss1 = self.get_cost_fun()(self.p)
        rss2 = fit2.get_cost_fun()(fit2.p)
        swapped = np.argmin((self.n_p, fit2.n_p))
        if swapped and rss1 > rss2:
            return -1
        elif not swapped and rss1 < rss2:
            return 1
        else:
            n = self.n_p if swapped else fit2.n_p
            dn = np.abs(self.n_p - fit2.n_p)
            f_value = (np.abs(rss1 - rss2) / dn) / ((rss1 if swapped else rss2) / (len(self.x) - n))
            p_value = stats.f(dn, len(self.x) - n).sf(f_value)
            return -p_value if swapped else p_value


class Exponential1(Fit):
    n_p = 2
    bounds = ((0, None), (0, None))

    @property
    def p0(self):
        """ y = a*exp(-t/tau)
            return a, tau
        """
        q = np.polyfit(*finite(self.x.astype('complex'), np.log(self.y.astype('complex'))), 1)
        return [np.clip(value.real, *bound) for value, bound in zip((np.exp(q[1]), -1 / q[0]), self.bounds)]

    @staticmethod
    def fun(p, x):
        return p[0] * np.exp(-x / p[1])


class Exponential2(Fit):
    n_p = 4
    bounds = ((0, None), (0, 1), (0, None), (0, None))

    @property
    def p0(self):
        """ y = A(a*exp(-t/tau_0) + (1-a)*exp(-t/tau_1)
            return A, tau_0, a, tau_1
        """
        n = len(self.x) // 2
        y0 = np.nanmax(self.y)
        q = Exponential1(self.x[n:], self.y[n:] / y0).p0
        return [np.clip(value, *bound)
                for value, bound in zip((y0, 1 - q[0], q[1] / 3, q[1]), self.bounds)]

    @staticmethod
    def fun(p, x):
        return p[0] * (p[1] * np.exp(-x / p[2]) + (1 - p[1]) * np.exp(-x / p[3]))


class Powerlaw(Fit):
    n_p = 2

    @property
    def p0(self):
        """ y = (x/tau)^alpha
            return alpha, tau
        """
        q = np.polyfit(*finite(np.log(self.x.astype('complex')), np.log(self.y.astype('complex'))), 1)
        return q[0].real, np.exp(-q[1] / q[0]).real

    @staticmethod
    def fun(p, x):
        return ((x.astype('complex') / p[1]) ** p[0]).real


def finite(*args):
    idx = np.prod([np.isfinite(arg) for arg in args], 0).astype(bool)
    return [arg[idx] for arg in args]


def nonnan(*args):
    idx = np.prod([~np.isnan(arg) for arg in args], 0).astype(bool)
    return [arg[idx] for arg in args]


def fminerr(fun, a, y, args=(), w=None, diffstep=1e-6):
    """ Error estimation of a fit

        Inputs:
        fun:  function which was fitted to data
        a:    function parameters
        y:    ydata
        args: extra arguments to fun
        w:    weights

        Outputs:
        chisq: Chi^2
        da:    95% confidence interval
        R2:    R^2

        Example:
        x = np.array((-3,-1,2,4,5))
        a = np.array((2,-3))
        y = (15,0,5,30,50)
        fun = lambda a: a[0]*x**2+a[1]
        chisq, dp, R2 = fminerr(fun, p, y)

        adjusted from Matlab version by Thomas Schmidt, Leiden University
        wp@tl2020
    """
    eps = np.spacing(1)
    a = np.array(a).flatten()
    y = np.array(y).flatten()
    if w is None:
        w = np.ones(np.shape(y))
    else:
        w = np.array(w).flatten()
    n_data = np.size(y)
    n_par = np.size(a)
    f0 = np.array(fun(a, *args)).flatten()
    chisq = np.sum(((f0 - y) * w) ** 2) / (n_data - n_par)

    # calculate R^2
    sstot = np.sum((y - np.nanmean(y)) ** 2)
    ssres = np.sum((y - f0) ** 2)
    r_squared = 1 - ssres / sstot

    # calculate derivatives
    deriv = np.zeros((n_data, n_par), dtype='complex')
    for i in range(n_par):
        ah = a.copy()
        ah[i] = a[i] * (1 + diffstep) + eps
        f = np.array(fun(ah, *args)).flatten()
        deriv[:, i] = (f - f0) / (ah[i] - a[i]) * w

    hesse = np.matmul(deriv.T, deriv)

    try:
        if np.linalg.matrix_rank(hesse) == np.shape(hesse)[0]:
            da = np.sqrt(chisq * np.diag(np.linalg.inv(hesse)))
        else:
            da = np.sqrt(chisq * np.diag(np.linalg.pinv(hesse)))
    except (Exception,):
        da = np.full_like(a, np.nan)
    return chisq.real, 1.96 * da.real, r_squared.real
