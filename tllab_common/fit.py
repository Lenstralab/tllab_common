from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import copy
from functools import cached_property
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from parfor import pmap
from scipy import special, stats
from scipy.optimize import OptimizeResult, minimize

Number = int | float | complex


class Fit(metaclass=ABCMeta):
    bounds = None

    def __init__(self, x: ArrayLike, y: ArrayLike,
                 w: [ArrayLike, None] = None, s: [ArrayLike, None] = None,
                 fit_window: Sequence[float] = None, log_scale: bool = False, fit_s: bool = True,
                 p0: ArrayLike = None) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        w = np.ones_like(x) if w is None else np.asarray(w)
        if log_scale:
            s = np.ones_like(x) if s is None else np.asarray(s) / np.abs(y)
        else:
            s = np.ones_like(x) if s is None else np.asarray(s)

        if fit_window:
            idx = (fit_window[0] <= x) & (x < fit_window[1])
            x, y, w, s = x[idx], y[idx], w[idx], s[idx]

        self.x, self.y, self.w, self.s = nonnan(x, y, w, s)
        self.log_scale = log_scale
        self.fit_s = fit_s
        self.n = np.sum(self.w)
        self.p_ci95 = None
        self.r_squared = None
        self.chi_squared = None
        self.r_squared_adjusted = None
        self.p0_manual = p0

    @property
    @abstractmethod
    def n_p(self) -> int:
        pass

    @property
    @abstractmethod
    def p0(self) -> ArrayLike:
        pass

    @staticmethod
    @abstractmethod
    def fun(p: ArrayLike, x: Number | ArrayLike) -> ArrayLike:
        pass

    def dfun(self, p: ArrayLike, x: ArrayLike, diffstep: float = 1e-6) -> np.ndarray:
        """ d fun / dp_i for each p_i in p, this default function will calculate it numerically """
        eps = np.spacing(1)
        deriv = np.zeros((len(p), len(x)))
        f0 = np.asarray(self.fun(p, x))
        p = np.asarray(p)
        for i in range(len(p)):
            ph = p.copy()
            ph[i] = p[i] * (1 + diffstep) + eps
            f = np.asarray(self.fun(ph, x))
            deriv[i] = (f - f0) / (ph[i] - p[i])
        return deriv

    def evaluate(self, x: Number | ArrayLike = None) -> tuple[ArrayLike, ArrayLike]:
        if x is None:
            x = np.linspace(np.nanmin(self.x), np.nanmax(self.x))
        else:
            x = np.asarray(x)
        return x.real, self.fun(self.p, x)

    def evaluate_ci(self, x: Number | ArrayLike = None) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        if x is None:
            x = np.linspace(np.nanmin(self.x), np.nanmax(self.x))
        else:
            x = np.asarray(x)
        f = self.fun(self.p, x)
        df = np.sqrt(np.sum((self.dfun(self.p, x).T * self.p_ci95).T ** 2, 0))
        return x.real, f - df, f + df

    def get_cost_fun(self) -> Callable[[ArrayLike], float]:
        s = self.s if self.fit_s else 1
        eps = np.spacing(0)
        if self.log_scale:
            def cost(p: ArrayLike) -> float:
                with np.errstate(divide='ignore'):
                    return np.nansum(np.abs(
                        self.w / s * (np.log(self.y) - np.log(np.clip(self.fun(p, self.x), eps, None))) ** 2))
        else:
            def cost(p: ArrayLike) -> float:
                with np.errstate(divide='ignore'):
                    return np.nansum(np.abs(self.w / s * (self.y - self.fun(p, self.x)) ** 2))
        return cost

    def fit(self) -> Fit:
        _ = self.r
        return self

    @cached_property
    def r(self) -> OptimizeResult:
        if not hasattr(self, 'p0_manual'):
            self.p0_manual = None
        with np.errstate(divide='ignore'):
            if len(self.x):
                r = minimize(self.get_cost_fun(), np.asarray(self.p0 if self.p0_manual is None else self.p0_manual),
                             method='Nelder-Mead', bounds=self.bounds, options=dict(maxiter=400 * self.n_p))
            else:
                r = OptimizeResult(fun=np.nan, message='Empty data', nfev=0, nit=0, status=1, success=False,
                                   x=np.full(self.n_p, np.nan))
            if self.log_scale:
                self.p_ci95, self.r_squared = fminerr(
                    lambda p, x: np.log(self.fun(p, x)), self.sort(r.x), np.log(self.y), (self.x,), self.w, self.s)
            else:
                self.p_ci95, self.r_squared = fminerr(self.fun, self.sort(r.x), self.y, (self.x,), self.w, self.s)
            if self.n - self.n_p - 1 > 0:
                self.r_squared_adjusted = 1 - (1 - self.r_squared) * (self.n - 1) / (self.n - self.n_p - 1)
            else:
                self.r_squared_adjusted = np.nan
            return r

    @staticmethod
    def sort(p: np.ndarray) -> np.ndarray:
        return p

    @property
    def p(self) -> np.ndarray:
        return np.full(self.n_p, np.nan) if self.r is None else self.sort(self.r.x)

    @property
    def log_likelihood(self) -> float:
        return -self.n * np.log(2 * np.pi * self.r.fun / (self.n - 1)) / 2 - (self.n - 1) / 2

    @property
    def bic(self) -> float:
        """ Bayesian Information Criterion: the fit with the smallest bic should be the best fit """
        return self.n_p * np.log(self.n) - 2 * self.log_likelihood

    def ftest(self, fit2) -> float:
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
            f_value = (np.abs(rss1 - rss2) / dn) / ((rss1 if swapped else rss2) / (self.n - n))
            p_value = stats.f(dn, self.n - n).sf(f_value)
            return -p_value if swapped else p_value

    def reset(self, log_scale: bool = None, fit_s: bool = True, p0: ArrayLike = None) -> Fit:
        new = copy(self)
        if log_scale is not None:
            if log_scale and not self.log_scale:
                new.s /= np.abs(new.y)
            if not log_scale and self.log_scale:
                new.s *= np.abs(new.y)
            new.log_scale = log_scale
        if fit_s is not None:
            new.fit_s = fit_s
        if p0 is not None:
            new.p0_manual = p0
        if hasattr(new, 'r'):
            delattr(new, 'r')
        return new

    def get_best_p0(self, space: np.ndarray) -> Fit:
        bounds = np.array([(-np.inf if i[0] is None else i[0], np.inf if i[1] is None else i[1])
                           for i in self.bounds]).T
        p0 = np.clip(np.stack(np.meshgrid(*self.n_p * (space,)), 0).reshape((self.n_p, -1)).T * self.p0, *bounds)
        return self.reset(p0=p0[np.argmin(pmap(self.get_cost_fun(), p0, desc='finding best p0', leave=False))])


class Exponential1(Fit):
    n_p = 2
    bounds = ((0, None), (0, None))

    @property
    def p0(self) -> ArrayLike:
        """ y = a*exp(-t/tau)
            return a, tau
        """
        x, y = finite(self.x.astype('complex'), np.log(self.y.astype('complex')))
        if len(x) < 2:
            return [1, 1]
        else:
            q = np.polyfit(x, y, 1)
            return [np.clip(value.real, *bound) for value, bound in zip((np.exp(q[1]), -1 / q[0]), self.bounds)]

    @staticmethod
    def fun(p: ArrayLike, x: Number | ArrayLike) -> ArrayLike:
        return p[0] * np.exp(-x / p[1])

    # def dfun(self, p, x, diffstep=None):
    #     e = np.exp(-x / p[1])
    #     return np.vstack((e, p[0] * x * e / p[1] ** 2))


class Exponential2(Fit):
    n_p = 4
    bounds = ((0, None), (0, 1), (0, None), (0, None))

    @property
    def p0(self) -> ArrayLike:
        """ y = A(a*exp(-t/tau_0) + (1-a)*exp(-t/tau_1))
            return A, a, tau_0, tau_1
        """
        n = len(self.x) // 2
        y0 = np.nanmax(self.y)
        q = Exponential1(self.x[n:], self.y[n:] / y0).p0
        return [np.clip(value, *bound)
                for value, bound in zip((y0, 1 - q[0], q[1] / 3, q[1]), self.bounds)]

    @staticmethod
    def fun(p: ArrayLike, x: Number | ArrayLike) -> ArrayLike:
        return p[0] * (p[1] * np.exp(-x / p[2]) + (1 - p[1]) * np.exp(-x / p[3]))

    # def dfun(self, p, x, diffstep=None):
    #     e0 = np.exp(-x / p[2])
    #     e1 = np.exp(-x / p[3])
    #     return np.vstack((p[1] * e0 + (1 - p[1]) * e1, p[0] * (e0 - e1),
    #                       p[0] * p[1] * e0 / p[2] ** 2, p[0] * (1 - p[1]) * e1 / p[3] ** 2))


class Exponential3(Fit):
    n_p = 6
    bounds = ((0, None), (0, 1), (0, 1), (0, None), (0, None), (0, None))

    @property
    def p0(self) -> ArrayLike:
        """ y = A(a*exp(-t/tau_0) + b*exp(-t/tau_1) + (1-a-b)*exp(-t/tau-2))
            return A, a, b, tau_0, tau_1, tau_2
        """
        n = len(self.x) // 2
        y0 = np.nanmax(self.y)
        q = Exponential2(self.x[n:], self.y[n:] / y0).p0
        return [np.clip(value, *bound)
                for value, bound in zip((y0, 0.3, 0.3, q[2] / 3, q[3] / 3, q[3]), self.bounds)]

    @staticmethod
    def fun(p: ArrayLike, x: Number | ArrayLike) -> ArrayLike:
        return p[0] * (p[1] * np.exp(-x / p[3]) + p[2] * np.exp(-x / p[4]) +
                       (1 - p[1] - p[2]) * np.exp(-x / p[5]))


class Powerlaw(Fit):
    n_p = 2

    @property
    def p0(self) -> ArrayLike:
        """ y = (x/tau)^alpha
            return alpha, tau
        """
        q = np.polyfit(*finite(np.log(self.x.astype('complex')), np.log(self.y.astype('complex'))), 1)
        return q[0].real, np.exp(-q[1] / q[0]).real

    @staticmethod
    def fun(p: ArrayLike, x: Number | ArrayLike) -> ArrayLike:
        return ((np.asarray(x).astype('complex') / p[1]) ** p[0]).real


class GammaCDF(Fit):
    n_p = 2

    @property
    def p0(self) -> ArrayLike:
        """ y = γ(k, x / θ) / Γ(k)
        """
        m = np.sum(-self.x[1:] * np.diff(self.y))
        v = np.sum(-(self.x[1:] - m) ** 2 * np.diff(self.y))
        return m ** 2 / v, v / m  # A, k, theta

    @staticmethod
    def fun(p: ArrayLike, x: Number | ArrayLike) -> ArrayLike:
        """ p: k, theta """
        return 1 - special.gammainc(p[0], x / p[1])


def finite(*args: ArrayLike) -> list[np.ndarray]:
    idx = np.prod([np.isfinite(arg) for arg in args], 0).astype(bool)
    return [np.asarray(arg)[idx] for arg in args]


def nonnan(*args: ArrayLike) -> list[np.ndarray]:
    idx = np.prod([~np.isnan(arg) for arg in args], 0).astype(bool)
    return [np.asarray(arg)[idx] for arg in args]


def fminerr(fun: Callable[[ArrayLike, Any], float], a: ArrayLike, y: ArrayLike,
            args: tuple[Any] = (), w: ArrayLike = None, s: ArrayLike = None,
            diffstep: float = 1e-6) -> tuple[np.ndarray, float]:
    """ Error estimation of a fit

        Inputs:
        fun:  function which was fitted to data
        a:    function parameters
        y:    ydata
        args: extra arguments to fun
        w:    weights
        s:    error on y (std)

        Outputs:
        da:    95% confidence interval
        R2:    R^2

        Example:
        x = np.array((-3,-1,2,4,5))
        a = np.array((2,-3))
        y = (15,0,5,30,50)
        fun = lambda a: a[0]*x**2+a[1]
        dp, R2 = fminerr(fun, p, y)

        adjusted from Matlab version by Thomas Schmidt, Leiden University
        wp@tl2020
    """
    eps = np.spacing(1)
    a = np.array(a).flatten()
    y = np.array(y).flatten()
    w = np.ones_like(y) if w is None else np.asarray(w).flatten()
    s = np.zeros_like(y) if s is None else np.asarray(s).flatten()

    n_data = np.size(y)
    n_par = np.size(a)

    if n_data > n_par:
        f0 = np.array(fun(a, *args)).flatten()
        var_res = (np.sum(w * (f0 - y) ** 2) + np.sum(w * s ** 2)) / (np.sum(w) - n_par)

        # calculate R^2
        ss_tot = np.sum(w * (y - np.nanmean(y)) ** 2)
        ss_res = np.sum(w * (y - f0) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # calculate derivatives
        jac = np.zeros((n_data, n_par), dtype='complex')
        for i in range(n_par):
            ah = a.copy()
            ah[i] = np.clip(a[i] * (1 + diffstep), eps, None)
            f = np.array(fun(ah, *args)).flatten()
            jac[:, i] = (f - f0) / (ah[i] - a[i])

        hesse = np.matmul(jac.T, jac)

        try:
            if np.linalg.matrix_rank(hesse) == np.shape(hesse)[0]:
                da = np.sqrt(var_res * np.diag(np.linalg.inv(hesse)))
            else:
                da = np.sqrt(var_res * np.diag(np.linalg.pinv(hesse)))
        except (Exception,):
            da = np.full_like(a, np.nan)
        return 1.96 * da.real, r_squared.real
    else:
        return np.full_like(a, np.nan), np.nan
