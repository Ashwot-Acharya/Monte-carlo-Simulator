"""
Target Distributions
=====================
A library of log-probability functions to test your sampler on.

Each function returns log p(x) — we work in log space to avoid
numerical underflow (tiny probabilities become very negative numbers,
not zero).

These range from trivial (Gaussian) to pathological (banana, donut)
which stress-test your sampler in different ways.
"""

import numpy as np
from typing import Callable


def gaussian_2d(mu: np.ndarray = None, sigma: np.ndarray = None) -> Callable:
    """
    Multivariate Gaussian. The "hello world" of MCMC.
    If your sampler can't get this right, nothing else will work.
    
    True posterior: known analytically, so R-hat and ESS are checkable.
    """
    if mu is None:
        mu = np.array([2.0, -1.0])
    if sigma is None:
        sigma = np.array([[1.0, 0.8], [0.8, 1.0]])   # correlated!
    
    inv_sigma = np.linalg.inv(sigma)
    log_det   = np.log(np.linalg.det(sigma))

    def log_prob(x: np.ndarray) -> float:
        diff = x - mu
        return -0.5 * (diff @ inv_sigma @ diff + log_det)
    
    log_prob.__name__ = "2D Correlated Gaussian"
    log_prob.true_mean = mu
    log_prob.n_dims = 2
    return log_prob


def banana(b: float = 0.1, sigma: float = 10.0) -> Callable:
    """
    Rosenbrock / Banana distribution. 
    
    The classic MCMC stress test. The posterior is curved like a banana,
    meaning x1 and x2 are highly correlated in a nonlinear way.
    Simple random-walk MH struggles here — this is WHY HMC was invented.
    
    p(x1, x2) ∝ exp(-x1²/200 - 0.5*(x2 + b*x1² - 100b)²)
    """
    def log_prob(x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return -x1**2 / (2 * sigma**2) - 0.5 * (x2 + b * x1**2 - b * sigma**2)**2
    
    log_prob.__name__ = f"Banana (b={b})"
    log_prob.n_dims = 2
    return log_prob


def donut(radius: float = 3.0, sigma: float = 0.5) -> Callable:
    """
    Ring / Donut distribution.
    
    Probability mass concentrated on a ring of given radius.
    Kills random-walk samplers because you need to "find" the ring.
    The chain gets stuck inside or outside the ring.
    
    Tests: does your sampler explore the full circle, or get stuck?
    """
    def log_prob(x: np.ndarray) -> float:
        r = np.sqrt(np.sum(x**2))
        return -0.5 * ((r - radius) / sigma)**2
    
    log_prob.__name__ = f"Donut (r={radius})"
    log_prob.n_dims = 2
    return log_prob


def mixture_gaussians(
    means: list = None,
    weights: list = None,
    sigma: float = 0.5
) -> Callable:
    """
    Gaussian mixture model — multimodal distribution.
    
    The nightmare scenario: probability mass in multiple separated modes.
    A chain can get trapped in one mode and never find the others.
    R-hat will flag this: different chains find different modes → high variance between chains.
    
    This is where you'd need parallel tempering or other tricks.
    """
    if means is None:
        means = [np.array([-4, -4]), np.array([4, 4]), np.array([-4, 4])]
    if weights is None:
        weights = [0.4, 0.35, 0.25]
    
    log_weights = np.log(weights)

    def log_prob(x: np.ndarray) -> float:
        log_components = []
        for mean, lw in zip(means, log_weights):
            diff = x - mean
            log_components.append(lw - 0.5 * np.dot(diff, diff) / sigma**2)
        # Log-sum-exp for numerical stability
        return float(np.logaddexp.reduce(log_components))
    
    log_prob.__name__ = "3-Mode Mixture Gaussian"
    log_prob.n_dims = 2
    return log_prob


def funnel(n_dims: int = 8) -> Callable:
    """
    Neal's Funnel — the hardest standard benchmark.
    
    v  ~ N(0, 9)
    xi ~ N(0, exp(v))  for i = 1..n-1
    
    The variance of x_i depends exponentially on v.
    When v is large (negative), the funnel is extremely narrow.
    MH either: (a) uses big steps and rejects everything in the narrow part,
               (b) uses small steps and takes forever in the wide part.
    
    This is a real problem in hierarchical Bayesian models.
    """
    def log_prob(x: np.ndarray) -> float:
        v    = x[0]
        rest = x[1:]
        lp_v    = -0.5 * v**2 / 9.0
        lp_rest = -0.5 * np.sum(rest**2) * np.exp(-v) - 0.5 * (n_dims - 1) * v
        return lp_v + lp_rest
    
    log_prob.__name__ = f"Neal's Funnel ({n_dims}D)"
    log_prob.n_dims = n_dims
    return log_prob


def bayesian_linear_regression(X: np.ndarray, y: np.ndarray,
                                prior_sigma: float = 10.0) -> Callable:
    """
    Posterior of a Bayesian linear regression: p(β | X, y).
    
    Likelihood: y ~ N(Xβ, σ²I)  with σ=1 for simplicity
    Prior:      β ~ N(0, prior_sigma² I)
    
    This is a REAL use case — MCMC for Bayesian inference.
    True posterior is Gaussian (can verify analytically).
    
    x = [β_0, β_1, ..., β_p]
    """
    n, p = X.shape

    def log_prob(beta: np.ndarray) -> float:
        residuals  = y - X @ beta
        log_like   = -0.5 * np.dot(residuals, residuals)
        log_prior  = -0.5 * np.dot(beta, beta) / prior_sigma**2
        return log_like + log_prior
    
    # Analytical posterior for verification
    prior_prec  = np.eye(p) / prior_sigma**2
    post_cov    = np.linalg.inv(X.T @ X + prior_prec)
    post_mean   = post_cov @ X.T @ y
    
    log_prob.__name__ = f"Bayesian Linear Regression ({p}D)"
    log_prob.n_dims = p
    log_prob.true_mean = post_mean
    log_prob.true_cov  = post_cov
    return log_prob