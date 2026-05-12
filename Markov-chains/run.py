"""
MCMC Demo Runner
=================
Run all experiments and generate diagnostic plots.

Usage:
    python run_experiments.py
    python run_experiments.py --target banana
    python run_experiments.py --target all
"""

import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))   # adds the script's own directory
from sampler import MetropolisHastings
from plots import plot_diagnostics, plot_joint
from distributions import (
    gaussian_2d, banana, donut, mixture_gaussians, funnel, bayesian_linear_regression
)


def run_gaussian():
    """Baseline experiment — simple 2D correlated Gaussian."""
    print("\n── Experiment 1: 2D Correlated Gaussian ──")
    target = gaussian_2d()
    
    sampler = MetropolisHastings(
        log_prob   = target,
        n_dims     = 2,
        step_size  = 0.5,
        adapt_step_size = True,
        seed       = 42,
    )
    result = sampler.run(n_samples=5000, n_chains=4)
    
    _print_summary(result, target.__name__)
    plot_diagnostics(
        result,
        target_name  = target.__name__,
        true_values  = target.true_mean,
        dim_names    = ['μ₁', 'μ₂'],
        save_path    = 'output/gaussian_diagnostics.png'
    )
    plot_joint(result, target_name=target.__name__,
               save_path='output/gaussian_joint.png')
    return result


def run_banana():
    """Challenging curved posterior — tests mixing."""
    print("\n── Experiment 2: Banana Distribution ──")
    target = banana(b=0.1)
    
    sampler = MetropolisHastings(
        log_prob   = target,
        n_dims     = 2,
        step_size  = 1.0,
        adapt_step_size = True,
        seed       = 42,
    )
    result = sampler.run(n_samples=8000, n_chains=4)
    
    _print_summary(result, target.__name__)
    plot_diagnostics(
        result,
        target_name = target.__name__,
        dim_names   = ['x₁', 'x₂'],
        save_path   = 'output/banana_diagnostics.png'
    )
    plot_joint(result, target_name=target.__name__,
               save_path='output/banana_joint.png')
    return result


def run_donut():
    """Ring distribution — tests global exploration."""
    print("\n── Experiment 3: Donut Distribution ──")
    target = donut(radius=3.0, sigma=0.5)
    
    sampler = MetropolisHastings(
        log_prob   = target,
        n_dims     = 2,
        step_size  = 0.3,
        adapt_step_size = True,
        seed       = 42,
    )
    result = sampler.run(n_samples=10000, n_chains=4)
    
    _print_summary(result, target.__name__)
    plot_diagnostics(
        result,
        target_name = target.__name__,
        dim_names   = ['x₁', 'x₂'],
        save_path   = 'output/donut_diagnostics.png'
    )
    plot_joint(result, target_name=target.__name__,
               save_path='output/donut_joint.png')
    return result


def run_mixture():
    """Multimodal — the hardest test for random-walk MH."""
    print("\n── Experiment 4: 3-Mode Gaussian Mixture ──")
    target = mixture_gaussians()
    
    sampler = MetropolisHastings(
        log_prob   = target,
        n_dims     = 2,
        step_size  = 1.5,           # needs larger steps to hop between modes
        adapt_step_size = True,
        seed       = 42,
    )
    result = sampler.run(n_samples=15000, n_chains=4, init_spread=5.0)
    
    _print_summary(result, target.__name__)
    plot_diagnostics(
        result,
        target_name = target.__name__,
        dim_names   = ['x₁', 'x₂'],
        save_path   = 'output/mixture_diagnostics.png'
    )
    plot_joint(result, target_name=target.__name__,
               save_path='output/mixture_joint.png')
    return result


def run_bayesian_regression():
    """Real Bayesian inference problem."""
    print("\n── Experiment 5: Bayesian Linear Regression ──")
    rng = np.random.default_rng(0)
    n, p = 100, 3
    X      = np.column_stack([np.ones(n), rng.normal(size=(n, p-1))])
    true_beta = np.array([1.5, -2.0, 0.8])
    y      = X @ true_beta + rng.normal(scale=0.5, size=n)
    
    target = bayesian_linear_regression(X, y)
    
    sampler = MetropolisHastings(
        log_prob   = target,
        n_dims     = p,
        step_size  = 0.1,
        adapt_step_size = True,
        seed       = 42,
    )
    result = sampler.run(n_samples=6000, n_chains=4)
    
    _print_summary(result, target.__name__)
    print(f"  True β:     {true_beta}")
    print(f"  Posterior μ:{target.true_mean.round(3)}  (analytical)")
    print(f"  MCMC mean:  {result.flat_samples.mean(axis=0).round(3)}")
    
    plot_diagnostics(
        result,
        target_name  = target.__name__,
        true_values  = target.true_mean,
        dim_names    = ['β₀', 'β₁', 'β₂'],
        save_path    = 'output/regression_diagnostics.png'
    )
    return result


def _print_summary(result, name):
    from plots import r_hat
    rhat = r_hat(result)
    ess  = result.effective_sample_size
    print(f"  Target:      {name}")
    print(f"  Time:        {result.elapsed_time:.2f}s")
    print(f"  Acceptance:  {result.acceptance_rate.mean():.1%} (target: 23.4%)")
    print(f"  R-hat:       {rhat}  (< 1.01 = converged)")
    print(f"  ESS:         {ess.round(0)}  (want >> 100)")
    convergence = "✓ CONVERGED" if all(rhat < 1.01) else "⚠ NOT CONVERGED"
    print(f"  Status:      {convergence}")


EXPERIMENTS = {
    'gaussian':    run_gaussian,
    'banana':      run_banana,
    'donut':       run_donut,
    'mixture':     run_mixture,
    'regression':  run_bayesian_regression,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMC Experiment Runner")
    parser.add_argument(
        '--target', default='gaussian',
        choices=list(EXPERIMENTS.keys()) + ['all'],
        help="Which experiment to run"
    )
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    if args.target == 'all':
        for name, fn in EXPERIMENTS.items():
            fn()
    else:
        EXPERIMENTS[args.target]()

    print("\nDone. Check output/ for plots.")