"""
MCMC Diagnostics
=================
Research-grade convergence diagnostics.

The three things every paper checks:
  1. R-hat  — did chains converge to the SAME distribution?
  2. Trace plots — does the chain look like "a hairy caterpillar"?
  3. Autocorrelation — how correlated are consecutive samples?

Reference: Vehtari et al. (2021) "Rank-normalization, folding, and 
localization: An improved R-hat for assessing convergence of MCMC"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional
from sampler import MCMCResult


# ─── R-hat (Gelman-Rubin Statistic) ──────────────────────────────────────────

def r_hat(result: MCMCResult) -> np.ndarray:
    """
    Rank-normalized split R-hat (Vehtari 2021).
    
    R-hat ≈ 1.00 → converged
    R-hat > 1.01 → run longer or check your model
    R-hat > 1.10 → definitely not converged
    
    The "split" part: each chain is split in half, doubling the number of chains.
    This catches chains that haven't mixed within themselves.
    
    The "rank-normalized" part: makes it robust to non-normal posteriors.
    """
    burnin = result.n_samples // 2
    draws  = result.samples[:, burnin:, :]   # (chains, draws, dims)
    
    # Split each chain in half → 2*n_chains pseudo-chains
    n_chains, n_draws, n_dims = draws.shape
    half = n_draws // 2
    split = np.concatenate([draws[:, :half, :], draws[:, half:, :]], axis=0)
    # split shape: (2*n_chains, half, n_dims)
    
    rhat_vals = np.zeros(n_dims)
    for d in range(n_dims):
        rhat_vals[d] = _rhat_scalar(split[:, :, d])
    return rhat_vals


def _rhat_scalar(chains: np.ndarray) -> float:
    """R-hat for a single parameter. chains shape: (n_chains, n_draws)"""
    # Rank-normalize
    chains = _rank_normalize(chains)
    
    m, n = chains.shape    # chains, draws
    chain_means = chains.mean(axis=1)
    grand_mean  = chains.mean()
    
    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)
    # Within-chain variance
    W = chains.var(axis=1, ddof=1).mean()
    
    var_hat = ((n - 1) / n) * W + B / n
    return np.sqrt(var_hat / W) if W > 0 else np.nan


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map values to normal scores via ranks — makes R-hat robust."""
    from scipy.stats import rankdata, norm
    flat  = x.flatten()
    ranks = rankdata(flat)
    z     = norm.ppf((ranks - 0.375) / (len(ranks) + 0.25))
    return z.reshape(x.shape)


# ─── Autocorrelation ─────────────────────────────────────────────────────────

def autocorrelation(result: MCMCResult, max_lag: int = 100) -> np.ndarray:
    """
    Mean autocorrelation across chains per dimension.
    Returns shape: (max_lag, n_dims)
    
    Well-mixed chain: drops to ~0 within a few lags.
    Poorly-mixed:     stays high → you need more samples or better step size.
    """
    burnin  = result.n_samples // 2
    draws   = result.samples[:, burnin:, :]
    n_draws = draws.shape[1]
    max_lag = min(max_lag, n_draws // 2)
    
    acf_all = np.zeros((result.n_chains, max_lag, result.n_dims))
    
    for c in range(result.n_chains):
        for d in range(result.n_dims):
            acf_all[c, :, d] = _acf(draws[c, :, d], max_lag)
    
    return acf_all.mean(axis=0)   # average over chains


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation function via FFT."""
    n = len(x)
    x = x - x.mean()
    fft   = np.fft.rfft(x, n=2 * n)
    power = fft * np.conj(fft)
    acf   = np.fft.irfft(power)[:max_lag]
    return acf / acf[0]


# ─── Plotting ─────────────────────────────────────────────────────────────────

# Dark research aesthetic — like a real paper but actually readable
COLORS = ['#4FC3F7', '#81C784', '#FFB74D', '#F06292', '#CE93D8', '#80DEEA']
BG     = '#0D1117'
PANEL  = '#161B22'
TEXT   = '#E6EDF3'
GRID   = '#21262D'


def plot_diagnostics(
    result: MCMCResult,
    target_name: str = "Target Distribution",
    true_values: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dim_names: Optional[list] = None,
):
    """
    Full diagnostic dashboard. One figure with 4 panels per dimension:
      - Trace plot (time series of chain)
      - Marginal posterior histogram
      - Autocorrelation function
      - Running mean (convergence check)
    Plus a summary stats panel.
    """
    n_dims = min(result.n_dims, 4)   # cap at 4 dims for readability
    if dim_names is None:
        dim_names = [f"θ_{i+1}" for i in range(n_dims)]
    
    rhat_vals = r_hat(result)
    acf_vals  = autocorrelation(result, max_lag=80)
    ess_vals  = result.effective_sample_size

    fig = plt.figure(figsize=(20, 5 * n_dims + 3), facecolor=BG)
    fig.suptitle(
        f"MCMC Diagnostics — {target_name}",
        color=TEXT, fontsize=18, fontweight='bold', y=0.98
    )

    outer = gridspec.GridSpec(n_dims + 1, 1, figure=fig, hspace=0.5)

    # ── Summary stats row ────────────────────────────────────────────────────
    ax_sum = fig.add_subplot(outer[0])
    ax_sum.set_facecolor(PANEL)
    ax_sum.set_xlim(0, 1)
    ax_sum.set_ylim(0, 1)
    ax_sum.axis('off')

    accept_rates = result.acceptance_rate
    stats_text = (
        f"  Chains: {result.n_chains}   "
        f"Samples/chain: {result.n_samples:,}   "
        f"Step size: {result.step_size:.4f}   "
        f"Time: {result.elapsed_time:.2f}s   "
        f"Acceptance: " + "  ".join(
            [f"chain {i+1}: {r:.1%}" for i, r in enumerate(accept_rates)]
        )
    )
    ax_sum.text(
        0.5, 0.6, stats_text,
        color=TEXT, fontsize=10, ha='center', va='center',
        transform=ax_sum.transAxes
    )

    # R-hat summary
    rhat_str = "  R-hat: " + "   ".join(
        [f"{dim_names[d]}: {rhat_vals[d]:.4f} {'✓' if rhat_vals[d] < 1.01 else '⚠'}"
         for d in range(n_dims)]
    )
    ess_str = "  ESS:   " + "   ".join(
        [f"{dim_names[d]}: {ess_vals[d]:.0f}" for d in range(n_dims)]
    )
    ax_sum.text(
        0.5, 0.35, rhat_str, color='#81C784', fontsize=10,
        ha='center', va='center', transform=ax_sum.transAxes, family='monospace'
    )
    ax_sum.text(
        0.5, 0.1, ess_str, color='#4FC3F7', fontsize=10,
        ha='center', va='center', transform=ax_sum.transAxes, family='monospace'
    )

    # ── Per-dimension rows ───────────────────────────────────────────────────
    burnin = result.n_samples // 2

    for d in range(n_dims):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[d + 1], wspace=0.35
        )

        # 1. Trace plot
        ax_trace = fig.add_subplot(inner[0])
        _style_ax(ax_trace, f"Trace — {dim_names[d]}", "Iteration", "Value")
        for c in range(result.n_chains):
            ax_trace.plot(
                result.samples[c, :, d],
                color=COLORS[c % len(COLORS)],
                alpha=0.6, lw=0.5
            )
        ax_trace.axvline(burnin, color='#FF6B6B', lw=1.5, ls='--', alpha=0.8, label='burn-in')
        ax_trace.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

        # 2. Posterior histogram
        ax_hist = fig.add_subplot(inner[1])
        _style_ax(ax_hist, f"Posterior — {dim_names[d]}", "Value", "Density")
        post_samples = result.flat_samples[:, d]
        ax_hist.hist(
            post_samples, bins=60, density=True,
            color=COLORS[d % len(COLORS)], alpha=0.75, edgecolor='none'
        )
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(post_samples)
        xs  = np.linspace(post_samples.min(), post_samples.max(), 300)
        ax_hist.plot(xs, kde(xs), color='white', lw=1.5, alpha=0.9)
        if true_values is not None and d < len(true_values):
            ax_hist.axvline(
                true_values[d], color='#FF6B6B', lw=2, ls='--',
                label=f'true={true_values[d]:.2f}'
            )
            ax_hist.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
        mean_val = post_samples.mean()
        std_val  = post_samples.std()
        ax_hist.set_title(
            f"Posterior — {dim_names[d]}\nμ={mean_val:.3f}  σ={std_val:.3f}",
            color=TEXT, fontsize=9, pad=4
        )

        # 3. Autocorrelation
        ax_acf = fig.add_subplot(inner[2])
        _style_ax(ax_acf, f"Autocorrelation — {dim_names[d]}", "Lag", "ACF")
        lags = np.arange(acf_vals.shape[0])
        ax_acf.bar(lags, acf_vals[:, d], color=COLORS[d % len(COLORS)], alpha=0.7, width=0.8)
        ax_acf.axhline(0, color=TEXT, lw=0.5)
        # 95% confidence band for white noise
        ci = 1.96 / np.sqrt(result.n_samples // 2)
        ax_acf.axhline(ci,  color='#FF6B6B', lw=1, ls='--', alpha=0.6)
        ax_acf.axhline(-ci, color='#FF6B6B', lw=1, ls='--', alpha=0.6)
        ax_acf.set_ylim(-0.3, 1.05)

        # 4. Running mean
        ax_run = fig.add_subplot(inner[3])
        _style_ax(ax_run, f"Running Mean — {dim_names[d]}", "Iteration", "Cumulative Mean")
        for c in range(result.n_chains):
            chain = result.samples[c, burnin:, d]
            running_mean = np.cumsum(chain) / (np.arange(len(chain)) + 1)
            ax_run.plot(
                running_mean, color=COLORS[c % len(COLORS)], alpha=0.8, lw=1
            )
        if true_values is not None and d < len(true_values):
            ax_run.axhline(true_values[d], color='#FF6B6B', lw=1.5, ls='--')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.set_title(title, color=TEXT, fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=7)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=7)
    ax.grid(color=GRID, lw=0.5, alpha=0.5)


def plot_joint(result: MCMCResult, dim_x: int = 0, dim_y: int = 1,
               target_name: str = "", save_path: Optional[str] = None):
    """2D joint posterior — pairs plot for 2D targets."""
    samples = result.flat_samples

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    fig.suptitle(f"Joint Posterior — {target_name}", color=TEXT, fontsize=14)

    # Scatter
    ax = axes[0]
    ax.set_facecolor(PANEL)
    ax.scatter(
        samples[:, dim_x], samples[:, dim_y],
        alpha=0.15, s=3, color='#4FC3F7'
    )
    ax.set_xlabel(f"θ_{dim_x+1}", color=TEXT)
    ax.set_ylabel(f"θ_{dim_y+1}", color=TEXT)
    ax.set_title("Scatter", color=TEXT)
    ax.tick_params(colors=TEXT)

    # 2D KDE density
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples[:, [dim_x, dim_y]].T)
    x_range = np.linspace(samples[:, dim_x].min(), samples[:, dim_x].max(), 100)
    y_range = np.linspace(samples[:, dim_y].min(), samples[:, dim_y].max(), 100)
    XX, YY = np.meshgrid(x_range, y_range)
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

    custom_cmap = LinearSegmentedColormap.from_list(
        'mcmc', ['#0D1117', '#1565C0', '#4FC3F7', '#E0F7FA']
    )
    ax2.contourf(XX, YY, Z, levels=20, cmap=custom_cmap)
    ax2.contour(XX, YY, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel(f"θ_{dim_x+1}", color=TEXT)
    ax2.set_ylabel(f"θ_{dim_y+1}", color=TEXT)
    ax2.set_title("Density (KDE)", color=TEXT)
    ax2.tick_params(colors=TEXT)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    else:
        plt.show()
    return fig