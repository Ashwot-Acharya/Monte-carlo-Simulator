import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
import time


@dataclass
class MCMCResult:
    """
    Container for all MCMC output.
    Mirrors what you'd get from Stan/PyMC — this is the research standard.
    """
    samples: np.ndarray          # shape (n_chains, n_samples, n_dims)
    log_probs: np.ndarray        # shape (n_chains, n_samples)
    accepted: np.ndarray         # shape (n_chains, n_samples) — bool
    n_chains: int
    n_samples: int
    n_dims: int
    elapsed_time: float
    step_size: float

    @property
    def acceptance_rate(self) -> np.ndarray:
        """Per-chain acceptance rate. Should be ~0.234 for optimal MH in high dims."""
        return self.accepted.mean(axis=1)

    @property
    def flat_samples(self) -> np.ndarray:
        """All chains concatenated after burn-in (first 50%). Shape: (total, dims)"""
        burnin = self.n_samples // 2
        return self.samples[:, burnin:, :].reshape(-1, self.n_dims)

    @property
    def effective_sample_size(self) -> np.ndarray:
        """
        ESS per dimension — how many *independent* samples you effectively have.
        Autocorrelated chains give ESS << n_samples.
        Uses Geyer's initial monotone sequence estimator.
        """
        burnin = self.n_samples // 2
        post = self.samples[:, burnin:, :]        # (chains, draws, dims)
        ess = np.zeros(self.n_dims)
        for d in range(self.n_dims):
            chain_ess = []
            for c in range(self.n_chains):
                chain_ess.append(_ess_geyer(post[c, :, d]))
            ess[d] = np.sum(chain_ess)
        return ess


def _ess_geyer(x: np.ndarray) -> float:
    """
    Geyer's initial positive sequence ESS estimate for a single chain.
    More robust than naive 1/(1+2*sum(rho)) because it stops summing
    autocorrelations when they become negative.
    """
    n = len(x)
    # Normalized autocorrelation via FFT — O(n log n) instead of O(n²)
    x_centered = x - x.mean()
    fft = np.fft.rfft(x_centered, n=2 * n)
    acf_full = np.fft.irfft(fft * np.conj(fft))[:n]
    acf = acf_full / acf_full[0]

    # Geyer: sum pairs Gamma_k = rho(2k) + rho(2k+1) while Gamma_k > 0
    rho_sum = 0.0
    for k in range(1, n // 2):
        gamma = acf[2 * k] + acf[2 * k + 1]
        if gamma <= 0:
            break
        rho_sum += gamma

    tau = -1 + 2 * rho_sum   # integrated autocorrelation time
    return n / max(tau, 1.0)


class MetropolisHastings:
    """
    General-purpose Metropolis-Hastings sampler.

    Usage:
        sampler = MetropolisHastings(log_prob_fn, n_dims=2)
        result  = sampler.run(n_samples=5000, n_chains=4)

    The only thing you need to provide is log_prob_fn(x) → scalar.
    Everything else is automatic.
    """

    def __init__(
        self,
        log_prob: Callable[[np.ndarray], float],
        n_dims: int,
        step_size: float = 0.5,
        proposal: str = "gaussian",          # "gaussian" | "uniform"
        adapt_step_size: bool = True,        # tune step_size during warmup
        target_acceptance: float = 0.234,    # Roberts et al. 1997 optimal rate
        seed: Optional[int] = None,
    ):
        self.log_prob = log_prob
        self.n_dims = n_dims
        self.step_size = step_size
        self.proposal = proposal
        self.adapt_step_size = adapt_step_size
        self.target_acceptance = target_acceptance
        self.rng = np.random.default_rng(seed)

    def _propose(self, x: np.ndarray) -> np.ndarray:
        if self.proposal == "gaussian":
            return x + self.rng.normal(0, self.step_size, size=self.n_dims)
        elif self.proposal == "uniform":
            return x + self.rng.uniform(-self.step_size, self.step_size, size=self.n_dims)
        else:
            raise ValueError(f"Unknown proposal: {self.proposal}")

    def _run_chain(
        self,
        n_samples: int,
        init: np.ndarray,
        warmup_frac: float = 0.5,
    ):
        """Run a single chain. Returns (samples, log_probs, accepted)."""
        n_warmup = int(n_samples * warmup_frac)
        samples  = np.zeros((n_samples, self.n_dims))
        lps      = np.zeros(n_samples)
        accepted = np.zeros(n_samples, dtype=bool)

        x      = init.copy()
        lp_x   = self.log_prob(x)
        step   = self.step_size

        # Dual averaging for step size adaptation (Nesterov 2009, used in Stan)
        accept_window = []

        for i in range(n_samples):
            x_prop  = self._propose(x) if self.proposal == "gaussian" else (
                x + self.rng.uniform(-step, step, size=self.n_dims))
            lp_prop = self.log_prob(x_prop)

            log_alpha = lp_prop - lp_x
            if np.log(self.rng.uniform()) < log_alpha:
                x, lp_x = x_prop, lp_prop
                accepted[i] = True

            samples[i] = x
            lps[i]     = lp_x

            # Adapt step size during warmup only
            if self.adapt_step_size and i < n_warmup:
                accept_window.append(accepted[i])
                if len(accept_window) == 50:
                    rate = np.mean(accept_window)
                    # Robbins-Monro style update
                    step *= np.exp(rate - self.target_acceptance)
                    step  = np.clip(step, 1e-4, 10.0)
                    accept_window = []

        self.step_size = step   # store adapted value
        return samples, lps, accepted

    def run(
        self,
        n_samples: int = 5000,
        n_chains: int = 4,
        init: Optional[np.ndarray] = None,
        init_spread: float = 2.0,
    ) -> MCMCResult:
        """
        Run n_chains parallel chains.
        Chains start from random dispersed positions (important for R-hat diagnosis).
        """
        t0 = time.time()

        all_samples  = np.zeros((n_chains, n_samples, self.n_dims))
        all_lps      = np.zeros((n_chains, n_samples))
        all_accepted = np.zeros((n_chains, n_samples), dtype=bool)

        for c in range(n_chains):
            if init is not None:
                x0 = init[c] if init.ndim == 2 else init
            else:
                x0 = self.rng.normal(0, init_spread, size=self.n_dims)

            s, lp, acc = self._run_chain(n_samples, x0)
            all_samples[c]  = s
            all_lps[c]      = lp
            all_accepted[c] = acc

        return MCMCResult(
            samples      = all_samples,
            log_probs    = all_lps,
            accepted     = all_accepted,
            n_chains     = n_chains,
            n_samples    = n_samples,
            n_dims       = self.n_dims,
            elapsed_time = time.time() - t0,
            step_size    = self.step_size,
        )