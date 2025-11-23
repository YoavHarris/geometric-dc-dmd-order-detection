import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union, Sequence


def get_ar1_noise(
    shape: Tuple[int, int],
    total_variance: float,
    phi: float = 0.8,
    tol: float = 0.05,
    max_tries: int = 100,
    burn_in: int | None = None,  # None → 5×time constant
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """
    Zero-mean AR(1):  w[n] = phi*w[n-1] + eps[n],  eps ~ N(0,1).

    • Draws a longer sequence, discards a burn-in section to ensure
      stationarity, then applies rejection rescaling to match
      `total_variance` within the requested tolerance.
    """
    if not (-1 < phi < 1):
        raise ValueError("phi must be in (-1, 1) for stationarity")

    rng = rng or np.random.default_rng()
    ns, nt = shape
    theo_var = 1.0 / (1.0 - phi**2)
    base_scale = np.sqrt(total_variance / theo_var)

    # Choose burn-in length ≈ 5 time constants if not provided
    if burn_in is None:
        burn_in = max(50, int(5 / (1 - abs(phi))))

    ext_len = nt + burn_in

    for _ in range(max_tries):
        eps = rng.standard_normal((ns, ext_len))
        w = np.empty_like(eps)
        w[:, 0] = eps[:, 0]
        for t in range(1, ext_len):
            w[:, t] = phi * w[:, t - 1] + eps[:, t]

        noise = (w[:, burn_in:] * base_scale).astype(float)
        var = np.mean(noise**2)
        if abs(var - total_variance) / total_variance <= tol:
            return noise

    # fall back: return last draw even if slightly off
    return noise


def get_heteroscedastic_noise(
    shape: Tuple[int, int],
    total_variance: float = 1.0,
    sigma_min: float = 0.5,
    sigma_max: float = 1.5,
    rng: Optional[np.random.Generator] = None,
):
    """
    Gaussian noise with variance ramping smoothly from sigma_min
    to sigma_max.  Result re-scaled so overall Var = total_variance.
    """
    rng = rng or np.random.default_rng()
    ns, nt = shape
    t = np.linspace(0.0, 1.0, nt)
    sigma = sigma_min + (sigma_max - sigma_min) * t  # length-nt
    noise = (
        rng.standard_normal(size=shape).astype(np.float64) * sigma
    )  # broadcast on rows
    raw_var = np.mean(noise**2)
    return noise * np.sqrt(total_variance / raw_var)


def get_student_t_noise(
    shape: Tuple[int, int],
    df: float = 2.01,
    total_variance: float = 1.0,
    tol: float = 0.05,  # ±5 % variance window
    max_tries: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """
    Student-t noise whose *empirical* variance is within `tol`
    of `total_variance`, obtained by repeated draws *without*
    per-sample renormalisation (“rejection rescaling”).

    Notes
    -----
    • For df ≤ 2 the variance is infinite; we fall back to MAD-based scaling.
    • The function never tweaks individual realisations, so the heavy tails
      remain intact once a draw is accepted.
    """
    if df <= 0:
        raise ValueError("df must be positive")

    rng = rng or np.random.default_rng()

    if df <= 2:
        # same robust branch as before: MAD scaling (no rejection needed)
        MAD_TO_STD = 1.4826
        raw = rng.standard_t(df, size=shape)
        mad = np.median(np.abs(raw - np.median(raw))) * MAD_TO_STD
        return raw * (np.sqrt(total_variance) / mad)

    # df > 2  → analytic variance exists
    theo_var = df / (df - 2.0)
    base_scale = np.sqrt(total_variance / theo_var)

    for _ in range(max_tries):
        raw = rng.standard_t(df, size=shape)
        noise = raw * base_scale
        var = np.mean(noise**2)
        if abs(var - total_variance) / total_variance <= tol:
            return noise

    # fall back: return the last draw (variance slightly off)
    return noise


def get_bi_gaussian_noise(
    shape: Tuple[int, int],
    p: float = 0.1,  # fraction of "high-mean" component
    total_variance: float = 1.0,
    q: float = 3.0,  # separation (Δμ in units of σ)
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    rng = rng or np.random.default_rng()

    sigma = np.sqrt(total_variance / (1 + p * (1 - p) * q**2))
    mu1 = +(1 - p) * q * sigma
    mu2 = -p * q * sigma

    # mixture draw
    choice = rng.random(shape) < p
    base = rng.standard_normal(shape) * sigma
    noise = np.where(choice, base + mu1, base + mu2)
    return noise


class DMDDataGenerator:
    """
    Synthesizes DMD-style data with controlled eigenvalues, amplitudes, and noise.
    """

    def __init__(
        self,
        eigenvalue_magnitude: Union[float, Sequence[float]],
        frequency_separation: float,
        snr_db: float,
        eigenvalue_magnitude_spread: Optional[float] = None,
        rho_mode: str = "linspace",
        top_amplitude: float = 1.0,
        dt: float = 1.0,
        noise_mode: str = "gaussian",
        force_mode_orthogonality: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        self.rho_spec = eigenvalue_magnitude  # scalar or sequence
        self.rho_spread = eigenvalue_magnitude_spread  # None → 0
        self.rho_mode = rho_mode.lower()
        self.f_sep = frequency_separation
        self.snr_db = snr_db
        self.top_amplitude = top_amplitude
        self.dt = dt
        self.noise_mode = noise_mode
        self.force_mode_orthogonality = force_mode_orthogonality
        self.rng = np.random.default_rng(random_seed)  # single RNG for everything

    # ───────────────────── private helpers (all use self.rng) ──────────────────
    def _build_rhos(self, n_modes: int) -> NDArray[np.floating]:
        """
        Return a length-n_modes vector of eigen-value magnitudes.

        Cases
        -----
        1. rho_spec is scalar → all modes share the same rho..
           Optional spread (default None → 0) introduces heterogeneity,
           drawn either linspace or uniform as selected by rho_mode.
        2. rho_spec is a list / array → use it verbatim (spread must be None).

        Raises
        ------
        ValueError on size mismatch or invalid magnitude / spread.
        """
        # ── explicit list/array given → use directly ───────────────────
        if not np.isscalar(self.rho_spec):
            if self.rho_spread is not None:
                raise ValueError(
                    "When eigenvalue_magnitude is a list, "
                    "eigenvalue_magnitude_spread must be None"
                )
            rho_vec = np.asarray(self.rho_spec, dtype=np.float64)
            if rho_vec.size != n_modes:
                raise ValueError(
                    "Length of eigenvalue_magnitude "
                    f"({rho_vec.size}) ≠ n_modes ({n_modes})"
                )
            return rho_vec

        # ── scalar magnitude, possibly with spread ─────────────────────
        mu = float(self.rho_spec)
        if not (0.0 < mu <= 1.0):
            raise ValueError("eigenvalue_magnitude must be in (0, 1]")

        spread = 0.0 if self.rho_spread is None else float(self.rho_spread)

        if not 0 <= spread <= min(mu, 1.0 - mu):
            print("Warning: eigenvalue magnitude spread is out of range.")
            print("Clipping to ensure eigenvalue magnitudes in (0,1].")
            spread = np.clip(spread, 0.0, min(mu, 1.0 - mu))
        if spread == 0.0:
            return mu * np.ones(n_modes, dtype=np.float64)

        lo, hi = mu - spread, mu + spread
        if self.rho_mode == "linspace":
            rhos = np.linspace(lo, hi, n_modes, dtype=np.float64)
            return np.random.permutation(rhos)
        if self.rho_mode == "random":
            return self.rng.uniform(lo, hi, size=n_modes).astype(np.float64)
        raise ValueError("rho_mode must be 'random' or 'linspace'")

    def build_omegas(
        self,
        n_modes: int,
        f_min: float = 0.0,
        f_max: float = 0.2,
    ) -> NDArray[np.floating]:
        domega = 2 * np.pi * self.f_sep  # convert to radians
        omega_min = 2 * np.pi * f_min
        omega_max = 2 * np.pi * f_max

        if (n_modes - 1) * domega > omega_max - omega_min:
            raise ValueError(
                f"Cannot fit {n_modes} frequencies with separation {self.f_sep} in given range."
            )

        omega0_upper = omega_max - (n_modes - 1) * domega
        omega0 = self.rng.uniform(omega_min, omega0_upper)
        omegas = omega0 + np.arange(n_modes) * domega
        return omegas

    def _build_eigenvalues(self, n_modes: int) -> NDArray[np.complexfloating]:
        rho_vec = self._build_rhos(n_modes)
        omega_vec = self.build_omegas(
            n_modes,
        )
        eigenvalues = rho_vec * np.exp(1j * omega_vec)
        return eigenvalues

    def _build_unit_modes(
        self, n_spatial: int, n_modes: int
    ) -> NDArray[np.complexfloating]:
        M_real = self.rng.standard_normal((n_spatial, n_modes))
        M_imag = self.rng.standard_normal((n_spatial, n_modes))
        M = M_real + 1j * M_imag

        if self.force_mode_orthogonality:
            Q, _ = np.linalg.qr(M)  # QR gives orthonormal columns
            return Q[:, :n_modes]

        norms = np.linalg.norm(M, axis=0, keepdims=True)
        return M / norms

    def _build_amplitudes(
        self, n_modes: int, log_scale: bool = False
    ) -> NDArray[np.floating]:
        if n_modes == 1:
            return np.array([1.0])

        base = np.linspace(0, 1, n_modes)
        quarter = 0.25 / (n_modes - 1)
        jitter = self.rng.uniform(-quarter, quarter, size=n_modes)
        jitter[0] = 0.0
        jitter[-1] = self.rng.uniform(-quarter, 0.0)
        beta = np.clip(base + jitter, 0.0, 1.0)

        if log_scale:
            # Map to log10 scale from 1 to top_amplitude
            amps = 10 ** (np.log10(self.top_amplitude) * beta)
        else:
            # Linear scale from 1 to top_amplitude
            amps = 1.0 + (self.top_amplitude - 1.0) * beta

        return np.random.permutation(amps)

    @staticmethod
    def _build_time_dynamics(
        eigs: NDArray[np.complexfloating], n_timesteps: int
    ) -> NDArray[np.complexfloating]:
        t = np.arange(n_timesteps)
        return np.stack([eig**t for eig in eigs], axis=0)

    # ───────────────────────────── noise helper ───────────────────────────────
    def _add_noise(
        self, X_clean: NDArray[np.complexfloating]
    ) -> NDArray[np.complexfloating]:
        power = np.mean(np.abs(X_clean) ** 2)
        var = power / (10 ** (self.snr_db / 10))
        ns, nt = X_clean.shape

        if self.noise_mode == "gaussian":
            real = self.rng.standard_normal((ns, nt))
            imag = self.rng.standard_normal((ns, nt))
            return np.sqrt(var / 2) * (real + 1j * imag)

        if self.noise_mode == "bi_gaussian":
            real = get_bi_gaussian_noise(
                (ns, nt), p=0.85, total_variance=var / 2, rng=self.rng
            )
            imag = get_bi_gaussian_noise(
                (ns, nt), p=0.85, total_variance=var / 2, rng=self.rng
            )
            return (real + 1j * imag).astype(np.complex128)

        if self.noise_mode == "uniform":
            real = self.rng.uniform(-1, 1, (ns, nt))
            imag = self.rng.uniform(-1, 1, (ns, nt))
            real *= np.sqrt(var / 2 / np.mean(real**2))
            imag *= np.sqrt(var / 2 / np.mean(imag**2))
            return real + 1j * imag

        if self.noise_mode == "student_t":
            real = get_student_t_noise((ns, nt), total_variance=var / 2, rng=self.rng)
            imag = get_student_t_noise((ns, nt), total_variance=var / 2, rng=self.rng)
            return (real + 1j * imag).astype(np.complex128)

        if self.noise_mode == "ar1":
            real_noise = get_ar1_noise(
                (ns, nt), total_variance=var / 2, phi=0.8, rng=self.rng
            )
            imag_noise = get_ar1_noise(
                (ns, nt), total_variance=var / 2, phi=0.8, rng=self.rng
            )
            return (real_noise + 1j * imag_noise).astype(np.complex128)

        if self.noise_mode == "hetero":
            real_noise = get_heteroscedastic_noise(
                (ns, nt),
                total_variance=var / 2,
                sigma_min=0.5,
                sigma_max=1.5,
                rng=self.rng,
            )
            imag_noise = get_heteroscedastic_noise(
                (ns, nt),
                total_variance=var / 2,
                sigma_min=0.5,
                sigma_max=1.5,
                rng=self.rng,
            )
            return (real_noise + 1j * imag_noise).astype(np.complex128)

        raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

    # ─────────────────────────── public interface ─────────────────────────────
    def generate(self, n_spatial: int, n_timesteps: int, n_modes: int) -> Tuple[
        NDArray[np.complexfloating],
        NDArray[np.complexfloating],
        NDArray[np.complexfloating],
        NDArray[np.complexfloating],
        NDArray[np.floating],
    ]:
        """
        Generate synthetic DMD data.

        Returns:
            X -- noisy data       (n_spatial, n_timesteps)
            X_clean -- clean data (n_spatial, n_timesteps)
            disc_time_eigs -- eigenvalues (n_modes,)
            modes -- modes * amplitudes  (n_spatial, n_modes)
            amplitudes -- (n_modes,)
        """
        eigs = self._build_eigenvalues(n_modes)
        unit_modes = self._build_unit_modes(n_spatial, n_modes)
        amps = self._build_amplitudes(n_modes)
        modes = unit_modes * amps[None, :]
        dynamics = self._build_time_dynamics(eigs, n_timesteps)
        X_clean = modes @ dynamics
        X = X_clean + self._add_noise(X_clean)
        return X, X_clean, eigs, modes, amps
