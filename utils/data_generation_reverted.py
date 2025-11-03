import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union


def get_ar1_noise(shape, total_variance, phi=0.8, rng=None):
    """
    Zero-mean AR(1) noise:  w(n) = phi*w(n-1) + eps(n)
    eps ~ N(0,1).  Returned array has Var = total_variance.
    """
    rng = rng or np.random.default_rng()
    ns, nt = shape
    eps = rng.standard_normal(size=shape)
    noise = np.empty(shape, dtype=float)
    noise[:, 0] = eps[:, 0]
    for t in range(1, nt):
        noise[:, t] = phi * noise[:, t - 1] + eps[:, t]
    raw_var = 1.0 / (1.0 - phi**2)  # stationary variance of AR(1)
    return noise * np.sqrt(total_variance / raw_var)


def get_heteroscedastic_noise(
    shape, total_variance, sigma_min=0.5, sigma_max=1.5, rng=None
):
    """
    Gaussian noise with variance ramping smoothly from sigma_min
    to sigma_max.  Result re-scaled so overall Var = total_variance.
    """
    rng = rng or np.random.default_rng()
    ns, nt = shape
    t = np.linspace(0.0, 1.0, nt)
    sigma = sigma_min + (sigma_max - sigma_min) * t  # length-nt
    noise = rng.standard_normal(size=shape) * sigma  # broadcast on rows
    raw_var = np.mean(noise**2)
    return noise * np.sqrt(total_variance / raw_var)


def get_student_t_noise(
    shape,
    df: float = 3.0,
    total_variance: float = 1.0,
    rng: Union[np.random.Generator, None] = None,
) -> np.ndarray:
    if df <= 2:
        raise ValueError("Student-t variance exists only for df > 2")
    rng = rng or np.random.default_rng()

    raw = rng.standard_t(df, size=shape)  # E[raw] = 0
    raw_var = df / (df - 2)  # Var[raw]
    return raw * np.sqrt(total_variance / raw_var)


def get_bi_gaussian_noise(
    shape: Tuple[int, int],
    p: float,
    total_variance: float,
    q: float = 1.0,
    rng: Union[np.random.Generator, None] = None,
) -> NDArray[np.floating]:
    """
    Zero-mean bi-Gaussian noise whose overall variance is `total_variance`.
    """
    rng = rng or np.random.default_rng()
    sigma = np.sqrt(
        total_variance / (1 + p * q**2 + (1 - p) * (p**2 * q**2) / (1 - p) ** 2)
    )
    a = q * sigma
    b = -(p / (1 - p)) * a

    Y = rng.standard_normal(shape, dtype=np.floating) * sigma
    choice = rng.random(shape) < p
    return np.where(choice, Y + a, Y + b).astype(np.floating)


class DMDDataGenerator:
    """
    Synthesizes DMD-style data with controlled eigenvalues, amplitudes, and noise.
    """

    def __init__(
        self,
        eigenvalue_magnitude: float,
        frequency_separation: float,
        snr_db: float,
        top_amplitude_exponent: float = 0.5,
        dt: float = 1.0,
        noise_mode: str = "gaussian",
        random_seed: Optional[int] = None,
    ) -> None:
        self.rho = eigenvalue_magnitude
        self.dtheta = frequency_separation
        self.snr_db = snr_db
        self.p = top_amplitude_exponent
        self.dt = dt
        self.noise_mode = noise_mode
        self.rng = np.random.default_rng(random_seed)  # single RNG for everything

    # ───────────────────── private helpers (all use self.rng) ──────────────────
    def _build_eigenvalues(self, n_modes: int) -> NDArray[np.complexfloating]:
        # ---- handle flexible magnitude -------------------------
        if np.isscalar(self.rho):
            rho_vec = float(self.rho) * np.ones(n_modes, dtype=np.floating)
        else:
            rho_vec = np.asarray(self.rho, dtype=np.floating)
            if rho_vec.size != n_modes:
                raise ValueError(
                    f"eigenvalue_magnitude length {rho_vec.size} "
                    f"does not match n_modes={n_modes}"
                )

        # ---- build evenly separated angles ---------------------
        if (n_modes - 1) * self.dtheta > 2 * np.pi:
            raise ValueError("frequency_separation too large for n_modes")
        max_off = 2 * np.pi - (n_modes - 1) * self.dtheta
        offset = self.rng.uniform(0, max_off)
        theta = offset + np.arange(n_modes) * self.dtheta
        return rho_vec * np.exp(1j * theta * self.dt)

    def _build_unit_modes(
        self, n_spatial: int, n_modes: int
    ) -> NDArray[np.complexfloating]:
        M_real = self.rng.standard_normal((n_spatial, n_modes))
        M_imag = self.rng.standard_normal((n_spatial, n_modes))
        M = M_real + 1j * M_imag
        norms = np.linalg.norm(M, axis=0, keepdims=True)
        return M / norms

    def _build_amplitudes(self, n_modes: int) -> NDArray[np.floating]:
        if n_modes == 1:
            return np.ones(1, np.floating)
        base = np.linspace(0, 1, n_modes)
        quarter = 0.25 / (n_modes - 1)
        jitter = self.rng.uniform(-quarter, quarter, size=n_modes)
        jitter[0] = 0.0
        jitter[-1] = self.rng.uniform(-quarter, 0)
        beta = np.clip(base + jitter, 0.0, 1.0)
        return 10 ** (self.p * beta)

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
            return (real + 1j * imag).astype(np.complexfloating)

        if self.noise_mode == "uniform":
            real = self.rng.uniform(-1, 1, (ns, nt))
            imag = self.rng.uniform(-1, 1, (ns, nt))
            real *= np.sqrt(var / 2 / np.mean(real**2))
            imag *= np.sqrt(var / 2 / np.mean(imag**2))
            return real + 1j * imag

        if self.noise_mode == "student_t":
            real = get_student_t_noise(
                (ns, nt), df=3.0, total_variance=var / 2, rng=self.rng
            )
            imag = get_student_t_noise(
                (ns, nt), df=3.0, total_variance=var / 2, rng=self.rng
            )
            return (real + 1j * imag).astype(np.complexfloating)

        if self.noise_mode == "ar1":
            real_noise = get_ar1_noise((ns, nt), total_variance=var / 2, phi=0.8)
            imag_noise = get_ar1_noise((ns, nt), total_variance=var / 2, phi=0.8)
            return (real_noise + 1j * imag_noise).astype(np.complexfloating)

        if self.noise_mode == "hetero":
            real_noise = get_heteroscedastic_noise(
                (ns, nt), total_variance=var / 2, sigma_min=0.5, sigma_max=1.5
            )
            imag_noise = get_heteroscedastic_noise(
                (ns, nt), total_variance=var / 2, sigma_min=0.5, sigma_max=1.5
            )
            return (real_noise + 1j * imag_noise).astype(np.complexfloating)

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
