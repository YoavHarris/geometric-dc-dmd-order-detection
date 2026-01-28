import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt


def matrix_imshow(mat, title=None, xlabel=None, ylabel=None):
    if np.iscomplexobj(mat):
        matrices = [mat.real, mat.imag]
        subtitles = ["Real", "Imag"]
    else:
        matrices = [mat]
        subtitles = [None]

    # Compute shared color limits
    vmin = min(np.min(m) for m in matrices)
    vmax = max(np.max(m) for m in matrices)

    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4))

    if len(matrices) == 1:
        axes = [axes]

    ims = []
    for ax, m, st in zip(axes, matrices, subtitles):
        im = ax.imshow(m, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
        ims.append(im)
        if st:
            ax.set_title(st)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    # Shared colorbar
    fig.colorbar(ims[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.04)

    if title:
        fig.suptitle(title)
    plt.show()


def plot_power_spectrum(
    aggregated_power: NDArray[np.floating], max_power_bins: NDArray[np.int_]
) -> None:
    """
    Plot aggregated power for each mode with vertical lines for peak bins.
    """
    _, L = aggregated_power.shape
    plt.figure()
    for i, row in enumerate(aggregated_power):
        plt.plot(range(L), row, alpha=0.5, label=f"Mode {i}")
        plt.axvline(x=max_power_bins[i].item(), linestyle="--", alpha=0.5)
    plt.xlabel("Frequency Bins")
    plt.ylabel("Power")
    plt.title("Aggregated Power Spectrum (with Theoretical Peaks)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_matrices_list(
    matrices: NDArray[np.complexfloating],
    row_mask: NDArray[np.bool_] | None = None,
    title: str = "Matrix",
    show_imag: bool = True,
    x_label: str = "",
    y_label: str = "",
    limit: int | None = None,
) -> None:
    """
    Plot a batch of 2D matrices (M, D, L) using imshow.
    Invalid rows can be masked with NaN using row_mask.
    """
    matrices_to_plot = matrices.copy()
    if limit is not None:
        assert limit > 0
        matrices_to_plot = matrices[:limit, ...]

    if row_mask is not None:
        if row_mask.shape[:2] != matrices_to_plot.shape[:2]:
            raise ValueError(
                f"row_mask shape {row_mask.shape} must match matrices[:2] {matrices_to_plot.shape[:2]}"
            )

        M, D, L = matrices_to_plot.shape
        if row_mask.ndim == 2:
            row_mask = row_mask[:, :, None]
        row_mask = np.repeat(row_mask, L, axis=2)
        matrices_to_plot = np.where(row_mask, matrices_to_plot, np.nan)

    M = matrices_to_plot.shape[0]
    nrows = 2 if show_imag else 1
    ncols = M

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True
    )
    if axs.ndim == 1:
        axs = axs[:, None]

    combined_vals = np.concatenate(
        [matrices_to_plot.real.ravel(), matrices_to_plot.imag.ravel()]
        if show_imag
        else [matrices_to_plot.real.ravel()]
    )
    vmin, vmax = np.nanmin(combined_vals), np.nanmax(combined_vals)
    im = None

    for idx in range(ncols):
        im = axs[0, idx].imshow(
            matrices_to_plot[idx].real,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, idx].set_title(f"Mode {idx} (Real)" if show_imag else f"Mode {idx}")
        axs[0, idx].set_xlabel(x_label)
        axs[0, idx].set_ylabel(y_label)

        if show_imag:
            axs[1, idx].imshow(
                matrices_to_plot[idx].imag,
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            axs[1, idx].set_title(f"Mode {idx} (Imag)")
            axs[1, idx].set_xlabel(x_label)
            axs[1, idx].set_ylabel(y_label)

    fig.suptitle(title)
    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
    plt.show()


def scatter_scores_2d(
    score_mat: NDArray[np.floating],
    score_names: list[str],
    title: str = "",
    show_id: bool = False,
):
    assert score_mat.shape[1] == 2, "score_mat must have 2 columns"
    n_candidates = score_mat.shape[0]
    score1 = score_mat[:, 0]
    score2 = score_mat[:, 1]
    plt.scatter(x=score1, y=score2, marker="+", alpha=0.8, s=10)
    if show_id:
        for i in range(n_candidates):
            plt.annotate(
                str(i),
                (score1[i].item(), score2[i].item()),
                textcoords="offset points",
                xytext=(5, 5),
                ha="center",
            )
    plt.title(title)
    plt.xlabel(score_names[0])
    plt.ylabel(score_names[1])
    plt.show()


def scatter_scores_1d(
    scores: NDArray[np.floating],
    score_name: str,
    title: str = "",
    show_id: bool = False,
):
    n_candidates = scores.shape[0]
    if show_id:
        idx = np.argsort(scores)
        x = np.arange(n_candidates)
        y = scores[idx]
        labels = idx
    else:
        x = np.arange(n_candidates)
        y = scores
        labels = None

    plt.scatter(x=x, y=y, marker="+", alpha=0.8, s=10)

    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(
                str(label),
                (float(i), y[i].item()),
                textcoords="offset points",
                xytext=(5, 5),
                ha="center",
            )
    else:
        plt.xlabel("Mode Index")

    plt.title(title)
    plt.ylabel(score_name)
    plt.show()


def plot_modes(modes: NDArray[np.complexfloating]) -> None:
    """
    Plot real and imaginary parts of each mode in separate subplots.
    """
    fig, axs = plt.subplots(modes.shape[1], 2)
    for i in range(modes.shape[1]):
        axs[i, 0].plot(modes[:, i].real)
        axs[i, 1].plot(modes[:, i].imag)
    plt.show()


def imshow_complex(im, title=None):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    fig.suptitle(title)
    axs[0].imshow(im.real, aspect="auto", interpolation="nearest")
    axs[0].set_title("Real Part")
    axs[1].imshow(im.imag, aspect="auto", interpolation="nearest")
    axs[1].set_title("Imaginary Part")
    plt.show()


def plot_mode_table(eigs: np.ndarray, amps: np.ndarray) -> None:
    """Standalone figure with a table of rho, theta (radians), and amplitude b."""
    rho = np.abs(eigs)
    theta = np.mod(np.angle(eigs), 2 * np.pi)  # 0 … 2π
    b = amps

    # format with 4 significant digits
    cell_text = [
        [f"{r:.4g}", f"{t:.4g}", f"{amp:.4g}"] for r, t, amp in zip(rho, theta, b)
    ]
    col_labels = ["rho", "theta (rad)", "b"]

    fig, ax = plt.subplots(figsize=(4, 0.4 * len(eigs) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)  # tidy row spacing
    fig.tight_layout()
    plt.show()
