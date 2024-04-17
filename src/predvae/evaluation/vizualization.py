import jax
import numpy as np
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt

from umap import UMAP
from jax.scipy.stats import norm
from functools import partial

colors = sns.color_palette("colorblind")


@partial(jax.vmap, in_axes=(0, None, None, None))
def gaussian_mixture_pdf(x, means, stds, weights):
    return jnp.sum(weights * norm.pdf(x, means, stds), axis=-1)


@partial(jax.vmap, in_axes=(0, None, None, None))
def log10_gaussian_mixture_pdf(x, means, stds, weights):
    weights = weights / (jnp.abs(x) * jnp.log(10))
    return jnp.sum(weights * norm.pdf(jnp.log10(x), means, stds), axis=-1)


def plot_latent_space(latent_means, labels, save_dir, filename="latent_space"):

    umap = UMAP(n_components=2)
    latent_space = umap.fit_transform(latent_means)
    n_latent = latent_space.shape[0]

    _labels = ["GALAXY", "QSO", "STAR"]
    idx_galaxy = labels == _labels[0]
    idx_qso = labels == _labels[1]
    idx_star = labels == _labels[2]
    idxes = [idx_galaxy[:n_latent], idx_qso[:n_latent], idx_star[:n_latent]]

    fig, ax = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)
    ax = ax.flatten()

    for i, idx in enumerate(idxes):

        ax[0].scatter(
            latent_space[idx, 0],
            latent_space[idx, 1],
            color=colors[i],
            label=_labels[i],
            alpha=0.5,
            s=5,
        )

        ax[i + 1].scatter(
            latent_space[idx, 0], latent_space[idx, 1], color=colors[i], s=5
        )
        ax[i + 1].set_title(_labels[i], fontsize=16)

    ax[0].set_title("All Classes", fontsize=16)
    ax[0].legend(fontsize=16)

    fig.suptitle("Latent Space Visualization", fontsize=20)
    fig.supxlabel("UMAP 1", fontsize=16)
    fig.supylabel("UMAP 2", fontsize=16)
    fig.tight_layout()

    fig.savefig(save_dir / f"{filename}.png", dpi=300)


def qq_plot(ppf_fractions, save_dir, filename="qq_plot", min=0.01, max=0.99):

    q = jnp.linspace(min, max, len(ppf_fractions))

    fig, ax = plt.subplots()
    ax.plot(q, q, linestyle="--", color="black")
    ax.plot(q, ppf_fractions, color=colors[0])
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title("QQ Plot")
    fig.savefig(save_dir / f"{filename}.png", dpi=300)


def plot_sample_redshift_dists(
    z_specs,
    object_classes,
    ppfs,
    dist_means,
    dist_stds,
    dist_weights,
    medians,
    lowers,
    uppers,
    save_dir,
    x_label,
    is_z_space=False,
    file_name="sample_distributions",
    n_rows=4,
    n_cols=4,
):

    means = np.sum(dist_weights * dist_means, axis=-1)
    stds = np.sqrt(
        np.sum(dist_weights * dist_stds**2, axis=-1)
        + np.sum(dist_weights * dist_means**2, axis=-1)
        - np.sum(dist_weights * dist_means, axis=-1) ** 2
    )

    pdf_fn = gaussian_mixture_pdf if not is_z_space else log10_gaussian_mixture_pdf
    if is_z_space:
        trans_dist_means = 10 ** (dist_means + 0.5 * dist_stds**2)
        trans_dist_std = trans_dist_means * np.sqrt(10**dist_stds**2 - 1)
        means = np.sum(dist_weights * trans_dist_means, axis=-1)
        stds = np.sqrt(
            np.sum(dist_weights * trans_dist_std**2, axis=-1)
            + np.sum(dist_weights * trans_dist_means**2, axis=-1)
            - np.sum(dist_weights * trans_dist_means, axis=-1) ** 2
        )

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(5 * n_cols, 5 * n_rows))
    ax = ax.flatten()

    for i in range(n_rows * n_cols):

        # z_lower = means[i] - 5 * stds[i]
        # if is_z_space:
        #     z_lower = 1e-10
        # z_upper = means[i] + 5 * stds[i]
        z_lower = ppfs[0, i]
        z_upper = ppfs[-1, i]
        if is_z_space:
            z_lower = 1e-1
            z_upper = 6
        z_plot = np.linspace(z_lower, z_upper, 1024)

        ax[i].plot(
            z_plot,
            pdf_fn(
                z_plot,
                dist_means[i],
                dist_stds[i],
                dist_weights[i],
            ),
            color=colors[0],
        )
        ax[i].axvline(z_specs[i], color="k", linestyle="--", label="Spec Z")

        y_lims = ax[i].get_ylim()
        y_median = (y_lims[0] + y_lims[1]) / 2
        median = medians[i]
        median_lower = lowers[i]
        median_upper = uppers[i]
        median_err = np.array([[median - median_lower], [median_upper - median]])
        ax[i].errorbar(
            median,
            y_median,
            xerr=median_err,
            fmt="o",
            color=colors[1],
            label="Median",
        )

        y_mean = y_median + 0.1 * (y_lims[1] - y_lims[0])
        mean = means[i]
        std = stds[i]  # 1.96 * stds[i]
        if not is_z_space:
            ax[i].errorbar(
                mean,
                y_mean,
                xerr=std,
                fmt="o",
                color=colors[2],
                label="Mean",
            )
        # else:
        #     ax[i].scatter(mean, y_mean, color=colors[2], label="Mean")

        ax[i].set_title(
            f"Object Class: {object_classes[i]}, True Value: {z_specs[i]:.2f}",
            fontsize=14,
        )
        ax[i].legend(loc="upper right")
    fig.supxlabel(x_label, fontsize=20)

    fig.tight_layout()
    fig.savefig(save_dir / f"{file_name}.png", dpi=300)


def plot_photo_vs_spec(
    z_specs,
    labels,
    z_photo,
    save_dir,
    filename="photo_vs_spec",
):

    n = len(z_photo)
    z_specs = z_specs[:n]
    _labels = ["GALAXY", "QSO", "STAR"]
    idx_galaxy = labels == _labels[0]
    idx_qso = labels == _labels[1]
    idx_star = labels == _labels[2]
    idxes = [idx_galaxy[:n], idx_qso[:n], idx_star[:n]]

    fig, ax = plt.subplots()

    for i, idx in enumerate(idxes):
        ax.scatter(
            z_specs[idx],
            z_photo[idx],
            color=colors[i],
            label=_labels[i],
            s=5,
            alpha=0.5,
        )

    z_max = np.max(z_specs) + 0.1
    ax.plot([0, z_max], [0, z_max], linestyle="--", color="black")
    ax.set_xlim(0, z_max)
    ax.set_ylim(0, z_max)
    ax.set_xlabel("Spec Z", fontsize=16)
    ax.set_ylabel("Photo Z", fontsize=16)
    ax.legend(fontsize=16)
    ax.set_title("Photo Z vs Spec Z", fontsize=20)

    fig.tight_layout()
    fig.savefig(save_dir / f"{filename}.png", dpi=300)
