import jax
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt

from umap import UMAP

colors = sns.color_palette("colorblind")


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
