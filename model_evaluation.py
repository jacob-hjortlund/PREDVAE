import os
import jax

jax.config.update("jax_enable_x64", True)
print(f"JAX backend: {jax.devices()}")

import time
import optax
import hydra

import numpy as np
import pandas as pd
import equinox as eqx
import seaborn as sns
import jax.numpy as jnp
import jax.random as jr
import src.predvae.nn as nn
import src.predvae.data as data
import matplotlib.pyplot as plt
import src.predvae.training as training
import src.predvae.evaluation as evaluation

from pathlib import Path
from functools import partial
from jax.tree_util import tree_map
from omegaconf import DictConfig, OmegaConf

colors = sns.color_palette("colorblind")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg), "\n")

    DATA_DIR = Path(cfg["data_config"]["data_dir"])
    SAVE_DIR = Path(cfg["save_dir"]) / cfg["run_name"]
    RNG_KEY = jax.random.PRNGKey(cfg["seed"])

    ###################################################################################
    ############################# LOAD DATA ###########################################
    ###################################################################################

    print(
        "\n--------------------------------- LOADING DATA ---------------------------------\n"
    )

    spec_df = pd.read_csv(DATA_DIR / "SDSS_spec_test.csv")

    (
        spec_psf_photometry,
        spec_psf_photometry_err,
        spec_model_photometry,
        spec_model_photometry_err,
        spec_additional_info,
        spec_z,
        spec_objid,
    ) = data.create_input_arrays(
        input_df=spec_df,
        psf_columns=cfg["data_config"]["psf_columns"],
        psf_err_columns=cfg["data_config"]["psf_err_columns"],
        model_columns=cfg["data_config"]["model_columns"],
        model_err_columns=cfg["data_config"]["model_err_columns"],
        additional_columns=cfg["data_config"]["additional_columns"],
        z_column=cfg["data_config"]["z_column"],
        objid_column=cfg["data_config"]["objid_column"],
        shuffle=False,
    )
    spec_psf_photometry = spec_psf_photometry.squeeze(axis=0)
    spec_psf_photometry_err = spec_psf_photometry_err.squeeze(axis=0)
    spec_model_photometry = spec_model_photometry.squeeze(axis=0)
    spec_model_photometry_err = spec_model_photometry_err.squeeze(axis=0)
    spec_additional_info = jnp.log10(spec_additional_info).squeeze(axis=0)
    spec_z = jnp.log10(spec_z).squeeze(axis=0)
    spec_objid = spec_objid.squeeze(axis=0)

    ###################################################################################
    ############################# CREATE DATASET ######################################
    ###################################################################################

    spec_dataset = data.SpectroPhotometricDataset(
        spec_psf_photometry,
        spec_psf_photometry_err,
        spec_model_photometry,
        spec_model_photometry_err,
        spec_additional_info,
        spec_z,
        spec_objid,
    )

    spec_dataset_statistics = data.SpectroPhotometricStatistics(
        spec_dataset, spec_dataset
    )
    dataset_statistics = training.load(
        SAVE_DIR / "train_dataset_statistics.pkl", spec_dataset_statistics
    )

    dataloader_key = jr.PRNGKey(420)
    (
        dataloader,
        dataloader_state,
    ) = data.make_dataloader(
        spec_dataset,
        batch_size=cfg["training_config"]["batch_size"],
        rng_key=dataloader_key,
        shuffle=False,
        drop_last=True,
    )

    _dataset_iterator = data.make_dataset_iterator(
        resample_photometry=False,
    )
    dataset_iterator = lambda state, rng_key: _dataset_iterator(
        dataloader, dataset_statistics, state, rng_key
    )

    descaled_log10_zspec = spec_dataset.log10_redshift.squeeze()
    normed_log10_zspec = (
        descaled_log10_zspec - dataset_statistics.log10_redshift_mean
    ) / dataset_statistics.log10_redshift_std
    zspec = 10**descaled_log10_zspec
    object_class = spec_df["class"].fillna("Unknown", inplace=False).values
    object_class = np.array(
        [
            object_class[spec_df["objid"].values == objid][0]
            for objid in spec_dataset.objid
        ]
    )

    spec_z_sample_idxes = jr.choice(
        jr.PRNGKey(420),
        64 * 1024,  # GET RID OF MAGIC NUMBER
        (cfg["evaluation_config"]["n_sample"],),
        replace=False,
    )
    sample_normed_log10_zspec = normed_log10_zspec[spec_z_sample_idxes]
    sample_descaled_log10_zspec = descaled_log10_zspec[spec_z_sample_idxes]
    sample_zspec = zspec[spec_z_sample_idxes]
    sample_object_classes = object_class[spec_z_sample_idxes]

    ###################################################################################
    ################################### MODEL #########################################
    ###################################################################################

    (
        predictor_key,
        encoder_input_key,
        encoder_key,
        decoder_input_key,
        decoder_key,
        RNG_KEY,
    ) = jr.split(RNG_KEY, 6)

    decoder_input_layer = nn.InputLayer(
        x_features=cfg["model_config"]["latent_size"],
        y_features=cfg["model_config"]["predictor_size"],
        out_features=cfg["model_config"]["latent_size"]
        + cfg["model_config"]["predictor_size"],
        key=decoder_input_key,
    )

    decoder = nn.GaussianCoder(
        input_size=cfg["model_config"]["latent_size"]
        + cfg["model_config"]["predictor_size"],
        output_size=cfg["model_config"]["input_size"],
        width=cfg["model_config"]["layers"],
        depth=len(cfg["model_config"]["layers"]),
        activation=getattr(jax.nn, cfg["model_config"]["activation"]),
        key=decoder_key,
        use_spectral_norm=cfg["model_config"]["use_spec_norm"],
        use_final_spectral_norm=cfg["model_config"]["use_final_spec_norm"],
        num_power_iterations=cfg["model_config"]["num_power_iterations"],
    )

    latent_prior = nn.Gaussian(
        mu=jnp.zeros(cfg["model_config"]["latent_size"]),
        log_sigma=jnp.zeros(cfg["model_config"]["latent_size"]),
    )

    target_prior = nn.Gaussian(
        mu=jnp.zeros(cfg["model_config"]["predictor_size"]),
        log_sigma=jnp.zeros(cfg["model_config"]["predictor_size"]),
    )

    if cfg["model_config"]["use_v2"]:

        encoder = nn.GaussianCoder(
            input_size=cfg["model_config"]["input_size"],
            output_size=cfg["model_config"]["latent_size"],
            width=cfg["model_config"]["layers"],
            depth=len(cfg["model_config"]["layers"]),
            activation=getattr(jax.nn, cfg["model_config"]["activation"]),
            key=encoder_key,
            use_spectral_norm=cfg["model_config"]["use_spec_norm"],
            use_final_spectral_norm=cfg["model_config"]["use_final_spec_norm"],
            num_power_iterations=cfg["model_config"]["num_power_iterations"],
        )

        predictor_input_layer = nn.InputLayer(
            x_features=cfg["model_config"]["latent_size"],
            y_features=cfg["model_config"]["input_size"],
            out_features=cfg["model_config"]["latent_size"]
            + cfg["model_config"]["input_size"],
            key=encoder_input_key,
        )

        predictor = nn.GaussianMixtureCoder(
            input_size=cfg["model_config"]["latent_size"]
            + cfg["model_config"]["input_size"],
            output_size=cfg["model_config"]["predictor_size"],
            width=cfg["model_config"]["layers"],
            depth=len(cfg["model_config"]["layers"]),
            num_components=cfg["model_config"]["num_mixture_components"],
            activation=getattr(jax.nn, cfg["model_config"]["activation"]),
            key=predictor_key,
            use_spectral_norm=cfg["model_config"]["use_spec_norm"],
            use_final_spectral_norm=cfg["model_config"]["use_final_spec_norm"],
            num_power_iterations=cfg["model_config"]["num_power_iterations"],
        )

        SSVAE = partial(
            nn.SSVAEv2,
            encoder=encoder,
            predictor=predictor,
            predictor_input_layer=predictor_input_layer,
        )

    else:

        encoder_input_layer = nn.InputLayer(
            x_features=cfg["model_config"]["input_size"],
            y_features=cfg["model_config"]["predictor_size"],
            out_features=cfg["model_config"]["input_size"]
            + cfg["model_config"]["predictor_size"],
            key=encoder_input_key,
        )

        encoder = nn.GaussianCoder(
            input_size=cfg["model_config"]["input_size"]
            + cfg["model_config"]["predictor_size"],
            output_size=cfg["model_config"]["latent_size"],
            width=cfg["model_config"]["layers"],
            depth=len(cfg["model_config"]["layers"]),
            activation=getattr(jax.nn, cfg["model_config"]["activation"]),
            key=encoder_key,
            use_spectral_norm=cfg["model_config"]["use_spec_norm"],
            use_final_spectral_norm=cfg["model_config"]["use_final_spec_norm"],
            num_power_iterations=cfg["model_config"]["num_power_iterations"],
        )

        predictor = nn.GaussianMixtureCoder(
            input_size=cfg["model_config"]["input_size"],
            output_size=cfg["model_config"]["predictor_size"],
            width=cfg["model_config"]["layers"],
            depth=len(cfg["model_config"]["layers"]),
            num_components=cfg["model_config"]["num_mixture_components"],
            activation=getattr(jax.nn, cfg["model_config"]["activation"]),
            key=predictor_key,
            use_spectral_norm=cfg["model_config"]["use_spec_norm"],
            use_final_spectral_norm=cfg["model_config"]["use_final_spec_norm"],
            num_power_iterations=cfg["model_config"]["num_power_iterations"],
        )

        SSVAE = partial(
            nn.SSVAE,
            encoder=encoder,
            predictor=predictor,
            encoder_input_layer=encoder_input_layer,
        )

    ssvae, input_state = eqx.nn.make_with_state(SSVAE)(
        decoder=decoder,
        latent_prior=latent_prior,
        target_prior=target_prior,
        decoder_input_layer=decoder_input_layer,
    )
    ssvae = eqx.nn.inference_mode(ssvae)

    ###################################################################################
    ############################# EVALUATION ##########################################
    ###################################################################################

    if cfg["training_config"]["pretrain_vae"]:

        pretrained_vae = training.load(SAVE_DIR / "best_pretrained_vae.pkl", ssvae)
        pretrained_vae_state = training.load(
            SAVE_DIR / "best_pretrained_vae_state.pkl", input_state
        )
        pretrained_vae = eqx.nn.inference_mode(pretrained_vae)

        latent_means, latent_log_stds, pretrained_vae_state, dataloader_state = (
            evaluation.latent_space_evaluator(
                pretrained_vae,
                dataset_iterator,
                pretrained_vae_state,
                dataloader_state,
                RNG_KEY,
            )
        )

        evaluation.plot_latent_space(
            latent_means,
            object_class,
            SAVE_DIR,
            filename="pretrained_vae_latent_space",
        )

    if cfg["training_config"]["pretrain_predictor"]:

        pretrained_predictor = training.load(
            SAVE_DIR / "best_pretrained_predictor.pkl", ssvae
        )
        pretrained_predictor_state = training.load(
            SAVE_DIR / "best_pretrained_predictor_state.pkl", input_state
        )
        pretrained_predictor = eqx.nn.inference_mode(pretrained_predictor)

        (
            target_values,
            latent_means,
            latent_log_stds,
            y_weights,
            y_means,
            y_stds,
            ppfs,
            ppf_fractions,
            pretrained_predictor_state,
            dataloader_state,
        ) = evaluation.full_model_evaluator(
            pretrained_predictor,
            dataset_iterator,
            pretrained_predictor_state,
            dataloader_state,
            RNG_KEY,
            min=cfg["evaluation_config"]["q_min"],
            max=cfg["evaluation_config"]["q_max"],
            n=cfg["evaluation_config"]["n_q"],
        )

        evaluation.plot_latent_space(
            latent_means,
            object_class,
            SAVE_DIR,
            filename="pretrained_predictor_latent_space",
        )
        evaluation.qq_plot(
            ppf_fractions, SAVE_DIR, filename="pretrained_predictor_qq_plot"
        )

        (means, stds), (medians, lower, upper) = evaluation.calculate_point_values(
            y_weights,
            y_means,
            y_stds,
            ppfs.T,
            cfg["evaluation_config"]["q_min"],
            cfg["evaluation_config"]["q_max"],
            cfg["evaluation_config"]["n_q"],
        )

        descaled_medians = (
            medians * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_lowers = (
            lower * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_uppers = (
            upper * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_dist_means = (
            y_means * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_dist_stds = y_stds * dataset_statistics.log10_redshift_std
        descaled_dist_weights = y_weights
        descaled_ppfs = (
            ppfs * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )

        z_medians = 10**descaled_medians
        z_lowers = 10**descaled_lowers
        z_uppers = 10**descaled_uppers
        z_ppfs = 10**descaled_ppfs

        sample_medians = medians[spec_z_sample_idxes]
        sample_lowers = lower[spec_z_sample_idxes]
        sample_uppers = upper[spec_z_sample_idxes]
        sample_ppfs = ppfs[:, spec_z_sample_idxes]
        sample_dist_means = y_means[spec_z_sample_idxes]
        sample_dist_stds = y_stds[spec_z_sample_idxes]
        sample_dist_weights = y_weights[spec_z_sample_idxes]

        sample_descaled_medians = descaled_medians[spec_z_sample_idxes]
        sample_descaled_lowers = descaled_lowers[spec_z_sample_idxes]
        sample_descaled_uppers = descaled_uppers[spec_z_sample_idxes]
        sample_descaled_ppfs = descaled_ppfs[:, spec_z_sample_idxes]
        sample_descaled_dist_means = descaled_dist_means[spec_z_sample_idxes]
        sample_descaled_dist_stds = descaled_dist_stds[spec_z_sample_idxes]
        sample_descaled_dist_weights = descaled_dist_weights[spec_z_sample_idxes]

        sample_z_medians = z_medians[spec_z_sample_idxes]
        sample_z_lowers = z_lowers[spec_z_sample_idxes]
        sample_z_uppers = z_uppers[spec_z_sample_idxes]
        sample_z_ppfs = z_ppfs[:, spec_z_sample_idxes]

        evaluation.plot_sample_redshift_dists(
            sample_normed_log10_zspec,
            sample_object_classes,
            sample_ppfs,
            sample_dist_means,
            sample_dist_stds,
            sample_dist_weights,
            sample_medians,
            sample_lowers,
            sample_uppers,
            SAVE_DIR,
            x_label=r"Rescaled log$_{10}(z)$",
            file_name="pretrained_predictor_rescaled_log10z_sample_distributions",
        )

        evaluation.plot_sample_redshift_dists(
            sample_descaled_log10_zspec,
            sample_object_classes,
            sample_descaled_ppfs,
            sample_descaled_dist_means,
            sample_descaled_dist_stds,
            sample_descaled_dist_weights,
            sample_descaled_medians,
            sample_descaled_lowers,
            sample_descaled_uppers,
            SAVE_DIR,
            x_label=r"log$_{10}(z)$",
            file_name="pretrained_predictor_log10z_sample_distributions",
        )

        evaluation.plot_sample_redshift_dists(
            sample_zspec,
            sample_object_classes,
            sample_z_ppfs,
            sample_descaled_dist_means,
            sample_descaled_dist_stds,
            sample_descaled_dist_weights,
            sample_z_medians,
            sample_z_lowers,
            sample_z_uppers,
            SAVE_DIR,
            x_label=r"$z$",
            is_z_space=True,
            file_name="pretrained_predictor_z_sample_distributions",
        )

        evaluation.plot_photo_vs_spec(
            zspec,
            object_class,
            z_medians,
            SAVE_DIR,
            filename="pretrained_predictor_photo_vs_spec",
        )

    if cfg["training_config"]["train_full_model"]:

        final_model = training.load(SAVE_DIR / "best_model.pkl", ssvae)
        final_model_state = training.load(
            SAVE_DIR / "best_model_state.pkl", input_state
        )
        final_model = eqx.nn.inference_mode(final_model)

        (
            target_values,
            latent_means,
            latent_log_stds,
            y_weights,
            y_means,
            y_stds,
            ppfs,
            ppf_fractions,
            final_model_state,
            dataloader_state,
        ) = evaluation.full_model_evaluator(
            final_model,
            dataset_iterator,
            final_model_state,
            dataloader_state,
            RNG_KEY,
            min=cfg["evaluation_config"]["q_min"],
            max=cfg["evaluation_config"]["q_max"],
            n=cfg["evaluation_config"]["n_q"],
        )

        evaluation.plot_latent_space(
            latent_means,
            object_class,
            SAVE_DIR,
            filename="full_model_latent_space",
        )
        evaluation.qq_plot(ppf_fractions, SAVE_DIR, filename="full_model_qq_plot")

        (means, stds), (medians, lower, upper) = evaluation.calculate_point_values(
            y_weights,
            y_means,
            y_stds,
            ppfs.T,
            cfg["evaluation_config"]["q_min"],
            cfg["evaluation_config"]["q_max"],
            cfg["evaluation_config"]["n_q"],
        )

        descaled_medians = (
            medians * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_lowers = (
            lower * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_uppers = (
            upper * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_dist_means = (
            y_means * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )
        descaled_dist_stds = y_stds * dataset_statistics.log10_redshift_std
        descaled_dist_weights = y_weights
        descaled_ppfs = (
            ppfs * dataset_statistics.log10_redshift_std
            + dataset_statistics.log10_redshift_mean
        )

        z_medians = 10**descaled_medians
        z_lowers = 10**descaled_lowers
        z_uppers = 10**descaled_uppers
        z_ppfs = 10**descaled_ppfs

        sample_medians = medians[spec_z_sample_idxes]
        sample_lowers = lower[spec_z_sample_idxes]
        sample_uppers = upper[spec_z_sample_idxes]
        sample_ppfs = ppfs[:, spec_z_sample_idxes]
        sample_dist_means = y_means[spec_z_sample_idxes]
        sample_dist_stds = y_stds[spec_z_sample_idxes]
        sample_dist_weights = y_weights[spec_z_sample_idxes]

        sample_descaled_medians = descaled_medians[spec_z_sample_idxes]
        sample_descaled_lowers = descaled_lowers[spec_z_sample_idxes]
        sample_descaled_uppers = descaled_uppers[spec_z_sample_idxes]
        sample_descaled_ppfs = descaled_ppfs[:, spec_z_sample_idxes]
        sample_descaled_dist_means = descaled_dist_means[spec_z_sample_idxes]
        sample_descaled_dist_stds = descaled_dist_stds[spec_z_sample_idxes]
        sample_descaled_dist_weights = descaled_dist_weights[spec_z_sample_idxes]

        sample_z_medians = z_medians[spec_z_sample_idxes]
        sample_z_lowers = z_lowers[spec_z_sample_idxes]
        sample_z_uppers = z_uppers[spec_z_sample_idxes]
        sample_z_ppfs = z_ppfs[:, spec_z_sample_idxes]

        evaluation.plot_sample_redshift_dists(
            sample_normed_log10_zspec,
            sample_object_classes,
            sample_ppfs,
            sample_dist_means,
            sample_dist_stds,
            sample_dist_weights,
            sample_medians,
            sample_lowers,
            sample_uppers,
            SAVE_DIR,
            x_label=r"Rescaled log$_{10}(z)$",
            file_name="full_model_rescaled_log10z_sample_distributions",
        )

        evaluation.plot_sample_redshift_dists(
            sample_descaled_log10_zspec,
            sample_object_classes,
            sample_descaled_ppfs,
            sample_descaled_dist_means,
            sample_descaled_dist_stds,
            sample_descaled_dist_weights,
            sample_descaled_medians,
            sample_descaled_lowers,
            sample_descaled_uppers,
            SAVE_DIR,
            x_label=r"log$_{10}(z)$",
            file_name="full_model_log10z_sample_distributions",
        )

        evaluation.plot_sample_redshift_dists(
            sample_zspec,
            sample_object_classes,
            sample_z_ppfs,
            sample_descaled_dist_means,
            sample_descaled_dist_stds,
            sample_descaled_dist_weights,
            sample_z_medians,
            sample_z_lowers,
            sample_z_uppers,
            SAVE_DIR,
            x_label=r"$z$",
            is_z_space=True,
            file_name="full_model_z_sample_distributions",
        )

        evaluation.plot_photo_vs_spec(
            zspec,
            object_class,
            z_medians,
            SAVE_DIR,
            filename="full_model_photo_vs_spec",
        )


if __name__ == "__main__":
    main()
