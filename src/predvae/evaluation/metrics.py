import numpy as np


def rmse(residual):

    return np.sqrt(np.mean(residual**2))


def mae(residual):

    return np.mean(np.abs(residual))


def iqr(x):
    # Graham et al. 2020 https://arxiv.org/abs/2004.07885

    upper = np.percentile(x, 75)
    lower = np.percentile(x, 25)
    iqrange = upper - lower

    return iqrange, lower, upper


def photo_bias(zspec, zphot):
    # Graham et al. 2020 https://arxiv.org/abs/2004.07885

    return (zspec - zphot) / (1 + zphot)


def spec_bias(zspec, zphot):

    return (zphot - zspec) / (1 + zspec)


def iqr_robust_bias(zspec, zphot):
    # Graham et al. 2020 https://arxiv.org/abs/2004.07885

    bias = photo_bias(zspec, zphot)
    _, lower, upper = iqr(bias)
    idx_bias = (bias > lower) & (bias < upper)

    return np.mean(bias[idx_bias])


def sigma_iqr(zspec, zphot):
    # Graham et al. 2020 https://arxiv.org/abs/2004.07885

    iqrange, _, _ = iqr(photo_bias(zspec, zphot))
    return iqrange / 1.349


def sigma_nmad(residuals):
    # Kodra et al 2018 https://arxiv.org/abs/2210.01140

    return 1.48 * np.median(np.abs(residuals - np.median(residuals)))


def iqr_point_outliers(zspec, zphot):
    # Graham et al. 2020 https://arxiv.org/abs/2004.07885

    bias = photo_bias(zspec, zphot)
    std_iqr = sigma_iqr(zspec, zphot)
    idx_outliers = (bias < -3 * std_iqr) | (bias > 3 * std_iqr) | (np.abs(bias) > 0.06)
    frac_outliers = np.sum(idx_outliers) / len(zspec)

    return frac_outliers, idx_outliers


def lsst_catastrophic_outliers(zspec, zphot):
    # Graham et al. 2020 https://arxiv.org/abs/2004.07885

    abs_diff = np.abs(zspec - zphot)
    idx_cat_outliers = abs_diff >= 0.2
    frac_cat_outliers = np.sum(idx_cat_outliers) / len(zspec)

    return frac_cat_outliers, idx_cat_outliers


def catastrophic_outliers(zspec, zphot, threshold=0.15):

    residual = np.abs(spec_bias(zspec, zphot))
    idx_cat_outliers = residual >= threshold
    frac_cat_outliers = np.sum(idx_cat_outliers) / len(zspec)

    return frac_cat_outliers, idx_cat_outliers


def qq_rmse(quantile_fraction, q_min=0.001, q_max=0.999, n_q=0.999):

    q_theory = np.linspace(q_min, q_max, n_q)
    residual = quantile_fraction - q_theory
    return rmse(residual)


def pdf_outliers(zspec, lower_3sig, upper_3sig):

    idx_outliers = (zspec < lower_3sig) | (zspec > upper_3sig)
    frac_outliers = np.sum(idx_outliers) / len(zspec)

    return frac_outliers, idx_outliers
