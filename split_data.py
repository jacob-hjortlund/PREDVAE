import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

BASE_PATH = Path("/groups/dark/osman/VAEPhotoZ/Data")
SAVE_PATH = BASE_PATH / "Processed"
PHOTO_PATH = BASE_PATH / "Base" / "SDSS_photo_xmatch.csv"
SPEC_PATH = BASE_PATH / "Base" / "SDSS_spec_xmatch.csv"

SAVE_PATH.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_VAL_FRACTION = 0.1
NO_SPEC_Z_FLAG = -9999.0
QUALITY_FLAGS = [-9999, 9999]
COLS = [
    "modelmag_u",
    "modelmag_g",
    "modelmag_r",
    "modelmag_i",
    "modelmag_z",
    "psfmag_u",
    "psfmag_g",
    "psfmag_r",
    "psfmag_i",
    "psfmag_z",
    "w1mag",
    "w2mag",
    "w1mpro",
    "w2mpro",
    "psfmagerr_u",
    "psfmagerr_g",
    "psfmagerr_r",
    "psfmagerr_i",
    "psfmagerr_z",
    "w1sigmag",
    "w2sigmag",
    "modelmagerr_u",
    "modelmagerr_g",
    "modelmagerr_r",
    "modelmagerr_i",
    "modelmagerr_z",
    "w1sigmpro",
    "w2sigmpro",
    "extinction_i",
]


def remove_quality_flagged_inputs(
    bands: list, input_flags: list, df: pd.DataFrame
) -> pd.DataFrame:
    """Each input band is checked for input flags given a list of flags. If a flag is
    present the object is removed.
    Args:
        bands (list(str)): Band column names
        input_flags (list(str)): Input flag values
        df (pd.DataFrame): Input data

    Returns:
        pd.DataFrame: Filtered input data
    """

    mag_columns = df[bands]
    is_flagged = np.full(len(mag_columns), False)
    for flag in input_flags:
        is_flagged = is_flagged | (np.any(mag_columns == flag, axis=1))
    is_not_flagged = ~is_flagged
    filtered_df = df[is_not_flagged].reset_index(drop=True)

    return filtered_df


def split_dataset(df, test_size, seed, val_size=None):

    if test_size < 1:
        test_size = int(test_size * len(df))
    if val_size is not None:
        if val_size < 1:
            val_size = int(val_size * len(df))
    else:
        val_size = test_size

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["class"],
    )

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train, df_val = train_test_split(
        df_train,
        test_size=val_size,
        random_state=seed,
        stratify=df_train["class"],
    )

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True)

    return df_train, df_val, df_test


###############################################################################################
##################################### LOAD DATA ###############################################
###############################################################################################

photo = pd.read_csv(PHOTO_PATH)
spec = pd.read_csv(SPEC_PATH)

###############################################################################################
############################## PREPROCESS SPEC DATA ###########################################
###############################################################################################

spec = remove_quality_flagged_inputs(bands=COLS, input_flags=QUALITY_FLAGS, df=spec)

spec["log10_z"] = np.log10(spec["z"])
spec["log10_extinction_i"] = np.log10(spec["extinction_i"])

spec_galaxies = spec[spec["class"] == "GALAXY"]
spec_galaxies = spec_galaxies.reset_index(drop=True)

spec_qso = spec[spec["class"] == "QSO"]
spec_qso = spec_qso.reset_index(drop=True)

spec_star = spec[spec["class"] == "STAR"]
spec_star = spec_star.reset_index(drop=True)

spec_train, spec_val, spec_test = split_dataset(
    spec, test_size=TEST_VAL_FRACTION, seed=SEED
)
spec_galaxies_train, spec_galaxies_val, spec_galaxies_test = split_dataset(
    spec_galaxies, test_size=TEST_VAL_FRACTION, seed=SEED
)
spec_qso_train, spec_qso_val, spec_qso_test = split_dataset(
    spec_qso, test_size=TEST_VAL_FRACTION, seed=SEED
)
spec_star_train, spec_star_val, spec_star_test = split_dataset(
    spec_star, test_size=TEST_VAL_FRACTION, seed=SEED
)

spec_train.to_csv(SAVE_PATH / "spec_train.csv", index=False)
spec_val.to_csv(SAVE_PATH / "spec_val.csv", index=False)
spec_test.to_csv(SAVE_PATH / "spec_test.csv", index=False)

spec_galaxies_train.to_csv(SAVE_PATH / "spec_galaxies_train.csv", index=False)
spec_galaxies_val.to_csv(SAVE_PATH / "spec_galaxies_val.csv", index=False)
spec_galaxies_test.to_csv(SAVE_PATH / "spec_galaxies_test.csv", index=False)

spec_qso_train.to_csv(SAVE_PATH / "spec_qso_train.csv", index=False)
spec_qso_val.to_csv(SAVE_PATH / "spec_qso_val.csv", index=False)
spec_qso_test.to_csv(SAVE_PATH / "spec_qso_test.csv", index=False)

spec_star_train.to_csv(SAVE_PATH / "spec_star_train.csv", index=False)
spec_star_val.to_csv(SAVE_PATH / "spec_star_val.csv", index=False)
spec_star_test.to_csv(SAVE_PATH / "spec_star_test.csv", index=False)

###############################################################################################
############################## PREPROCESS PHOTO DATA ##########################################
###############################################################################################

print("Preprocessing photo")
photo = remove_quality_flagged_inputs(bands=COLS, input_flags=QUALITY_FLAGS, df=photo)
photo["log10_z"] = NO_SPEC_Z_FLAG
photo["log10_extinction_i"] = np.log10(photo["extinction_i"])

photo_train, photo_val, photo_test = split_dataset(
    photo, test_size=TEST_VAL_FRACTION, seed=SEED
)

photo_train.to_csv(SAVE_PATH / "photo_train.csv", index=False)
photo_val.to_csv(SAVE_PATH / "photo_val.csv", index=False)
photo_test.to_csv(SAVE_PATH / "photo_test.csv", index=False)
