"""
SCI-RTW Predictor: Data Preprocessing Pipeline
Chapter 3, Section 3.5.3 - Data Preprocessing Implementation

Steps (per Table 3.16 in the proposal):
1. Data merging      - Form I + Form II merged via UniID
2. Variable selection - demographic, clinical, functional, employment variables
3. Missing value handling - sentinel codes (999/888/'U'/9/99) treated as missing
4. Categorical encoding
5. Target construction - Layer 1 (RTW/Non-RTW), Layer 2 (RTW timing category)
6. Train-test split

--------------------------------------------------------------------------
LABEL DEFINITION (updated after test_report.md findings)
--------------------------------------------------------------------------
The original "any follow-up shows RTW" label is right-censoring biased:
patients observed longer have more chances to be labelled RTW (20% RTW at
last-observed-year 1, rising to 82% at last-observed-year 45), and Layer 2
inherits the same problem (a patient last seen at Year 1 can only ever be
"Year 1"). Two features made this worse: PHQ-9 (depression score) acted as a
proxy for "recently injured, short follow-up" rather than a clinical signal
(same patient: P(RTW)=0.70 with PHQ blank vs 0.53 with PHQ=2 entered), and
race/hispanic_origin changed accuracy by ~0 while adding no value (Layer 1
AUC 0.723 with vs 0.719 without).

Fix implemented here (LABEL_MODE = "fixed_horizon", the default):
  - Label = "RTW within 10 years", restricted to a cohort where the outcome
    is actually knowable by year 10: a patient is INCLUDED if they either
    (a) had an RTW-positive record at BYear <= 10 (event observed), or
    (b) have a valid record at BYear >= 10 with no such event (followed long
        enough to say "no" with confidence).
    Patients whose last valid record is < 10 and never RTW-positive are
    EXCLUDED (censored - we don't actually know their year-10 outcome).
  - race, hispanic_origin, and the PHQ-9 depression score are dropped from
    the predictor set entirely (not just hidden in the UI).

The original definition is kept as LABEL_MODE = "any_followup" purely so the
Table 3.4/3.5/3.6 validation numbers in docs/data_validation_report.md remain
reproducible; it is not used for the deployed models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_data_dir():
    """Locate the NSCISC public-use CSVs. Accepts either a plain data/ folder
    or the dataset as distributed (NSCISC_Public_Data_2021/, which unzips to
    a folder of the same name inside itself)."""
    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "NSCISC_Public_Data_2021" / "NSCISC_Public_Data_2021",
        PROJECT_ROOT / "NSCISC_Public_Data_2021",
    ]
    for d in candidates:
        if (d / "f1_public_2021.csv").exists() and (d / "f2_public_2021.csv").exists():
            return d
    raise FileNotFoundError(
        "Could not find f1_public_2021.csv / f2_public_2021.csv. Looked in: "
        + ", ".join(str(d) for d in candidates))


# The raw data is looked up when it is read, not on import, so the constants
# here can be imported (by the tests, for one) on a machine without it.
OUT_DIR = PROJECT_ROOT / "processed"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
HORIZON_YEARS = 10
LABEL_MODE = "fixed_horizon"  # "fixed_horizon" (default/fixed) or "any_followup" (original, biased)

# ----------------------------------------------------------------------------
# 1. TARGET CONSTRUCTION (from Form II)
# ----------------------------------------------------------------------------
# BPrLvlSt = occupational status at each follow-up. Verified against the
# proposal's Table 3.4/3.5/3.6 under LABEL_MODE="any_followup" (exact match):
#   RTW-positive codes : {1, 3, 4, 10}
#   Non-RTW codes       : {2, 5, 6, 7, 8, 9}
#   Excluded            : 99 (unknown), blank
RTW_CODES = {1, 3, 4, 10}
NONRTW_CODES = {2, 5, 6, 7, 8, 9}


def build_targets(mode=LABEL_MODE, horizon=HORIZON_YEARS):
    f2 = pd.read_csv(str(find_data_dir() / "f2_public_2021.csv"),
                      usecols=["UniID", "BYear", "BPrLvlSt"], low_memory=False)

    valid = f2[f2["BPrLvlSt"].notna() & (f2["BPrLvlSt"] != 99)].copy()
    valid["is_rtw"] = valid["BPrLvlSt"].isin(RTW_CODES)

    if mode == "any_followup":
        # Original definition: RTW=1 if ANY valid follow-up record is
        # RTW-positive, whenever observed. Kept only for reproducing the
        # proposal's Tables 3.4-3.6; not used for the deployed models
        # (see module docstring for why).
        layer1 = valid.groupby("UniID")["is_rtw"].max().astype(int).rename("rtw_status")
        first_rtw_year = valid[valid["is_rtw"]].groupby("UniID")["BYear"].min()

        def timing_category(y):
            if y <= 1:
                return "Year 1"
            elif y <= 5:
                return "Year 5"
            else:
                return "Year 10 or later"

        layer2 = first_rtw_year.apply(timing_category).rename("rtw_timing")
        return layer1.to_frame().join(layer2, how="left")

    elif mode == "fixed_horizon":
        # Fixed-horizon definition: "did this patient return to work within
        # `horizon` years?", restricted to patients whose outcome by that
        # horizon is actually observable (event-before-horizon, or followed
        # to at least the horizon). This removes the observation-window bias.
        max_year = valid.groupby("UniID")["BYear"].max()
        event_by_horizon = (
            valid[valid["is_rtw"] & (valid["BYear"] <= horizon)]
            .groupby("UniID").size() > 0
        ).reindex(max_year.index, fill_value=False)
        observed_to_horizon = max_year >= horizon
        include = event_by_horizon | observed_to_horizon

        layer1 = event_by_horizon[include].astype(int).rename("rtw_status")

        first_rtw_year = (
            valid[valid["is_rtw"] & (valid["BYear"] <= horizon)]
            .groupby("UniID")["BYear"].min()
        )

        def timing_category(y):
            if y <= 1:
                return "Year 1"
            elif y <= 5:
                return "Year 5"
            else:
                return f"By Year {horizon}"

        layer2 = first_rtw_year.apply(timing_category).rename("rtw_timing")
        return layer1.to_frame().join(layer2, how="left")

    raise ValueError(f"Unknown LABEL_MODE: {mode}")


# ----------------------------------------------------------------------------
# 2. PREDICTOR VARIABLES (from Form I - baseline/rehab data)
# ----------------------------------------------------------------------------
# Selected per Table 3.12 (Variable Categories for Model Input), restricted to
# fields with acceptable completeness after inspecting Form I.
#
# race, hispanic_origin, and the PHQ-9 depression score were REMOVED after
# testing (see module docstring): PHQ acted as an injury-era proxy rather
# than a clinical signal, and race/hispanic_origin gave no accuracy benefit
# (Layer 1 AUC 0.723 -> 0.719 without them) while posing a fairness risk in
# a clinical decision-support tool. Both are also outside the variable list
# in the proposal's Table 3.12.
PREDICTOR_COLS = [
    "UniID",
    # Demographic
    "ASex", "AInjAge", "AEducLvl", "AMarStIj", "AVeteran",
    # Clinical / injury severity (discharge-time values used - most complete)
    "ATrmEtio", "AVertInj", "AAsscInj", "AWrkRltd",
    "AASAImDs", "ANurLvlD", "ANCatDis",
    # Functional (ASIA motor total score at discharge; sentinels handled below)
    "AASATotD",
    # Employment / work background (pre-injury)
    "APrLvlSt",
]

CATEGORICAL_COLS = [
    "ASex", "AEducLvl", "AMarStIj", "AVeteran",
    "ATrmEtio", "AVertInj", "AAsscInj", "AWrkRltd",
    "AASAImDs", "ANurLvlD", "ANCatDis", "APrLvlSt",
]

# Sentinel codes that mean "unknown / not available" for specific numeric fields
SENTINELS = {
    "AASATotD": [888, 999],
}

FRIENDLY_NAMES = {
    "ASex": "sex",
    "AInjAge": "age_at_injury",
    "AEducLvl": "education_level",
    "AMarStIj": "marital_status",
    "AVeteran": "veteran_status",
    "ATrmEtio": "injury_cause",
    "AVertInj": "vertebral_injury_level",
    "AAsscInj": "associated_injury",
    "AWrkRltd": "work_related_injury",
    "AASAImDs": "aisgrade_discharge",
    "ANurLvlD": "neuro_level_discharge",
    "ANCatDis": "injury_completeness_discharge",
    "AASATotD": "asia_motor_total_discharge",
    "APrLvlSt": "pre_injury_employment_status",
}


# Every categorical field's own "unknown" code(s), keyed by the FRIENDLY name.
# Used both to convert raw codes to NaN and, at inference time, to recognise
# "the user didn't know/enter this" regardless of which literal code a given
# field uses for it (9, 99, "U", "X99", ...).
UNKNOWN_CODES = {
    "sex": ["9"],
    "education_level": ["9"],
    "marital_status": ["9"],
    "veteran_status": [9],
    "injury_cause": ["99"],
    "vertebral_injury_level": [9],
    "associated_injury": [9],
    "work_related_injury": [9],
    "aisgrade_discharge": ["U", "9"],
    "neuro_level_discharge": ["X99"],
    "injury_completeness_discharge": [9],
    "pre_injury_employment_status": [99],
}

# AIS grade at discharge is recorded two ways: injuries up to 1993 carry the
# old numeric Frankel code, later ones the AIS letter (data dictionary,
# AASAImDs). Left as they were, the model saw ten categories and could tell
# the two eras apart: "Complete" was followed by return to work in 37% of
# patients coded 5 and 58% of those coded A. That is the era-proxy problem
# of TESTING FINDING #2 again, so each Frankel code is mapped to the AIS
# letter for the same grade (see docs/test_report.md section 6.7).
LEGACY_AIS_GRADES = {"5": "A", "1": "B", "2": "C", "3": "D", "4": "E"}


def build_predictors():
    f1 = pd.read_csv(str(find_data_dir() / "f1_public_2021.csv"), usecols=PREDICTOR_COLS, low_memory=False)

    # Numeric sentinel "unknown" codes -> NaN
    for col, sentinel_vals in SENTINELS.items():
        f1[col] = f1[col].replace(sentinel_vals, np.nan)

    f1 = f1.rename(columns=FRIENDLY_NAMES)

    # Categorical "unknown" codes -> NaN, for every field that has one (see
    # module docstring: TESTING FINDING #2 below explains why these are not
    # kept as their own "Unknown" category).
    #
    # Compared as strings deliberately: several of these source columns are
    # int64 (AEducLvl, AMarStIj, ATrmEtio have no existing NaNs, so pandas
    # reads them as int, not float) while others are object/float. A plain
    # .replace({9: np.nan}) silently matches nothing on an int64 column
    # storing 9 if the dict key is "9" (or vice versa) -- this was caught by
    # testing: education_level's "9" code was NOT being converted, leaving a
    # residual era-correlated category after the fields above were fixed.
    # .astype(str) comparison is robust to the underlying dtype either way.
    for field, codes in UNKNOWN_CODES.items():
        if field in f1.columns:
            str_codes = {str(c) for c in codes}
            mask = f1[field].notna() & f1[field].astype(str).isin(str_codes)
            f1.loc[mask, field] = np.nan

    ais = f1["aisgrade_discharge"]
    f1["aisgrade_discharge"] = ais.where(ais.isna(), ais.astype(str).replace(LEGACY_AIS_GRADES))

    return f1


MOTOR_SCORE = "asia_motor_total_discharge"


def fill_missing_motor_scores(train, test, rng):
    """Replace each missing ASIA motor score with one drawn at random from
    the training patients who have a score.

    The score is unrecorded for 38% of the cohort, mostly older records.
    Filling them all with the median put over a third of patients on exactly
    50, and the model learned that 50 means "older record": its estimate
    dropped by about 6 points at 50 and nowhere else, for a patient who
    really scored 50 and for a form left blank alike. Drawing from the
    recorded scores leaves no single value that marks a missing score, so
    the model can no longer read missingness from it
    (src/experiment_motor_score.py, docs/test_report.md section 6.8).

    The draws come from the training split only, for both splits, so no
    test patient's score reaches the training data."""
    train, test = train.copy(), test.copy()
    recorded = train[MOTOR_SCORE].dropna().to_numpy()
    for part in (train, test):
        missing = part[MOTOR_SCORE].isna()
        part.loc[missing, MOTOR_SCORE] = rng.choice(recorded, size=int(missing.sum()))
    return train, test


# ----------------------------------------------------------------------------
# 3. MERGE + FINAL CLEANING
# ----------------------------------------------------------------------------
def main():
    print(f"Building targets from Form II (LABEL_MODE={LABEL_MODE!r}) ...")
    targets = build_targets()
    print(f"  Patients in modeling cohort: {len(targets)}")
    print(f"  RTW=1: {(targets['rtw_status']==1).sum()}, RTW=0: {(targets['rtw_status']==0).sum()}")

    print("Building predictors from Form I ...")
    predictors = build_predictors()

    print("Merging on UniID ...")
    df = predictors.merge(targets, left_on="UniID", right_index=True, how="inner")
    print(f"  Final modeling dataset: {df.shape[0]} patients, {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # TESTING FINDING #2 (see module docstring): missingness in this dataset
    # is pervasive AND strongly correlated with how long a patient has been
    # followed (older intake forms collected fewer fields) -- e.g. patients
    # with a missing ASIA motor score were followed for a mean of 14.4 years
    # vs 8.4 for patients with a recorded score. Under the fixed-horizon
    # label this means "was this field even collected" becomes a proxy for
    # "which pathway got this patient into the cohort" rather than a
    # clinical signal. A prior version of this pipeline kept a separate
    # "Unknown" category (categorical) / "_missing" flag (numeric) per
    # field; testing showed the model exploited these heavily (one such
    # flag alone carried ~4x the importance of the next real feature).
    # Fix: missing values are imputed with the field's own mode/median
    # (computed here from the full cohort) with NO missingness indicator
    # exposed to the model, for every field except the ASIA motor score,
    # which is filled after the split (fill_missing_motor_scores explains
    # why a single fill value was not enough there). This removes the leakage
    # channel; the tradeoff is that any *genuinely* clinical missingness
    # (e.g. AIS grade not assessable because a patient was unstable) is no
    # longer distinguishable from administrative non-collection -- see
    # docs/test_report.md for the full discussion and the case for treating
    # this dataset with proper survival/censoring methods as future work.
    # ------------------------------------------------------------------
    df["age_at_injury"] = pd.to_numeric(df["age_at_injury"], errors="coerce")

    # The motor score's median is what the API uses for a field left blank
    # on the form. It is the median of the recorded scores, and it is not
    # filled in here.
    numeric_medians = {c: float(df[c].median()) for c in ["age_at_injury", MOTOR_SCORE]}
    df["age_at_injury"] = df["age_at_injury"].fillna(numeric_medians["age_at_injury"])
    motor_score_recorded = int(df[MOTOR_SCORE].notna().sum())

    cat_cols_final = [FRIENDLY_NAMES.get(c, c) for c in CATEGORICAL_COLS]
    categorical_modes = {}
    for c in cat_cols_final:
        df[c] = df[c].astype("object")
        mode_val = str(df[c].mode(dropna=True).iloc[0])
        categorical_modes[c] = mode_val
        df[c] = df[c].fillna(mode_val).astype(str)

    with open(str(OUT_DIR / "imputation_modes.json"), "w") as f:
        json.dump({"numeric": numeric_medians, "categorical": categorical_modes}, f, indent=2)

    # ------------------------------------------------------------------
    # Layer 1 dataset
    # ------------------------------------------------------------------
    layer1_df = df.drop(columns=["rtw_timing"])
    layer1_train, layer1_test = train_test_split(
        layer1_df, test_size=0.2, random_state=RANDOM_STATE, stratify=layer1_df["rtw_status"])
    rng = np.random.default_rng(RANDOM_STATE)
    layer1_train, layer1_test = fill_missing_motor_scores(layer1_train, layer1_test, rng)

    layer1_train.to_csv(str(OUT_DIR / "layer1_train.csv"), index=False)
    layer1_test.to_csv(str(OUT_DIR / "layer1_test.csv"), index=False)

    # ------------------------------------------------------------------
    # Layer 2 dataset (only RTW-positive patients)
    # ------------------------------------------------------------------
    layer2_df = df[df["rtw_status"] == 1].drop(columns=["rtw_status"]).copy()
    layer2_train, layer2_test = train_test_split(
        layer2_df, test_size=0.2, random_state=RANDOM_STATE, stratify=layer2_df["rtw_timing"])
    layer2_train, layer2_test = fill_missing_motor_scores(layer2_train, layer2_test, rng)

    layer2_train.to_csv(str(OUT_DIR / "layer2_train.csv"), index=False)
    layer2_test.to_csv(str(OUT_DIR / "layer2_test.csv"), index=False)

    # Full cleaned dataset (for reference / EDA). A missing motor score is
    # left blank here: it is filled per split, above.
    df.to_csv(str(OUT_DIR / "full_cleaned_dataset.csv"), index=False)

    # ------------------------------------------------------------------
    # Preprocessing report
    # ------------------------------------------------------------------
    timing_labels = sorted(layer2_df["rtw_timing"].unique().tolist())
    report = {
        "label_mode": LABEL_MODE,
        "horizon_years": HORIZON_YEARS if LABEL_MODE == "fixed_horizon" else None,
        "total_patients_form1": 35675,
        "valid_followup_records": 71292,
        "patients_with_valid_outcome": int(len(df)),
        "motor_score_recorded": motor_score_recorded,
        "layer1_rtw_positive": int((df["rtw_status"] == 1).sum()),
        "layer1_non_rtw": int((df["rtw_status"] == 0).sum()),
        "layer2_total_rtw_patients": int(len(layer2_df)),
        "layer2_timing_breakdown": {t: int((layer2_df["rtw_timing"] == t).sum()) for t in timing_labels},
        "layer1_train_size": len(layer1_train),
        "layer1_test_size": len(layer1_test),
        "layer2_train_size": len(layer2_train),
        "layer2_test_size": len(layer2_test),
        "predictor_columns": [c for c in df.columns if c not in ("UniID", "rtw_status", "rtw_timing")],
    }
    with open(str(OUT_DIR / "preprocessing_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Preprocessing complete ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
