"""Data cleaning and feature engineering for the Saudi used-car price dataset.

Every step here is row-wise (no statistics are learned from the data), except
`add_mileage_vs_age`, which is fitted on the training set only and then applied
to the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIDDEN_NULLS = ["-", "غير محدد"]  # "-" and "not specified" in Arabic
CAT_NULL_COLS = ["transmission", "trim", "specs", "seller_type", "color"]
DROP_COLS = ["id", "title", "listing_date"]
TARGET = "price"

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

CITY_MAP = {
    "riyadh": "Riyadh", "riyad": "Riyadh", "الرياض": "Riyadh",
    "jeddah": "Jeddah", "jedda": "Jeddah", "jiddah": "Jeddah", "جدة": "Jeddah",
    "dammam": "Dammam", "damam": "Dammam", "الدمام": "Dammam",
    "makkah": "Makkah", "mecca": "Makkah", "مكة": "Makkah",
    "madinah": "Madinah", "medina": "Madinah", "المدينة": "Madinah",
    "khobar": "Khobar", "al khobar": "Khobar", "الخبر": "Khobar",
    "buraydah": "Buraydah", "buraidah": "Buraydah", "بريدة": "Buraydah",
    "tabuk": "Tabuk", "tabouk": "Tabuk", "تبوك": "Tabuk",
    "abha": "Abha", "أبها": "Abha",
    "hail": "Hail", "ha'il": "Hail", "حائل": "Hail",
}


def extract_trim_from_title(text: str) -> str | float:
    """Recover a missing `trim` from the free-text listing title (Arabic or English)."""
    s = str(text).lower()
    # "Semi Full" must be checked first, otherwise "نص فل" would match "فل".
    if any(k in s for k in ["semi full", "semi-full", "نص فل", "نصف فل"]):
        return "Semi Full"
    if any(k in s for k in ["full option", "fully loaded", "فل اوبشن", "فل أوبشن", "فل كامل", "فل"]):
        return "Full Option"
    if any(k in s for k in ["standard", "base", "ستاندر", "عادي"]):
        return "Standard"
    return np.nan


def hijri_to_gregorian_year(year: pd.Series) -> pd.Series:
    """Some listings give the model year in the Hijri calendar (e.g. 1445).

    One Hijri year is ~0.970224 Gregorian years and the calendar starts in 622 CE.
    """
    return pd.Series(
        np.where(year < 1900, np.round(year * 0.970224 + 621.57).astype(int), year),
        index=year.index,
    )


def clean_mileage(mileage: pd.Series) -> pd.Series:
    """'70000 km', '٢٢٠٠٠', '1,200' -> integers."""
    return (
        mileage.astype(str)
        .str.translate(ARABIC_DIGITS)
        .str.replace(r"[^0-9]", "", regex=True)
        .astype(int)
    )


def clean_city(city: pd.Series) -> pd.Series:
    raw = city.astype(str).str.strip()
    return raw.str.lower().map(CITY_MAP).fillna(raw)


def title_condition_flags(title: pd.Series) -> pd.DataFrame:
    """Parse the car's condition out of the listing title."""
    t = title.astype(str).str.lower()
    minor = t.str.contains(r"minor accident|حادث بسيط|صدمة خفيفة|رش|سمكرة", regex=True)
    major = t.str.contains(r"damaged|مصدوم|تالف|حادث قوي", regex=True) | (
        t.str.contains(r"accident|حادث", regex=True) & ~minor
    )
    excellent = t.str.contains(r"excellent|نظيف جداً|صيانة وكالة|وكالة", regex=True)
    clean = t.str.contains(r"very clean|نظيف", regex=True) & ~excellent

    out = pd.DataFrame(
        {
            "is_damaged": major.astype(int),
            "has_minor_accident": minor.astype(int),
            "is_excellent": excellent.astype(int),
            "is_very_clean": clean.astype(int),
        },
        index=title.index,
    )
    out["condition_score"] = (
        -2 * out["is_damaged"] - out["has_minor_accident"]
        + out["is_very_clean"] + 2 * out["is_excellent"]
    )
    return out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning + feature engineering (safe to apply to train and test separately)."""
    d = df.copy()

    # 1. Hidden nulls -> NaN
    for col in CAT_NULL_COLS:
        d[col] = d[col].replace(HIDDEN_NULLS, np.nan)

    # 2-4. Calendar, mileage and city normalisation
    d["model_year"] = hijri_to_gregorian_year(d["model_year"])
    d["mileage"] = clean_mileage(d["mileage"])
    d["city"] = clean_city(d["city"])

    # 5. Age and listing-date features
    listing_dt = pd.to_datetime(d["listing_date"])
    d["listing_year"] = listing_dt.dt.year
    d["listing_month"] = listing_dt.dt.month
    d["listing_days"] = (listing_dt - pd.Timestamp("2025-01-01")).dt.days
    d["car_age"] = (d["listing_year"] - d["model_year"]).clip(lower=0)
    d["exact_age"] = (
        listing_dt.dt.year + (listing_dt.dt.dayofyear - 1) / 365.25 - d["model_year"]
    ).clip(lower=0)

    # 6. Mileage transforms and usage intensity
    d["log_mileage"] = np.log1p(d["mileage"])
    d["sqrt_mileage"] = np.sqrt(d["mileage"])
    d["km_per_year"] = d["mileage"] / (d["exact_age"] + 0.5)
    d["log_km_per_year"] = np.log1p(d["km_per_year"])

    # 7. Condition from title
    d = pd.concat([d, title_condition_flags(d["title"])], axis=1)

    # 8. Recover missing trim from the title, then keep remaining nulls as their own level
    missing_trim = d["trim"].isna()
    d.loc[missing_trim, "trim"] = d.loc[missing_trim, "title"].apply(extract_trim_from_title)
    for col in CAT_NULL_COLS:
        d[col] = d[col].fillna("Unknown")

    # 9. Domain features
    d["brand_model"] = d["brand"].astype(str) + "_" + d["model"].astype(str)
    d["is_luxury"] = d["brand"].isin(["BMW", "Lexus"]).astype(int)
    d["trim_ordinal"] = d["trim"].map(
        {"Standard": 1, "Semi Full": 2, "Full Option": 3, "Unknown": 2}
    )
    return d


def clean_train_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Drop placeholder prices (0 / 1 SAR) and duplicate listings (same car re-posted)."""
    train = train[~train[TARGET].isin([0, 1])].copy()
    dedup_cols = [c for c in train.columns if c not in ["id", "title"]]
    return train.drop_duplicates(subset=dedup_cols).reset_index(drop=True)


def add_mileage_vs_age(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Mileage relative to the median mileage of cars of the same age (fitted on train only)."""
    age_median = train.groupby("car_age")["mileage"].median()
    global_median = train["mileage"].median()
    for d in (train, test):
        expected = d["car_age"].map(age_median).fillna(global_median)
        d["mileage_vs_age_ratio"] = d["mileage"] / (expected + 1.0)


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Full pipeline: returns X, y, X_test (categoricals aligned to train categories)."""
    train = clean_train_rows(preprocess(train_raw))
    test = preprocess(test_raw)
    add_mileage_vs_age(train, test)

    X = train.drop(columns=DROP_COLS + [TARGET])
    y = train[TARGET]
    X_test = test.drop(columns=DROP_COLS)

    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
        X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)
    return X, y, X_test
