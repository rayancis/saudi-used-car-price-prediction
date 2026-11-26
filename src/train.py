"""Train the 2-level stacking ensemble and write a Kaggle submission.

Level 1: LightGBM, XGBoost, CatBoost and an MLP, each trained with K-fold CV on
log1p(price) to produce out-of-fold (OOF) predictions for train and averaged
fold predictions for test.
Level 2: RidgeCV fitted on the OOF predictions learns how much to trust each model.

Usage:
    python src/train.py --data-dir data --out submissions/submission.csv --folds 5
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

try:  # works both as `python src/train.py` and `from src.train import ...`
    from features import build_features
except ImportError:  # pragma: no cover
    from src.features import build_features

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

MODELS = ["lgb", "xgb", "cat", "mlp"]
SEED = 42


def rmsle_log(y_true_log, y_pred_log) -> float:
    """RMSE in log1p space == RMSLE in price space."""
    return float(np.sqrt(np.mean((np.asarray(y_true_log) - np.asarray(y_pred_log)) ** 2)))


def make_mlp(cat_cols, num_cols, seed):
    """MLP with scaling + one-hot encoding fitted inside each fold (no leakage)."""
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3,
        learning_rate_init=1e-3, max_iter=300, early_stopping=True,
        n_iter_no_change=15, random_state=seed,
    )
    return make_pipeline(prep, mlp)


def run_stacking(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
                 n_splits: int = 5, verbose: bool = True):
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # CatBoost and the sklearn MLP want plain string categoricals
    X_str, X_test_str = X.copy(), X_test.copy()
    for c in cat_cols:
        X_str[c] = X_str[c].astype(str)
        X_test_str[c] = X_test_str[c].astype(str)

    y_log = np.log1p(y.values)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(X)) for m in MODELS}
    test = {m: np.zeros(len(X_test)) for m in MODELS}

    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        seed = SEED + fold

        lgb = LGBMRegressor(
            n_estimators=3000, learning_rate=0.025, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, min_child_samples=25,
            random_state=seed, verbosity=-1,
        )
        lgb.fit(X.iloc[tr], y_log[tr], eval_set=[(X.iloc[va], y_log[va])],
                callbacks=[early_stopping(100, verbose=False)])
        oof["lgb"][va] = lgb.predict(X.iloc[va])
        test["lgb"] += lgb.predict(X_test) / n_splits

        xgb = XGBRegressor(
            n_estimators=3000, learning_rate=0.025, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0, tree_method="hist",
            enable_categorical=True, early_stopping_rounds=100, random_state=seed, verbosity=0,
        )
        xgb.fit(X.iloc[tr], y_log[tr], eval_set=[(X.iloc[va], y_log[va])], verbose=False)
        oof["xgb"][va] = xgb.predict(X.iloc[va])
        test["xgb"] += xgb.predict(X_test) / n_splits

        cat = CatBoostRegressor(
            iterations=4000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
            loss_function="RMSE", random_seed=seed, verbose=0,
        )
        cat.fit(X_str.iloc[tr], y_log[tr], cat_features=cat_cols,
                eval_set=(X_str.iloc[va], y_log[va]), early_stopping_rounds=100)
        oof["cat"][va] = cat.predict(X_str.iloc[va])
        test["cat"] += cat.predict(X_test_str) / n_splits

        mlp = make_mlp(cat_cols, num_cols, seed)
        mlp.fit(X_str.iloc[tr], y_log[tr])
        oof["mlp"][va] = mlp.predict(X_str.iloc[va])
        test["mlp"] += mlp.predict(X_test_str) / n_splits

        if verbose:
            scores = " | ".join(f"{m.upper()} {rmsle_log(y_log[va], oof[m][va]):.4f}" for m in MODELS)
            print(f"Fold {fold}/{n_splits}: {scores}")

    meta_X = np.column_stack([oof[m] for m in MODELS])
    meta_X_test = np.column_stack([test[m] for m in MODELS])
    meta = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(meta_X, y_log)
    oof_stack = meta.predict(meta_X)
    test_stack = meta.predict(meta_X_test)

    if verbose:
        print("\nOOF RMSLE per model:")
        for m in MODELS:
            print(f"  {m.upper():4} {rmsle_log(y_log, oof[m]):.5f}")
        print(f"  STACK {rmsle_log(y_log, oof_stack):.5f}")
        print("Meta weights (LGB, XGB, CAT, MLP):", np.round(meta.coef_, 3))

    return {"oof": oof, "test": test, "oof_stack": oof_stack,
            "test_stack": test_stack, "meta": meta, "y_log": y_log}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="submissions/submission.csv")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    data = Path(a.data_dir)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")

    X, y, X_test = build_features(train, test)
    res = run_stacking(X, y, X_test, n_splits=a.folds)

    sub = pd.DataFrame({"id": test["id"].values,
                        "price": np.clip(np.expm1(res["test_stack"]), 0, None)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(a.out, index=False)
    print(f"\nSaved {a.out} {sub.shape}")


if __name__ == "__main__":
    main()
