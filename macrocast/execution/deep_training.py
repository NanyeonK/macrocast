from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ..tuning import HPDistribution, TuningSpec, run_tuning
from ..tuning.hp_spaces import MODEL_HP_SPACES


def make_model_instance(model_family: str, hp: dict[str, Any] | None = None):
    hp = dict(hp or {})
    if model_family == "ols":
        return LinearRegression()
    if model_family == "ridge":
        return Ridge(alpha=float(hp.get("alpha", 1.0)))
    if model_family == "lasso":
        return Lasso(alpha=float(hp.get("alpha", 1e-4)), max_iter=10000)
    if model_family == "elasticnet":
        return ElasticNet(alpha=float(hp.get("alpha", 1e-4)), l1_ratio=float(hp.get("l1_ratio", 0.5)), max_iter=10000)
    if model_family == "bayesianridge":
        return BayesianRidge()
    if model_family == "huber":
        return HuberRegressor(epsilon=float(hp.get("epsilon", 1.35)), alpha=float(hp.get("alpha", 0.0001)))
    if model_family == "svr_linear":
        return LinearSVR(C=float(hp.get("C", 1.0)), epsilon=float(hp.get("epsilon", 0.01)), max_iter=50000, random_state=42)
    if model_family == "svr_rbf":
        return SVR(kernel="rbf", C=float(hp.get("C", 1.0)), epsilon=float(hp.get("epsilon", 0.01)), gamma=hp.get("gamma", "scale"))
    if model_family == "randomforest":
        return RandomForestRegressor(n_estimators=int(hp.get("n_estimators", 200)), max_depth=None if hp.get("max_depth") is None else int(hp.get("max_depth")), random_state=42)
    if model_family == "extratrees":
        return ExtraTreesRegressor(n_estimators=int(hp.get("n_estimators", 200)), max_depth=None if hp.get("max_depth") is None else int(hp.get("max_depth")), random_state=42)
    if model_family == "gbm":
        return GradientBoostingRegressor(n_estimators=int(hp.get("n_estimators", 100)), learning_rate=float(hp.get("learning_rate", 0.05)), max_depth=int(hp.get("max_depth", 3)), random_state=42)
    if model_family == "xgboost":
        return XGBRegressor(n_estimators=int(hp.get("n_estimators", 100)), max_depth=int(hp.get("max_depth", 3)), learning_rate=float(hp.get("learning_rate", 0.05)), random_state=42, verbosity=0)
    if model_family == "lightgbm":
        return LGBMRegressor(n_estimators=int(hp.get("n_estimators", 100)), num_leaves=int(hp.get("num_leaves", 31)), learning_rate=float(hp.get("learning_rate", 0.05)), random_state=42, verbosity=-1)
    if model_family == "catboost":
        return CatBoostRegressor(iterations=int(hp.get("iterations", 100)), learning_rate=float(hp.get("learning_rate", 0.05)), depth=int(hp.get("depth", 4)), verbose=False, random_seed=42)
    if model_family == "mlp":
        return MLPRegressor(hidden_layer_sizes=hp.get("hidden_layer_sizes", (32,)), alpha=float(hp.get("alpha", 1e-4)), learning_rate_init=float(hp.get("learning_rate_init", 1e-3)), max_iter=500, random_state=42)
    raise ValueError(f"unsupported model_family {model_family!r}")


def fit_adaptive_lasso(X: np.ndarray, y: np.ndarray, hp: dict[str, Any] | None = None) -> object:
    hp = dict(hp or {})
    gamma = float(hp.get("gamma", 1.0))
    init_estimator = str(hp.get("init_estimator", "ridge"))
    alpha = float(hp.get("alpha", 1e-3))
    init = Ridge(alpha=1.0).fit(X, y) if init_estimator == "ridge" else LinearRegression().fit(X, y)
    weights = 1.0 / (np.abs(init.coef_) ** gamma + 1e-6)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X / weights, y)
    model._adaptive_weights = weights
    return model


def predict_adaptive_lasso(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X / model._adaptive_weights)


def _default_hp_space(model_family: str, X_train: np.ndarray) -> dict[str, HPDistribution]:
    hp_space = dict(MODEL_HP_SPACES.get(model_family, {}))
    if model_family in {"pcr", "pls"}:
        max_comp = max(1, min(10, X_train.shape[0], X_train.shape[1]))
        hp_space = {"n_components": HPDistribution("int", 1, max_comp)}
    return hp_space


def fit_with_optional_tuning(model_family: str, X_train: np.ndarray, y_train: np.ndarray, training_spec: dict[str, Any]):
    algo = training_spec.get("search_algorithm", "grid_search")
    convergence_handling = training_spec.get("convergence_handling", "mark_fail")
    if not bool(training_spec.get("enable_tuning", False)) or algo not in {"grid_search", "random_search", "bayesian_optimization", "genetic_algorithm"}:
        try:
            if model_family == "adaptivelasso":
                return fit_adaptive_lasso(X_train, y_train), {}
            model = make_model_instance(model_family)
            model.fit(X_train, y_train)
            return model, {}
        except Exception:
            if convergence_handling == "fallback_to_safe_hp":
                if model_family == "adaptivelasso":
                    return fit_adaptive_lasso(X_train, y_train, {"gamma": 1.0}), {"fallback": True}
                model = make_model_instance(model_family)
                model.fit(X_train, y_train)
                return model, {"fallback": True}
            raise
    tuning_spec = TuningSpec(
        search_algorithm=algo,
        tuning_objective=training_spec.get("tuning_objective", "validation_mse"),
        tuning_budget={
            "max_trials": int(training_spec.get("max_trials", 6)),
            "max_time_seconds": float(training_spec.get("max_time_seconds", 15.0)),
            "early_stop_trials": int(training_spec.get("early_stop_trials", 3)),
        },
        hp_space=_default_hp_space(model_family, X_train),
        validation_size_rule=training_spec.get("validation_size_rule", "ratio"),
        validation_size_config={
            "ratio": float(training_spec.get("validation_ratio", 0.2)),
            "n": int(training_spec.get("validation_n", 5)),
            "years": int(training_spec.get("validation_years", 1)),
            "obs_per_year": int(training_spec.get("obs_per_year", 12)),
        },
        validation_location=training_spec.get("validation_location", "last_block"),
        embargo_gap=training_spec.get("embargo_gap", "none"),
        embargo_gap_size=int(training_spec.get("embargo_gap_size", 0)),
        seed=int(training_spec.get("random_seed", 42)),
    )
    if model_family == "adaptivelasso":
        factory = lambda hp: fit_adaptive_lasso(X_train, y_train, hp)
        # tuning engine expects unfit model factory; adapt via tiny wrapper class
        class _AdaptiveWrap:
            def __init__(self, hp):
                self.hp = hp
            def fit(self, X, y):
                self.model = fit_adaptive_lasso(X, y, self.hp)
                return self
            def predict(self, X):
                return predict_adaptive_lasso(self.model, X)
        try:
            result = run_tuning(model_family, lambda hp: _AdaptiveWrap(hp), X_train, y_train, tuning_spec)
            return fit_adaptive_lasso(X_train, y_train, result.best_hp), {"best_hp": result.best_hp, "best_score": result.best_score, "total_trials": result.total_trials}
        except Exception:
            if convergence_handling == "fallback_to_safe_hp":
                return fit_adaptive_lasso(X_train, y_train, {"gamma": 1.0}), {"fallback": True}
            raise
    try:
        result = run_tuning(model_family, lambda hp: make_model_instance(model_family, hp), X_train, y_train, tuning_spec)
        model = make_model_instance(model_family, result.best_hp)
        model.fit(X_train, y_train)
        return model, {"best_hp": result.best_hp, "best_score": result.best_score, "total_trials": result.total_trials}
    except Exception:
        if convergence_handling == "fallback_to_safe_hp":
            model = make_model_instance(model_family)
            model.fit(X_train, y_train)
            return model, {"fallback": True}
        raise


def resolve_factor_count(X_train: np.ndarray, y_train: np.ndarray, training_spec: dict[str, Any]) -> int:
    mode = training_spec.get("factor_count", "fixed")
    max_k = max(1, min(int(training_spec.get("max_factors", 5)), X_train.shape[0], X_train.shape[1]))
    if mode == "fixed":
        return max(1, min(int(training_spec.get("fixed_factor_count", 3)), max_k))
    if mode == "cv_select":
        best_k, best_sse = 1, math.inf
        for k in range(1, max_k + 1):
            scores = PCA(n_components=k).fit_transform(X_train)
            recon = PCA(n_components=k).fit(X_train).inverse_transform(scores)
            sse = float(np.mean((X_train - recon) ** 2))
            if sse < best_sse:
                best_k, best_sse = k, sse
        return best_k
    if mode == "BaiNg_rule":
        Xc = X_train - X_train.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        T, N = X_train.shape[0], X_train.shape[1]
        penalties = []
        for k in range(1, max_k + 1):
            sigma2 = float(np.sum(s[k:] ** 2) / (T * N)) if k < len(s) else 0.0
            ic = math.log(max(sigma2, 1e-12)) + k * ((N + T) / (N * T)) * math.log((N * T) / (N + T))
            penalties.append((ic, k))
        return min(penalties)[1]
    return max(1, min(3, max_k))


def build_factor_panel(X_train_df: pd.DataFrame, y_train: np.ndarray, X_pred_df: pd.DataFrame, training_spec: dict[str, Any], include_ar_lags: bool = False) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train_df.to_numpy(dtype=float)
    X_pred = X_pred_df.to_numpy(dtype=float)
    n_components = resolve_factor_count(X_train, y_train, training_spec)
    pca = PCA(n_components=n_components)
    F_train = pca.fit_transform(X_train)
    F_pred = pca.transform(X_pred)
    if not include_ar_lags:
        return F_train, F_pred
    lag_order = int(training_spec.get("factor_ar_lags", 1))
    if len(y_train) <= lag_order:
        raise ValueError("insufficient target history for factors_plus_AR")
    y_lags = []
    for idx in range(lag_order, len(y_train)):
        y_lags.append(y_train[idx-lag_order:idx][::-1])
    F_train = F_train[lag_order:]
    lag_arr = np.asarray(y_lags, dtype=float)
    X_aug = np.concatenate([F_train, lag_arr], axis=1)
    pred_lags = np.asarray(y_train[-lag_order:][::-1], dtype=float).reshape(1, -1)
    X_pred_aug = np.concatenate([F_pred, pred_lags], axis=1)
    return X_aug, X_pred_aug


def fit_factor_model(model_family: str, X_train_df: pd.DataFrame, y_train: np.ndarray, X_pred_df: pd.DataFrame, training_spec: dict[str, Any], include_ar_lags: bool = False) -> tuple[float, dict[str, Any]]:
    if model_family == "pcr":
        X_train, X_pred = build_factor_panel(X_train_df, y_train, X_pred_df, {**training_spec, "factor_count": training_spec.get("factor_count", "fixed")}, include_ar_lags=False)
        model, tuning = fit_with_optional_tuning("ols", X_train, y_train, training_spec)
        return float(model.predict(X_pred)[0]), {"tuning": tuning}
    if model_family == "pls":
        n_components = resolve_factor_count(X_train_df.to_numpy(dtype=float), y_train, training_spec)
        model = PLSRegression(n_components=n_components)
        model.fit(X_train_df.to_numpy(dtype=float), y_train)
        return float(model.predict(X_pred_df.to_numpy(dtype=float))[0]), {"n_components": n_components}
    if model_family == "factor_augmented_linear":
        X_train, X_pred = build_factor_panel(X_train_df, y_train, X_pred_df, training_spec, include_ar_lags=True)
        model, tuning = fit_with_optional_tuning("ols", X_train, y_train[1:], training_spec)
        return float(model.predict(X_pred)[0]), {"tuning": tuning}
    X_train, X_pred = build_factor_panel(X_train_df, y_train, X_pred_df, training_spec, include_ar_lags=include_ar_lags)
    y_used = y_train[1:] if include_ar_lags else y_train
    base_family = "ols" if model_family == "factor_pca" else model_family
    model, tuning = fit_with_optional_tuning(base_family, X_train, y_used, training_spec)
    pred = model.predict(X_pred)
    return float(pred[0]), {"tuning": tuning}
