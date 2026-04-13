from __future__ import annotations

from typing import Any

from macrocast.pipeline.components import (
    CVScheme,
    CVSchemeType,
    LossFunction,
    Nonlinearity,
    Regularization,
)
from macrocast.pipeline.experiment import FeatureSpec, ModelSpec
from macrocast.pipeline.models import (
    GBModel,
    KRRModel,
    LSTMModel,
    NNModel,
    RFModel,
    SVRLinearModel,
    SVRRBFModel,
    XGBoostModel,
)
from macrocast.pipeline.r_models import (
    AdaptiveLassoModel,
    ARDIModel,
    ARModel,
    BoogingModel,
    BVARModel,
    ElasticNetModel,
    GroupLassoModel,
    LassoModel,
    RidgeModel,
    TVPRidgeModel,
)

_MODEL_REGISTRY: dict[str, tuple] = {
    'krr': (KRRModel, Nonlinearity.KRR, Regularization.FACTORS),
    'svr_rbf': (SVRRBFModel, Nonlinearity.SVR_RBF, Regularization.FACTORS),
    'svr_linear': (SVRLinearModel, Nonlinearity.SVR_LINEAR, Regularization.NONE),
    'rf': (RFModel, Nonlinearity.RANDOM_FOREST, Regularization.NONE),
    'xgboost': (XGBoostModel, Nonlinearity.XGBOOST, Regularization.NONE),
    'gb': (GBModel, Nonlinearity.GRADIENT_BOOSTING, Regularization.NONE),
    'nn': (NNModel, Nonlinearity.NEURAL_NET, Regularization.NONE),
    'lstm': (LSTMModel, Nonlinearity.LSTM, Regularization.NONE),
    'ar': (ARModel, Nonlinearity.LINEAR, Regularization.NONE),
    'ardi': (ARDIModel, Nonlinearity.LINEAR, Regularization.FACTORS),
    'ridge': (RidgeModel, Nonlinearity.LINEAR, Regularization.RIDGE),
    'lasso': (LassoModel, Nonlinearity.LINEAR, Regularization.LASSO),
    'adaptive_lasso': (AdaptiveLassoModel, Nonlinearity.LINEAR, Regularization.ADAPTIVE_LASSO),
    'group_lasso': (GroupLassoModel, Nonlinearity.LINEAR, Regularization.GROUP_LASSO),
    'elastic_net': (ElasticNetModel, Nonlinearity.LINEAR, Regularization.ELASTIC_NET),
    'tvp_ridge': (TVPRidgeModel, Nonlinearity.LINEAR, Regularization.RIDGE),
    'booging': (BoogingModel, Nonlinearity.LINEAR, Regularization.NONE),
    'bvar': (BVARModel, Nonlinearity.LINEAR, Regularization.NONE),
}

_MODEL_ALIASES: dict[str, str] = {
    'al': 'adaptive_lasso',
    'en': 'elastic_net',
    'gl': 'group_lasso',
    'tvp': 'tvp_ridge',
    'boog': 'booging',
    'gbm': 'gb',
}

_REGULARIZATION_MAP: dict[str, Regularization] = {r.value: r for r in Regularization}
_LOSS_MAP: dict[str, LossFunction] = {
    'l2': LossFunction.L2,
    'epsilon_insensitive': LossFunction.EPSILON_INSENSITIVE,
}


def parse_feature_specs(feat_sec: dict) -> list[FeatureSpec]:
    preset_name = (feat_sec.get('preset') or '').lower()
    if preset_name == 'clss2021':
        from macrocast.replication.clss2021 import CLSS2021
        return list(CLSS2021.info_sets().values())
    if preset_name and preset_name != 'none':
        raise ValueError(f"Unknown preset '{preset_name}'. Valid values: clss2021, none.")
    spec = FeatureSpec(
        factor_type=feat_sec.get('factor_type', 'X'),
        n_factors=feat_sec.get('n_factors', 8),
        n_lags=feat_sec.get('n_lags', 4),
        p_marx=feat_sec.get('p_marx', 12),
        append_x_factors=feat_sec.get('append_x_factors', False),
        append_marx=feat_sec.get('append_marx', False),
        append_raw_x=feat_sec.get('append_raw_x', False),
        append_levels=feat_sec.get('append_levels', False),
        standardize_X=feat_sec.get('standardize_X', True),
        standardize_Z=feat_sec.get('standardize_Z', False),
        lookback=feat_sec.get('lookback', 12),
    )
    return [spec]


def normalise_model_list(models: list) -> list[dict]:
    return [{'name': m} if isinstance(m, str) else m for m in models]


def resolve_model_name(name: str) -> str:
    key = name.strip().lower()
    return _MODEL_ALIASES.get(key, key)


def parse_model_spec(m: dict) -> ModelSpec:
    raw_name = m.get('name', '')
    name = resolve_model_name(raw_name)
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{raw_name}'. Valid names: {sorted(_MODEL_REGISTRY)} and aliases: {sorted(_MODEL_ALIASES)}.")
    model_cls, _default_nonlin, default_reg = _MODEL_REGISTRY[name]
    reg_str = m.get('regularization', default_reg.value)
    regularization = _REGULARIZATION_MAP.get(reg_str, default_reg)
    cv_str = (m.get('cv_scheme', 'kfold') or 'kfold').lower()
    k = int(m.get('kfold_k', 5))
    if cv_str == 'kfold':
        cv_scheme: CVSchemeType = CVScheme.KFOLD(k=k)
    elif cv_str == 'bic':
        cv_scheme = CVScheme.BIC
    elif cv_str == 'poos':
        cv_scheme = CVScheme.POOS
    else:
        raise ValueError(f"Unknown cv_scheme '{cv_str}'. Valid: kfold, bic, poos.")
    loss_str = (m.get('loss_function', 'l2') or 'l2').lower()
    loss_function = _LOSS_MAP.get(loss_str, LossFunction.L2)
    kwargs: dict[str, Any] = m.get('kwargs', {}) or {}
    model_id = m.get('model_id')
    return ModelSpec(
        model_cls=model_cls,
        regularization=regularization,
        cv_scheme=cv_scheme,
        loss_function=loss_function,
        model_kwargs=kwargs,
        model_id=model_id,
    )
