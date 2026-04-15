from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='model_family',
    layer='3_training',
    axis_type='enum',
    default_policy='sweep',
    entries=(
        EnumRegistryEntry(id='ar', description='ar', status='operational', priority="A"),
        EnumRegistryEntry(id='ols', description='ols', status='operational', priority="A"),
        EnumRegistryEntry(id='ridge', description='ridge', status='operational', priority="A"),
        EnumRegistryEntry(id='lasso', description='lasso', status='operational', priority="A"),
        EnumRegistryEntry(id='elasticnet', description='elasticnet', status='operational', priority="A"),
        EnumRegistryEntry(id='bayesianridge', description='bayesianridge', status='operational', priority="A"),
        EnumRegistryEntry(id='adaptivelasso', description='adaptivelasso', status='planned', priority="A"),
        EnumRegistryEntry(id='svr_linear', description='svr linear', status='planned', priority="A"),
        EnumRegistryEntry(id='svr_rbf', description='svr rbf', status='planned', priority="A"),
        EnumRegistryEntry(id='randomforest', description='randomforest', status='operational', priority="A"),
        EnumRegistryEntry(id='extratrees', description='extratrees', status='operational', priority="A"),
        EnumRegistryEntry(id='gbm', description='gbm', status='operational', priority="A"),
        EnumRegistryEntry(id='xgboost', description='xgboost', status='operational', priority="A"),
        EnumRegistryEntry(id='lightgbm', description='lightgbm', status='operational', priority="A"),
        EnumRegistryEntry(id='mlp', description='mlp', status='operational', priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
