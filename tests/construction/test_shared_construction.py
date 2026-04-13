from macrocast.construction import normalise_model_list, parse_feature_specs, parse_model_spec


def test_shared_normalise_model_list() -> None:
    out = normalise_model_list(['rf', {'name': 'ar'}])
    assert out == [{'name': 'rf'}, {'name': 'ar'}]


def test_shared_parse_feature_specs() -> None:
    specs = parse_feature_specs({'factor_type': 'X', 'n_factors': 4, 'n_lags': 2})
    assert len(specs) == 1
    assert specs[0].factor_type == 'X'
    assert specs[0].n_factors == 4


def test_shared_parse_model_spec() -> None:
    spec = parse_model_spec({'name': 'rf'})
    assert spec.model_cls.__name__ == 'RFModel'
