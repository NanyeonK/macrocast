"""Tests for macrocast.evaluation.variable_importance.

Covers: group inference, VI extraction, normalisation, averaging.
"""

from __future__ import annotations

import pandas as pd
import pytest

from macrocast.interpretation.variable_importance import (
    CLSS_VI_GROUPS,
    _infer_group,
    average_vi_by_horizon,
    extract_vi_dataframe,
    vi_by_group,
)
from macrocast.pipeline.results import ForecastRecord, ResultSet
from macrocast.pipeline.components import (
    CVScheme,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    model_id: str = "rf_fx_marx",
    feature_set: str = "F-X-MARX",
    horizon: int = 12,
    forecast_date: str = "2010-01-01",
    feature_importances: dict[str, float] | None = None,
) -> ForecastRecord:
    """Construct a minimal ForecastRecord for testing."""
    return ForecastRecord(
        experiment_id="test-exp",
        model_id=model_id,
        nonlinearity=Nonlinearity.RANDOM_FOREST,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.KFOLD(),
        loss_function=LossFunction.L2,
        window=Window.EXPANDING,
        horizon=horizon,
        train_end=pd.Timestamp("2009-12-01"),
        forecast_date=pd.Timestamp(forecast_date),
        y_hat=0.5,
        y_true=0.6,
        n_train=120,
        n_factors=4,
        n_lags=2,
        feature_importances=feature_importances,
    )


# Shared synthetic importance dict used across several tests
_IMPORTANCES = {
    "y_lag_1":       0.10,
    "y_lag_2":       0.05,
    "y_marx_lag_1":  0.08,
    "MAF_factor_1":  0.20,
    "factor_1":      0.07,
    "MARX_0":        0.15,
    "MARX_1":        0.10,
    "X_5":           0.25,
}


# ---------------------------------------------------------------------------
# Tests: _infer_group
# ---------------------------------------------------------------------------


class TestInferGroup:
    def test_ar_lags(self) -> None:
        assert _infer_group("y_lag_1") == "ar"
        assert _infer_group("y_lag_12") == "ar"

    def test_ar_marx_lags(self) -> None:
        assert _infer_group("y_marx_lag_1") == "ar_marx"
        assert _infer_group("y_marx_lag_3") == "ar_marx"

    def test_maf_factor(self) -> None:
        assert _infer_group("MAF_factor_1") == "factors"
        assert _infer_group("MAF_factor_10") == "factors"

    def test_plain_factor(self) -> None:
        assert _infer_group("factor_1") == "factors"
        assert _infer_group("factor_20") == "factors"

    def test_marx(self) -> None:
        assert _infer_group("MARX_0") == "marx"
        assert _infer_group("MARX_5") == "marx"

    def test_x(self) -> None:
        assert _infer_group("X_5") == "x"
        assert _infer_group("X_100") == "x"

    def test_level(self) -> None:
        assert _infer_group("level_INDPRO") == "levels"

    def test_other(self) -> None:
        assert _infer_group("unknown_feature") == "other"
        assert _infer_group("") == "other"

    def test_ar_marx_not_confused_with_ar(self) -> None:
        # y_marx_lag_ must NOT fall through to "ar"
        assert _infer_group("y_marx_lag_2") == "ar_marx"


# ---------------------------------------------------------------------------
# Tests: extract_vi_dataframe
# ---------------------------------------------------------------------------


class TestExtractViDataframe:
    def test_extract_vi_skips_none_importances(self) -> None:
        """Records with feature_importances=None must not appear in output."""
        records = [
            _make_record(
                forecast_date="2010-01-01",
                feature_importances=_IMPORTANCES,
            ),
            _make_record(
                forecast_date="2010-02-01",
                feature_importances=None,  # should be skipped
            ),
            _make_record(
                forecast_date="2010-03-01",
                feature_importances={"y_lag_1": 0.5, "MARX_0": 0.5},
            ),
        ]
        df = extract_vi_dataframe(records)
        # Only records from 2010-01-01 and 2010-03-01 contribute rows
        dates = df["date"].dt.strftime("%Y-%m-%d").unique().tolist()
        assert "2010-02-01" not in dates
        assert "2010-01-01" in dates
        assert "2010-03-01" in dates

    def test_extract_vi_skips_none_all_none(self) -> None:
        """All-None importances must yield an empty DataFrame."""
        records = [
            _make_record(feature_importances=None),
            _make_record(feature_importances=None),
        ]
        df = extract_vi_dataframe(records)
        assert df.empty

    def test_extract_vi_group_inference(self) -> None:
        """Feature names in output must map to the expected groups."""
        records = [_make_record(feature_importances=_IMPORTANCES)]
        df = extract_vi_dataframe(records)

        expected = {
            "y_lag_1":       "ar",
            "y_lag_2":       "ar",
            "y_marx_lag_1":  "ar_marx",
            "MAF_factor_1":  "factors",
            "factor_1":      "factors",
            "MARX_0":        "marx",
            "MARX_1":        "marx",
            "X_5":           "x",
        }
        for feat, grp in expected.items():
            row = df[df["feature_name"] == feat]
            assert len(row) == 1, f"Feature {feat!r} not found"
            assert row.iloc[0]["group"] == grp, (
                f"Expected group {grp!r} for {feat!r}, "
                f"got {row.iloc[0]['group']!r}"
            )

    def test_extract_vi_columns(self) -> None:
        """Output must have the required column set."""
        records = [_make_record(feature_importances={"y_lag_1": 0.3})]
        df = extract_vi_dataframe(records)
        expected_cols = {
            "model_id",
            "feature_set",
            "horizon",
            "date",
            "feature_name",
            "importance",
            "group",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_extract_vi_accepts_result_set(self) -> None:
        """extract_vi_dataframe must accept a ResultSet, not just a list."""
        rs = ResultSet()
        rs.add(_make_record(feature_importances={"y_lag_1": 1.0}))
        rs.add(_make_record(feature_importances=None))
        df = extract_vi_dataframe(rs)
        assert len(df) == 1

    def test_extract_vi_date_field(self) -> None:
        """The ``date`` column must reflect forecast_date."""
        fc_date = "2015-06-01"
        records = [
            _make_record(
                forecast_date=fc_date,
                feature_importances={"y_lag_1": 1.0},
            )
        ]
        df = extract_vi_dataframe(records)
        assert df.iloc[0]["date"] == pd.Timestamp(fc_date)


# ---------------------------------------------------------------------------
# Tests: vi_by_group
# ---------------------------------------------------------------------------


class TestViByGroup:
    def _make_vi_df(self) -> pd.DataFrame:
        """Two dates, same model/feature_set/horizon, same importance dict."""
        records = [
            _make_record(
                forecast_date="2010-01-01",
                feature_importances=_IMPORTANCES,
            ),
            _make_record(
                forecast_date="2010-02-01",
                feature_importances=_IMPORTANCES,
            ),
        ]
        return extract_vi_dataframe(records)

    def test_vi_by_group_shares_sum_to_one(self) -> None:
        """With normalize=True, shares must sum to 1.0 per cell."""
        vi_df = self._make_vi_df()
        agg = vi_by_group(vi_df, normalize=True)

        cell_keys = ["model_id", "feature_set", "horizon", "date"]
        totals = agg.groupby(cell_keys)["importance_share"].sum()
        for val in totals:
            assert abs(val - 1.0) < 1e-10, f"Shares do not sum to 1: {val}"

    def test_vi_by_group_no_normalize(self) -> None:
        """With normalize=False, the importance_share column holds raw sums."""
        vi_df = self._make_vi_df()
        agg = vi_by_group(vi_df, normalize=False)

        # The "x" group has only X_5 = 0.25 in _IMPORTANCES
        x_row = agg[agg["group"] == "x"]
        assert len(x_row) == 2  # two dates
        for val in x_row["importance_share"]:
            assert abs(val - 0.25) < 1e-10

        # The "ar" group has y_lag_1=0.10 + y_lag_2=0.05 = 0.15
        ar_row = agg[agg["group"] == "ar"]
        for val in ar_row["importance_share"]:
            assert abs(val - 0.15) < 1e-10

    def test_vi_by_group_columns(self) -> None:
        """Output must have the required column set."""
        vi_df = self._make_vi_df()
        agg = vi_by_group(vi_df)
        expected = {
            "model_id",
            "feature_set",
            "horizon",
            "date",
            "group",
            "importance_share",
        }
        assert expected.issubset(set(agg.columns))

    def test_vi_by_group_custom_group_map(self) -> None:
        """group_map override reassigns feature names to custom groups."""
        vi_df = self._make_vi_df()
        custom_map = {"X_5": "custom_group"}
        agg = vi_by_group(vi_df, group_map=custom_map, normalize=False)
        assert "custom_group" in agg["group"].values

    def test_vi_by_group_empty_input(self) -> None:
        """Empty input must return an empty DataFrame with correct columns."""
        empty = pd.DataFrame(
            columns=[
                "model_id",
                "feature_set",
                "horizon",
                "date",
                "feature_name",
                "importance",
                "group",
            ]
        )
        agg = vi_by_group(empty)
        assert agg.empty
        assert "importance_share" in agg.columns


# ---------------------------------------------------------------------------
# Tests: average_vi_by_horizon
# ---------------------------------------------------------------------------


class TestAverageViByHorizon:
    def _make_group_df(self) -> pd.DataFrame:
        """Two dates, two horizons (12 and 6), single model."""
        dates = ["2010-01-01", "2010-02-01", "2010-03-01"]
        horizons = [12, 6]
        importances = _IMPORTANCES

        records = [
            _make_record(
                horizon=h,
                forecast_date=d,
                feature_importances=importances,
            )
            for h in horizons
            for d in dates
        ]
        vi_df = extract_vi_dataframe(records)
        return vi_by_group(vi_df, normalize=True)

    def test_average_vi_by_horizon_shape(self) -> None:
        """Output must have one row per (model, feature_set, horizon, group)."""
        grp_df = self._make_group_df()
        avg = average_vi_by_horizon(grp_df)

        # Number of unique (model, feature_set, horizon, group) combos
        expected_rows = (
            grp_df[["model_id", "feature_set", "horizon", "group"]]
            .drop_duplicates()
            .shape[0]
        )
        assert avg.shape[0] == expected_rows

    def test_average_vi_by_horizon_values(self) -> None:
        """Averaged importance_share must equal the mean over dates."""
        grp_df = self._make_group_df()
        avg = average_vi_by_horizon(grp_df)

        # Since every date has the same _IMPORTANCES, per-date shares are
        # identical — the average must equal each individual share.
        for _, avg_row in avg.iterrows():
            per_date = grp_df[
                (grp_df["model_id"] == avg_row["model_id"])
                & (grp_df["feature_set"] == avg_row["feature_set"])
                & (grp_df["horizon"] == avg_row["horizon"])
                & (grp_df["group"] == avg_row["group"])
            ]["importance_share"]
            expected_mean = per_date.mean()
            assert abs(avg_row["importance_share"] - expected_mean) < 1e-10

    def test_average_vi_by_horizon_filter(self) -> None:
        """horizons parameter must restrict output to the requested horizons."""
        grp_df = self._make_group_df()
        avg = average_vi_by_horizon(grp_df, horizons=[12])
        assert set(avg["horizon"].unique()) == {12}

    def test_average_vi_by_horizon_columns(self) -> None:
        """Output must have the required column set."""
        grp_df = self._make_group_df()
        avg = average_vi_by_horizon(grp_df)
        expected = {"model_id", "feature_set", "horizon", "group", "importance_share"}
        assert expected.issubset(set(avg.columns))

    def test_average_vi_by_horizon_empty_filter(self) -> None:
        """Filtering to a non-existent horizon must yield an empty DataFrame."""
        grp_df = self._make_group_df()
        avg = average_vi_by_horizon(grp_df, horizons=[99])
        assert avg.empty
