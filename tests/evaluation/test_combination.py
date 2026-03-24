"""Tests for macrocast/evaluation/combination.py — forecast combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.evaluation.combination import combine_forecasts


def _make_result_df(
    n_models: int = 3,
    T: int = 50,
    horizons: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic result_df with controllable structure."""
    if horizons is None:
        horizons = [1, 3]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=T, freq="MS")
    rows = []
    for h in horizons:
        y_true = rng.standard_normal(T)
        for m in range(n_models):
            model_id = f"model_{m}"
            y_hat = y_true + rng.standard_normal(T) * (0.5 * (m + 1))
            for i, d in enumerate(dates):
                rows.append(
                    {
                        "experiment_id": "test_exp",
                        "model_id": model_id,
                        "nonlinearity": "linear",
                        "regularization": "none",
                        "cv_scheme": "bic",
                        "loss_function": "l2",
                        "horizon": h,
                        "train_end": d - pd.offsets.MonthBegin(1),
                        "forecast_date": d,
                        "y_hat": float(y_hat[i]),
                        "y_true": float(y_true[i]),
                        "n_train": 100,
                        "n_factors": None,
                    }
                )
    return pd.DataFrame(rows)


class TestCombineForecastsBasic:
    def test_returns_dataframe(self) -> None:
        df = _make_result_df()
        out = combine_forecasts(df)
        assert isinstance(out, pd.DataFrame)

    def test_model_id_set_correctly(self) -> None:
        df = _make_result_df()
        out = combine_forecasts(df, method="mean")
        assert (out["model_id"] == "COMBO_MEAN").all()

    def test_output_rows_equal_horizon_times_dates(self) -> None:
        """One row per (horizon, forecast_date)."""
        df = _make_result_df(n_models=3, T=50, horizons=[1, 3])
        out = combine_forecasts(df, method="mean")
        assert len(out) == 2 * 50  # 2 horizons × 50 dates

    def test_y_true_preserved(self) -> None:
        """y_true in output should match the input y_true at each (horizon, date)."""
        df = _make_result_df()
        out = combine_forecasts(df, method="mean")
        # Pick first row and verify
        row = out.iloc[0]
        match = df[
            (df["horizon"] == row["horizon"])
            & (df["forecast_date"] == row["forecast_date"])
        ]["y_true"].iloc[0]
        assert row["y_true"] == pytest.approx(match)

    def test_output_schema_matches_input(self) -> None:
        """Output columns should be a subset of input columns."""
        df = _make_result_df()
        out = combine_forecasts(df, method="mean")
        for col in out.columns:
            assert col in df.columns

    def test_metadata_columns_set_to_combo(self) -> None:
        df = _make_result_df()
        out = combine_forecasts(df, method="mean")
        for col in ["nonlinearity", "regularization", "cv_scheme", "loss_function"]:
            assert (out[col] == "combo").all()


class TestCombineMethodsMean:
    def test_mean_of_two_models_is_average(self) -> None:
        """With 2 models at a single date/horizon, mean = average of y_hats."""
        df = pd.DataFrame(
            [
                {"model_id": "a", "horizon": 1, "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": 1.0, "y_true": 0.0},
                {"model_id": "b", "horizon": 1, "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": 3.0, "y_true": 0.0},
            ]
        )
        out = combine_forecasts(df, method="mean")
        assert out["y_hat"].iloc[0] == pytest.approx(2.0)

    def test_mean_single_model_identity(self) -> None:
        """With 1 model, mean combo = that model's forecast."""
        df = _make_result_df(n_models=1, T=20, horizons=[1])
        out = combine_forecasts(df, method="mean")
        merged = out.merge(
            df[["horizon", "forecast_date", "y_hat"]],
            on=["horizon", "forecast_date"],
            suffixes=("_combo", "_orig"),
        )
        np.testing.assert_allclose(
            merged["y_hat_combo"].values, merged["y_hat_orig"].values, rtol=1e-10
        )


class TestCombineMethodsMedian:
    def test_model_id_median(self) -> None:
        df = _make_result_df()
        out = combine_forecasts(df, method="median")
        assert (out["model_id"] == "COMBO_MEDIAN").all()

    def test_median_of_three_models(self) -> None:
        """Median of [1, 2, 3] = 2."""
        df = pd.DataFrame(
            [
                {"model_id": "a", "horizon": 1, "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": 1.0, "y_true": 0.0},
                {"model_id": "b", "horizon": 1, "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": 2.0, "y_true": 0.0},
                {"model_id": "c", "horizon": 1, "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": 3.0, "y_true": 0.0},
            ]
        )
        out = combine_forecasts(df, method="median")
        assert out["y_hat"].iloc[0] == pytest.approx(2.0)


class TestCombineMethodsTrimmedMean:
    def test_model_id_trimmed_mean(self) -> None:
        df = _make_result_df()
        out = combine_forecasts(df, method="trimmed_mean")
        assert (out["model_id"] == "COMBO_TRIMMED_MEAN").all()

    def test_trimmed_mean_removes_extremes(self) -> None:
        """With 5 models and 20% trim (k=1 each side), removes min and max."""
        df = pd.DataFrame(
            [
                {"model_id": str(i), "horizon": 1,
                 "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": float(i), "y_true": 0.0}
                for i in range(5)  # y_hat = 0, 1, 2, 3, 4
            ]
        )
        out = combine_forecasts(df, method="trimmed_mean", trim_pct=0.20)
        # Removes 0 and 4, mean of [1, 2, 3] = 2.0
        assert out["y_hat"].iloc[0] == pytest.approx(2.0)

    def test_trimmed_mean_fallback_to_mean_for_small_n(self) -> None:
        """With 2 models, trim_pct=0.10 → k=0 → falls back to mean."""
        df = pd.DataFrame(
            [
                {"model_id": "a", "horizon": 1, "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": 1.0, "y_true": 0.0},
                {"model_id": "b", "horizon": 1, "forecast_date": pd.Timestamp("2000-01-01"),
                 "y_hat": 3.0, "y_true": 0.0},
            ]
        )
        out_trimmed = combine_forecasts(df, method="trimmed_mean", trim_pct=0.10)
        out_mean = combine_forecasts(df, method="mean")
        assert out_trimmed["y_hat"].iloc[0] == pytest.approx(out_mean["y_hat"].iloc[0])


class TestCombineMethodsInvMSFE:
    def test_model_id_inv_msfe(self) -> None:
        df = _make_result_df(T=30)
        out = combine_forecasts(df, method="inv_msfe")
        assert (out["model_id"] == "COMBO_INV_MSFE").all()

    def test_inv_msfe_fallback_equal_weights_early(self) -> None:
        """First few dates: no history → equal weights → same as mean."""
        df = _make_result_df(n_models=3, T=5, horizons=[1], seed=10)
        out_inv = combine_forecasts(df, method="inv_msfe")
        out_mean = combine_forecasts(df, method="mean")
        # First date has no history → equal weights → should equal mean
        first_date = sorted(df["forecast_date"].unique())[0]
        inv_first = out_inv[out_inv["forecast_date"] == first_date]["y_hat"].iloc[0]
        mean_first = out_mean[out_mean["forecast_date"] == first_date]["y_hat"].iloc[0]
        assert inv_first == pytest.approx(mean_first)

    def test_inv_msfe_better_model_gets_higher_weight(self) -> None:
        """Over many periods, inv_msfe should weight the accurate model more."""
        rng = np.random.default_rng(99)
        T = 100
        dates = pd.date_range("2000-01-01", periods=T, freq="MS")
        y_true = rng.standard_normal(T)
        # model_good: small error; model_bad: large error
        rows = []
        for i, d in enumerate(dates):
            rows.append({"model_id": "good", "horizon": 1, "forecast_date": d,
                         "y_hat": y_true[i] + rng.standard_normal() * 0.1,
                         "y_true": y_true[i]})
            rows.append({"model_id": "bad", "horizon": 1, "forecast_date": d,
                         "y_hat": y_true[i] + rng.standard_normal() * 3.0,
                         "y_true": y_true[i]})
        df = pd.DataFrame(rows)
        out = combine_forecasts(df, method="inv_msfe")
        # For late dates, combo y_hat should be much closer to good model's y_hat
        late = dates[80:]
        late_df = df[df["forecast_date"].isin(late)]
        late_out = out[out["forecast_date"].isin(late)]
        good_yhat = late_df[late_df["model_id"] == "good"]["y_hat"].values
        combo_yhat = late_out["y_hat"].values
        y_t = late_df[late_df["model_id"] == "good"]["y_true"].values
        msfe_combo = np.mean((combo_yhat - y_t) ** 2)
        msfe_bad = np.mean(
            (late_df[late_df["model_id"] == "bad"]["y_hat"].values - y_t) ** 2
        )
        assert msfe_combo < msfe_bad

    def test_inv_msfe_rolling_window(self) -> None:
        """Rolling window should produce finite output."""
        df = _make_result_df(T=40)
        out = combine_forecasts(df, method="inv_msfe", window=10)
        assert out["y_hat"].notna().all()

    def test_inv_msfe_expanding_vs_rolling_differ(self) -> None:
        df = _make_result_df(T=60, seed=7)
        out_exp = combine_forecasts(df, method="inv_msfe", window=None)
        out_roll = combine_forecasts(df, method="inv_msfe", window=5)
        assert not out_exp["y_hat"].equals(out_roll["y_hat"])


class TestCombineForecastsInputValidation:
    def test_missing_column_raises(self) -> None:
        df = _make_result_df()
        df = df.drop(columns=["y_hat"])
        with pytest.raises(ValueError, match="missing required columns"):
            combine_forecasts(df)

    def test_unknown_method_raises(self) -> None:
        df = _make_result_df()
        with pytest.raises(ValueError, match="Unknown method"):
            combine_forecasts(df, method="harmonic_mean")  # type: ignore[arg-type]


class TestCombineForecastsExport:
    def test_importable_from_evaluation(self) -> None:
        from macrocast.evaluation.combination import combine_forecasts  # noqa: F401

    def test_importable_from_package(self) -> None:
        from macrocast.evaluation import combine_forecasts  # noqa: F401
