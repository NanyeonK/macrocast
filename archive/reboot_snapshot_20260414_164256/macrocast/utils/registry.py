"""Experiment registry for persisting and retrieving ResultSets.

Provides a lightweight file-based registry that stores each experiment's
forecast records as a parquet file under a configurable root directory
(default ``~/.macrocast/results/{experiment_id}/``).

Typical usage::

    from macrocast.utils.registry import ExperimentRegistry

    reg = ExperimentRegistry()
    reg.save(result_set)                     # saves to disk
    rs = reg.load("clss2021_rf_INDPRO")      # retrieve by id
    print(reg.list())                        # DataFrame of all experiments
    reg.compare(["exp_a", "exp_b"])          # RMSFE comparison table
    reg.delete("exp_a")                      # remove from disk
"""

from __future__ import annotations

import contextlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from macrocast.pipeline.results import ResultSet

# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class ExperimentRegistry:
    """File-based registry for macrocast ResultSets.

    Each experiment is stored as::

        {root_dir}/{experiment_id}/
            results.parquet    — forecast records
            meta.json          — experiment metadata + creation timestamp

    Parameters
    ----------
    root_dir : str or Path, optional
        Root directory for all stored experiments.
        Defaults to ``~/.macrocast/results``.
    """

    _RESULTS_FILE = "results.parquet"
    _META_FILE = "meta.json"

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            root_dir = Path.home() / ".macrocast" / "results"
        self.root_dir = Path(root_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def save(
        self,
        result_set: ResultSet,
        experiment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist a ResultSet to disk.

        Parameters
        ----------
        result_set : ResultSet
            The forecast records to save.
        experiment_id : str, optional
            Override the ResultSet's own ``experiment_id``.  Useful when you
            want a human-readable name (e.g. ``"clss2021_rf_INDPRO"``) rather
            than the auto-generated UUID.
        metadata : dict, optional
            Additional key-value metadata to store alongside the records
            (e.g. ``{"target": "INDPRO", "vintage": "2018-02"}``).  Merged
            with ``result_set.metadata``.

        Returns
        -------
        Path
            Directory where the experiment was saved.

        Raises
        ------
        ValueError
            If the ResultSet contains no records.
        """
        eid = experiment_id or result_set.experiment_id
        if not eid:
            raise ValueError("experiment_id must be non-empty.")

        df = result_set.to_dataframe_cached()
        if df.empty:
            raise ValueError(
                f"ResultSet '{eid}' is empty; nothing to save."
            )

        exp_dir = self.root_dir / eid
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Write parquet
        df.to_parquet(exp_dir / self._RESULTS_FILE, index=False)

        # Write metadata
        meta: dict[str, Any] = {
            "experiment_id": eid,
            "created_at": datetime.utcnow().isoformat(),
            "n_records": len(df),
            "horizons": sorted(df["horizon"].unique().tolist()) if "horizon" in df.columns else [],
            "model_ids": sorted(df["model_id"].unique().tolist()) if "model_id" in df.columns else [],
        }
        meta.update(result_set.metadata or {})
        if metadata:
            meta.update(metadata)

        (exp_dir / self._META_FILE).write_text(
            json.dumps(meta, default=str, indent=2)
        )

        return exp_dir

    def load(self, experiment_id: str) -> ResultSet:
        """Load a previously saved ResultSet.

        Parameters
        ----------
        experiment_id : str
            The id used when the experiment was saved.

        Returns
        -------
        ResultSet
            A ResultSet with an empty ``records`` list but a populated
            ``_df_cache`` (parquet data).  Use ``rs.to_dataframe()`` for
            downstream evaluation.

        Raises
        ------
        FileNotFoundError
            If ``experiment_id`` does not exist in the registry.
        """
        exp_dir = self.root_dir / experiment_id
        parquet_path = exp_dir / self._RESULTS_FILE
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}' not found in registry at "
                f"{self.root_dir}."
            )

        df = pd.read_parquet(parquet_path)
        meta: dict[str, Any] = {}
        meta_path = exp_dir / self._META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        rs = ResultSet(experiment_id=experiment_id, metadata=meta)
        rs._df_cache = df  # type: ignore[attr-defined]
        return rs

    def exists(self, experiment_id: str) -> bool:
        """Return True if ``experiment_id`` is present in the registry."""
        return (self.root_dir / experiment_id / self._RESULTS_FILE).exists()

    def delete(self, experiment_id: str) -> None:
        """Remove an experiment and all its files from the registry.

        Parameters
        ----------
        experiment_id : str

        Raises
        ------
        FileNotFoundError
            If ``experiment_id`` does not exist.
        """
        exp_dir = self.root_dir / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}' not found in registry at "
                f"{self.root_dir}."
            )
        shutil.rmtree(exp_dir)

    # ------------------------------------------------------------------
    # Listing and comparison
    # ------------------------------------------------------------------

    def list(self) -> pd.DataFrame:
        """Return a summary DataFrame of all stored experiments.

        Returns
        -------
        pd.DataFrame
            Columns: ``experiment_id``, ``n_records``, ``horizons``,
            ``model_ids``, ``created_at``, plus any extra keys from the
            experiment's metadata.  Sorted by ``created_at`` descending.
        """
        rows = []
        if not self.root_dir.exists():
            return pd.DataFrame(columns=["experiment_id", "n_records", "created_at"])

        for exp_dir in sorted(self.root_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            parquet_path = exp_dir / self._RESULTS_FILE
            if not parquet_path.exists():
                continue

            meta_path = exp_dir / self._META_FILE
            row: dict[str, Any] = {"experiment_id": exp_dir.name}
            if meta_path.exists():
                with contextlib.suppress(json.JSONDecodeError):
                    row.update(json.loads(meta_path.read_text()))
            else:
                # Fall back to reading parquet for basic stats
                try:
                    df = pd.read_parquet(parquet_path, columns=["horizon", "model_id"])
                    row["n_records"] = len(df)
                    row["horizons"] = sorted(df["horizon"].unique().tolist())
                    row["model_ids"] = sorted(df["model_id"].unique().tolist())
                except Exception:
                    pass

            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["experiment_id", "n_records", "created_at"])

        result = pd.DataFrame(rows)
        if "created_at" in result.columns:
            result = result.sort_values("created_at", ascending=False)
        return result.reset_index(drop=True)

    def compare(
        self,
        experiment_ids: list[str],
        benchmark_id: str | None = None,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compare MSFE across multiple experiments.

        Parameters
        ----------
        experiment_ids : list of str
            Experiments to compare.
        benchmark_id : str, optional
            If provided, MSFE values are expressed relative to this
            experiment's MSFE (ratio < 1 means better than benchmark).
            If None, raw MSFE values are returned.
        horizons : list of int, optional
            Restrict comparison to these horizons.  Uses all horizons found
            in the first experiment if None.

        Returns
        -------
        pd.DataFrame
            Rows = experiment_ids, columns = horizons.
            Values are MSFE (absolute) or relative MSFE (when
            ``benchmark_id`` is given).
        """
        msfe: dict[str, dict[int, float]] = {}
        all_horizons: set[int] = set()

        for eid in experiment_ids:
            rs = self.load(eid)
            df = rs.to_dataframe_cached()
            if df.empty:
                continue
            df = df.copy()
            df["_se"] = (df["y_true"] - df["y_hat"]) ** 2
            h_msfe = df.groupby("horizon")["_se"].mean().to_dict()
            msfe[eid] = {int(k): float(v) for k, v in h_msfe.items()}
            all_horizons.update(msfe[eid].keys())

        if horizons is not None:
            use_horizons = sorted(h for h in horizons if h in all_horizons)
        else:
            use_horizons = sorted(all_horizons)

        result = pd.DataFrame(
            {eid: {h: msfe.get(eid, {}).get(h, float("nan")) for h in use_horizons}
             for eid in experiment_ids}
        ).T
        result.index.name = "experiment_id"

        if benchmark_id is not None and benchmark_id in msfe:
            bm = msfe[benchmark_id]
            for h in use_horizons:
                bm_val = bm.get(h, float("nan"))
                if bm_val and bm_val > 0:
                    result[h] = result[h] / bm_val

        return result
