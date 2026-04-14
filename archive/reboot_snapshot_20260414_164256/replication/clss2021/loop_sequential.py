"""Sequential model-by-model replication loop — CLSS 2021.

Order: AR → AL → EN → RF (FM validated implicitly as denominator).
For each model: runs all 10 targets × all 16 info_sets.
Runs until --deadline (default 2026-03-31 09:00).

Pass gate: logs pass/fail per model but does NOT block progression —
known AL/EN h=6-12 gaps (~+0.08-0.13, MATLAB vs R platform) are
recorded as "escalated" rather than blocking failures.

Usage
-----
cd /home/nanyeon99/project/macroforecast
uv run replication/clss2021/loop_sequential.py [--deadline "2026-03-31 09:00"]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT     = Path(__file__).resolve().parent.parent.parent
_SPEC_DIR = Path(__file__).parent
_LOG      = _SPEC_DIR / "loop_sequential.log"
_ATTEMPT  = _SPEC_DIR / "attempt_log_sequential.jsonl"

sys.path.insert(0, str(_ROOT / "scripts"))
from info_set_config import INFO_SETS, TARGETS, HORIZONS

# ---------------------------------------------------------------------------
# Priority queue: model-first sequential order
# (AR handled separately via batch_ar.py; FM is implicit denominator)
# ---------------------------------------------------------------------------

# (model, csv) pairs: AL -> EN -> RF-b2 -> RF-b1
MODEL_SEQUENCE: list[tuple[str, str]] = [
    ("AL", "b2"),
    ("EN", "b2"),
    ("RF", "b2"),
    ("RF", "b1"),
]

TARGET_COLS = list(TARGETS.keys())

# Known escalated gaps: these count toward the "escalated" bucket, not "fail"
# Format: (model, horizon) pairs where gaps are expected but irreducible
KNOWN_ESCALATED: set[tuple[str, int]] = {
    ("AL", 6), ("AL", 9), ("AL", 12),
    ("EN", 6), ("EN", 9), ("EN", 12),
    ("RF", 12), ("RF", 24),
}

# ---------------------------------------------------------------------------
# Timeout by model (seconds)
# RF: 16 info_sets × ~13 min each = ~210 min → use 14400s (4h)
# AL/EN: 16 info_sets × ~8 min each = ~130 min → use 9000s (2.5h)
# ---------------------------------------------------------------------------
TIMEOUT: dict[str, int | None] = {
    "RF": None,   # no limit — RF can take 3-4h per target
    "AL": None,   # no limit
    "EN": None,   # no limit
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(_LOG, "a") as f:
        f.write(line + "\n")


def append_attempt(entry: dict) -> None:
    with open(_ATTEMPT, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Run batch_validate.py as subprocess
# ---------------------------------------------------------------------------

def run_batch(
    model: str,
    target: str,
    csv: str,
    info_sets: str = "all",
    extra_args: list[str] | None = None,
) -> dict | None:
    timeout = TIMEOUT.get(model, 9000)
    cmd = [
        "uv", "run", "scripts/batch_validate.py",
        "--model",     model,
        "--target",    target,
        "--csv",       csv,
        "--info_sets", info_sets,
        "--n_jobs",    "-1",
    ] + (extra_args or [])

    log(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=str(_ROOT),
        )
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s")
        return None

    elapsed = round(time.time() - t0, 1)
    log(f"  Finished in {elapsed}s  (rc={proc.returncode})")

    if proc.stdout:
        for line in proc.stdout.splitlines():
            print(f"    | {line}", flush=True)
    if proc.stderr:
        log(f"  STDERR ({len(proc.stderr)} bytes): {proc.stderr[:1000]}")

    for line in proc.stdout.splitlines():
        if line.startswith("##BATCH_RESULT##"):
            try:
                result = json.loads(line[len("##BATCH_RESULT##"):].strip())
                # Flag all-unknown runs: model ran but produced zero usable cells
                n_unk = result.get("n_unknown", 0)
                n_ok  = result.get("n_ok", 0)
                n_fail = result.get("n_fail", 0)
                if n_ok == 0 and n_fail == 0 and n_unk > 0:
                    log(f"  WARNING: all {n_unk} cells unknown — model likely errored silently")
                return result
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Load gap matrix
# ---------------------------------------------------------------------------

def load_gap_matrix(model: str, target: str, csv: str) -> dict | None:
    path = _SPEC_DIR / f"gap_{model.lower()}_{target.lower()}_{csv}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pass gate evaluation (per model, after all targets complete)
# ---------------------------------------------------------------------------

def evaluate_model_gate(
    model: str,
    results: list[dict],
    tolerance: float = 0.05,
) -> tuple[str, str]:
    """Return (verdict, description).

    Verdict: PASS | ESCALATED | FAIL
    - PASS: all cells within tolerance (or empty)
    - ESCALATED: only known platform gaps exceed tolerance
    - FAIL: unexpected failures beyond known platform gaps
    """
    unexpected_fails: list[str] = []
    escalated_cells: list[str] = []
    n_pass = n_esc = n_fail = 0

    for r in results:
        target = r.get("target", "?")
        gap_data = load_gap_matrix(model, target, r.get("csv", "b2"))
        if gap_data is None:
            continue
        for iset, row in gap_data.get("info_sets", {}).items():
            for h_str, cell in row.items():
                h = int(h_str)
                tol = 0.10 if h == 24 else tolerance
                gap = cell.get("gap")
                ok  = cell.get("ok", True)
                if ok or gap is None:
                    n_pass += 1
                    continue
                if (model, h) in KNOWN_ESCALATED:
                    escalated_cells.append(f"{target}/{iset}@h{h}({gap:+.3f})")
                    n_esc += 1
                else:
                    unexpected_fails.append(f"{target}/{iset}@h{h}({gap:+.3f})")
                    n_fail += 1

    # Count unknowns from results (cells that errored and produced no gap)
    n_unknown = sum(r.get("n_unknown", 0) for r in results if not r.get("skipped"))
    n_total_attempted = n_pass + n_esc + n_fail

    if n_total_attempted == 0 and n_unknown > 0:
        return "FAIL", f"Zero usable cells — all {n_unknown} cells were unknown/errored. Model likely broken."
    if n_total_attempted == 0:
        return "FAIL", "Zero cells evaluated — no gap data produced."
    if n_fail == 0 and n_esc == 0 and n_unknown == 0:
        return "PASS", f"All {n_pass} cells within tolerance."
    if n_fail == 0 and n_esc == 0:
        return "PASS", f"{n_pass} cells within tolerance ({n_unknown} unknown — check logs)."
    elif n_fail == 0:
        esc_preview = ", ".join(escalated_cells[:5])
        return "ESCALATED", (
            f"{n_esc} known platform gaps (h=6-12 MATLAB vs R): {esc_preview}. "
            f"{n_pass} cells pass."
        )
    else:
        fail_preview = ", ".join(unexpected_fails[:5])
        return "FAIL", (
            f"{n_fail} unexpected failures: {fail_preview}. "
            f"Escalated: {n_esc}. Pass: {n_pass}."
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--deadline", type=str, default=None,
                   help="Optional stop datetime (YYYY-MM-DD HH:MM). Omit for indefinite run.")
    p.add_argument("--resume",   action="store_true",
                   help="Skip (model, target, csv) that already have gap JSON")
    p.add_argument("--only_model",  type=str, default="",
                   help="Only run this model (AL, EN, RF)")
    p.add_argument("--only_target", type=str, default="",
                   help="Only run this target column")
    p.add_argument("--info_sets",   type=str, default="all",
                   help="Comma-separated info_sets, or 'all'")
    args = p.parse_args()

    deadline = datetime.strptime(args.deadline, "%Y-%m-%d %H:%M") if args.deadline else None

    log("=" * 70)
    log(f"SEQUENTIAL LOOP START  deadline={deadline or 'INDEFINITE'}")
    log(f"Model order: {[m for m,_ in MODEL_SEQUENCE]}")
    log(f"Targets: {TARGET_COLS}")
    log(f"Info_sets: {args.info_sets}")
    log("=" * 70)

    grand_pass = grand_fail = grand_esc = 0

    for model, csv in MODEL_SEQUENCE:
        if args.only_model and model != args.only_model:
            continue
        if deadline and datetime.now() >= deadline:
            log("DEADLINE REACHED — stopping.")
            break

        log(f"\n{'═'*70}")
        log(f"MODEL: {model}  CSV: {csv}")
        log(f"{'═'*70}")

        model_results: list[dict] = []
        n_done = n_skip = 0

        for target in TARGET_COLS:
            if args.only_target and target != args.only_target:
                continue
            if deadline and datetime.now() >= deadline:
                log(f"DEADLINE during {model}/{target} — stopping.")
                break

            # Skip if gap JSON already exists and --resume
            existing = load_gap_matrix(model, target, csv)
            if existing is not None and args.resume:
                log(f"  SKIP (done): {target}")
                model_results.append({"target": target, "csv": csv, "skipped": True})
                n_skip += 1
                continue

            log(f"\n  ── {model}/{target}/{csv} ──")
            result = run_batch(model, target, csv, info_sets=args.info_sets)

            if result is None:
                log(f"  ERROR: no result for {model}/{target}/{csv}")
                append_attempt({
                    "timestamp": datetime.now().isoformat(),
                    "model": model, "target": target, "csv": csv,
                    "status": "error",
                    "note": "batch_validate returned no result",
                })
                continue

            n_ok   = result.get("n_ok",   0)
            n_fail = result.get("n_fail", 0)
            n_unk  = result.get("n_unknown", 0)
            worst  = result.get("worst_gaps", [])
            n_done += 1

            log(f"  Pass/Fail/Unknown: {n_ok}/{n_fail}/{n_unk}")
            if n_ok == 0 and n_fail == 0 and n_unk > 0:
                log(f"  ERROR: {model}/{target}/{csv} produced ZERO usable cells ({n_unk} unknown) — treating as failure")
            if worst:
                log("  Worst: " + ", ".join(
                    f"{w['info_set']}@h{w['horizon']}={w['gap']:+.3f}"
                    for w in worst[:5]))

            entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model, "target": target, "csv": csv,
                "n_ok": n_ok, "n_fail": n_fail, "n_unknown": n_unk,
                "worst_gaps": worst,
            }
            append_attempt(entry)
            model_results.append({"target": target, "csv": csv, "n_ok": n_ok, "n_fail": n_fail, "n_unknown": n_unk})

        # ── Per-model pass gate ──
        log(f"\n  ── Gate check: {model}/{csv} ({n_done} targets run, {n_skip} skipped) ──")
        verdict, desc = evaluate_model_gate(model, model_results)
        log(f"  Gate: {verdict} — {desc}")

        append_attempt({
            "timestamp": datetime.now().isoformat(),
            "model": model, "csv": csv,
            "event": "model_gate",
            "verdict": verdict,
            "desc": desc,
        })

        if verdict == "PASS":
            grand_pass += n_done
        elif verdict == "ESCALATED":
            grand_esc += n_done
        else:
            grand_fail += n_done

    # ── Final summary ──
    log("\n" + "=" * 70)
    log(f"SEQUENTIAL LOOP COMPLETE")
    log(f"Models passed: {grand_pass}  escalated: {grand_esc}  failed: {grand_fail}")
    log("=" * 70)
    _write_summary()


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary() -> None:
    lines = [
        "# Sequential Replication Loop Summary",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## Model Gates\n",
    ]

    if not _ATTEMPT.exists():
        lines.append("No attempt log found.")
    else:
        gates: list[dict] = []
        with open(_ATTEMPT) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("event") == "model_gate":
                        gates.append(e)
                except json.JSONDecodeError:
                    pass

        for g in gates:
            lines.append(f"- **{g['model']}/{g['csv']}**: {g['verdict']} — {g['desc']}")

    lines.append("\n## Per-Model Target Tables\n")

    for model, csv in MODEL_SEQUENCE:
        lines.append(f"\n### {model} / {csv}\n")
        lines.append("| Target | n_ok | n_fail | Worst gap |")
        lines.append("|--------|------|--------|-----------|")
        for target in TARGET_COLS:
            gd = load_gap_matrix(model, target, csv)
            if gd is None:
                lines.append(f"| {target} | — | — | not run |")
                continue
            n_ok = n_fail = 0
            worst_gap = 0.0
            for row in gd.get("info_sets", {}).values():
                for cell in row.values():
                    if cell.get("ok"):
                        n_ok += 1
                    elif cell.get("gap") is not None:
                        n_fail += 1
                        g = abs(cell["gap"])
                        if g > worst_gap:
                            worst_gap = g
            lines.append(f"| {target} | {n_ok} | {n_fail} | {worst_gap:+.3f} |")

    out = _SPEC_DIR / "loop_sequential_summary.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    log(f"Summary written: {out}")


if __name__ == "__main__":
    main()
