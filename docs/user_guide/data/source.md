# §1.1 Source & frame

Declares **where the data comes from and which information-set regime applies**. These five axes together answer: which dataset, at what frequency, over which real-time regime, under what domain label — before the task (§1.2) or the evaluation window (§1.3) is fixed.

| § | axis | Role |
|---|---|---|
| 1.1.1 | [`dataset`](#111-dataset) | Which FRED-family dataset schema to load |
| 1.1.2 | [`dataset_source`](#112-dataset_source) | Where the actual bytes came from (usually same as dataset; provenance field) |
| 1.1.3 | [`frequency`](#113-frequency) | Series frequency (monthly/quarterly); dataset-derived |
| 1.1.4 | [`data_domain`](#114-data_domain) | Coarse subject-area label (macro / macro_finance / ...) |
| 1.1.5 | [`information_set_type`](#115-information_set_type) | Real-time regime (revised vs. vintage-aware) |

---

## 1.1.1 `dataset`

**Selects the schema of the dataset loaded.** Every recipe picks exactly one.

### Value catalog

| Value | Status | Loader | Content |
|---|---|---|---|
| `fred_md` | operational | `macrocast.raw.load_fred_md` | FRED-MD monthly macro panel (McCracken & Ng) |
| `fred_qd` | operational | `macrocast.raw.load_fred_qd` | FRED-QD quarterly macro panel |
| `fred_sd` | operational | `macrocast.raw.load_fred_sd` | FRED-SD state-monthly panel |

All three values are fully wired end-to-end: the registry entry is used by the compiler, the loader is chosen at run time via `_get_dataset_loader`, and the resulting panel flows into every downstream axis.

### Functions & features

- `macrocast.load_fred_md()` / `load_fred_qd()` / `load_fred_sd()` — public loaders.
- `macrocast.raw.datasets.fred_md` / `fred_qd` / `fred_sd` — per-dataset modules with cache + manifest logic.
- Compiler reads `dataset` via `_selection_value(selection_map, "dataset")` → propagated into `CompiledRecipeSpec.dataset` and every downstream spec.
- `_DATASET_DEFAULT_FREQUENCY` in `compiler/build.py` maps each dataset to its default `frequency` value.

### Recipe usage

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
```

---

## 1.1.2 `dataset_source`

**Provenance label** for where the dataset bytes were loaded from. In v1.0 this is largely a recording-only field; the actual loader is dispatched by `dataset`, not by `dataset_source`.

### Value catalog

| Value | Status | Semantics |
|---|---|---|
| `fred_md` | operational | FRED-MD canonical source (St. Louis Fed) |
| `fred_qd` | operational | FRED-QD canonical source |
| `fred_sd` | operational | FRED-SD canonical source |
| `fred_api_custom` | registry_only (v1.1) | User-supplied FRED API pull; adapter pending |
| `bea` / `bls` / `census` / `oecd` / `imf_ifs` / `ecb_sdw` / `bis` / `world_bank` / `wrds_macro_finance` / `survey_spf` | registry_only (v1.1) | Third-party source adapters; all registered as reserved slots, none wired yet |
| `custom_csv` | registry_only (v1.1) | User-supplied CSV matching a FRED-* schema; adapter pending |
| `custom_parquet` | registry_only (v1.1) | Same, Parquet format |
| `market_prices` / `news_text` / `custom_sql` | future (v2) | Alt-data + SQL adapters, phase-11 scope |

### Why most values are registry_only in v1.0

`dataset_source` is designed for a world where the schema (`dataset`) and the byte source can differ — e.g., `dataset=fred_md` + `dataset_source=custom_csv` meaning "load a user CSV that conforms to the FRED-MD schema." That dispatch layer is not built in v1.0; the loader is chosen by `dataset` alone. The 3 FRED operational values exist because they coincide with the canonical FRED loader paths; setting any other value triggers no special loader today, so they remain `registry_only` until the custom-adapter dispatcher lands.

### Functions & features

- Compiler: `dataset_source` default = the value of `dataset` (`_selection_value(..., default=dataset)`). No downstream branch.
- Manifest: `dataset_source` recorded for provenance audit.

### Recipe usage

Usually omitted; the default mirrors `dataset`. Explicit:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      dataset_source: fred_md   # redundant but explicit
```

---

## 1.1.3 `frequency`

**Series frequency of the dataset panel.** Dataset-derived in v1.0; user override has no runtime effect.

### Value catalog

| Value | Status | Which dataset uses this |
|---|---|---|
| `monthly` | operational | `fred_md`, `fred_sd` |
| `quarterly` | operational | `fred_qd` |
| `daily` / `weekly` / `yearly` | registry_only (v1.1) | Reserved for future daily/weekly/yearly FRED-variant loaders |
| `mixed_frequency` | future (v2) | MIDAS-style mixed-frequency infra (phase-11) |

### Functions & features

- Compiler default: `_DATASET_DEFAULT_FREQUENCY.get(dataset, "monthly")` — so `dataset=fred_md` implies `frequency=monthly`, `dataset=fred_qd` implies `frequency=quarterly`, etc.
- User can place `frequency` in `fixed_axes`, but downstream execution does not dispatch on it — the actual data cadence comes from the loaded panel's index, not this axis.
- Manifest records `frequency` for provenance.

### Recipe usage

Usually omitted (dataset implies the frequency). Explicit only when the manifest needs to carry an override tag — but the override does not change runtime behaviour in v1.0.

---

## 1.1.4 `data_domain`

**Coarse subject-area label** for the panel — declarative tagging, not a runtime switch.

### Value catalog

| Value | Status | Rough meaning |
|---|---|---|
| `macro` | operational | Standard macroeconomic panel (default) |
| `macro_finance` | registry_only (v1.1) | Macro + finance mixed panel |
| `housing` / `energy` / `labor` / `regional` | registry_only (v1.1) | Specialised sub-domains; no dedicated loaders yet |
| `panel_macro` / `text_macro` / `mixed_domain` | future (v2) | Cross-sectional panels, NLP-augmented macro, multi-domain fusion |

### Functions & features

- Compiler reads `data_domain` into the manifest with default `"macro"`.
- No downstream dispatch; purely declarative.
- Downgrading `macro_finance` to `registry_only` in this §1.1 pass reflects the honest status — v1.0 does not produce different behaviour for `macro` vs. `macro_finance`.

### Recipe usage

Usually omitted; default `"macro"` covers FRED-MD / FRED-QD / FRED-SD. Not a load-time dispatcher.

---

## 1.1.5 `information_set_type`

**Real-time regime** that governs which version of each observation the model is allowed to see at each forecast origin. Fully wired — this is the only §1.1 axis with compile-time validation AND runtime dispatch across its operational values.

### Value catalog

| Value | Status | Contract |
|---|---|---|
| `revised` | operational | Latest revised values (post-revision truth). Default. |
| `real_time_vintage` | operational | Load the vintage available at each forecast origin. Requires `leaf_config.data_vintage` at compile time. |
| `pseudo_oos_revised` | operational | Pseudo out-of-sample: latest revised values but masked according to (fake) release-lag discipline. |
| `pseudo_oos_vintage_aware` | registry_only (v1.1) | Vintage-aware pseudo-OOS; needs release-calendar infrastructure |
| `release_calendar_aware` | future (v2) | Full publication-calendar-driven data feed |
| `publication_lag_aware` | future (v2) | Richer publication-lag metadata beyond `release_lag_rule` |

### Functions & features

- Compile-time validation (`compiler/build.py:514-515`): `information_set_type == "real_time_vintage"` requires `leaf_config.data_vintage`. Missing vintage → `CompileValidationError`.
- Runtime: loaders (`raw/datasets/fred_md.py` etc.) dispatch on this axis to pick the correct vintage source.
- Compat mirror: the older recipe alias `info_set` is canonicalised to `information_set_type` (compiler/build.py alias map).
- `information_set_type` also interacts with `vintage_policy` (§1.5) — `real_time_vintage` defaults `vintage_policy` to `single_vintage`, `revised` defaults to `latest_only`.

### Recipe usage

```yaml
# Revised (post-revision) default
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: revised
    leaf_config:
      target: INDPRO

# Real-time vintage — requires data_vintage
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: real_time_vintage
    leaf_config:
      target: INDPRO
      data_vintage: "2023-06-01"

# Pseudo-OOS on revised data
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      information_set_type: pseudo_oos_revised
```

---

## §1.1 takeaways

- **`dataset`** and **`information_set_type`** are the two axes the user actually decides. Every operational value dispatches.
- **`dataset_source`**, **`frequency`**, **`data_domain`** are declarative / provenance-only in v1.0 — most values demoted to `registry_only` in this pass to match reality. Keep them for manifest audit and future adapter work (v1.1+).

Next group: [§1.2 Task & target](task.md) (coming) — what exactly is being forecast.
