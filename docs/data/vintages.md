# Vintage Management

Real-time macroeconomic data is subject to revisions. FRED-MD and FRED-QD publish monthly vintage snapshots that capture the state of the data as it was known at a given point in time. macrocast provides utilities to enumerate, load, and work with these vintages.

---

## Vintage Identifiers

Vintages follow the `"YYYY-MM"` format. For example, `"2020-01"` represents the data release from January 2020.

- FRED-MD vintages are available from 1999-01 onward.
- FRED-QD vintages are available from 2005-01 onward.
- FRED-SD vintages are available from 2005-01 onward.

---

## Listing Available Vintages

`list_available_vintages` enumerates expected vintage identifiers without making a network request:

```python
import macrocast as mc

# All FRED-MD vintages from 2010 to 2020
vintages = mc.list_available_vintages(
    "fred_md",
    start="2010-01",
    end="2020-12",
)
print(len(vintages))   # 132
print(vintages[:3])    # ['2010-01', '2010-02', '2010-03']
```

Not every generated vintage identifier necessarily has a file on the FRED server; discontinued months are silently absent when loading.

---

## Loading a Single Vintage

Pass the vintage identifier to any loader function:

```python
md_2020 = mc.load_fred_md(vintage="2020-01")
print(md_2020.vintage)   # '2020-01'
```

---

## Loading Multiple Vintages

`load_vintage_panel` loads a collection of vintage snapshots and returns a `dict[str, MacroFrame]`:

```python
panel = mc.load_vintage_panel(
    "fred_md",
    vintages=["2019-01", "2020-01", "2021-01"],
)
print(panel["2020-01"])
# MacroFrame(dataset='FRED-MD', vintage='2020-01', ...)
```

Each vintage is downloaded and cached separately. Historical vintage files never expire.

---

## RealTimePanel

`RealTimePanel` is a lightweight wrapper around a dict of vintage MacroFrames:

```python
panel_dict = mc.load_vintage_panel(
    "fred_md",
    vintages=["2019-01", "2020-01", "2021-01"],
)
rt = mc.RealTimePanel(panel_dict)

print(rt)
# RealTimePanel(n_vintages=3, range=2019-01 to 2021-01)

print(rt.vintages)
# ['2019-01', '2020-01', '2021-01']

# Access a specific vintage
md_2020 = rt["2020-01"]
```

!!! note "RealTimePanel is a v0.1 stretch goal"
    The current `RealTimePanel` provides basic access and iteration. Alignment utilities (pseudo-out-of-sample construction, revision analysis) are planned for v0.2.

---

## Caching

Each vintage file is cached at `~/.macrocast/cache/{dataset}/{vintage}.csv`. Historical vintages never expire; only the `current.csv` file refreshes after 30 days.

```python
# Load multiple vintages with a custom cache directory
panel = mc.load_vintage_panel(
    "fred_md",
    vintages=["2019-01", "2020-01"],
    cache_dir="/data/fred",
)
```

---

## Function Reference

### `list_available_vintages(dataset, start=None, end=None)`

Generate the list of expected vintage identifiers for a dataset. No network request is made.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | `str` | `"fred_md"`, `"fred_qd"`, or `"fred_sd"` |
| `start` | `str` or `None` | Start vintage (`"YYYY-MM"`). Defaults to dataset minimum. |
| `end` | `str` or `None` | End vintage. Defaults to current month. |

Returns `list[str]`.

### `load_vintage_panel(dataset, vintages, target=None, cache_dir=None)`

Load multiple vintage snapshots.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | `str` | `"fred_md"`, `"fred_qd"`, or `"fred_sd"` |
| `vintages` | `list[str]` | List of `"YYYY-MM"` vintage identifiers |
| `target` | `str` or `None` | Not yet implemented; currently ignored |
| `cache_dir` | `str` or `Path` or `None` | Override for cache directory |

Returns `dict[str, MacroFrame]`.
