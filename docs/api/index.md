# API Reference

This section provides the full auto-generated API documentation for macrocast v0.1.

---

## Module Structure

| Module | Contents |
|--------|----------|
| `macrocast` | Top-level namespace; re-exports all public symbols from the data layer |
| `macrocast.data` | FRED-MD/QD/SD loaders, MacroFrame, transforms, missing, vintages |
| `macrocast.utils.cache` | Cache management utilities |

---

## Top-Level Namespace

All public symbols from the data layer are re-exported at the package level:

```python
import macrocast as mc

# Loaders
mc.load_fred_md()
mc.load_fred_qd()
mc.load_fred_sd()

# Core container
mc.MacroFrame
mc.MacroFrameMetadata
mc.VariableMetadata

# Transforms
mc.TransformCode
mc.apply_tcode
mc.apply_tcodes

# Missing value utilities
mc.classify_missing
mc.handle_missing

# Vintage management
mc.list_available_vintages
mc.load_vintage_panel
mc.RealTimePanel
```

---

## Auto-Generated Reference

::: macrocast
    options:
      members:
        - load_fred_md
        - load_fred_qd
        - load_fred_sd
        - MacroFrame
        - MacroFrameMetadata
        - VariableMetadata
        - TransformCode
        - apply_tcode
        - apply_tcodes
        - classify_missing
        - handle_missing
        - list_available_vintages
        - load_vintage_panel
        - RealTimePanel
      show_root_heading: true
      show_source: false
