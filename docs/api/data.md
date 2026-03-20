# macrocast.data

Full API reference for the data layer modules.

---

## macrocast.data.fred_md

::: macrocast.data.fred_md
    options:
      members:
        - load_fred_md
      show_root_heading: true

---

## macrocast.data.fred_qd

::: macrocast.data.fred_qd
    options:
      members:
        - load_fred_qd
      show_root_heading: true

---

## macrocast.data.fred_sd

::: macrocast.data.fred_sd
    options:
      members:
        - load_fred_sd
      show_root_heading: true

---

## macrocast.data.schema

::: macrocast.data.schema
    options:
      members:
        - MacroFrame
        - MacroFrameMetadata
        - VariableMetadata
      show_root_heading: true

---

## macrocast.data.transforms

::: macrocast.data.transforms
    options:
      members:
        - TransformCode
        - apply_tcode
        - apply_tcodes
        - inverse_tcode
      show_root_heading: true

---

## macrocast.data.missing

::: macrocast.data.missing
    options:
      members:
        - classify_missing
        - handle_missing
        - detect_missing_type
      show_root_heading: true

---

## macrocast.data.vintages

::: macrocast.data.vintages
    options:
      members:
        - list_available_vintages
        - load_vintage_panel
        - RealTimePanel
      show_root_heading: true
