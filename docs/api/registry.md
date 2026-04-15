# Registry API Reference

## Import surface

```python
from macrocast import (
    AxisRegistryEntry,
    AxisSelection,
    get_canonical_layer_order,
    get_axis_registry,
    get_axis_registry_entry,
    axis_governance_table,
)
```

## Objects

### `AxisRegistryEntry`

Fields:
- `axis_name`
- `layer`
- `axis_type`
- `allowed_values`
- `current_status`
- `default_policy`
- `compatible_with`
- `incompatible_with`

### `AxisSelection`

Fields:
- `axis_name`
- `layer`
- `selection_mode`
- `selected_values`
- `selected_status`

## Functions

### `get_canonical_layer_order()`
Returns the fixed study-path layer order.

### `get_axis_registry()`
Returns the full current axis registry.

### `get_axis_registry_entry(axis_name)`
Returns one registry entry by axis name.

### `axis_governance_table()`
Returns the registry in table-like dict form for plans, inspection, and tests.

## Notes

The registry layer defines representable choice space.
It does not imply that every value is executable now.
That split is intentional and encoded in `current_status`.
