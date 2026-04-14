# macrocast_single_run

`macrocast_single_run()` is the single-path guided YAML builder and preview runner.

Design rule:
- one call manages one path family only
- default call starts the wizard
- the wizard asks for the YAML filename first
- then every step shows the available options
- every step writes the YAML immediately after selection
- Stage 0 is evaluated before the rest of the stack continues
- if Stage 0 selects a wrapper-owned `experiment_unit`, the wizard should stop after writing the handoff-ready YAML metadata
- custom handling exists through the `custom` option path in the wizard design, but the current implementation first stabilizes the default enumerated flow and YAML writing
- if a prewritten YAML path is supplied, auto-run compile/tree/runs/manifest preview is allowed only for implemented single-run units

## Default wizard mode

```python
from macrocast import macrocast_single_run

out = macrocast_single_run()
```

Wizard behavior:
1. ask which YAML file to write
2. ask Stage 0 routing choices first
3. classify the selected `experiment_unit`
4. if it is an implemented single-run path, continue through later choices
5. if it is a future single-run extension or wrapper-owned family, stop with routing guidance after writing YAML

## Auto-run prewritten YAML

```python
out = macrocast_single_run(yaml_path='/path/to/user_recipe.yaml')
```

Behavior:
- if `meta.experiment_unit == single_target_single_model`
  - compile/tree/runs/manifest preview may proceed
- if `meta.experiment_unit` belongs to a future single-run extension or wrapper-owned family
  - auto-run preview should be blocked with an explicit routing message

## Read this before making choices

- Stage 0 Meta Layer: `stage-0-meta-layer.md`

Stage 0 determines what kind of run the wizard is building. In particular, `0.1 experiment_unit` decides whether the rest of the flow remains a true `macrocast_single_run()` path or must hand off to a future wrapper/orchestrator.
