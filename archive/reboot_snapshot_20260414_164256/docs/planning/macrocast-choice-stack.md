# macrocast choice stack

`macrocast_single_run()` is a staged YAML-building and inspection process with Stage 0 routing up front.

There are two levels of choice:

1. Start-stage choices
- axes
- registries
- choice_stack
- yaml_preview
- compile
- tree_context
- runs_preview
- manifest_preview

2. Ordered experiment/YAML choices

## Ordered choice stack

Read first:
- `stage-0-meta-layer.md`

Stage 0 is not a small preface. It changes the structure of the rest of the wizard.

## Universal Stage 0 prefix
1. experiment_unit
2. reproducibility_mode
3. failure_policy
4. compute_mode

## If `experiment_unit == single_target_single_model`
5. domain
6. data
7. frequency
8. info_set
9. sample
10. oos
11. task
12. target
13. x_map
14. target_prep
15. x_prep
16. features
17. framework
18. validation
19. model
20. tuning
21. metric
22. benchmark
23. stat
24. importance
25. output
26. numeric_params
27. outputs

## If `experiment_unit == single_target_model_grid`
- Stage 0 owner remains `macrocast_single_run()`
- later stack should split into:
  - fixed single-target path choices
  - sweep-enabled model/tuning substack
- current implementation does not yet branch this path after Stage 0

## If `experiment_unit == single_target_full_sweep`
- Stage 0 owner remains `macrocast_single_run()`
- later stack should split into:
  - fixed experiment axes
  - sweep axes
  - nested/conditional axes
- current implementation does not yet branch this path after Stage 0

## If `experiment_unit` is wrapper-owned
Wrapper-owned families:
- `multi_target_separate_runs`
- `multi_target_shared_design`
- `replication_recipe`
- `benchmark_suite`
- `ablation_study`

Then `macrocast_single_run()` should:
- record Stage 0 metadata in YAML
- stop the single-run wizard
- hand off to a future wrapper/orchestrator entry point

Interpretation:
- earlier choices may fix or narrow later choices
- Stage 0 can also terminate the single-run path early and redirect ownership
- some later choices are dataset-specific or user-specified even if the taxonomy gives broad families only
- numeric/free parameters are separate from enumerated taxonomy choices

Recommended testing order:
1. `macrocast_single_run()`
2. `macrocast_single_run(stages=['axes'])`
3. `macrocast_single_run(stages=['registries'])`
4. `macrocast_single_run(stages=['choice_stack'])`
5. `macrocast_single_run(stages=['yaml_preview'], selections={...})`
6. `macrocast_single_run(yaml_path='user_recipe.yaml', stages=['compile','tree_context'])`
7. `macrocast_single_run(yaml_path='user_recipe.yaml', stages=['runs_preview','manifest_preview'])`
