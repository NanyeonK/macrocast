# macrocast wizard implementation plan

Objective:
Implement a hierarchical, docs-linked wizard that can eventually cover the full atomic choice inventory from `archive/legacy-plans/source/plan_2026_04_09_2358.md` without confusing single-run path construction with future wrapper/orchestration work.

## Current gap
Current wizard is still a flat stack.
Source plan requires:
- 8 main stages
- many substages
- ~136 atomic choices
- docs-linked explanations for each choice
- custom option at every step
- YAML write-through at every step
- Stage 0 routing that can either continue single-run flow or stop and hand off

## Locked redesign decision
- `macrocast_single_run()` owns Stage 0 plus the single-target family
- currently executable path in that entry point: `single_target_single_model`
- future but still same entry point:
  - `single_target_model_grid`
  - `single_target_full_sweep`
- future wrapper/orchestrator owns:
  - multi-target families
  - benchmark suites
  - ablation studies
  - replication bundles

Contract consequence:
- `experiment_unit` and related Stage 0 controls belong in recipe `meta`
- they should not be treated as ordinary taxonomy-path leaves
- compile/run preview should be blocked when the selected route is not yet executable in `macrocast_single_run()`

## Required build order
1. Stage 0 ownership and route-spec model
2. docs-linked metadata for every atomic choice
3. single-run-compatible subset filter
4. branch-aware wizard renderer using stage -> substage -> atomic choice traversal
5. custom payload schema and serializers
6. resume/continue support
7. wrapper/orchestrator contract for multi-run families
8. expanded coverage of remaining atomic choices

## Immediate next milestone
Build canonical hierarchical spec for:
- Stage 0 Meta routing
- Stage 1 Data/Task for `single_target_single_model`
- Stage 1-4 branch deltas for `single_target_model_grid`
- handoff contract for wrapper-owned units
