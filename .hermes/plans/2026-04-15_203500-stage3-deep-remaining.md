# Stage 3 Deep Expansion — Remaining Work

Goal: implement all remaining items from `plans/stage3-deep-expansion.md` beyond already-completed deep model slice.

Remaining scope:
1. Factor / factor-builder slice
   - `factor_pca`
   - `factors_plus_AR`
   - `pcr`
   - `pls`
   - `factor_augmented_linear`
   - `factor_count` runtime (`fixed`, `cv_select`, `BaiNg_rule`)
2. Tuning engine
   - core types
   - grid/random/optuna bayes/genetic search
   - budget enforcement
   - model HP spaces
3. Validation engine
   - temporal splitters
   - validation size/location
   - scorers
4. Framework/runtime expansion
   - `anchored_rolling`
   - `refit_every_k_steps`
   - `fit_once_predict_many`
   - `blocked_kfold`
   - `expanding_cv`
   - `rolling_cv`
   - `early_stopping`
   - `convergence_handling`
5. Tests/docs/manifests

Execution order:
- First factor slice + feature-builder widening so Stage 3 model layer is complete.
- Second tuning/validation engine under new `macrocast/tuning/` package.
- Third wire compiler/runtime to consume training axes instead of provenance-only defaults.
- Finally update tests/docs and run broad regression.
