#' @title Linear regularised forecasting models for macrocastR
#'
#' @description
#' Each function fits one model on a single training window and returns a
#' named list with elements:
#'
#' * `model`  -- fitted model object
#' * `y_hat`  -- point forecast for the test row(s)
#' * `hp`     -- selected hyperparameter values (named list)
#'
#' All functions follow the **direct** h-step forecasting convention:
#' the caller passes `y_train` already shifted by horizon h so that row t of
#' `Z_train` aligns with `y_train[t] = y_{t+h}`.
#'
#' @name models
NULL


# ---------------------------------------------------------------------------
# Internal helper: BIC-based lag selection for AR
# ---------------------------------------------------------------------------

.select_ar_lag_bic <- function(y, max_lag = 12L) {
  bic_vals <- numeric(max_lag)
  for (p in seq_len(max_lag)) {
    if (length(y) <= p + 1) {
      bic_vals[p] <- Inf
      next
    }
    # Build lag matrix manually for OLS (avoids stats::ar overhead)
    T_eff <- length(y) - p
    Y  <- y[seq(p + 1, length(y))]
    X  <- matrix(NA_real_, nrow = T_eff, ncol = p + 1)
    X[, 1] <- 1  # intercept
    for (lag in seq_len(p)) {
      X[, lag + 1] <- y[seq(p + 1 - lag, length(y) - lag)]
    }
    fit    <- lm.fit(X, Y)
    sigma2 <- sum(fit$residuals^2) / T_eff
    k      <- p + 2  # intercept + p lags + 1 variance param
    bic_vals[p] <- T_eff * log(sigma2) + k * log(T_eff)
  }
  which.min(bic_vals)
}


# ---------------------------------------------------------------------------
# AR(p) with BIC lag selection
# ---------------------------------------------------------------------------

#' Fit AR(p) with BIC lag order selection
#'
#' Benchmark model.  Lag order selected by BIC up to `max_lag`.
#'
#' @param Z_train Ignored.  AR uses only the target series for features.
#' @param y_train Numeric vector.  Target series (NOT shifted; the function
#'   applies the shift internally for direct forecasting when `h > 0`).
#' @param y_test_lags Numeric vector (length >= max_lag).  Last observations
#'   of the target series used to construct the AR feature row for forecasting.
#' @param h Integer.  Forecast horizon (used to align target internally).
#'   Pass `h = 1` for one-step-ahead, etc.
#' @param max_lag Integer.  Maximum lag order considered by BIC.
#'
#' @return Named list: `model` (list with coef and lag order), `y_hat`
#'   (scalar forecast), `hp` (named list with `p`).
#'
#' @export
fit_ar <- function(Z_train = NULL, y_train, y_test_lags, h = 1L, max_lag = 12L) {
  # Direct h-step: align X_t with y_{t+h}
  T_full <- length(y_train)
  if (T_full <= h) stop("y_train too short for horizon h.")

  # y target for direct forecasting
  y_target <- y_train[seq(h + 1, T_full)]  # y_{1+h} .. y_T
  T_eff    <- length(y_target)

  p <- .select_ar_lag_bic(y_target, max_lag = min(max_lag, floor(T_eff / 3)))

  # Design matrix for training
  X_mat <- matrix(NA_real_, nrow = T_eff, ncol = p + 1)
  X_mat[, 1] <- 1
  for (lag in seq_len(p)) {
    # y_{t+h - lag} for t=1..T_eff: shift back by lag from y_target
    X_mat[, lag + 1] <- y_train[seq(h + 1 - lag, T_full - lag)]
  }
  fit  <- lm.fit(X_mat, y_target)
  coef <- fit$coefficients

  # Forecast: use y_test_lags as the AR feature row
  x_test <- c(1, rev(tail(y_test_lags, p)))  # [intercept, y_{T}, y_{T-1}, ...]
  y_hat  <- as.numeric(coef %*% x_test)

  list(
    model = list(coef = coef, p = p),
    y_hat = y_hat,
    hp    = list(p = p)
  )
}


# ---------------------------------------------------------------------------
# ARDI (PCA diffusion index + AR lags)
# ---------------------------------------------------------------------------

#' Fit ARDI (Autoregressive Diffusion Index) model via OLS
#'
#' Z_train must already contain PCA factor columns + AR lag columns, as
#' returned by `build_features(use_factors = TRUE)`.
#'
#' @param Z_train Numeric matrix (T_z, n_features).  Feature matrix with
#'   factor and AR lag columns.  Rows aligned with `y_train`.
#' @param y_train Numeric vector (T_z).  Already h-shifted target.
#' @param Z_test  Numeric matrix (T_test, n_features).  Test feature row(s).
#' @param intercept Logical.  Include intercept in OLS.
#'
#' @return Named list: `model`, `y_hat`, `hp` (empty — OLS has no HP).
#'
#' @export
fit_ardi <- function(Z_train, y_train, Z_test, intercept = TRUE) {
  if (intercept) {
    X_tr  <- cbind(1, Z_train)
    X_te  <- cbind(1, Z_test)
  } else {
    X_tr  <- Z_train
    X_te  <- Z_test
  }
  fit   <- lm.fit(X_tr, y_train)
  coef  <- fit$coefficients
  y_hat <- as.numeric(X_te %*% coef)

  list(
    model = list(coef = coef),
    y_hat = y_hat,
    hp    = list()
  )
}


# ---------------------------------------------------------------------------
# Ridge (glmnet alpha = 0)
# ---------------------------------------------------------------------------

#' Fit Ridge regression via glmnet with CV penalty selection
#'
#' @param Z_train Numeric matrix (T_z, n_features).
#' @param y_train Numeric vector (T_z).
#' @param Z_test  Numeric matrix (T_test, n_features).
#' @param cv_folds Integer.  Number of folds for glmnet's cross-validation.
#' @param nlambda Integer.  Number of lambda values in the path.
#'
#' @return Named list: `model` (cv.glmnet fit), `y_hat`, `hp` (lambda.min).
#'
#' @export
fit_ridge <- function(Z_train, y_train, Z_test, cv_folds = 5L, nlambda = 50L) {
  cv_fit <- glmnet::cv.glmnet(
    x         = Z_train,
    y         = y_train,
    alpha     = 0,
    nfolds    = cv_folds,
    nlambda   = nlambda,
    type.measure = "mse"
  )
  y_hat <- as.numeric(predict(cv_fit, newx = Z_test, s = "lambda.min"))

  list(
    model = cv_fit,
    y_hat = y_hat,
    hp    = list(lambda = cv_fit$lambda.min)
  )
}


# ---------------------------------------------------------------------------
# LASSO (glmnet alpha = 1)
# ---------------------------------------------------------------------------

#' Fit LASSO via glmnet with CV penalty selection
#'
#' @inheritParams fit_ridge
#' @return Named list: `model`, `y_hat`, `hp` (lambda.min).
#'
#' @export
fit_lasso <- function(Z_train, y_train, Z_test, cv_folds = 5L, nlambda = 50L) {
  cv_fit <- glmnet::cv.glmnet(
    x         = Z_train,
    y         = y_train,
    alpha     = 1,
    nfolds    = cv_folds,
    nlambda   = nlambda,
    type.measure = "mse"
  )
  y_hat <- as.numeric(predict(cv_fit, newx = Z_test, s = "lambda.min"))

  list(
    model = cv_fit,
    y_hat = y_hat,
    hp    = list(lambda = cv_fit$lambda.min)
  )
}


# ---------------------------------------------------------------------------
# Adaptive LASSO
# ---------------------------------------------------------------------------

#' Fit Adaptive LASSO via glmnet with oracle penalty weights
#'
#' Two-step procedure:
#'   1. Fit Ridge to get initial coefficient estimates beta_hat.
#'   2. Compute penalty factors w_j = 1 / |beta_hat_j|^gamma (gamma = 1 default).
#'   3. Fit LASSO with penalty.factor = w.
#'
#' @inheritParams fit_ridge
#' @param gamma Numeric.  Exponent for penalty weight computation.  Default 1.
#'
#' @return Named list: `model`, `y_hat`, `hp` (lambda.min, gamma).
#'
#' @export
fit_adaptive_lasso <- function(Z_train, y_train, Z_test,
                               cv_folds = 5L, nlambda = 50L, gamma = 1) {
  # Step 1: Ridge for initial estimates
  ridge_fit <- glmnet::glmnet(Z_train, y_train, alpha = 0, nlambda = 50L)
  # Pick the Ridge coefficient at a moderate lambda (use lambda at 10% from end)
  lambda_idx  <- max(1L, round(0.1 * length(ridge_fit$lambda)))
  beta_init   <- as.numeric(coef(ridge_fit, s = ridge_fit$lambda[lambda_idx]))[-1]

  # Step 2: Penalty factors (protect against zero)
  eps      <- .Machine$double.eps^0.5
  pen_wts  <- 1 / pmax(abs(beta_init), eps)^gamma

  # Step 3: Adaptive LASSO
  cv_fit <- glmnet::cv.glmnet(
    x              = Z_train,
    y              = y_train,
    alpha          = 1,
    penalty.factor = pen_wts,
    nfolds         = cv_folds,
    nlambda        = nlambda,
    type.measure   = "mse"
  )
  y_hat <- as.numeric(predict(cv_fit, newx = Z_test, s = "lambda.min"))

  list(
    model = cv_fit,
    y_hat = y_hat,
    hp    = list(lambda = cv_fit$lambda.min, gamma = gamma)
  )
}


# ---------------------------------------------------------------------------
# Group LASSO (grpreg)
# ---------------------------------------------------------------------------

#' Fit Group LASSO via grpreg with CV penalty selection
#'
#' Group structure follows FRED variable categories:
#' output_income, labor, housing, prices, money, interest_rates, stock_market.
#' An additional group "ar_lags" is created for the AR lag columns appended
#' by `build_features()`.
#'
#' @param Z_train Numeric matrix (T_z, n_features).  Feature matrix where
#'   column names encode group membership (prefix before "_").
#' @param y_train Numeric vector (T_z).
#' @param Z_test  Numeric matrix (T_test, n_features).
#' @param group   Integer or character vector (n_features).  Group membership
#'   vector passed directly to grpreg.  If NULL, inferred from column name
#'   prefixes.
#' @param cv_folds Integer.
#'
#' @return Named list: `model`, `y_hat`, `hp` (lambda, group summary).
#'
#' @export
fit_group_lasso <- function(Z_train, y_train, Z_test,
                            group = NULL, cv_folds = 5L) {
  if (is.null(group)) {
    # Infer groups from column name prefix (e.g. "output_income_f1" → "output_income")
    cnames <- colnames(Z_train)
    if (is.null(cnames)) {
      stop("Z_train must have column names for automatic group inference.")
    }
    group <- sub("_[^_]+$", "", cnames)  # strip last _segment
  }

  cv_fit <- grpreg::cv.grpreg(
    X      = Z_train,
    y      = y_train,
    group  = group,
    nfolds = cv_folds,
    penalty = "grLasso"
  )
  y_hat <- as.numeric(predict(cv_fit, X = Z_test, lambda = cv_fit$lambda.min))

  list(
    model = cv_fit,
    y_hat = y_hat,
    hp    = list(lambda = cv_fit$lambda.min, groups = unique(group))
  )
}


# ---------------------------------------------------------------------------
# Elastic Net (glmnet alpha = 0.5)
# ---------------------------------------------------------------------------

#' Fit Elastic Net via glmnet with CV penalty selection
#'
#' @inheritParams fit_ridge
#' @param alpha Numeric.  Mixing parameter between Ridge (0) and LASSO (1).
#'   Default 0.5.
#'
#' @return Named list: `model`, `y_hat`, `hp` (lambda.min, alpha).
#'
#' @export
fit_elastic_net <- function(Z_train, y_train, Z_test,
                            cv_folds = 5L, nlambda = 50L, alpha = 0.5) {
  cv_fit <- glmnet::cv.glmnet(
    x         = Z_train,
    y         = y_train,
    alpha     = alpha,
    nfolds    = cv_folds,
    nlambda   = nlambda,
    type.measure = "mse"
  )
  y_hat <- as.numeric(predict(cv_fit, newx = Z_test, s = "lambda.min"))

  list(
    model = cv_fit,
    y_hat = y_hat,
    hp    = list(lambda = cv_fit$lambda.min, alpha = alpha)
  )
}


# ---------------------------------------------------------------------------
# TVP-Ridge (Time-Varying Parameters via two-step Ridge)
# ---------------------------------------------------------------------------

#' Fit TVP-Ridge model
#'
#' Implements time-varying parameter estimation via the two-step Ridge approach
#' of Coulombe (2024, philgoucou/tvpridge).  The idea: expand the feature
#' matrix by interacting each predictor with a polynomial time trend basis
#' (Legendre polynomials), then apply Ridge.  This allows coefficients to
#' drift smoothly over time without state-space estimation.
#'
#' Step 1: Build time-expanded design matrix.
#'   Z_tvp[t, :] = [Z[t,1]*B(t), Z[t,2]*B(t), ..., Z[t,N]*B(t)]
#'   where B(t) = Legendre polynomial basis evaluated at rescaled t in [-1,1].
#'
#' Step 2: Apply Ridge (glmnet alpha = 0) to Z_tvp.
#'
#' @param Z_train Numeric matrix (T_z, n_features).
#' @param y_train Numeric vector (T_z).
#' @param Z_test  Numeric matrix (T_test, n_features).
#' @param t_train Integer vector (T_z).  Time indices for training rows
#'   (used to evaluate polynomial basis).  If NULL, uses 1:T_z.
#' @param t_test  Integer vector (T_test).  Time indices for test rows.
#'   If NULL, uses T_z + 1.
#' @param poly_degree Integer.  Degree of Legendre polynomial basis.  Default 3.
#' @param cv_folds Integer.
#'
#' @references
#'   Coulombe, P.G. (2024). "A TVP Interpretation via the Ridge."
#'   philgoucou/tvpridge GitHub repository.
#'
#' @return Named list: `model`, `y_hat`, `hp` (lambda, poly_degree).
#'
#' @export
fit_tvp_ridge <- function(Z_train, y_train, Z_test,
                          t_train = NULL, t_test = NULL,
                          poly_degree = 3L, cv_folds = 5L) {
  T_z  <- nrow(Z_train)
  N_f  <- ncol(Z_train)

  if (is.null(t_train)) t_train <- seq_len(T_z)
  if (is.null(t_test))  t_test  <- T_z + seq_len(nrow(Z_test))

  # Rescale time to [-1, 1] using training range
  t_min <- min(t_train)
  t_max <- max(t_train)
  t_scale <- function(t) 2 * (t - t_min) / max(t_max - t_min, 1) - 1

  # Legendre polynomial basis B(t): columns are P_0(t), P_1(t), ..., P_d(t)
  .legendre_basis <- function(t_vec, degree) {
    t_sc <- t_scale(t_vec)
    B    <- matrix(NA_real_, nrow = length(t_vec), ncol = degree + 1)
    B[, 1] <- 1          # P_0
    if (degree >= 1) B[, 2] <- t_sc  # P_1
    for (k in seq(2, degree)) {
      # Recurrence: P_k = ((2k-1)*t*P_{k-1} - (k-1)*P_{k-2}) / k
      B[, k + 1] <- ((2 * k - 1) * t_sc * B[, k] - (k - 1) * B[, k - 1]) / k
    }
    B
  }

  B_train <- .legendre_basis(t_train, poly_degree)   # (T_z, deg+1)
  B_test  <- .legendre_basis(t_test,  poly_degree)   # (T_test, deg+1)

  # Build TVP design matrix: Kronecker row-wise expansion
  # Z_tvp[t, :] = vec(Z[t,] ⊗ B[t,]) — one row of Z interacted with one row of B
  # Result: (T_z, N_f * (poly_degree + 1))
  .interact <- function(Z_mat, B_mat) {
    T_obs  <- nrow(Z_mat)
    n_b    <- ncol(B_mat)
    Z_tvp  <- matrix(NA_real_, nrow = T_obs, ncol = ncol(Z_mat) * n_b)
    for (j in seq_len(ncol(Z_mat))) {
      col_start <- (j - 1) * n_b + 1
      col_end   <- j * n_b
      Z_tvp[, col_start:col_end] <- Z_mat[, j] * B_mat
    }
    Z_tvp
  }

  Z_train_tvp <- .interact(Z_train, B_train)
  Z_test_tvp  <- .interact(Z_test,  B_test)

  # Ridge on expanded matrix
  cv_fit <- glmnet::cv.glmnet(
    x        = Z_train_tvp,
    y        = y_train,
    alpha    = 0,
    nfolds   = cv_folds,
    type.measure = "mse"
  )
  y_hat <- as.numeric(predict(cv_fit, newx = Z_test_tvp, s = "lambda.min"))

  list(
    model = cv_fit,
    y_hat = y_hat,
    hp    = list(lambda = cv_fit$lambda.min, poly_degree = poly_degree)
  )
}


# ---------------------------------------------------------------------------
# Booging (Bootstrap Aggregating + Pruning)
# ---------------------------------------------------------------------------

#' Fit Booging (Bootstrap Aggregating of OLS with pruning)
#'
#' Implements the Booging ensemble from Coulombe (2025, philgoucou/bagofprunes).
#' Algorithm:
#'   1. Draw B bootstrap samples from the training window.
#'   2. On each bootstrap, fit OLS on a random subset of predictors (size
#'      floor(sqrt(N_f))), producing coefficient vector beta_b.
#'   3. Prune: keep only the bootstrap models whose in-sample MSFE is below
#'      the `prune_quantile` quantile (default 0.5, i.e., top half).
#'   4. Average the surviving forecasts (equal weights).
#'
#' The predictor subsampling combined with bootstrap aggregation acts as
#' implicit regularisation without requiring a penalty parameter.
#'
#' @param Z_train Numeric matrix (T_z, n_features).
#' @param y_train Numeric vector (T_z).
#' @param Z_test  Numeric matrix (T_test, n_features).
#' @param n_boot  Integer.  Number of bootstrap draws.  Default 200.
#' @param prune_quantile Numeric in (0, 1].  Fraction of bootstrap models to
#'   keep (those with lowest in-sample MSFE).  Default 0.5.
#' @param seed    Integer or NULL.  Random seed for reproducibility.
#'
#' @references
#'   Coulombe, P.G. (2025). "Bag of Prunes."
#'   philgoucou/bagofprunes GitHub repository.
#'
#' @return Named list: `model` (list with surviving forecasts and weights),
#'   `y_hat`, `hp` (n_boot, prune_quantile, n_survivors).
#'
#' @export
fit_booging <- function(Z_train, y_train, Z_test,
                        n_boot = 200L, prune_quantile = 0.5, seed = 42L) {
  if (!is.null(seed)) set.seed(seed)

  T_z  <- nrow(Z_train)
  N_f  <- ncol(Z_train)
  k    <- max(1L, floor(sqrt(N_f)))  # predictor subset size per bootstrap

  boot_forecasts  <- numeric(n_boot)
  boot_msfe       <- numeric(n_boot)

  for (b in seq_len(n_boot)) {
    # Bootstrap sample (with replacement)
    idx_boot <- sample(T_z, T_z, replace = TRUE)
    # Random predictor subset
    col_idx  <- sample(N_f, k, replace = FALSE)

    X_b <- cbind(1, Z_train[idx_boot, col_idx, drop = FALSE])
    y_b <- y_train[idx_boot]

    fit_b <- tryCatch(
      lm.fit(X_b, y_b),
      error = function(e) NULL
    )
    if (is.null(fit_b)) {
      boot_msfe[b]      <- Inf
      boot_forecasts[b] <- NA_real_
      next
    }
    coef_b <- fit_b$coefficients

    # In-sample MSFE on original training data (not bootstrap sample)
    X_train_sub <- cbind(1, Z_train[, col_idx, drop = FALSE])
    resid_b      <- y_train - as.numeric(X_train_sub %*% coef_b)
    boot_msfe[b] <- mean(resid_b^2)

    # Forecast on test
    X_test_sub      <- cbind(1, Z_test[, col_idx, drop = FALSE])
    boot_forecasts[b] <- mean(as.numeric(X_test_sub %*% coef_b))
  }

  # Pruning: keep top (prune_quantile * n_boot) models by in-sample MSFE
  msfe_threshold <- quantile(boot_msfe[is.finite(boot_msfe)], prune_quantile)
  keep_idx       <- which(boot_msfe <= msfe_threshold & is.finite(boot_msfe))

  if (length(keep_idx) == 0) {
    # Fallback: keep all finite
    keep_idx <- which(is.finite(boot_msfe))
  }

  y_hat <- mean(boot_forecasts[keep_idx], na.rm = TRUE)

  list(
    model = list(
      forecasts    = boot_forecasts[keep_idx],
      n_survivors  = length(keep_idx)
    ),
    y_hat = y_hat,
    hp    = list(
      n_boot         = n_boot,
      prune_quantile = prune_quantile,
      n_survivors    = length(keep_idx)
    )
  )
}
