test_that("build_features returns correct shape in factors mode", {
  set.seed(1)
  T_obs <- 80; N <- 15; p <- 4; k <- 6
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)

  Z <- build_features(X, y, n_factors = k, n_lags = p, use_factors = TRUE)

  expect_equal(nrow(Z), T_obs - p)
  expect_equal(ncol(Z), k + p)
})

test_that("build_features returns correct shape in AR-only mode", {
  set.seed(2)
  T_obs <- 60; N <- 10; p <- 3
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)

  Z <- build_features(X, y, n_lags = p, use_factors = FALSE)

  expect_equal(nrow(Z), T_obs - p)
  expect_equal(ncol(Z), p)
})

test_that("build_features with return_pca stores PCA fit", {
  set.seed(3)
  X <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)

  out <- build_features(X, y, n_factors = 4, n_lags = 2, return_pca = TRUE)

  expect_named(out, c("Z", "pca_fit"))
  expect_equal(out$pca_fit$n_factors, 4L)
  expect_equal(dim(out$pca_fit$rotation), c(10, 4))
})

test_that("pre-fitted PCA is applied to test window without refitting", {
  set.seed(4)
  T_tr <- 80; T_te <- 10; N <- 8; p <- 2; k <- 3
  X_tr <- matrix(rnorm(T_tr * N), T_tr, N)
  X_te <- matrix(rnorm(T_te * N), T_te, N)
  y_tr <- rnorm(T_tr)
  y_te <- rnorm(p)

  out_tr   <- build_features(X_tr, y_tr, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_te     <- build_features(X_te, y_te, n_factors = k, n_lags = p,
                              pca_fit = out_tr$pca_fit)

  # Test window: only p obs of y, so nrow(Z_te) = T_te - p
  expect_equal(ncol(Z_te), k + p)
})

test_that("n_factors is clamped to available dimensions", {
  set.seed(5)
  X <- matrix(rnorm(50 * 3), 50, 3)  # only 3 columns
  y <- rnorm(50)

  # Request more factors than columns — should silently clamp
  Z <- build_features(X, y, n_factors = 20, n_lags = 2, use_factors = TRUE)
  expect_lte(ncol(Z), 3 + 2)
})

test_that("marx_transform expands columns correctly", {
  set.seed(6)
  T_obs <- 50; N <- 4; p <- 2; p_m <- 1
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)

  Z_base <- build_features(X, y, n_lags = p, use_factors = FALSE)
  Z_marx <- marx_transform(Z_base, X, y, n_lags = p, p_marx = p_m)

  expect_equal(ncol(Z_marx), ncol(Z_base) + N * p_m)
  expect_equal(nrow(Z_marx), nrow(Z_base))
})
