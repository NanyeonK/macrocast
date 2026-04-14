test_that("fit_ar returns scalar forecast and hp with p", {
  set.seed(10)
  y <- cumsum(rnorm(80))
  res <- fit_ar(y_train = y, y_test_lags = tail(y, 12), h = 1L, max_lag = 6L)

  expect_named(res, c("model", "y_hat", "hp"))
  expect_length(res$y_hat, 1)
  expect_true(is.numeric(res$y_hat))
  expect_true("p" %in% names(res$hp))
})

test_that("fit_ardi returns scalar forecast", {
  set.seed(11)
  T_obs <- 60; N <- 8; p <- 3; k <- 4
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)

  out_tr <- build_features(X, y, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_tr   <- out_tr$Z
  Z_te   <- build_features(X[1, , drop = FALSE], tail(y, p),
                           n_factors = k, n_lags = p, pca_fit = out_tr$pca_fit)
  y_tr   <- y[seq(p + 1, T_obs)]

  res <- fit_ardi(Z_tr, y_tr, Z_te)

  expect_length(res$y_hat, 1)
  expect_true(is.numeric(res$y_hat))
})

test_that("fit_ridge returns forecast and lambda hp", {
  skip_if_not_installed("glmnet")
  set.seed(12)
  T_obs <- 80; N <- 6; p <- 2; k <- 3
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)

  out_tr <- build_features(X, y, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_tr   <- out_tr$Z; y_tr <- y[seq(p + 1, T_obs)]
  Z_te   <- build_features(X[1, , drop = FALSE], tail(y, p), n_factors = k,
                            n_lags = p, pca_fit = out_tr$pca_fit)

  res <- fit_ridge(Z_tr, y_tr, Z_te, cv_folds = 3L, nlambda = 20L)
  expect_length(res$y_hat, 1)
  expect_true("lambda" %in% names(res$hp))
})

test_that("fit_lasso returns forecast and lambda hp", {
  skip_if_not_installed("glmnet")
  set.seed(13)
  T_obs <- 80; N <- 6; p <- 2; k <- 3
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)
  out_tr <- build_features(X, y, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_tr <- out_tr$Z; y_tr <- y[seq(p + 1, T_obs)]
  Z_te <- build_features(X[1, , drop = FALSE], tail(y, p), n_factors = k,
                          n_lags = p, pca_fit = out_tr$pca_fit)

  res <- fit_lasso(Z_tr, y_tr, Z_te, cv_folds = 3L, nlambda = 20L)
  expect_length(res$y_hat, 1)
  expect_true("lambda" %in% names(res$hp))
})

test_that("fit_adaptive_lasso returns forecast", {
  skip_if_not_installed("glmnet")
  set.seed(14)
  T_obs <- 80; N <- 6; p <- 2; k <- 3
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)
  out_tr <- build_features(X, y, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_tr <- out_tr$Z; y_tr <- y[seq(p + 1, T_obs)]
  Z_te <- build_features(X[1, , drop = FALSE], tail(y, p), n_factors = k,
                          n_lags = p, pca_fit = out_tr$pca_fit)

  res <- fit_adaptive_lasso(Z_tr, y_tr, Z_te, cv_folds = 3L, nlambda = 20L)
  expect_length(res$y_hat, 1)
})

test_that("fit_elastic_net returns forecast", {
  skip_if_not_installed("glmnet")
  set.seed(15)
  T_obs <- 80; N <- 6; p <- 2; k <- 3
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)
  out_tr <- build_features(X, y, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_tr <- out_tr$Z; y_tr <- y[seq(p + 1, T_obs)]
  Z_te <- build_features(X[1, , drop = FALSE], tail(y, p), n_factors = k,
                          n_lags = p, pca_fit = out_tr$pca_fit)

  res <- fit_elastic_net(Z_tr, y_tr, Z_te, cv_folds = 3L, nlambda = 20L)
  expect_length(res$y_hat, 1)
})

test_that("fit_tvp_ridge returns forecast and poly_degree hp", {
  skip_if_not_installed("glmnet")
  set.seed(16)
  T_obs <- 80; N <- 6; p <- 2; k <- 3
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)
  out_tr <- build_features(X, y, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_tr <- out_tr$Z; y_tr <- y[seq(p + 1, T_obs)]
  Z_te <- build_features(X[1, , drop = FALSE], tail(y, p), n_factors = k,
                          n_lags = p, pca_fit = out_tr$pca_fit)

  res <- fit_tvp_ridge(Z_tr, y_tr, Z_te, poly_degree = 2L, cv_folds = 3L)
  expect_length(res$y_hat, 1)
  expect_equal(res$hp$poly_degree, 2L)
})

test_that("fit_booging returns forecast and n_survivors hp", {
  set.seed(17)
  T_obs <- 60; N <- 6; p <- 2; k <- 3
  X <- matrix(rnorm(T_obs * N), T_obs, N)
  y <- rnorm(T_obs)
  out_tr <- build_features(X, y, n_factors = k, n_lags = p, return_pca = TRUE)
  Z_tr <- out_tr$Z; y_tr <- y[seq(p + 1, T_obs)]
  Z_te <- build_features(X[1, , drop = FALSE], tail(y, p), n_factors = k,
                          n_lags = p, pca_fit = out_tr$pca_fit)

  res <- fit_booging(Z_tr, y_tr, Z_te, n_boot = 20L, seed = 42L)
  expect_length(res$y_hat, 1)
  expect_true("n_survivors" %in% names(res$hp))
  expect_gt(res$hp$n_survivors, 0)
})
