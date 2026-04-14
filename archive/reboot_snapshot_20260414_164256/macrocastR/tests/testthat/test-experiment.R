test_that("run_experiment returns data.table with expected columns", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("data.table")
  skip_if_not_installed("arrow")

  set.seed(99)
  T_obs <- 100; N <- 8
  dates <- seq.Date(as.Date("2005-01-01"), by = "month", length.out = T_obs)
  panel <- matrix(rnorm(T_obs * N), T_obs, N)
  colnames(panel) <- paste0("x", seq_len(N))
  target <- cumsum(rnorm(T_obs)) * 0.5

  dt <- run_experiment(
    panel       = panel,
    target      = target,
    dates       = dates,
    horizons    = 1L,
    models      = c("ar", "ardi", "ridge"),
    n_factors   = 3L,
    n_lags      = 2L,
    oos_start   = as.Date("2013-01-01"),
    oos_end     = as.Date("2013-03-01"),
    cv_folds    = 3L,
    experiment_id = "test-exp-R-001"
  )

  expect_true(data.table::is.data.table(dt))
  expect_true(nrow(dt) > 0)
  required_cols <- c("experiment_id", "model_id", "nonlinearity",
                     "regularization", "horizon", "y_hat", "y_true")
  for (col in required_cols) {
    expect_true(col %in% names(dt), info = paste("missing column:", col))
  }
})

test_that("run_experiment model_id matches expected values", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("data.table")

  set.seed(100)
  T_obs <- 80; N <- 6
  dates <- seq.Date(as.Date("2005-01-01"), by = "month", length.out = T_obs)
  panel <- matrix(rnorm(T_obs * N), T_obs, N)
  target <- rnorm(T_obs)

  dt <- run_experiment(
    panel     = panel,
    target    = target,
    dates     = dates,
    horizons  = 1L,
    models    = c("ar", "lasso"),
    n_factors = 2L, n_lags = 2L,
    oos_start = as.Date("2011-01-01"),
    oos_end   = as.Date("2011-02-01"),
    cv_folds  = 3L
  )

  model_ids <- unique(dt$model_id)
  expect_true(any(grepl("none", model_ids)))    # AR has reg = none
  expect_true(any(grepl("lasso", model_ids)))
})

test_that("run_experiment writes parquet when output_path provided", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("data.table")

  set.seed(101)
  T_obs <- 80; N <- 5
  dates <- seq.Date(as.Date("2005-01-01"), by = "month", length.out = T_obs)
  panel <- matrix(rnorm(T_obs * N), T_obs, N)
  target <- rnorm(T_obs)
  tmp_file <- tempfile(fileext = ".parquet")

  dt <- run_experiment(
    panel     = panel,
    target    = target,
    dates     = dates,
    horizons  = 1L,
    models    = "ar",
    n_lags    = 2L,
    oos_start = as.Date("2011-01-01"),
    oos_end   = as.Date("2011-02-01"),
    output_path = tmp_file
  )

  expect_true(file.exists(tmp_file))
  dt_read <- arrow::read_parquet(tmp_file)
  expect_equal(nrow(dt_read), nrow(dt))
})

test_that("run_experiment errors on rolling without rolling_size", {
  T_obs <- 60
  dates <- seq.Date(as.Date("2005-01-01"), by = "month", length.out = T_obs)
  panel <- matrix(rnorm(T_obs * 5), T_obs, 5)
  target <- rnorm(T_obs)

  expect_error(
    run_experiment(panel, target, dates, window = "rolling", rolling_size = NULL),
    "rolling_size"
  )
})
