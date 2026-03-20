test_that(".check_pkgs passes for installed packages", {
  # stats is always available
  expect_invisible(macrocastR:::.check_pkgs("stats", "test_fn"))
})

test_that(".check_pkgs errors for missing packages", {
  expect_error(
    macrocastR:::.check_pkgs("__nonexistent_pkg__", "test_fn"),
    "requires the following packages"
  )
})

test_that(".check_no_all_na_cols passes for clean matrix", {
  X <- matrix(1:12, 4, 3)
  expect_invisible(macrocastR:::.check_no_all_na_cols(X))
})

test_that(".check_no_all_na_cols errors for all-NA column", {
  X <- matrix(c(1, 2, 3, NA, NA, NA, 7, 8, 9), 3, 3)
  expect_error(
    macrocastR:::.check_no_all_na_cols(X),
    "entirely NA"
  )
})

test_that(".msfe returns zero for perfect forecast", {
  y <- c(1, 2, 3)
  expect_equal(macrocastR:::.msfe(y, y), 0)
})

test_that(".msfe returns positive for non-perfect forecast", {
  y_true <- c(1, 2, 3)
  y_hat  <- c(0, 2, 4)
  expect_gt(macrocastR:::.msfe(y_true, y_hat), 0)
})
