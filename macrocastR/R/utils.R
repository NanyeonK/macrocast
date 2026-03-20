#' @title macrocastR internal utilities
#' @name utils
#' @keywords internal
NULL


#' Check that required packages are installed
#'
#' Called at the start of functions that depend on optional packages.
#'
#' @param pkgs Character vector of package names.
#' @param fn_name Character.  Name of the calling function (for error message).
#'
#' @return Invisible NULL.  Stops with an informative message if any package
#'   is missing.
#'
#' @keywords internal
.check_pkgs <- function(pkgs, fn_name = "") {
  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop(
      fn_name, " requires the following packages: ",
      paste(missing_pkgs, collapse = ", "),
      ". Install with: install.packages(c(",
      paste0('"', missing_pkgs, '"', collapse = ", "), "))",
      call. = FALSE
    )
  }
  invisible(NULL)
}


#' Validate that a matrix has no all-NA columns
#'
#' @param X Numeric matrix.
#' @param name Character.  Variable name for error messages.
#'
#' @keywords internal
.check_no_all_na_cols <- function(X, name = "X") {
  all_na <- apply(X, 2, function(col) all(is.na(col)))
  if (any(all_na)) {
    n_bad <- sum(all_na)
    stop(
      name, " has ", n_bad, " column(s) that are entirely NA. ",
      "Remove or impute these before fitting.",
      call. = FALSE
    )
  }
  invisible(NULL)
}


#' Validate horizon h against training sample length
#'
#' @param T_full Integer.  Full training sample length.
#' @param h Integer.  Forecast horizon.
#'
#' @keywords internal
.check_horizon <- function(T_full, h) {
  if (T_full <= h) {
    stop(
      "Training sample (T=", T_full, ") must be longer than horizon h=", h, ".",
      call. = FALSE
    )
  }
  invisible(NULL)
}


#' Compute mean squared error
#'
#' @param y_true Numeric vector.
#' @param y_hat Numeric vector.
#'
#' @return Scalar MSFE.
#'
#' @keywords internal
.msfe <- function(y_true, y_hat) {
  mean((y_true - y_hat)^2, na.rm = TRUE)
}
