#' macrocastR: Linear Regularised Models for the macrocast Pipeline
#'
#' R companion to the macrocast Python package.  Implements the linear
#' regularised forecasting models — AR, ARDI, Ridge, LASSO, Adaptive LASSO,
#' Group LASSO, Elastic Net, TVP-Ridge, and Booging — for the
#' Coulombe-Leroux-Stevanovic-Surprenant (2022) four-component decomposition
#' framework.
#'
#' Results are exported to parquet files so that the macrocast Python
#' evaluation layer (Layer 3) can merge Python and R forecasts transparently.
#'
#' @section Main functions:
#' \describe{
#'   \item{\code{\link{run_experiment}}}{Outer pseudo-OOS loop for all linear models.}
#'   \item{\code{\link{build_features}}}{Construct PCA factor + AR lag feature matrix.}
#'   \item{\code{\link{marx_transform}}}{MARX cross-product feature expansion.}
#'   \item{\code{\link{fit_ar}}}{AR(p) with BIC lag selection.}
#'   \item{\code{\link{fit_ardi}}}{ARDI (PCA diffusion index + OLS).}
#'   \item{\code{\link{fit_ridge}}}{Ridge via glmnet (alpha=0).}
#'   \item{\code{\link{fit_lasso}}}{LASSO via glmnet (alpha=1).}
#'   \item{\code{\link{fit_adaptive_lasso}}}{Adaptive LASSO (two-step).}
#'   \item{\code{\link{fit_group_lasso}}}{Group LASSO via grpreg.}
#'   \item{\code{\link{fit_elastic_net}}}{Elastic Net via glmnet.}
#'   \item{\code{\link{fit_tvp_ridge}}}{TVP-Ridge (Legendre polynomial expansion).}
#'   \item{\code{\link{fit_booging}}}{Booging (bootstrap aggregating + pruning).}
#' }
#'
#' @section References:
#' Coulombe, P.G., Leroux, M., Stevanovic, D., and Surprenant, S. (2022).
#' "How is Machine Learning Useful for Macroeconomic Forecasting?"
#' \emph{Journal of Applied Econometrics}, 37(5), 920-964.
#'
#' @docType package
#' @name macrocastR-package
#' @aliases macrocastR
"_PACKAGE"
