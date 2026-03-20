#' macrocastR: Linear Regularised Models for the macrocast Pipeline
#'
#' R companion to the macrocast Python package.  Provides linear model fitting
#' functions — AR, ARDI, Ridge, LASSO, Adaptive LASSO, Group LASSO, Elastic
#' Net, TVP-Ridge, and Booging — called by Python's RModelEstimator bridge via
#' \code{macrocastR/inst/bridge.R}.
#'
#' Feature construction and experiment orchestration are handled entirely by
#' the Python pipeline (\code{macrocast.pipeline.FeatureBuilder} and
#' \code{macrocast.pipeline.ForecastExperiment}).  R is responsible for model
#' fitting only.
#'
#' @section Primary model fitting functions:
#' \describe{
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
#' @section Deprecated functions:
#' \describe{
#'   \item{\code{\link{build_features}}}{Replaced by Python FeatureBuilder.}
#'   \item{\code{\link{marx_transform}}}{Replaced by Python FeatureBuilder (use_marx=True).}
#'   \item{\code{\link{run_experiment}}}{Replaced by Python ForecastExperiment.}
#'   \item{\code{\link{forecast_record_to_df}}}{Replaced by Python ForecastRecord/ResultSet.}
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
