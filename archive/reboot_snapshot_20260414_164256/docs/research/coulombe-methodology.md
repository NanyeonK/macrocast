# Coulombe Methodology: ML for Macroeconomic Forecasting

*This document synthesizes the research program of Philippe Goulet Coulombe (McGill University) on machine learning for macroeconomic forecasting. It serves as the design specification for macrocast's Pipeline Layer (Layer 2). All architectural decisions in `macrocast/pipeline/` trace to the evidence reviewed here.*

---

## Section 1: The Central Question — Why Does ML Work in Macroeconomics?

The growing use of machine learning in macroeconomic forecasting invites a deceptively simple question: which properties of ML algorithms produce forecast gains? The answer is not obvious. ML encompasses a heterogeneous collection of techniques, and attributing improved forecast accuracy to "machine learning" as a category obscures more than it reveals.

Coulombe, Leroux, Stevanovic, and Surprenant (2022) provide the first systematic answer. Their framework decomposes any ML estimator into four orthogonal components: the nonlinearity of the functional form, the regularization strategy applied to parameter estimation, the cross-validation scheme used for hyperparameter selection, and the loss function optimized during training. By activating and deactivating each component independently in a pseudo-out-of-sample experiment on FRED-MD monthly data, they estimate treatment effects for each component separately.

The central finding is unambiguous. Nonlinearity is the dominant driver of ML forecast gains. The treatment effect of nonlinearity is large and statistically significant; the treatment effects of the other three components are smaller and more context-dependent. Critically, the nonlinearity gains are not uniformly distributed over time. They concentrate during periods of elevated macroeconomic uncertainty, financial stress, and housing market instability — precisely the episodes when linear approximations to the data-generating process are most strained. During tranquil periods, nonlinear and linear models perform comparably.

This finding has a precise interpretation. ML methods are not useful in macroeconomics because real economies are governed by complex universal nonlinearities of the kind that deep networks are theoretically capable of approximating. They are useful because macroeconomic parameters drift over time, and nonlinear estimators can implicitly track this drift in a data-driven way. The distinction matters for model design. A model targeting generic nonlinearity will differ fundamentally from a model designed to capture parameter time variation.

The implication for macrocast is architectural. The four components of the JAE 2022 decomposition — nonlinearity, regularization, cross-validation scheme, and loss function — should be the primary axes of variation in the pipeline layer. Each is an independent design choice. The `ForecastExperiment` class takes configuration objects for each component and computes forecast accuracy across the full factorial of combinations, replicating the treatment-effect experimental design of the foundational paper.

---

## Section 2: Nonlinearity — Time-Varying Parameters, Not Universal Approximation

The evidence that nonlinearity drives ML gains points toward a specific mechanism: time-varying parameters. Coulombe (2020, published 2024) establishes this connection formally by proving the equivalence between two superficially different model classes.

### The TVP-Ridge Equivalence

The standard state-space time-varying parameter model consists of an observation equation `y_t = X_t β_t + ε_t` and a state equation `β_t = β_{t-1} + u_t`, where `ε_t ~ N(0, σ²_ε)` and `u_t ~ N(0, σ²_u I_K)`. Maximum likelihood estimation of this system is equivalent to solving the fused-ridge optimization problem: minimize the sum of squared residuals subject to a quadratic penalty on successive parameter differences, with penalty parameter `λ = σ²_ε / (σ²_u · 1/K)`.

The equivalence does not end there. Reparametrizing via the summation matrix `C` (lower triangular with ones, so that `β_k = Cθ_k`) transforms the fused-ridge problem into a standard ridge regression in `θ`: minimize `(y − Zθ)'(y − Zθ) + λθ'θ`, where `Z = WC` restructures the design matrix. Prediction uses the dual formula `β̂ = CZ'(ZZ' + λI_T)^{-1}y`, which requires inverting only a `T × T` matrix rather than the `KT × KT` matrix of the primal problem. Computational complexity falls from `O(K³T³)` to `O(KT³)`, enabling estimation of thousands of time-varying parameters simultaneously.

Three implications follow immediately. First, TVP models belong in the regularized regression family; the state-space apparatus is not required. Second, the sole hyperparameter `λ` encodes the entire amount of time variation: large `λ` implies slow-moving parameters, small `λ` allows rapid drift. Third, and most importantly, `λ` can be selected by cross-validation, resolving the "pile-up problem" that plagues likelihood-based variance estimation, where `σ²_u` frequently collapses to zero at the boundary of the parameter space.

Extensions handle heteroskedasticity via a two-step ridge regression (2SRR): estimate homogeneous-variance ridge in the first step, compute time-varying residual volatility via GARCH(1,1), then re-estimate with heterogeneous weight matrices in the second step. Sparsity extensions allow soft-thresholding on which parameters vary. Reduced-rank extensions connect time variation to a factor model structure.

### The Bayesian Counterpart: HKKO2022

Hauzenberger, Huber, Koop, and Onorante (2022) independently derive the same static representation of the TVP model (`y = Zβ + η`, where `Z` is a `T × KT` design matrix), but pursue a Bayesian MCMC route rather than ridge regression and cross-validation. Their computational contribution is to apply the singular value decomposition `Z = UΛV'` to the static design matrix. Because `rank(Z) = T ≪ KT`, the full posterior covariance `V_β̃ = D₀ - D₀V(diag(λ⊙λ)^{-1} + V'D₀V)^{-1}V'D₀` is computed in `T`-dimensional space, achieving the same dimensionality reduction as Coulombe's dual formula but within a MCMC sampler.

The paper introduces three hierarchical priors — Minnesota (differential shrinkage by lag order), g-prior (single hyperparameter `ξ`), and ridge-type — as alternatives to the random walk assumption. The most distinctive contribution is the Sparse Finite Mixture (SFM) prior: `β̃_t ~ Σ_g w_g N(μ_g, σ²_t Ψ)`, which groups time periods into `G ≤ T` clusters. Rather than smooth drift, this prior allows abrupt regime changes and structural breaks to emerge endogenously from the data, closely mimicking Markov-switching and break-point models without pre-specifying the number of regimes.

The contrast with Coulombe's ridge approach is instructive. Both write TVP as high-dimensional static regression and both reduce computational complexity to `O(T)` in `K`. HKKO achieves uncertainty quantification through the posterior distribution but requires MCMC and user-specified priors. Coulombe's 2SRR achieves point estimates and cross-validated tuning without MCMC, at the cost of approximate inference. For macrocast's pipeline layer, Coulombe's frequentist 2SRR is the primary implementation; HKKO's SFM clustering prior is contextually relevant as the Bayesian alternative when regime-change modeling is the primary objective.

### The Macroeconomic Random Forest

Coulombe (2020, published 2024, JAE) develops the Macroeconomic Random Forest (MRF), which extends the TVP interpretation to the random forest estimator. The MRF has two distinct input sets and two estimation stages.

In the first stage, a standard random forest is applied to a set of state variables `S_t` — which may include lagged dependent variables, uncertainty indicators, or financial conditions — that index the state of the economy. The forest generates observation-level kernel weights `w_t(τ)` for each test period `τ`: observations that fall in the same terminal node as the test period receive positive weight proportional to `1/leaf_size`; others receive zero. Averaging across trees produces smooth weights.

In the second stage, the linear regressors `X_t` are used in a weighted ridge regression. For a given test point `x_0`, the kernel weights generated by the forest are `α_t(x_0) = (1/B) Σ_{b=1}^B 1{X_t ∈ L_b(x_0)} / |L_b(x_0)|`, and the GTVPs are estimated by solving `∀t : argmin_{β_t} Σ_τ α_t(s_τ)(Y_τ − X_τ β_τ)² + λ||β_t||₂`. The resulting estimates are the Generalized Time-Varying Parameters (GTVPs). Because each test period produces a different weight vector, the GTVPs are functions of the state rather than of calendar time alone, nesting threshold models, smooth transition autoregressions, and structural break models without pre-specifying the functional form.

MRF introduces a second regularization parameter `ζ < 1` via a "podium kernel": observations `t−1` and `t+1` from a given leaf receive weight `ζ`, observations `t−2` and `t+2` receive `ζ²`, encoding a random-walk smoothness prior on `β_t` (shrinking toward the neighborhood of `β_{t-1}` and `β_{t+1}`, not toward zero). This contrasts with plain ridge, which shrinks toward zero. The standard RF is MRF with `X_t = 1`, `λ = 0`, and `ζ = 0`: the sole regressor is a constant, so the tree fits a piecewise constant function. Feature engineering for `S_t` uses Moving Average Factors (MAFs), which compress the lag polynomial of each predictor into a small number of weighted averages, analogous to PCA across the time dimension rather than the cross-sectional dimension.

The GTVPs are interpretable. Coulombe demonstrates that the MRF detects the 2008 unemployment surge in real time because term spreads and housing starts receive large GTVPs before every recession: the model identifies the recession indicator endogenously. Similarly, the estimated Phillips curve GTVP is cyclical, declining during low-inflation periods and reviving during high-inflation ones, consistent with what theoretical models predict but difficult to identify with standard rolling-window estimation.

The MRF result reinforces the TVP interpretation of ML gains: random forests are not useful because they approximate arbitrary functions; they are useful because they generate proximity weights that allow downstream linear estimation to track parameter drift in a flexible, data-driven way.

### AlbaMA: Adaptive Moving Average via TVP-Ridge

Coulombe and Klieber (2025) apply the TVP-Ridge framework to construct an Adaptive Moving Average (AlbaMA). The procedure runs a random forest on a single predictor — a simple time trend — to generate locally weighted smoothing kernels. Because the tree assigns proximity weights based only on calendar proximity, the resulting forecast is a locally adaptive moving average. The bandwidth (effective window length) is determined data-adaptively via the forest's minimum node size, rather than by the researcher. AlbaMA generalizes classical moving averages in the same way that kernel regression generalizes local constant regression: the window width and shape are estimated from the data rather than fixed ex ante. This application demonstrates that even the simplest TVP-Ridge setup produces practically useful estimators that outperform standard exponential smoothing benchmarks.

### Design Implications for `NonlinearityType`

The `NonlinearityType` enum in `macrocast/pipeline/` should distinguish at minimum: linear (baseline), kernel ridge regression, standard random forest, and MRF. The MRF requires a separate API because it takes two distinct input sets (`X_regressors` and `S_state_variables`). The Hemisphere Neural Network (discussed in Section 7) constitutes a fifth type, relevant for theory-constrained applications.

---

## Section 3: Regularization — The Unifying Concept

Regularization is the second component in the four-part decomposition and, empirically, the most important choice after nonlinearity. The JAE 2022 results indicate that factor model regularization — standard PCA/diffusion index extraction from FRED-MD, implemented as the ARDI model of Stock and Watson (2002b) — outperforms ridge, lasso, and elastic net as standalone regularization strategies in the data-rich environment. In the data-poor environment, plain ridge performs comparably. The practical recommendation from the paper: first reduce dimensionality using principal components, then augment the diffusion index model with a generic ML nonlinear function approximator.

For KRR, the JAE 2022 paper uses specifically the radial basis function (RBF) kernel `K_σ(x, x') = exp(−||x − x'||² / 2σ²)`, where σ is a tuning parameter selected by CV. The KRR dual forecast is `Ê(y_{t+h}|Z_t) = K_σ(Z_t, Z)(K_σ(Z_t, Z) + λI_T)^{-1}y_t`, with the full tuning parameter set `τ = {λ, σ, p_y, p_f}` for the data-rich model.

### Implicit Priors

The connection between regularization and Bayesian priors is well established: ridge corresponds to a Gaussian prior `β ~ N(0, σ²_β I_K)`, lasso corresponds to a Laplace prior, and elastic net interpolates between the two. The TVP-Ridge result adds a third member to this taxonomy: TVP corresponds to a random-walk prior over time, `β_t = β_{t-1} + u_t`, which is a prior on parameter dynamics rather than on parameter magnitudes.

Coulombe, Leroux, Stevanovic, and Surprenant (2021) establish a fourth implicit prior through the Moving Average Rotation of X (MARX) transformation. MARX replaces raw lag values with expanding-window moving averages, imposing a smoothness constraint on the lag polynomial. The implicit prior is `β(p) = β(p-1) + u(p)` — a random walk over the lag index — contrasting with ridge's iid shrinkage across all lags. The practical consequence is that MARX performs better than raw lags at longer lag orders, where moving averages smooth out high-frequency noise. The transformation requires only a few lines of code and is compatible with any ML algorithm.

### Bagging as Regularization

The paper "To Bag Is to Prune" (Coulombe, 2025, SNDE; arXiv 2020) proves that bootstrap aggregation and random feature selection in random forests automatically prune a latent true tree. Randomized ensembles of greedily optimized learners implicitly perform optimal early stopping. Completely overfitting ensembles achieve out-of-sample performance equivalent to tuned counterparts. This establishes bagging as a form of implicit regularization rather than variance reduction through averaging.

The practical consequence for macrocast is that the number of trees in a random forest is not a hyperparameter requiring cross-validation. It should be set to a large fixed default (e.g., 500 or 1000) and excluded from the CV search. The tunable parameters are `mtry` (the feature subset fraction at each split) and `min_node_size` (which controls tree depth and thus the degree of implicit regularization). This simplifies the hyperparameter optimization problem.

### Design Implications for `RegularizationType`

The `RegularizationType` enum should include: ridge, lasso, elastic net, factor model (PCA), and TVP-Ridge (implemented as ridge on the reparametrized `Z = WC` design matrix without a Kalman filter). The MARX transformation is a preprocessing step that interacts with the regularization component and should be configurable independently. Factor augmentation — including PCA-extracted factors alongside other regressors — should be a first-class option.

---

## Section 4: Cross-Validation and Loss Function — The Overlooked Components

The third and fourth components of the decomposition — cross-validation scheme and loss function — receive less attention in the empirical ML literature but are consequential in macroeconomic applications.

### Cross-Validation in Time Series

Standard k-fold cross-validation randomly assigns observations to folds. This is inappropriate for time series because it creates look-ahead contamination: a model trained on post-2008 data evaluated on pre-2008 folds appears to predict the past from the future. The appropriate alternatives are expanding window CV (train on all data up to period t, evaluate on period t+1) and rolling window CV (train on a fixed-length window ending at period t).

The JAE 2022 paper compares four hyperparameter selection strategies: AIC, BIC, pseudo-OOS CV (POOS-CV, which uses the expanding window with one-step-ahead evaluation), and K-fold CV. The result is that K-fold CV and BIC (when applicable) outperform POOS-CV on average. K-fold is preferred because POOS-CV uses only the end of the training sample for evaluation, whereas K-fold exploits the full time series, yielding a more reliable ranking of hyperparameter values. This is the primary tuning default in macrocast.

The expanding versus rolling window choice applies to the pseudo-OOS evaluation period, not the CV inner loop. An expanding window uses all available history back to the start and is the macrocast default. A rolling window discards distant history and is relevant when structural changes make early data uninformative. Both should be supported as `CVScheme` options.

### HPO Strategy

Hyperparameter optimization is the most computationally intensive aspect of the pipeline. The key principles from Coulombe's blog and papers are: (1) the number of trees in random forests is not a tuning parameter and should not be searched; (2) lambda in ridge and TVP-Ridge should be searched on a logarithmic grid; (3) the primary hyperparameters for MRF are `mtry`, `min_node_size`, and the ridge penalty in the WLS step; (4) grid search over a coarse logarithmic grid is usually sufficient for macroeconomic applications where the objective function is smooth in `λ`.

### Loss Function

The JAE 2022 results favor the L2 (squared error) loss over epsilon-insensitive SVR-type losses. L2 is computationally convenient, statistically efficient under Gaussian errors, and consistent with the standard MSFE evaluation criterion for point forecasts. Asymmetric losses are relevant for specific applications (e.g., forecasting tail events or optimizing under asymmetric decision costs) but are not the default.

### Design Implications for `CVScheme` and `LossFunction`

The `CVScheme` enum should include: expanding window, rolling window, and k-fold with temporal ordering. The `LossFunction` enum should include: L2, L1 (absolute error, robust to outliers), and potentially asymmetric variants. Both enums should have sensible defaults (expanding window, L2) that replicate the best-performing configuration in the JAE 2022 results.

---

## Section 5: The Observation-Weight (Dual) Interpretation

The fourth intellectual pillar of Coulombe's research program is the primal-dual unification of ML forecasting methods, developed fully in Coulombe, Gobel, and Klieber (2024) and Coulombe (2025).

### The Portfolio Representation

Any out-of-sample ML forecast can be written as `ŷ_j = w_j · y`, where `w_j` is an N-vector of data portfolio weights and `y` is the vector of in-sample outcomes. The weights `w_j` reflect pairwise proximity between the forecast period j and each training observation in the model's feature space. This "dual" representation is equivalent to the standard "primal" representation in terms of estimated coefficients; neither is more fundamental.

For ridge regression, the dual weights are `w_j = K_j(K + λI_N)^{-1}`, where `K` is the `N × N` Gram matrix of training features (`K_{i,i'} = x_i'x_{i'}`) and `K_j` is the vector of inner products between the test observation and each training observation. For kernel ridge regression, the same formula applies with a nonlinear kernel function. For neural networks with a linear output layer, the weights are extracted by treating the penultimate layer's activations as features and solving the same ridge dual formula, achieving greater than 99% replication accuracy. For random forests and boosting, the weights are leaf co-membership counts divided by leaf size, averaged over trees; these are inherently non-negative because observations that share a terminal node with the forecast period receive positive weight, while all others receive zero.

The non-negativity constraint on tree-based weights has an economic interpretation: tree ensembles cannot extrapolate beyond the training data. Every forecast is a convex combination of historical outcomes. Linear models do not have this constraint and can produce negative weights (equivalent to short-selling in the portfolio analogy), allowing extrapolation beyond observed history.

### The Attention Mechanism Connection

Coulombe (2025, arXiv) establishes the same result from a different starting point. OLS out-of-sample predictions can be rewritten as `ŷ_test = F_test F_train' y_train`, where `F` are orthonormalized feature encodings derived from minimizing squared prediction error in an optimal embedding space. This is structurally identical to an attention module where the test observation is the query, the training observations are the keys, and the outcomes are the values. Ridge regression deviates from orthonormality through shrinkage, implementing a bias-variance tradeoff that is interpretable through the geometry of the embedding space. Adding a softmax nonlinearity restricts weights to a probability simplex, preventing negative attention and paralleling the tree-ensemble non-negativity constraint.

### Portfolio Diagnostics

The portfolio analogy generates a set of model-agnostic diagnostics applicable to any estimator. Forecast concentration measures how many historical observations substantially drive a given prediction; a forecast dominated by a single historical episode (e.g., 1970s stagflation) is less reliable than one drawing on many episodes. Short positions (negative weights) indicate extrapolation beyond historical data; they are possible for linear models but impossible for tree ensembles. Leverage measures the distance between the forecast period and the training data in feature space; high leverage signals out-of-distribution prediction. Turnover measures how rapidly weights change as new data arrive; high turnover may indicate instability.

Coulombe (2025, ECB WP) applies these diagnostics to local projections and finds that fiscal multiplier estimates are dominated by World War II, monetary policy estimates by 1970s stagflation, and climate responses by the Mount Agung eruption. The external validity implications are serious: many canonical empirical results in macroeconomics rest on a handful of historical episodes, a fact that standard regression output conceals.

### Design Implications for macrocast

Every `MacrocastEstimator` in the pipeline should expose a `.get_proximity_weights(X_test)` method returning the N-vector `w_j` for a given test observation. For ridge and kernel ridge, this follows from the closed-form dual formula. For random forests and MRF, weights are leaf co-membership counts normalized over trees. For neural networks, weights are extracted from the penultimate layer via an auxiliary ridge solve. The evaluation layer should compute portfolio diagnostics — concentration, leverage, short positions, turnover — as standard outputs alongside MSFE and Diebold-Mariano statistics.

---

## Section 6: Evaluation Philosophy — Skepticism and Statistical Rigor

A consistent thread running through Coulombe's research program is skepticism toward naive out-of-sample evaluation. Several specific pitfalls recur.

### In-Sample Fit vs. Out-of-Sample Accuracy

Random forests with default settings achieve near-perfect in-sample fit on macroeconomic data: in-sample R-squared values of 0.99 are common. The corresponding out-of-sample R-squared may be negative. This gap is toxic for interpretation: feature importance measures (SHAP values, permutation importance) computed from an overfit model analyze noise rather than signal. The model confidence set (MCS) of Hansen, Lunde, and Nason (2011) and Diebold-Mariano tests should always be computed on genuine out-of-sample periods, never on training data.

The remedy proposed in Coulombe's blog is to calibrate regularization (minimum node size for random forests, `λ` for ridge) until out-of-bag or out-of-sample R-squared begins declining, aligning in-sample and out-of-sample fit. This is equivalent to choosing the regularization level by cross-validation, consistent with the four-component framework.

### Predictor Importance: In-Sample vs. Out-of-Sample

Coulombe, Borup, Rapach, Schütte, and Schwenk-Nebbe (2022) propose the Performance-Based Shapley Value (PBSV) as an out-of-sample counterpart to SHAP. The PBSV decomposes out-of-sample loss into exact predictor contributions that sum to the total loss, are model-agnostic, and are loss-agnostic. An empirical application to US inflation forecasting finds significant disagreements between in-sample SHAP importance and out-of-sample PBSV importance, directly cautioning against using in-sample feature importance as a guide to out-of-sample variable relevance.

### The Satisficing Framework

A "satisficing" ML forecast (Coulombe, blog 2023) is one that passes a minimal set of diagnostic tests rather than maximizing performance on a single metric. The tests are: out-of-sample accuracy competitive with a well-specified AR benchmark; proximity weights that are not excessively concentrated; regularization calibrated to match in-sample and out-of-sample fit; and no reliance on look-ahead bias in the CV scheme. A model that satisfices is more reliable for structural analysis than one that maximizes IS R-squared.

### The Model Confidence Set

The MCS of Hansen, Lunde, and Nason (2011) provides a rigorous framework for selecting the set of models that cannot be distinguished from the best model at a given confidence level. Unlike Diebold-Mariano tests that compare two models, the MCS operates on a collection of models and identifies the subset of superior models. The MCS is the appropriate evaluation tool for the full factorial of component configurations in the `ForecastExperiment` class: rather than declaring one configuration the winner, the MCS identifies the set of configurations that are statistically indistinguishable.

### Structural Instability in Forecast Performance

Giacomini and Rossi (2010) show that the relative forecasting performance of two models may itself be time-varying, rendering global tests of equal predictive accuracy misleading. Their Fluctuation test plots the path of a local Diebold-Mariano statistic over the out-of-sample period and provides critical values for detecting whether one model dominates another at specific points in time. This is directly relevant to macrocast: because ML gains concentrate during high-uncertainty regimes (Section 1), global MSFE comparisons that average over calm and turbulent periods will understate ML gains during recessions and overstate them during expansions. The Fluctuation test should be a default output of the evaluation layer.

For nested model comparisons — where the linear benchmark AR(p) is a restricted version of a larger ML model — Clark and West (2007) establish that the standard Diebold-Mariano statistic has a non-standard limiting distribution under the null of equal predictive accuracy. An adjusted statistic (MSPE-adjusted) is required for valid inference. Giacomini and White (2006) extend equal predictive ability testing to conditional evaluations: rather than asking which model is better on average, their test asks whether there exist conditioning variables that predict which model will be superior in the next period. Both adjustments should be implemented in Layer 3.

### Design Implications for Layer 3

The evaluation layer should implement: relative MSFE (model MSFE / AR MSFE), Diebold-Mariano tests with Newey-West standard errors, the Clark-West MSPE-adjusted statistic for nested model comparisons, the Giacomini-Rossi Fluctuation test for time-varying relative performance, the MCS with block bootstrap, PBSV for predictor contribution, proximity weight portfolio diagnostics, and regime-conditional evaluation (accuracy during NBER recessions vs. expansions, high vs. low VIX periods). The AR(p) with BIC-selected lag order is the canonical benchmark.

---

## Section 7: Bridging ML and Structural Macroeconomics

Coulombe's research program is unusual in the ML-for-macro literature because it maintains a connection to economic interpretation at each step. Three specific bridges between ML and structural macroeconomics are relevant for macrocast.

### Theory-Constrained Neural Networks

The Hemisphere Neural Network (HNN), introduced in Coulombe (2022, arXiv; published 2025, JBES), partitions network inputs into groups corresponding to economic concepts — long-run inflation expectations, short-run expectations, a real activity index, and commodity prices — processed in separate sub-networks before combining in a final additive linear layer. The output is the sum of hemisphere contributions, each interpretable as a latent economic state.

The HNN architecture embeds theoretical structure as an architectural constraint. The additively separable final layer acts as a strong regularizer in macroeconomic settings where sample sizes are small (T ≈ 200-400 quarters). Theory is not a prior on coefficients but a constraint on functional form, a fundamentally different approach to incorporating economic structure. The HNN correctly anticipated the 2021 inflation surge by identifying a large positive deep output gap beginning in late 2020, whereas standard models failed because they could not model the COVID-era combination of supply constraints and demand stimulus.

### Regime-Dependent Forecasting via GTVPs

The MRF's GTVPs provide a natural framework for regime-dependent analysis. Because GTVPs are functions of the state variables `S_t`, the model can identify periods where a particular relationship (e.g., the Phillips curve, the term structure-recession nexus) is especially strong or weak. This is more flexible than Markov-switching models — which require pre-specifying the number of regimes as in Hamilton's (1989) seminal Markov-switching framework — or smooth transition models, which require pre-specifying the transition function. The GTVP approach nests all of these: Markov switching is a special case where `S_t` is a discrete state indicator, smooth transition is a case where the kernel function is a logistic sigmoid, and structural breaks are cases where the kernel has a single sharp discontinuity. GTVPs infer the functional form from data rather than imposing it.

The regime indicator approach used in macrocast's evaluation layer — conditioning on NBER recession indicators, VIX, financial conditions indices — is complementary to GTVP-based analysis. GTVPs estimate the functional relationship between regime and model behavior endogenously; regime indicators evaluate whether forecast accuracy differs across exogenously classified states. Both perspectives are informative.

### Assemblage Regression and Forward-Looking Targets

Coulombe et al. (2024) introduce Assemblage Regression as a method for constructing maximally forward-looking core inflation. The target variable is defined not by excluding volatile components ex ante (as in traditional core CPI) but by solving a constrained optimization: find the nonnegative weights on disaggregated price components that produce the forecast-maximally informative linear combination. Technically, this is a nonnegative ridge regression where the response variable is future headline inflation and the predictors are current price subcomponents. Nonnegativity ensures interpretability — the resulting index is a weighted average of price categories — and ridge regularization prevents overfitting to the small number of components. The resulting "Assemblage" index outperforms standard core measures as a leading indicator of future inflation. For macrocast, this illustrates how regularized regression with economic constraints can construct theory-consistent aggregate indicators, complementing the standard practice of forecasting headline series directly.

### Iterated vs. Direct Forecasting

A subtle methodological point from Coulombe's blog ("Careful With That Axe," 2021) is that iterated multi-step forecasts from nonlinear models must be obtained by simulation, not by direct recursion. For a nonlinear model `f`, the h-step expectation `E[y_{t+h} | y_t]` is not equal to `f(f(...f(y_t)...))` because the expectation of a nonlinear function differs from the nonlinear function applied to the expectation. The correct procedure iterates the model forward with randomized residuals and averages over many simulations.

For macrocast's pipeline, this implies that the "iterated" forecasting mode must use a simulation engine, not analytical recursion. The "direct" mode — which trains a separate model for each horizon h — avoids this issue entirely and is the computationally simpler default.

---

## Appendix A: Bibliography

| Key | Citation |
|-----|----------|
| CLSS2022 | Goulet Coulombe, P., Leroux, M., Stevanovic, D., and Surprenant, S. "How Is Machine Learning Useful for Macroeconomic Forecasting?" *Journal of Applied Econometrics*, 37(5), 2022, pp. 920–964. DOI: 10.1002/jae.2910. |
| C2024tvp | Goulet Coulombe, P. "Time-Varying Parameters as Ridge Regressions." *International Journal of Forecasting*, 41(3), 2025, pp. 982–1002. DOI: 10.1016/j.ijforecast.2024.08.006. |
| C2024mrf | Goulet Coulombe, P. "The Macroeconomy as a Random Forest." *Journal of Applied Econometrics*, 39(3), 2024, pp. 401–421. DOI: 10.1002/jae.3030. arXiv:2006.12724. |
| CGK2024 | Goulet Coulombe, P., Gobel, M., and Klieber, K. "Dual Interpretation of Machine Learning Forecasts." arXiv:2412.13076, December 2024. |
| C2025ols | Goulet Coulombe, P. "Ordinary Least Squares as an Attention Mechanism." arXiv:2504.09663, April 2025. |
| C2025hnn | Goulet Coulombe, P. "A Neural Phillips Curve and a Deep Output Gap." *Journal of Business and Economic Statistics*, 43(3), 2025. arXiv:2202.04146. |
| CLSS2021 | Goulet Coulombe, P., Leroux, M., Stevanovic, D., and Surprenant, S. "Macroeconomic Data Transformations Matter." *International Journal of Forecasting*, 37(4), 2021, pp. 1338–1354. DOI: 10.1016/j.ijforecast.2021.05.005. arXiv:2008.01714. |
| C2025bag | Goulet Coulombe, P. "To Bag Is to Prune." *Studies in Nonlinear Dynamics and Econometrics*, 29(6), 2025, pp. 669–697. DOI: 10.1515/snde-2023-0030. arXiv:2008.07063. |
| CMS2021 | Goulet Coulombe, P., Marcellino, M., and Stevanovic, D. "Can Machine Learning Catch the COVID-19 Recession?" *National Institute Economic Review*, 256, 2021, pp. 1–10. DOI: 10.1017/nie.2021.12. |
| CBRSS2022 | Goulet Coulombe, P., Borup, D., Rapach, D., Schütte, E.C.M., and Schwenk-Nebbe, S. "The Anatomy of Out-of-Sample Forecasting Accuracy." Federal Reserve Bank of Atlanta Working Paper 2022-16, November 2022. |
| CGK2025 | Goulet Coulombe, P., and Klieber, K. "Opening the Black Box of Local Projections." ECB Working Paper 3105, May 2025. arXiv:2505.12422. |
| HLN2011 | Hansen, P.R., Lunde, A., and Nason, J.M. "The Model Confidence Set." *Econometrica*, 79(2), 2011, pp. 453–497. |
| MN2016 | McCracken, M.W., and Ng, S. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business and Economic Statistics*, 34(4), 2016, pp. 574–589. |
| CW2007 | Clark, T.E., and West, K.D. "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models." *Journal of Econometrics*, 138(1), 2007, pp. 291–311. DOI: 10.1016/j.jeconom.2006.05.023. |
| GW2006 | Giacomini, R., and White, H. "Tests of Conditional Predictive Ability." *Econometrica*, 74(6), 2006, pp. 1545–1578. DOI: 10.1111/j.1468-0262.2006.00718.x. |
| GR2010 | Giacomini, R., and Rossi, B. "Forecast Comparisons in Unstable Environments." *Journal of Applied Econometrics*, 25(4), 2010, pp. 595–620. DOI: 10.1002/jae.1177. |
| H1989 | Hamilton, J.D. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 1989, pp. 357–384. |
| CK2025 | Goulet Coulombe, P., and Klieber, K. "AlbaMA: Adaptive Moving Average via Machine Learning." arXiv:2501.13222, January 2025. |
| C2024mfl | Goulet Coulombe, P., et al. "Maximally Forward-Looking Core Inflation." Working Paper, 2024. |
| CRMS2026 | Goulet Coulombe, P., Rapach, D., Montes Schütte, E.C., and Schwenk-Nebbe, S. "The Anatomy of Machine Learning-Based Portfolio Performance." SSRN 4628462, February 2026. |
| HKKO2022 | Hauzenberger, N., Huber, F., Koop, G., and Onorante, L. "Fast and Flexible Bayesian Inference in Time-varying Parameter Regression Models." *Journal of Business and Economic Statistics*, 40(4), 2022, pp. 1904–1918. DOI: 10.1080/07350015.2021.1990772. |

---

## Appendix B: Blog Post Index

The following blog posts at `https://philippegouletcoulombe.com/blog` provide accessible expositions of the ideas documented above.

| Post Title | Approximate Date | Corresponding Paper(s) |
|------------|-----------------|------------------------|
| Time-Varying Parameters as Ridge Regressions | 2021-12 | C2024tvp |
| Random Forests for Time-Varying Parameters | 2022-02 | C2024mrf |
| The Macroeconomy as a Random Forest | 2022-07 | C2024mrf |
| On the Link between Machine Learning & Macroeconomic Forecasting | 2023-07 | CLSS2022 |
| How to Make a Satisficing ML Forecast | 2023-04 | CLSS2022, CBRSS2022 |
| Neural Networks/Random Forests are Observation-Weighted Regressions | 2024-01 | CGK2024 |
| On Finding the Holy Grail of ML for Macro | 2024-06 | CLSS2022, C2024mrf |
| Out-of-Sample Prediction in the Age of AI | 2024-12 | CGK2024, C2025ols |
| Don't Let Your Model Overfit Your Intuitions | 2024-09 | CBRSS2022 |
| On Overfitting | 2023-10 | C2025bag |
| How to Put Prior Beliefs into Regression-Based Predictions | 2023-09 | C2024tvp, CLSS2021 |
| On Macro Forecasting with Random Forests | 2023-02 | C2024mrf |
| The Number of Trees in Random Forest Is Not a Tuning Parameter | 2022-04 | C2025bag |
| Careful With That Axe: ML, Recursive Forecasting, and IRFs | 2021-09 | C2024mrf |
| Econometricians Should Know About Double Descent | 2021-08 | General ML theory |

---

## Appendix C: Design Decision Mapping

The following table maps each major architectural decision in `macrocast/pipeline/` to its evidential basis.

| macrocast Design Decision | Basis | Core Argument |
|--------------------------|-------|---------------|
| `NonlinearityType` as first-class enum | CLSS2022 (JAE) | Nonlinearity is the dominant driver of ML forecast gains; it should be the primary axis of variation |
| `RegularizationType` includes factor model (PCA) | CLSS2022 (JAE) | Factor model regularization outperforms ridge and lasso as standalone strategies |
| TVP-Ridge implemented as ridge on `Z = WC`, no Kalman filter | C2024tvp (IJF) | State-space TVP is equivalent to ridge on reparametrized design matrix; CV replaces likelihood estimation |
| MRF requires separate `X_regressors` and `S_state_variables` inputs | C2024mrf (JAE) | MRF has two distinct input sets by construction; forest acts on state space, WLS acts on regressor space |
| Number of trees in RF is fixed, not tuned | C2025bag (SNDE) | Bagging is implicit regularization; adding trees beyond a threshold does not change model complexity |
| `CVScheme` defaults to expanding window | CLSS2022 (JAE) | Expanding window performs best on average in FRED-MD experiments |
| `LossFunction` defaults to L2 | CLSS2022 (JAE) | L2 loss preferred over SVR-type alternatives in systematic comparison |
| `MacrocastEstimator` exposes `.get_proximity_weights()` | CGK2024, C2025ols | All estimators admit dual weight representation; portfolio diagnostics require it |
| Benchmark is AR(p) with BIC lag selection | Standard; implicit in CLSS2022 | AR(p) is the canonical forecasting benchmark in macroeconomics; relative MSFE is the primary evaluation metric |
| MCS (Hansen et al. 2011) for model comparison | HLN2011; CLSS2022 | MCS identifies the set of models indistinguishable from the best; appropriate for full factorial experiments |
| PBSV for predictor contribution | CBRSS2022 | In-sample SHAP and out-of-sample PBSV disagree; PBSV is the correct out-of-sample measure |
| Iterated forecasting uses simulation, not recursion | C2024mrf (blog companion) | Expectation of nonlinear function ≠ nonlinear function of expectation; simulation is required for correctness |
| Direct forecasting as default (trains separate model per horizon) | Standard; C2024mrf | Computationally simpler and avoids the iterated-forecasting simulation requirement |
| Regime-conditional evaluation (NBER, VIX) | CLSS2022 (JAE) | ML nonlinearity gains concentrate during high-uncertainty regimes; unconditional MSFE obscures regime-specific performance |
| Clark-West MSPE-adjusted test for AR vs. ML | CW2007 (JE) | DM statistic has non-standard distribution under null when comparing nested models; adjusted statistic required |
| Fluctuation test for time-varying relative performance | GR2010 (JAE) | Global MSFE averages over calm and turbulent periods; local performance path reveals when ML gains are significant |
| GW conditional EPA test as optional diagnostic | GW2006 (ECTA) | Tests whether conditioning variables predict which model will be superior next period; complements unconditional evaluation |
| Markov-switching benchmark in `NonlinearityType` | H1989 (ECTA) | GTVPs nest Markov switching; including MS-AR as a benchmark tests whether the additional flexibility of GTVPs is warranted |
| AlbaMA as `NonlinearityType.adaptive_ma` option | CK2025 | RF on time-trend-only predictor produces data-adaptive moving average; bandwidth estimated by min_node_size CV |
| Nonnegative ridge as constrained `RegularizationType` | C2024mfl | Assemblage Regression requires nonnegativity + ridge to produce interpretable, forward-looking aggregate indicators |
| SFM clustering prior as Bayesian TVP option (not default) | HKKO2022 | MCMC + sparse finite mixture allows abrupt regime changes; complementary to Coulombe's frequentist 2SRR |
