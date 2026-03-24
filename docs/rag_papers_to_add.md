# RAG Papers To Add (Coulombe KB)

Papers confirmed NOT indexed in the Coulombe KB RAG as of 2026-03-23.
All use FRED-MD or FRED-QD as sole data source and are replicable with macrocast.

---

## Grade A — Immediately replicable with macrocast

### 1. Medeiros, Vasconcelos, Veiga & Zilberman (2021)
- **Title:** Forecasting Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods
- **Journal:** Journal of Business & Economic Statistics
- **Volume/Pages:** 39(1):98-119
- **Year:** 2021
- **DOI:** 10.1080/07350015.2019.1637745
- **Data:** FRED-MD (130 monthly series)
- **Targets:** CPI, PCE, PPI (US inflation)
- **OOS period:** 1990-2017, h=1,3,6,12
- **Models:** Random Forest, Boosted Trees, Elastic Net, Ridge, LASSO, AR benchmark
- **macrocast models:** RFModel, GBModel, ElasticNetModel, LassoModel, RidgeModel
- **Notes:** Canonical "RF beats AR for US inflation" paper. Directly extended by Naghi et al. (2024).
  Cited in every major paper in the RAG but NOT indexed as a standalone document.

---

### 2. Naghi, O'Neill & Zaharieva (2024)
- **Title:** The Benefits of Forecasting Inflation with Machine Learning: New Evidence
- **Journal:** Journal of Applied Econometrics
- **Volume/Pages:** 39(7):1321-1331
- **Year:** 2024
- **DOI:** 10.1002/jae.3088
- **PDF:** https://pure.eur.nl/ws/files/162496227/J_of_Applied_Econometrics_-_2024_-_Naghi_-_The_benefits_of_forecasting_inflation_with_machine_learning_New_evidence.pdf
- **Data:** FRED-MD (US), plus Canada and UK CPI
- **Targets:** CPI inflation (US, CA, UK)
- **OOS period:** 1990-2022 (extended through Oct 2022 vs Medeiros 2021)
- **Models:** 30+ methods: RF variants, Gradient Boosted Trees, BART, SVM, Neural Nets, penalized regressions
- **macrocast models:** RFModel, GBModel, KRRModel, SVRModel, ElasticNetModel, LassoModel
- **Notes:** Direct replication + extension of Medeiros et al. (2021). Finds RF not uniquely dominant;
  other methods (GBT, EN) competitive. Adds coverage rates and prediction intervals.

---

### 3. Bae (2024)
- **Title:** Factor-Augmented Forecasting in Big Data
- **Journal:** International Journal of Forecasting
- **Volume/Pages:** 40(4):1660-1688
- **Year:** 2024
- **DOI:** 10.1016/j.ijforecast.2024.01.002
- **PDF:** https://eprints.gla.ac.uk/336044/1/336044.pdf
- **Data:** FRED-MD (130 monthly series)
- **Targets:** Multiple macro variables (broad OOS experiment)
- **Models:** PLS, PCA, IPCA, POET, SEM and 2 others; 13 rules for number of factors
- **macrocast models:** FeatureSpec(use_factors=True) — extends factor method comparison
- **Notes:** Key finding: 1-PLS factor dominates PCA across most targets and horizons.
  Directly relevant for FeatureBuilder factor extraction design choices.

---

### 4. Chu & Qureshi (2023)
- **Title:** Comparing Out-of-Sample Performance of Machine Learning Methods to Forecast U.S. GDP Growth
- **Journal:** Computational Economics
- **Volume/Pages:** 62(4):1567-1609
- **Year:** 2023
- **DOI:** 10.1007/s10614-022-10312-z
- **Data:** FRED-QD (224 quarterly predictors)
- **Targets:** Real GDP growth (quarterly)
- **OOS period:** 2000Q1-2019Q4
- **Models:** RF, Gradient Boosting, EN, Ridge, LASSO, PCR, AR; walk-forward CV
- **macrocast models:** load_fred_qd() + RFModel, GBModel, ElasticNetModel, LassoModel
- **Notes:** FRED-QD companion to Medeiros et al. (2021). Uses large quarterly predictor set (224 series).
  Useful benchmark for quarterly forecasting experiment.

---

## Grade B — FRED-based, partially replicable

### 5. Hauzenberger, Huber & Klieber (2023)
- **Title:** Real-Time Inflation Forecasting Using Non-Linear Dimension Reduction Techniques
- **Journal:** International Journal of Forecasting
- **Volume/Pages:** 39(2):901-921
- **Year:** 2023
- **DOI:** 10.1016/j.ijforecast.2022.08.004
- **PDF:** https://strathprints.strath.ac.uk/86919/1/Hauzenberger_etal_IJF_2023_Real_time_inflation_forecasting_using_non_linear_dimension_reduction_techniques.pdf
- **Data:** FRED-MD real-time vintages (McCracken vintage files)
- **Targets:** US CPI inflation (monthly)
- **OOS period:** 1990-2019, h=1,3
- **Models:** Autoencoder, squared PCs, kernel PCA; combined with TVP-regression + shrinkage
- **macrocast replicable:** load_vintage_panel() + apply_pca() (linear PCA part); autoencoder needs PyTorch
- **Notes:** Uses real-time vintages from McCracken's archive. Demonstrates squared-PC transformation
  as simple nonlinear feature engineering competitive with autoencoders.

---

### 6. Eraslan & Schroeder (2023)
- **Title:** Nowcasting GDP with a Pool of Factor Models and a Fast Estimation Algorithm
- **Journal:** International Journal of Forecasting
- **Volume/Pages:** 39(3):1460-1476
- **Year:** 2023
- **DOI:** 10.1016/j.ijforecast.2022.06.009
- **Data:** FRED-MD (monthly), mixed-frequency for GDP nowcasting
- **Targets:** US GDP QoQ growth (nowcast)
- **Models:** Time-varying DFM pool, stochastic volatility, Kalman filter, model averaging
- **macrocast replicable:** Static DFM/PCA factor part; TVP/SV component needs macrocastR
- **Notes:** Uses FRED-MD monthly indicators to nowcast quarterly GDP. SV improves point forecast
  accuracy, especially during COVID. Relevant for vintage + mixed-frequency design.

---

### 7. Gruber & Kastner (2025)
- **Title:** Forecasting Macroeconomic Data with Bayesian VARs: Sparse or Dense? It Depends!
- **Journal:** International Journal of Forecasting
- **Volume/Pages:** 41(4):1589-1619
- **Year:** 2025
- **DOI:** 10.1016/j.ijforecast.2025.01.001
- **arXiv:** https://arxiv.org/abs/2206.04902
- **Data:** FRED-MD and FRED-QD
- **Targets:** US macroeconomic panel (point + density forecasts)
- **Models:** Bayesian VARs with semi-global shrinkage priors (sparse vs. dense)
- **macrocast replicable:** Needs macrocastR BVAR extension; not Python ML pipeline
- **Notes:** Settles the "illusion of sparsity" debate. Sparse vs. dense prior performance varies
  by variable and time period. Relevant for macrocastR model grid extension.

---

## Previously identified (pre-2022, also not in RAG)

These are older papers heavily cited throughout the RAG but not indexed as standalone documents.
Lower priority for RAG addition but useful for completeness.

### 8. Smeekes & Wijler (2018)
- **Title:** Macroeconomic Forecasting Using Penalized Regression Methods
- **Journal:** International Journal of Forecasting
- **Volume/Pages:** 34(3):408-430
- **Year:** 2018
- **DOI:** 10.1016/j.ijforecast.2018.01.001
- **Data:** FRED-QD (quarterly)
- **Targets:** GDP, inflation, unemployment
- **Models:** Ridge, LASSO, EN, adaptive LASSO, group LASSO vs. DFM benchmark
- **macrocast models:** RidgeModel, LassoModel, ElasticNetModel (all in macrocastR)
- **Notes:** Canonical penalized regression comparison on FRED-QD. Cited in CBRSS2022.

### 9. Stock & Watson (2002b)
- **Title:** Macroeconomic Forecasting Using Diffusion Indexes
- **Journal:** Journal of Business & Economic Statistics
- **Volume/Pages:** 20(2):147-162
- **Year:** 2002
- **Data:** Large monthly US macro panel (precursor to FRED-MD)
- **Models:** Diffusion indexes (PCA factors) + AR lags
- **macrocast models:** FeatureSpec(use_factors=True, use_ar=True)
- **Notes:** Foundational factor forecasting paper. Defines the ARDI/DI forecasting framework
  implemented in macrocast's FeatureBuilder.

### 10. Bai & Ng (2009)
- **Title:** Boosting Diffusion Indices
- **Journal:** Journal of Applied Econometrics
- **Volume/Pages:** 24(4):607-629
- **Year:** 2009
- **Data:** Monthly US macro panel
- **Models:** Boosting applied to factor-augmented regression (ARDI with boosting)
- **macrocast models:** BoogingModel (L2Boosting already in macrocastR)
- **Notes:** Combines boosting with diffusion indexes. Implements boosting stopping rule
  relevant to BoogingModel tuning in macrocastR.

---

## Summary table

| # | Authors | Year | Journal | Grade | macrocast models |
|---|---------|------|---------|-------|-----------------|
| 1 | Medeiros et al. | 2021 | JBES | A | RF, GB, EN, LASSO, Ridge |
| 2 | Naghi et al. | 2024 | JAE | A | RF, GB, KRR, SVR, EN, LASSO |
| 3 | Bae | 2024 | IJF | A | FeatureSpec(factors) |
| 4 | Chu & Qureshi | 2023 | Comp.Econ | A | RF, GB, EN (FRED-QD) |
| 5 | Hauzenberger et al. | 2023 | IJF | B | vintage_panel + PCA |
| 6 | Eraslan & Schroeder | 2023 | IJF | B | DFM factor part |
| 7 | Gruber & Kastner | 2025 | IJF | B | macrocastR BVAR |
| 8 | Smeekes & Wijler | 2018 | IJF | A | macrocastR penalized |
| 9 | Stock & Watson | 2002b | JBES | A | FeatureSpec(factors) |
| 10 | Bai & Ng | 2009 | JAE | A | BoogingModel |
