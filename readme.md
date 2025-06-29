# Credit Risk Scoring Project

This repository will contain my work on developing a credit-risk scoring model in alignment with the Basel II framework. Below, I will present the business-understanding section that will motivate my modeling choices and regulatory considerations.

## Credit Scoring Business Understanding

### 1. Basel II’s emphasis on risk measurement and the need for interpretability  
Basel II will require banks to quantify credit-risk exposures (Probability of Default, Loss Given Default, Exposure at Default) and hold capital accordingly. Under the Internal Ratings-Based (IRB) approaches, regulators will need to review, validate, and challenge model outputs—so every assumption, parameter choice, and transformation will have to be clear. This will drive me to choose models that are inherently interpretable (or can be decomposed into human-readable pieces), accompanied by full documentation of data sources, feature-engineering logic (e.g. WoE binning), and performance diagnostics.

### 2. Necessity and risks of proxy default labels  
Because “actual default” events will occur infrequently and will not be directly observed in most historical data, I will have to construct a proxy target—commonly arrears beyond 90 days, charge-off status, or bankruptcy filings. While this will enable me to train a classifier, it will introduce **model risk**: if the proxy behavior diverges from true default (e.g. successful restructurings or write-offs), my predictions could misestimate risk, leading to under- or over-capitalization, poor credit decisions, and potential regulatory censure.

### 3. Trade-offs: simple interpretable vs. complex high-performance models  
- **Logistic Regression + WoE**  
  + Will be fully transparent: each coefficient will correspond to a well-understood predictor’s weight of evidence.  
  + Will be easier to validate and document; will deploy faster in production and explain to auditors.  
  – May underperform if relationships become highly non-linear or involve complex feature interactions.  
- **Gradient Boosting Machines (GBMs)**  
  + Will often deliver higher predictive accuracy by capturing non-linearities and feature interactions automatically.  
  – Will behave as a “black box,” complicating explanation; will require surrogate modeling (e.g. SHAP) and additional validation steps.  
  – Will involve longer development and review cycles under strict governance—potentially slowing time-to-market.

In a regulated context, I will balance regulatory acceptability and operational efficiency (favoring simpler models) against the potential uplift in risk differentiation and profitability (favoring complex models).
