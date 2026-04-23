# Real Estate Housing Prices Analysis Report

## 1. Introduction
This project involved the comprehensive analysis of a synthetic real estate dataset containing 10,000 properties, utilizing various regression architectures developed from scratch. The primary goal was to predict housing prices while overcoming standard data irregularities, including 5% missing values (Missing Completely at Random) and 2% outlier contamination. We approached the prediction task using three distinct modelling paradigms: Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression.

## 2. Preprocessing Strategies
Prior to prediction modelling, robust data cleaning and scaling pipelines were implemented without external library dependencies aside from NumPy and Pandas. Missing values were addressed using central tendency imputation mechanisms: mean imputation for continuous variables with low skew, and median imputation for discrete or skewed metrics (like bedrooms and age). We subsequently scaled our inputs using standardization (Z-score normalization) to uniformly constrain the feature distributions, eliminating numerical instability during optimization. 

Extremal price points representing 2% of the distribution were processed via the Interquartile Range (IQR) technique. Comparing results, executing IQR stripping drastically improved our Simple Linear Regression's coefficient of determination ($R^2$) by smoothing residual variance that would otherwise heavily bias unregularized objective functions.

## 3. Comparative Modelling Analysis

### Simple Linear Regression
Utilizing `area` as the singular explanatory feature, we trained a gradient descent optimizer. The unifactorial model yielded an $R^2$ of approximately 0.40, revealing that living area alone explains 40% of the price variance. A $95\%$ confidence band was established across predictions. The residual plots highlighted clear heteroscedasticity—indicating that as property size grew, error margins widened. Thus, assuming constant variance (homoscedasticity) inherently fractures when modeling complex housing paradigms strictly linearly.

### Multiple Linear Regression
To capture more variance, we expanded to a multifactor equation solved using the Normal Equation, supported by $L2$ (Ridge) Regularization. This stabilized iterations against multicollinearity heavily present between variables like `bedrooms` and `bathrooms` (where our VIF algorithm yielded inflation scores exceeding 5.0). Using Stepwise Forward Selection guided by analytical $p$-value metrics, we successfully truncated redundant signals. The multivariate baseline enhanced the $R^2$ metric substantially to ~$0.90$, a staggering improvement capturing interactions spanning location (`distance_city`) to amenities (`garage`, `basement`). 

### Polynomial Regression
By mapping the retained top-tier features into quadratic and cubic spaces ($Degree \le 3$), Polynomial interpolation isolated non-linear elasticity—especially beneficial given the underlying quadratic `area` synergy generated in our baseline. While Degree 2 captured intrinsic interactions peaking at a comparable cross-validated pseudo-$R^2$, extending beyond Degree 3 forced extreme overfitting spikes mapped on our learning curves, a classic illustration of the Bias-Variance tradeoff where training precision rapidly diverged from validation logic.

## 4. Final Ensemble and Business Interpretations
Our final architecture leverages Blending Regression stacked representations of Polynomial features ($Degree\ 1$ and $Degree\ 2$) to optimize stability. For stakeholders and business executives:
- **Major Value Driver**: Every square foot increment scales exponentially in localized sectors; marketing larger assets yields non-linear returns.
- **Urban Decline**: Proximity distances drastically diminish appraisal value; investments naturally favor immediate radius mapping.
- **Aesthetic Assumptions**: Standard linear interpretations failed to encapsulate nuanced depreciation. Our approach verifies that although properties deprecate linearly by age structurally, interaction with proximity compounds value loss—old homes strictly distant from hubs perform far worse statistically. 

## 5. Limitations
Our algorithms assumed linear combinations and unmeasured features holding constant (Ceteris Paribus). Additionally, implementing regularization directly on analytical solutions may still skew true intrinsic beta weights if distribution skewness exceeds simple Z-score adjustments. Future iterations should incorporate tree-based topological splits for deeper categorisation over traditional algebraic curves.
