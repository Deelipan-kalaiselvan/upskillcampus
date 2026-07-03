"""
Week 3 Project: Probability & Statistics Foundations
UCT / upskill Campus Internship - Deelipan

Covers: probability distributions, hypothesis testing, confidence intervals,
regression analysis, correlation/covariance, and A/B testing methodology —
applied to the Week 2 healthcare readmission dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

# ---------------------------------------------------------------------
# 1. LOAD DATA (reusing Week 2 healthcare dataset for applied statistics)
# ---------------------------------------------------------------------
df = pd.read_csv("data/hospital_discharges.csv")

# ---------------------------------------------------------------------
# 2. PROBABILITY DISTRIBUTIONS
# ---------------------------------------------------------------------
def plot_distribution(series, name):
    plt.figure(figsize=(7, 4))
    sns.histplot(series.dropna(), kde=True, stat="density")
    plt.title(f"Distribution of {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/dist_{name}.png")
    plt.close()

for col in ["age", "length_of_stay", "comorbidity_count"]:
    if col in df.columns:
        plot_distribution(df[col], col)
        skewness = stats.skew(df[col].dropna())
        kurtosis = stats.kurtosis(df[col].dropna())
        print(f"{col}: skew={skewness:.2f}, kurtosis={kurtosis:.2f}")

# Fit and compare theoretical distributions (Normal, Poisson) to length_of_stay
los = df["length_of_stay"].dropna()
print(f"\nLength of stay — mean: {los.mean():.2f}, var: {los.var():.2f}")
print("Poisson comparison (mean ≈ variance check):",
      "Poisson-like" if abs(los.mean() - los.var()) < 2 else "Overdispersed (not Poisson)")

# ---------------------------------------------------------------------
# 3. HYPOTHESIS TESTING — Age-related readmission gradient (65+ vs <65)
# ---------------------------------------------------------------------
group_65plus = df[df["age"] >= 65]["readmitted_30d"]
group_under65 = df[df["age"] < 65]["readmitted_30d"]

count = np.array([group_65plus.sum(), group_under65.sum()])
nobs = np.array([len(group_65plus), len(group_under65)])

z_stat, p_value = proportions_ztest(count, nobs)
print(f"\nAge group readmission test: z={z_stat:.3f}, p-value={p_value:.4f}")
print("Result:", "Statistically significant (reject H0)" if p_value < 0.05
      else "Not significant (fail to reject H0)")
print(f"65+ readmission rate: {group_65plus.mean():.1%}")
print(f"<65 readmission rate: {group_under65.mean():.1%}")

# ---------------------------------------------------------------------
# 4. CONFIDENCE INTERVALS
# ---------------------------------------------------------------------
def proportion_ci(successes, n, confidence=0.95):
    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat - margin, p_hat + margin

ci_low, ci_high = proportion_ci(group_65plus.sum(), len(group_65plus))
print(f"\n95% CI for 65+ readmission rate: ({ci_low:.1%}, {ci_high:.1%})")

# ---------------------------------------------------------------------
# 5. CORRELATION & COVARIANCE
# ---------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Matrix — Clinical Features")
plt.tight_layout()
plt.savefig("outputs/figures/correlation_matrix.png")
plt.close()

print("\nTop correlations with readmission:\n",
      corr_matrix["readmitted_30d"].sort_values(ascending=False).head(6))

# ---------------------------------------------------------------------
# 6. REGRESSION ANALYSIS (simple, multiple, logistic)
# ---------------------------------------------------------------------
# Simple linear regression: length_of_stay ~ comorbidity_count
X_simple = sm.add_constant(df["comorbidity_count"])
y_simple = df["length_of_stay"]
simple_model = sm.OLS(y_simple, X_simple, missing="drop").fit()
print("\nSimple Linear Regression (LOS ~ comorbidity_count):\n", simple_model.summary().tables[1])

# Multiple logistic regression: readmission ~ age + comorbidity_count + length_of_stay
X_multi = sm.add_constant(df[["age", "comorbidity_count", "length_of_stay"]].fillna(0))
y_multi = df["readmitted_30d"]
logit_model = sm.Logit(y_multi, X_multi).fit(disp=0)
print("\nMultiple Logistic Regression Summary:\n", logit_model.summary())

# ---------------------------------------------------------------------
# 7. A/B TESTING METHODOLOGY — living alone vs family support intervention
# ---------------------------------------------------------------------
if "lives_alone" in df.columns:
    group_alone = df[df["lives_alone"] == 1]["readmitted_30d"]
    group_family = df[df["lives_alone"] == 0]["readmitted_30d"]

    t_stat, p_val = stats.ttest_ind(group_alone, group_family, equal_var=False)
    print(f"\nA/B test — living alone vs family support:")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    print(f"Living alone readmission rate: {group_alone.mean():.1%}")
    print(f"Family support readmission rate: {group_family.mean():.1%}")

print("\nStatistical foundations analysis complete.")
