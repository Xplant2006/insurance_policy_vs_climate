from model_train import policy_risk,policy_data
import matplotlib.pyplot as plt
import seaborn as sns

summary=policy_risk[["PolicyID","OverallRiskScore","RiskScore100","NormalizedRisk","RiskCategory"]].copy()
summary=summary.sort_values("RiskScore100",ascending=False)

print(summary.head(10).to_string(index=False))

plt.figure(figsize=(6,4))
sns.countplot(data=summary, x="RiskCategory", palette=["#88cc88","#ffcc66","#ff6666"])
plt.title("Distribution of Policy Risk Levels")
plt.xlabel("Risk Category")
plt.ylabel("Number of Policies")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(summary["RiskScore100"], bins=20, kde=True, color="steelblue")
plt.title("Distribution of Risk Scores (0–100)")
plt.xlabel("Risk Score")
plt.ylabel("Number of Policies")
plt.show()

plt.figure(figsize=(7,4))
sns.boxplot(data=policy_data.merge(policy_risk, on="PolicyID"), x="PropertyType", y="RiskScore100")
plt.title("Risk Scores by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Risk Score (0–100)")
plt.show()

top_risky = summary.nlargest(10, "RiskScore100")[["PolicyID", "RiskScore100", "RiskCategory"]]
print("\n Top 10 High-Risk Policies:")
print(top_risky.to_string(index=False))