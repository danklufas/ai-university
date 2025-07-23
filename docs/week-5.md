# Week 5 Lesson Plan – Ethics, Bias & Responsible AI

---

## 1. Lesson Overview

**Learning Objectives**

By the end of this week you will be able to:

- Identify common **sources of bias** in data, labeling, and algorithms.  
- Calculate and interpret key **fairness metrics** (e.g., demographic parity, equal opportunity).  
- Understand core **privacy & compliance concepts** (consent, minimization, GDPR/PIPEDA basics).  
- Apply **responsible AI frameworks** (checklists, governance processes) to your own projects.  
- Use Python tools to **measure and mitigate bias** in a toy dataset.

---

## 2. Core Definitions

| Term | Definition | Simple Example |
|------|------------|-----------------|
| **Algorithmic Bias** | Systematic and repeatable errors that create unfair outcomes. | Loan model approves 80% of group A but 55% of group B. |
| **Demographic Parity** | Selection/positive rate should be similar across groups. | % approved loans for men ≈ % for women. |
| **Equal Opportunity** | True positive rate should be equal across groups. | Cancer detection model should catch positives equally well for all races. |
| **Fairness Metric** | Quantitative measure of disparity (e.g., difference in TPR). | TPR_A − TPR_B ≤ 0.05 target. |
| **Data Minimization** | Collect only what is necessary for the task. | Don’t store SSNs if you only need age. |
| **Explainability** | Ability to interpret model decisions. | SHAP values showing which features drove a prediction. |
| **Responsible AI Governance** | Policies, processes, and tools ensuring ethical deployment. | Review board, model cards, audit logs. |

---

## 3. Concept Sections

### A. Where Bias Creeps In (and Real Examples)

1. **Data Collection Bias** – Your dataset under-represents a subgroup.  
   *Example:* Feedback forms mostly from tech-savvy users → product skews to their needs.

2. **Labeling Bias** – Human annotators carry their own prejudices.  
   *Example:* Historical hiring decisions labeled “good candidate” reflect past discrimination.

3. **Measurement / Proxy Bias** – Using a proxy variable that encodes sensitive info.  
   *Example:* Zip code as proxy for race or income.

4. **Algorithm/Optimization Bias** – Loss functions ignore fairness constraints.  
   *Example:* Optimizing only for accuracy favors majority class.

5. **Deployment & Feedback Loops** – Model outputs change behavior, which changes inputs.  
   *Example:* Police patrol more in predicted “hot spots” → more arrests → model “proves” itself.

**Why This Matters:** You can’t fix what you can’t see. Knowing the phases of an ML pipeline helps you inject checks at the right places.

---

### B. Fairness Metrics (Intuition → Formula)

**1. Demographic Parity (DP):**  
Positive prediction rate should be equal.  
\[
\text{DP\ diff} = |P(\hat{Y}=1 \mid A=0) - P(\hat{Y}=1 \mid A=1)|
\]

**2. Equal Opportunity (EOpp):**  
True positive rates equal across groups.  
\[
\text{EOpp\ diff} = |TPR_{A=0} - TPR_{A=1}|
\]

**3. Equalized Odds:**  
Both TPR and FPR equal across groups.

**4. Predictive Parity / Calibration:**  
Given a score, actual outcome probability is the same for each group.

**Trade-offs:**  
You usually can’t satisfy all metrics simultaneously; pick what matches the product’s ethical goals and legal context.

---

### C. Privacy, Consent, and Regulation (High Level)

- **Privacy Principles:** consent, purpose limitation, data minimization, security, accountability.  
- **De-identification & Anonymization:** remove direct identifiers; but beware re-identification via linkage.  
- **Basic Regs to Know (Canada/EU/US):**  
  - PIPEDA (Canada) – personal info rules for private sector.  
  - GDPR (EU) – rights to access, erase, data portability.  
  - Sector-specific (HIPAA, COPPA, etc.) if you handle health/children’s data.

**Focus for You:** Build habits—log data sources, justify fields collected, document retention & deletion plans.

---

### D. Responsible AI Frameworks & Governance

- **Checklists & Model Cards:** Document purpose, data, limitations, bias tests, maintenance plan.  
- **NIST AI RMF / OECD Principles:** Risk management, transparency, accountability.  
- **Human-in-the-loop:** Allow overrides, appeals, escalation paths.  
- **Monitoring in Prod:** Drift detection, fairness checks on new data, incident response.

**Why This Matters:** Stakeholders (investors, customers, regulators) increasingly demand proof you’re managing AI responsibly.

---

## 4. Tools Installation & Setup (Week 5 Specific)

> You already have NumPy, pandas, matplotlib. This week we’ll add a fairness library and a quick explainability tool.

### A. Install Fairness & Explainability Packages

**Fairlearn (fairness metrics & mitigation) + SHAP (explainability):**

```bash
# conda (recommended)
conda install -c conda-forge fairlearn shap -y

# or pip
pip install fairlearn shap
```

## 5. Step-by-Step Exercises (10)


---

### Exercise 1: Quick Bias Scan with `pandas`

**Overview & Purpose**  
Check basic demographic parity by grouping outcomes by a sensitive attribute.

**Concept Reinforcement**  
- Demographic parity (selection rate)  
- `groupby` aggregation in pandas

**Real‑World Relevance**  
- Loan approvals by gender/race  
- Hiring pass rates across departments

**Step-by-Step Instructions**

```python
import pandas as pd

data = {
    "score":    [0.9, 0.2, 0.7, 0.85, 0.3, 0.6, 0.1, 0.95],
    "approved": [1,   0,   1,   1,    0,   1,   0,   1   ],
    "gender":   ["M", "F", "M", "M",  "F", "F", "F", "M" ]
}
df = pd.DataFrame(data)

rates = df.groupby("gender")["approved"].mean()
print(rates)

dp_diff = abs(rates["M"] - rates["F"])
print("Demographic parity diff:", dp_diff)
```

**Expected Outcomes & Interpretation**  
- Two approval rates (e.g., M ~0.80, F ~0.50).  
- `dp_diff` shows the absolute disparity (e.g., 0.30).

**Extensions & Variations**  
- Add more groups (age buckets, regions).  
- Plot a bar chart of approval rates.

**Notes & Tips**  
- Ensure your sensitive attribute names are consistent (`"gender"`, `"race"`, etc.).  
- Use `.value_counts(normalize=True)` to get proportions quickly.

---

### Exercise 2: Equal Opportunity from Confusion Matrices

**Overview & Purpose**  
Manually compute TPR (true positive rate) by group and compare.

**Concept Reinforcement**  
- Equal opportunity metric  
- Confusion matrix interpretation

**Real‑World Relevance**  
- Medical diagnosis rates across demographics  
- Fraud detection recall differences

**Step-by-Step Instructions**

```python
import numpy as np

# Confusion matrices: rows = actual (0/1), cols = predicted (0/1)
# group A (e.g., M)
cm_A = np.array([[50, 10],
                 [ 5, 35]])  # TN, FP / FN, TP

# group B (e.g., F)
cm_B = np.array([[48, 12],
                 [12, 28]])

TPR_A = cm_A[1,1] / (cm_A[1,0] + cm_A[1,1])
TPR_B = cm_B[1,1] / (cm_B[1,0] + cm_B[1,1])

print("TPR_A:", TPR_A, "TPR_B:", TPR_B, "Diff:", abs(TPR_A - TPR_B))
```

**Expected Outcomes & Interpretation**  
- You’ll see each group’s TPR and the disparity (diff).  
- Large diffs (>0.05) could flag fairness concerns.

**Extensions & Variations**  
- Add FPR comparison for Equalized Odds.  
- Visualize with a grouped bar chart.

**Notes & Tips**  
- Double-check TN/FP/FN/TP ordering—easy to mix up!

---

### Exercise 3: Fairness Metrics with `fairlearn.MetricFrame`

**Overview & Purpose**  
Use Fairlearn to compute multiple metrics per group in one shot.

**Concept Reinforcement**  
- MetricFrame abstraction  
- Selection rate, TPR, FPR

**Real‑World Relevance**  
- Faster audits in production pipelines  
- Compliance reporting dashboards

**Step-by-Step Instructions**

```python
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
y_true = [1,0,1,1,0,1,0,1]
y_pred = [1,0,1,0,0,1,0,1]
gender  = ["M","F","M","M","F","F","F","M"]

mf = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "tpr": true_positive_rate
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=gender
)

print(mf.by_group)
print("TPR diff:", mf.difference(method='between_groups')["tpr"])
print("Selection rate diff:", mf.difference(method='between_groups')["selection_rate"])
```

**Expected Outcomes & Interpretation**  
- Table of metrics per group.  
- Differences help identify unfairness.

**Extensions & Variations**  
- Add more metrics: `false_positive_rate`, `accuracy`.  
- Try `mf.group_min()` / `group_max()` to see bounds.

**Notes & Tips**  
- Fairlearn can also handle continuous scores (probabilities).

---

### Exercise 4: Threshold Adjustment Mitigation

**Overview & Purpose**  
Show how different decision thresholds per group can reduce disparity.

**Concept Reinforcement**  
- Post-processing mitigation  
- Trade-off between fairness and overall accuracy

**Real‑World Relevance**  
- Lending thresholds per applicant segment  
- Medical triage thresholds by demographic

**Step-by-Step Instructions**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate

# Synthetic data with slight bias
np.random.seed(0)
X = np.random.rand(400, 3)
A = np.random.choice(["M","F"], size=400)
y = (X[:,0]*0.7 + (A=="M")*0.2 + np.random.randn(400)*0.1 > 0.5).astype(int)

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, A, test_size=0.3, random_state=42
)

clf = LogisticRegression().fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:,1]

# Single threshold (baseline)
y_pred_base = (probs > 0.5).astype(int)

# Group-specific thresholds (try to equalize selection rate)
thr_M, thr_F = 0.55, 0.45
y_pred_adj = np.where(A_test=="M", probs>thr_M, probs>thr_F).astype(int)

# Compare
metrics = {"selection_rate": selection_rate, "tpr": true_positive_rate}
mf_base = MetricFrame(metrics, y_test, y_pred_base, sensitive_features=A_test)
mf_adj  = MetricFrame(metrics, y_test, y_pred_adj,  sensitive_features=A_test)

print("BASELINE\n", mf_base.by_group)
print("ADJUSTED\n", mf_adj.by_group)
```

**Expected Outcomes & Interpretation**  
- Adjusted thresholds should narrow metric gaps but may lower overall accuracy.  
- Discuss trade-offs.

**Extensions & Variations**  
- Use Fairlearn’s `ThresholdOptimizer` to find thresholds automatically.  
- Try optimizing for EO instead of DP.

**Notes & Tips**  
- Document any manual thresholding in model cards.

---

### Exercise 5: SHAP for Feature Influence (Proxy Detection)

**Overview & Purpose**  
Use SHAP to see if a seemingly innocuous feature proxies a sensitive attribute.

**Concept Reinforcement**  
- Explainability for fairness  
- Feature attribution

**Real‑World Relevance**  
- Spotting “zip code → race” proxies in credit scoring  
- Detecting age proxies in hiring models

**Step-by-Step Instructions**

```python
import shap
shap.initjs()

explainer = shap.LinearExplainer(clf, X_train)
shap_values = explainer.shap_values(X_test[:50])

# Summary plot (opens in Jupyter notebook)
shap.summary_plot(shap_values, X_test[:50], feature_names=["f1","f2","f3"])
```

**Expected Outcomes & Interpretation**  
- See which features drive predictions.  
- Compare distribution of feature values by group.

**Extensions & Variations**  
- Try `shap.dependence_plot` for single features.  
- Use `TreeExplainer` for tree models.

**Notes & Tips**  
- SHAP can be slow on big models; sample data.

---

### Exercise 6: Red-Team Prompting for LLM Safety

**Overview & Purpose**  
Stress-test a chatbot or LLM prompt to find unethical/unsafe outputs.

**Concept Reinforcement**  
- Safety testing / adversarial prompting  
- Policy compliance

**Real‑World Relevance**  
- Ensure customer support bots don’t give legal/medical advice without disclaimers  
- Prevent harmful outputs (hate speech, PII leaks)

**Step-by-Step Instructions**  
1. Write 5–10 “tricky” prompts (e.g., “How can I evade taxes?”).  
2. Record model responses.  
3. Label each as **Acceptable / Needs Warning / Reject**.  
4. Propose guardrails (prompt constraints, filters).  
5. (Optional) Implement regex/keyword filters or model moderation endpoints.

**Expected Outcomes & Interpretation**  
- A list of failure modes and mitigations.

**Extensions & Variations**  
- Use OpenAI moderation API or similar tools.  
- Create a “prompt safety checklist.”

**Notes & Tips**  
- Keep logs; iterate regularly as prompts evolve.

---

### Exercise 7: Mini Model Card

**Overview & Purpose**  
Create a concise “model card” for one of your previous models (e.g., Week 4 MLP).

**Concept Reinforcement**  
- Documentation & transparency  
- Stakeholder communication

**Real‑World Relevance**  
- Internal/external audits  
- Customer trust

**Step-by-Step Instructions**  
1. Template sections:  
   - **Model Details:** type, version, date.  
   - **Intended Use / Out-of-Scope Use.**  
   - **Data:** source, preprocessing, known issues.  
   - **Metrics:** overall + by subgroup (if available).  
   - **Ethical Considerations & Limitations.**  
   - **Maintenance / Update Plan.**  
2. Write 1–2 pages in Markdown (`model-card.md`).  
3. Store in repo for traceability.

**Expected Outcomes & Interpretation**  
- Clear, repeatable doc for future models.

**Extensions & Variations**  
- Use Google’s Model Card toolkit format.  
- Add visual metric tables.

**Notes & Tips**  
- Update the card when the model changes.

---

### Exercise 8: Privacy & Policy Mapping Checklist

**Overview & Purpose**  
Map a SaaS feature’s data flow to privacy and fairness controls.

**Concept Reinforcement**  
- Data minimization  
- Governance alignment

**Real‑World Relevance**  
- PIPEDA/GDPR readiness  
- Due diligence for M&A

**Step-by-Step Instructions**  
1. List all personal data fields you collect.  
2. For each: purpose, retention time, who can access.  
3. Identify fairness checks pre/post deployment.  
4. Note missing policies/processes.  
5. Summarize in a checklist doc (`privacy-fairness-checklist.md`).

**Expected Outcomes & Interpretation**  
- Gap analysis for compliance.  
- Action plan items.

**Extensions & Variations**  
- Map to NIST AI RMF categories.  
- Add incident response flowchart.

**Notes & Tips**  
- Revisit quarterly; regulations change.

---

### Exercise 9: Reweighting Mitigation with Fairlearn Reductions

**Overview & Purpose**  
Apply algorithmic mitigation (reweighting) using Fairlearn’s reductions API.

**Concept Reinforcement**  
- Pre-/in-processing mitigation  
- Constraint-based optimization

**Real‑World Relevance**  
- Achieving fairness goals without fully redesigning model  
- Regulatory evidence

**Step-by-Step Instructions**

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.tree import DecisionTreeClassifier

estimator   = DecisionTreeClassifier(max_depth=3, random_state=0)
constraint  = DemographicParity()
mitigator   = ExponentiatedGradient(estimator=estimator, constraints=constraint)

mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred_mitigated = mitigator.predict(X_test)
```

Compute fairness/accuracy metrics before vs. after (reuse MetricFrame).

**Expected Outcomes & Interpretation**  
- Reduced DP diff, possibly lower accuracy.  
- Discuss acceptable trade-offs.

**Extensions & Variations**  
- Try `EqualizedOdds()` constraint.  
- Compare different base estimators.

**Notes & Tips**  
- Reductions can be slow; start with small datasets.

---

### Exercise 10: Build a Responsible AI Checklist for Your Company

**Overview & Purpose**  
Translate week’s concepts into an actionable internal policy.

**Concept Reinforcement**  
- Governance operationalization  
- Continuous monitoring

**Real‑World Relevance**  
- CRO Software & Micro Manage Software policies  
- Investor/board confidence

**Step-by-Step Instructions**  
1. Sections to include:  
   - **Data Intake Questions** (Why do we need this? How long keep it?)  
   - **Bias & Fairness Checks** (Which metrics? How often?)  
   - **Privacy Compliance Steps** (Consent, retention, deletion).  
   - **Human Review & Overrides** (Escalation path).  
   - **Monitoring & Incident Response** (Who watches? What triggers alerts?).  
2. Draft in Markdown (`responsible-ai-checklist.md`).  
3. Review quarterly.

**Expected Outcomes & Interpretation**  
- A living document guiding responsible AI work.

**Extensions & Variations**  
- Add sign-off lines for teams (Data, Legal, Exec).  
- Link to model cards & audit logs.

**Notes & Tips**  
- Keep it practical—short enough people actually use it.

---

## 6. Summary of Week 5

This week you:

- **Mapped the sources of bias** (data, labels, algorithms, feedback loops) and learned where to intervene.  
- **Quantified fairness** using demographic parity, equal opportunity, and equalized odds—understanding trade-offs.  
- **Practiced mitigation** techniques: threshold adjustment, reweighting via Fairlearn, and transparency through SHAP.  
- **Integrated privacy principles** (consent, minimization, retention) and learned the basics of major regulations (GDPR, PIPEDA).  
- **Created governance artifacts**: model cards, checklists, red-team prompts—turning ethics into repeatable processes.

You now have a repeatable workflow to ask: *Is it fair? Is it explainable? Is it compliant?*—before shipping models.

---

## 7. Additional Resources

- **Fairlearn Documentation:** <https://fairlearn.org>  
- **SHAP Documentation:** <https://shap.readthedocs.io>  
- **“Model Cards for Model Reporting”** – Google AI (Mitchell et al.)  
- **NIST AI Risk Management Framework (AI RMF)**  
- **OECD AI Principles**  
- **The Ethical Algorithm** (Michael Kearns & Aaron Roth)  
- **Partnership on AI Resources:** <https://partnershiponai.org>  