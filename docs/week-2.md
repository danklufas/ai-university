# Week 2 Lesson Plan — Supervised Learning: Regression & Classification

## 1. Lesson Overview

**Learning Objectives**  
By the end of Week 2, you will be able to:

- Explain what **supervised learning** is and how it differs from unsupervised learning.  
- Build and evaluate **linear regression** and **logistic regression** models using scikit‑learn.  
- Understand and compute key **evaluation metrics** (MSE, MAE, R² for regression; Accuracy, Precision, Recall, F1, ROC–AUC for classification).  
- Recognize **overfitting vs. underfitting** and apply basic remedies (train/test split, regularization).  
- Implement a clean **ML workflow**: load data → split → train → evaluate → iterate.

## 2. Core Definitions

| Term | Definition & Example |
|---|---|
| **Supervised Learning** | Learning a mapping from inputs **X** to an output **y** using labeled data. *Example:* Predicting house price (y) from size and bedrooms (X). |
| **Regression** | Supervised learning where **y is numeric/continuous**. *Example:* Predicting steel prices or monthly revenue. |
| **Classification** | Supervised learning where **y is categorical**. *Example:* Predicting if a lead will convert (“yes/no”). |
| **Loss Function** | A formula that measures how wrong a prediction is. Models try to minimize this. *Example:* Mean Squared Error (MSE). |
| **Overfitting / Underfitting** | Overfitting: model memorizes noise, performs poorly on new data. Underfitting: model too simple, misses patterns. |
| **Train/Test Split** | Partition data into a training set (to fit the model) and test set (to evaluate generalization). |
| **Precision / Recall / F1** | Metrics for classification. Precision: “Of predicted positives, how many were right?” Recall: “Of actual positives, how many did we catch?” F1 balances both. |

## 3. Concept Sections

### A. Supervised Learning: Big Picture

**A1. Introduction (Plain English)**  
- Supervised learning is **teaching by example**. You give the algorithm many input–output pairs, and it learns the relationship.  
- Two main flavors:  
  - **Regression**: Predict numbers (price, demand).  
  - **Classification**: Predict categories (spam/not spam, churn/no churn).

**A2. The Basic Pipeline**  
1. Collect & clean data (features **X**, target **y**)  
2. **Split** into training and test sets  
3. **Train** a model on training data  
4. **Evaluate** on test data (metrics)  
5. **Tune** hyperparameters or choose another model  
6. **Deploy/Use** the model

---

### B. Linear Regression Deep Dive

**B1. Intuition (Grade‑10 friendly)**  
- Plot points on a graph (e.g., ad spend vs. sales). Draw the “best” straight line through them.  
- “Best” usually = the line with the **smallest average squared error**.

**B2. Formalism**  
- Model: \\(\hat{y} = w_0 + w_1 x_1 + \dots + w_n x_n\\)  
- **MSE**: \\(\frac{1}{m}\sum_{i=1}^m (\hat{y}_i - y_i)^2\\)

**B3. Why You Care**  
- Baseline model for many business problems.  
- Builds intuition for linear layers in neural networks later.

---

### C. Logistic Regression (Classification)

**C1. Intuition (Plain English)**  
- Linear combo of features → pass through **sigmoid** to get probability (0–1).  
- Threshold (e.g., 0.5) to decide class.

**C2. Formal Bits**  
- Sigmoid: \\(\sigma(z) = \frac{1}{1 + e^{-z}}\\), where \\(z = w^\top x + b\\).  
- **Cross‑entropy loss** penalizes confident wrong predictions more.

**C3. Odds & Log‑Odds**  
- Logistic regression models **log‑odds**: \\(\log \frac{p}{1-p} = w^\top x + b\\).  
- Coefficients show how features push probability up/down.

---

### D. Evaluation Metrics & Overfitting

**D1. Regression Metrics**  
- **MSE / RMSE**: Squared errors (sensitive to outliers).  
- **MAE**: Absolute errors (robust to outliers).  
- **R²**: Variance explained.

**D2. Classification Metrics**  
- **Accuracy**: Overall correctness (misleading on imbalance).  
- **Precision / Recall / F1**: Balance false positives vs. false negatives.  
- **ROC–AUC**: Performance across thresholds.

**D3. Overfitting vs. Underfitting**  
- **Overfit**: Great on train, bad on test.  
- **Underfit**: Bad everywhere.  
- **Fixes**: More data, simpler model, regularization (L1/L2), cross‑validation.

## 4. Tools Installation & Setup (Week 2 Specific)

> Assumes Python, Jupyter, NumPy, and pandas are already installed from Week 1.

### A. Download/Install (only if you don’t already have them)

- **Python 3.x (official):** <https://www.python.org/downloads/>  
- **Anaconda (optional, easy package mgmt):** <https://www.anaconda.com/download>  
- **Git (already installed, but link for reference):** <https://git-scm.com/downloads>

### B. Install scikit-learn & matplotlib

Pick **ONE** method (conda **or** pip).

Official docs:  
- scikit-learn: <https://scikit-learn.org/stable/install.html>  
- matplotlib: <https://matplotlib.org/stable/users/installing.html>

**Using conda (recommended if you installed Anaconda):**

~~~bash
conda install scikit-learn matplotlib -y
~~~

**Or with pip:**

~~~bash
pip install scikit-learn matplotlib
# Windows alternative if 'pip' isn't found:
py -m pip install scikit-learn matplotlib
~~~

---

### C. (Optional) Install seaborn for nicer plots

Official docs: <https://seaborn.pydata.org/installing.html>

**With conda:**

~~~bash
conda install seaborn -y
~~~

**Or with pip:**

~~~bash
pip install seaborn
py -m pip install seaborn
~~~

---

### D. Verify inside Python/Jupyter

~~~python
import sklearn, matplotlib, seaborn
print("sklearn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", seaborn.__version__)
~~~

### **5 — Step-by-Step Exercises (10 total)**

## 5. Step by Step Exercises

> Same template as Week 1: Purpose → Concept → Real‑World → Steps → Expected Outcome → Extensions → Notes.

### Regression (Exercises 1–4)

???+ example "Exercise 1: Simple Linear Regression (Synthetic Data)"
    **1. Overview & Purpose**  
    Fit a linear regression on generated data (`y = 3x + noise`) to see how the model recovers the relationship.

    **2. Concept Reinforcement**  
    - Linear regression basics  
    - Train/test split, MSE, R²

    **3. Real‑World Relevance**  
    - Forecasting revenue vs. ad spend  
    - KPI prediction

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    np.random.seed(0)
    X = np.random.rand(100, 1) * 10
    y = 3 * X.squeeze() + np.random.randn(100) * 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R² :", r2)
    print("Coef:", model.coef_, "Intercept:", model.intercept_)
    ```

    **5. Expected Outcomes & Interpretation**  
    - Slope ≈ 3; high R².  

    **6. Extensions & Variations**  
    - Increase noise  
    - Add polynomial features

    **7. Notes**  
    - Always keep a test set to gauge generalization.


???+ example "Exercise 2: Visualizing Fit & Residuals"
    **Purpose**: See where the model makes errors.  
    **Concepts**: Residual plots reveal structure; random scatter = good.

    ```python
    import matplotlib.pyplot as plt

    # Uses X_test, y_test, y_pred, model from Exercise 1
    plt.scatter(X_test, y_test, label='Actual')
    plt.scatter(X_test, y_pred, label='Predicted')
    plt.plot(sorted(X_test[:,0]), model.predict(np.sort(X_test, axis=0)), color='red', label='Line')
    plt.legend(); plt.title("Actual vs. Predicted")
    plt.show()

    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted"); plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.show()
    ```

    **Expected**: Residuals roughly centered around 0 with no clear pattern.  
    **Extensions**: Histogram of residuals; log-transform y if needed.


???+ example "Exercise 3: Multiple Linear Regression with pandas"
    **Purpose**: Use several features to predict a target.  
    **Concepts**: Multivariate regression, MAE.

    ```python
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    data = {
        'size': [1400,1600,1700,1875,1100,1550,2350,2450,1425,1700],
        'bedrooms': [3,3,3,2,2,3,4,4,3,3],
        'age': [20,15,18,12,30,15,7,5,24,18],
        'price': [245000,312000,279000,308000,199000,219000,405000,324000,319000,255000]
    }
    df = pd.DataFrame(data)

    X = df[['size','bedrooms','age']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    lr = LinearRegression().fit(X_train, y_train)
    preds = lr.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    print("MAE:", mae)
    ```

    **Expected**: Reasonable MAE; coefficients show feature influence.  
    **Extensions**: Standardize features; add interaction terms.  
    **Notes**: Watch multicollinearity.


???+ example "Exercise 4: Polynomial Regression & Overfitting Demo"
    **Purpose**: Show how higher-degree polynomials can overfit.  
    **Concepts**: Bias–variance trade‑off.

    ```python
    import numpy as np, matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    np.random.seed(0)
    X = np.linspace(0, 1, 20).reshape(-1,1)
    y = np.sin(2*np.pi*X).ravel() + np.random.randn(20)*0.2

    for degree in [1, 3, 9]:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        mse = mean_squared_error(y, y_pred)
        plt.scatter(X, y, label='data' if degree==1 else None)
        plt.plot(X, y_pred, label=f'deg {degree} (MSE={mse:.2f})')
    plt.legend(); plt.show()
    ```

    **Expected**: deg=1 underfits, deg=9 overfits.  
    **Extensions**: Add train/test split; try Ridge/Lasso.  
    **Notes**: Visual demos make concepts stick.

---

### Classification (Exercises 5–7)

???+ example "Exercise 5: Logistic Regression on a Toy Dataset"
    **Purpose**: Fit logistic regression and read metrics.  
    **Concepts**: Classification, probabilities, classification_report.

    ```python
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X, y = make_classification(n_samples=500, n_features=4, n_informative=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    ```

    **Expected**: Accuracy ~0.8–0.9; precision/recall shown.  
    **Extensions**: Change threshold; class weights.  
    **Notes**: Increase `max_iter` if convergence warning.


???+ example "Exercise 6: Confusion Matrix & ROC Curve"
    **Purpose**: Visualize errors and threshold performance.  
    **Concepts**: Confusion matrix, ROC, AUC.

    ```python
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

    disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.title("Confusion Matrix"); plt.show()

    y_prob = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.show()
    ```

    **Expected**: Curved ROC; AUC > 0.5.  
    **Extensions**: Precision–Recall curve for imbalance.  
    **Notes**: Use multiple metrics.


???+ example "Exercise 7: Class Imbalance & Threshold Tuning"
    **Purpose**: Show why accuracy fails with rare positives.  
    **Concepts**: Threshold tuning, precision/recall trade‑off.

    ```python
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support

    X, y = make_classification(n_samples=1000, n_features=5, weights=[0.95, 0.05], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_probs = clf.predict_proba(X_test)[:,1]

    for thresh in [0.5, 0.3, 0.1]:
        preds = (y_probs >= thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
        print(f"Threshold {thresh}: precision={p:.2f}, recall={r:.2f}, f1={f1:.2f}")
    ```

    **Expected**: Lower threshold → higher recall, lower precision.  
    **Extensions**: Class weights, PR curves.  
    **Notes**: Choose threshold by business cost.

---

### Workflow & Tuning (Exercises 8–10)

???+ example "Exercise 8: Cross-Validation vs. Single Split"
    **Purpose**: Compare metrics stability.  
    **Concepts**: KFold, variance of scores.

    ```python
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Reuse X, y from Exercise 1 or make new
    model = LinearRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, scoring='r2', cv=kf)

    print("R² scores:", scores)
    print("Mean R²:", np.mean(scores), "Std:", np.std(scores))
    ```

    **Expected**: Slight variation across folds.  
    **Extensions**: cross_val_predict.  
    **Notes**: Vital for limited data.


???+ example "Exercise 9: Hyperparameter Tuning with GridSearchCV"
    **Purpose**: Systematically find best hyperparameters.  
    **Concepts**: Grid search, scoring.

    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Ridge

    params = {'alpha': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(Ridge(), params, scoring='neg_mean_squared_error', cv=5)
    grid.fit(X, y)

    print("Best params:", grid.best_params_)
    print("Best score (MSE):", -grid.best_score_)
    ```

    **Expected**: One alpha gives lowest MSE.  
    **Extensions**: RandomizedSearchCV for speed.  
    **Notes**: Don’t overfit to validation by repeated peeking.


???+ example "Exercise 10: Mini End‑to‑End Project"
    **Purpose**: Practice full pipeline on a real CSV.  
    **Concepts**: Clean → Split → Train → Evaluate → Save.

    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import joblib

    df = pd.read_csv("your_data.csv")  # replace with a real dataset
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))

    joblib.dump(model, "model.joblib")
    ```

    **Expected**: Working model + `model.joblib` saved.  
    **Extensions**: Use Pipeline/ColumnTransformer.  
    **Notes**: Document steps for your capstone.

    ## 6. Week 2 Summary & What You Can Now Do

!!! success "You can now…"
    - **Build baseline models** for regression and classification with scikit‑learn.  
    - **Choose appropriate metrics** and interpret them in context (business costs, imbalance).  
    - **Diagnose over/underfitting** and apply simple fixes (regularization, cross‑validation).  
    - **Run a full ML workflow** end‑to‑end on small datasets.

### A. Core Takeaways
- Supervised learning = labeled data, mapping X→y.  
- Linear regression → numeric predictions; logistic regression → categorical predictions via probabilities.  
- Metrics matter: pick ones aligned with the problem (e.g., F1 for imbalance).  
- Generalization beats memorization.

### B. Practical Wins
- You used **GridSearchCV**, cross‑validation, and plotted diagnostics.  
- You can prototype business ML problems quickly.

### C. Next Week Prep
- Skim Week 3 (k‑means, PCA) to see unsupervised learning.  
- Get comfortable with pandas & plotting—visualizations increase.

## 7. Additional Resources

!!! tip "Guides & Tutorials"
- **scikit‑learn User Guide – Supervised Learning**  
  <https://scikit-learn.org/stable/supervised_learning.html>
- **StatQuest (YouTube)** – Excellent clear videos on regression, classification, metrics  
  <https://www.youtube.com/user/joshstarmer>

!!! tip "Metrics & Evaluation"
- “Precision, Recall and F1 Score for Dummies” (various blog guides)  
- ROC & AUC explainers (blog posts, Coursera ML course notes)

!!! note "Books & Chapters"
- *An Introduction to Statistical Learning* (ISLR) – Free PDF, chapters on regression & classification  
  <https://www.statlearning.com/>

!!! info "Cheat Sheets"
- scikit‑learn, matplotlib, seaborn cheat sheets (quick Google finds)

