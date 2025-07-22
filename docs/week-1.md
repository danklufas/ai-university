# History of AI & Math Foundations
## 1. Lesson Overview

!!! sucess "Learning Objectives"
By the end of this lesson, you will be able to:

- Describe three pivotal AI milestones and their lasting impact.
- Define Artificial Intelligence (AI), Machine Learning (ML), and Data Science with clear examples.
- Understand probability fundamentals—random variables, expectation, variance—and compute them by hand and in Python.
- Grasp key linear algebra concepts—vectors, matrices, dot products, matrix multiplication—and see how they underpin AI algorithms.
- Install and launch the required tools (Python, Jupyter Notebook, NumPy, pandas) and execute basic code.

---

## 2. Core Definitions

| Term | Definition & Citation | Example |
|:---- |:----------------------|:--------|
|**Artificial Intelligence (AI)**|	“The science and engineering of making intelligent machines, especially intelligent computer programs.” — John McCarthy, 1956 |  A chatbot that interprets questions and crafts human‑like responses.
**Machine Learning (ML)**|	Algorithms that improve performance on tasks by learning from data rather than explicit programming.|A regression model that learns to predict steel prices from historical sales.
**Data Science**|	Interdisciplinary practice of using statistics, programming, and domain knowledge to extract insights from data.|Cleaning and visualizing e‑commerce logs to uncover purchasing trends.

## 3. Concept Sections

### A. AI Milestones

???+ info "The Dartmouth Workshop (1956)"
    **What happened:**  
    In summer 1956, John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon met at Dartmouth College to ask: *“Can machines simulate human intelligence?”* They coined **“Artificial Intelligence”** and proposed studying how machines might “learn from experience,” “make abstractions,” and “use language.”

    **Context & significance:**  
    - Pre‑1956, computers = number crunchers. Dartmouth reframed them as **potential thinking machines**.  
    - Sparked optimism (and funding) that small teams could crack “every aspect of learning.”

 **First programs:**  
    - **Logic Theorist (1955)** – Newell & Simon proved logic theorems with a program.  
    - **General Problem Solver (1957)** – Early universal reasoning attempt.

    !!! note "Why this still matters"
        - Understanding the **hype → disappointment → AI winters** cycle helps you stay realistic about today’s claims.  
        - Symbolic reasoning/search ideas from this era live on in **knowledge graphs** and **constraint solvers**.

---

???+ info "Expert Systems Era (1970s–1980s)"
    **Core idea:** Encode expert knowledge as **IF–THEN rules**.

    ```text
    IF symptom = fever AND symptom = rash
    THEN suggest = measles
    ```

    **MYCIN (1972–1980):**  
    - ~600 rules to diagnose bacterial infections & suggest antibiotics  
    - Matched/surpassed human experts in blind tests

    **Strengths vs. limits:**  
    - ✅ Transparent logic (traceable to specific rules)  
    - ❌ Hard to scale (thousands of hand‑written rules), weak with uncertainty

    !!! tip "Modern relevance"
        - Rule‑based logic still used in finance/healthcare compliance.  
        - Today’s **hybrid systems**: rules for regulation + ML models for scoring.

---

???+ info "Deep Learning Boom (2010s–Present)"
    **Key breakthrough – AlexNet (2012):**  
    - 8‑layer CNN, cut ImageNet error rate in half (1.2M images, 1,000 classes)  
    - Used ReLU, dropout, and **GPU training**.

    **Why deep learning emerged:**  
    1. **Data:** Huge labeled datasets (images, text, speech)  
    2. **Compute:** GPUs = fast parallel matrix ops  
    3. **Algorithms:** Batch norm, better backprop, new architectures

    **Transformative apps:**  
    - Computer vision: self‑driving cars, medical imaging  
    - NLP: translation, GPT‑style generation  
    - Speech: voice assistants, real‑time translation

    !!! success "Why this matters for you"
        - Modern frameworks (TensorFlow, PyTorch) are built around neural nets.  
        - Explains why later terms focus on coding deep models & leveraging **pretrained architectures** quickly.

### C. Probability Basics

!!! abstract "Definition"
    **Probability Theory** – “The mathematical framework for quantifying uncertainty and modeling random phenomena.”

#### C1. Gentle Introduction 

???+ tip "1. What is Chance?"
    **Analogy:** Flipping a coin—two outcomes, but you can’t predict which.  
    **Key idea:** Probability measures how likely something is (0 = impossible, 1 = certain).  
    **Example:** A fair coin → P(heads) = 0.5.

???+ tip "2. Simple Data & Averages"
    **Real example:** Test scores: 80, 90, 70, 100, 60.  
    **Mean (average):**
    ```text
    (80 + 90 + 70 + 100 + 60) / 5 = 80
    ```
    **Why it matters:** The mean tells you what’s “typical.”

???+ tip "3. Measuring Spread (Variance)"
    **Analogy:** Scores all near 80% → small spread; scores all over the place → big spread.  
    **Steps (using the score list above):**
    1. Subtract the mean (80): e.g. 60 − 80 = −20  
    2. Square them: (−20)² = 400  
    3. Average the squares → variance ≈ 280  
    4. Square root of variance → standard deviation ≈ 16.7  
    **Why we care:** Spread tells you how consistent or noisy data is—critical for risk or quality control.

#### C2. Formal Definitions & Deep Dive

???+ info "1. Random Variables"
    A **random variable (RV)** assigns numbers to random outcomes.

    - **Discrete RV:** countable values (die roll, number of returns)  
      Example: Fair die →  
      ```text
      X ∈ {1,2,3,4,5,6},   P(X = k) = 1/6
      ```
    - **Continuous RV:** any value in a range (time between failures)  
      Example: Exponential distribution for time \( t ≥ 0 \):  
      ```text
      f(t) = λ e^{−λ t}
      ```

???+ info "2. Expectation (Mean)"
    Long‑run average outcome if you repeat forever.

    - **Discrete:**  
      ```text
      E[X] = Σ x_i · P(X = x_i)
      ```
    - **Continuous:**  
      ```text
      E[X] = ∫ x f(x) dx
      ```
    **Worked example (die):**  
    ```text
    E[X] = (1+2+3+4+5+6) / 6 = 3.5
    ```
    **Relevance:** Loss functions (e.g., MSE) minimize expected error → expectation is baked into training.

???+ info "3. Variance & Standard Deviation"
    **Variance:** average squared distance from the mean.  
    ```text
    Var(X) = E[(X − E[X])^2]
    ```
    **Std. dev.:**  
    ```text
    σ = √Var(X)
    ```
    **Die example:**  
    ```text
    Var ≈ 2.92,  σ ≈ 1.71
    ```
    **Why it matters:** Tells you how uncertain predictions are, helps build confidence intervals, drives anomaly detection.

!!! note "Why Probability Matters in AI"
    - **Model Training:** Errors are expectations (means) over data.  
    - **Uncertainty:** Variance underpins confidence, risk, anomaly flags.  
    - **Feature Engineering:** Understanding distributions guides transformations (e.g., log scales for skewed data).


### D. Linear Algebra Basics

!!! abstract "Definition"
    **Linear Algebra** – “The branch of mathematics concerned with vectors, vector spaces, and linear transformations.”

#### D1. Gentle Introduction 

???+ tip "1. Vectors as Lists"
    **Analogy:** A grocery list: `[2 bananas, 1 loaf bread, 500 g cheese]`  
    **Key idea:** A **vector** is just a list of numbers representing features.

???+ tip "2. Matrices as Tables"
    **Analogy:** A seating chart (rows = tables, columns = seats):  
    ```text
           S1  S2  S3
        T1  A   B   C
        T2  D   E   F
    ```
    **Key idea:** A **matrix** stacks many vectors into rows or columns.

???+ tip "3. Dot Product Intuition"
    **Example (bill splitting):**  
    - You & a friend order appetizers `[3, 2]` and drinks `[1, 2]`.  
    - Prices: appetizers = \$5, drinks = \$2 →  
    ```text
    [3, 2] · [5, 2] = 3×5 + 2×2 = 19
    ```
    **Why it matters:** Same math as a simple regression prediction (weights × features).

???+ tip "4. Real‑World Matrix Uses"
    - **Recipe scaling:** Multiply ingredient matrix by 1.5 to go from 4 to 6 servings.  
    - **School timetable:** Days × hours grid to schedule classes.

#### D2. Formal Definitions & Deep Dive

???+ info "1. Vectors & Their Interpretation"
    A vector **x ∈ ℝⁿ** is an ordered list of n numbers (features).  
    **Example:**
    ```text
    x = [age, monthly_spend, num_orders] = [45, 320.5, 12]
    ```

???+ info "2. Matrices & Batch Operations"
    A matrix **X ∈ ℝ^{m×n}** stacks m row‑vectors of length n.  
    **Example (customer table):**
    ```text
    X = [
      [45, 320.5, 12],
      [23, 150.0,  5],
      ...
    ]
    ```

???+ info "3. Dot Product & Linear Transformations"
    **Dot product:**
    ```text
    a · b = Σ (a_i * b_i)
    ```
    **Use in AI:**
    - **Regression:**  ŷ = w · x + b  
    - **Neural nets:**  z = w · x + b, then apply activation (e.g., ReLU)

???+ info "4. Matrix Multiplication"
    ```text
    C = A × B,   C_{ij} = Σ_k A_{ik} B_{kj}
    ```
    **Example:** Combine/transform features or chain neural network layers.

???+ success "Why Linear Algebra Matters in AI"
    - **Speed:** GPUs/NumPy rely on vectorized (matrix) ops for efficiency.  
    - **Model Insight:** Weights, activations, attention maps are matrices/vectors.  
    - **Dimensionality Reduction:** PCA, SVD use eigenvectors/values to compress data.

## 4. Tools Installation & Setup

!!! info "You’ll do this once, then reuse the environment all term."

### A. Install Python & Anaconda (Windows & Mac)
```bash
# Visit this in your browser:
https://www.anaconda.com/products/distribution
```
1. Download the **Python 3.x** installer for your OS.  
2. Run the installer, accept defaults.  
3. Open **Anaconda Navigator** (Start Menu on Windows / Applications on Mac).

### B. Launch Jupyter Notebook
1. In Anaconda Navigator, click **Launch** under **Jupyter Notebook**.  
2. A browser window opens showing your files.  
3. Click **New → Python 3**.  
4. Rename it to **Week1_AI_Math.ipynb**.

### C. Install & Import NumPy & pandas
In a notebook cell, run:
```bash
!conda install numpy pandas -y
```
Then import:
```python
import numpy as np
import pandas as pd
```

!!! tip "Why these tools?"
    - **Python/Jupyter:** interactive coding & math demos  
    - **NumPy:** fast vectors/matrices (used everywhere in ML)  
    - **pandas:** quick data tables, cleaning, summaries
  
## 5. Step by Step Exercises

???+ example "Exercise 1: Die Roll Simulation & Statistics"
    **1. Overview & Purpose**  
    Simulate 1,000 rolls of a fair six‑sided die in Python and compute the empirical mean and variance.  
    **Why:** Reinforces theoretical vs. empirical probability, builds NumPy familiarity, and demonstrates sampling variability.

    **2. Concept Reinforcement**  
    - Probability & random variables  
    - Expectation (mean) & variance  
    - Sampling variability / Law of Large Numbers

    **3. Real‑World Relevance**  
    - **Quality control:** simulate defect rates in a batch  
    - **Risk modeling:** Monte Carlo estimates of portfolio variance  
    - **Game design:** balance randomness in mechanics

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np

    # Simulate 1,000 die rolls
    np.random.seed(42)           # optional: reproducibility
    rolls = np.random.randint(1, 7, size=1000)

    # Compute statistics
    mean_rolls = rolls.mean()
    var_rolls  = rolls.var()

    print("Simulated Mean:    ", mean_rolls)
    print("Simulated Variance:", var_rolls)
    ```

    **Notes:**  
    - `np.random.randint(1, 7, size=1000)` → integers 1–6  
    - `.mean()`, `.var()` → empirical mean & variance (population variance by default)

    **5. Expected Outcomes & Interpretation**  
    - Mean ≈ 3.5, Variance ≈ 2.92 (± sampling noise)  
    - Larger sample sizes converge closer to the theoretical values

    **6. Extensions & Variations**  
    - Try sample sizes 100, 10,000 and compare stats  
    - Simulate a **weighted/unfair die**  
    - Plot histogram with `matplotlib`

    **7. Additional Notes & Tips**  
    - Use `np.random.seed(...)` if you want the same results every run  
    - Avoid Python loops when possible—NumPy vectorization is faster

???+ example "Exercise 2: Coin Flip Probability Estimation"
    **1. Overview & Purpose**  
    Simulate 10,000 coin flips to estimate the probability of heads and tails.

    **2. Concept Reinforcement**  
    - Discrete random variables  
    - Empirical vs. theoretical probability

    **3. Real‑World Relevance**  
    - **A/B testing:** success/failure rates  
    - **Clinical trials:** treatment vs. control outcomes

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np

    np.random.seed(0)                      # optional: reproducibility
    flips = np.random.choice(['H', 'T'], size=10000)

    p_heads = np.mean(flips == 'H')
    p_tails = np.mean(flips == 'T')

    print(f"P(heads): {p_heads:.3f}, P(tails): {p_tails:.3f}")
    ```

    **5. Expected Outcomes & Interpretation**  
    - Both ≈ 0.5, with random fluctuation ~±0.01  
    - Larger samples narrow the gap to 0.5

    **6. Extensions & Variations**  
    - Weighted coin: `p=['H':0.3, 'T':0.7]`  
    - Plot counts with a bar chart

    **7. Additional Notes & Tips**  
    - Set `np.random.seed(...)` when you want reproducible runs

???+ example "Exercise 3: Histogram of Die Rolls"
    **1. Overview & Purpose**  
    Visualize the distribution of the 1,000 die rolls from Exercise 1.

    **2. Concept Reinforcement**  
    - Frequency vs. probability  
    - Basic data visualization

    **3. Real‑World Relevance**  
    - Sales distribution by category  
    - Error counts per batch in manufacturing

    **4. Step-by-Step Instructions**
    ```python
    import matplotlib.pyplot as plt

    plt.hist(rolls, bins=range(1, 8), align='left', rwidth=0.8)
    plt.xlabel('Die Face')
    plt.ylabel('Frequency')
    plt.title('Histogram of 1,000 Die Rolls')
    plt.show()
    ```

    **5. Expected Outcomes & Interpretation**  
    - Bars roughly equal for faces 1–6 (random noise is okay)

    **6. Extensions & Variations**  
    - Normalized histogram: `plt.hist(..., density=True)`  
    - Overlay the theoretical PMF as a line/bar plot

    **7. Additional Notes & Tips**  
    - `bins=range(1,8)` centers bars on integer faces  
    - If you reused `rolls` from Exercise 1, you don’t need to re‑simulate

???+ example "Exercise 4: Exponential Distribution Simulation"
    **1. Overview & Purpose**  
    Simulate 5,000 samples from an exponential distribution (mean = 2) and compute mean/variance.

    **2. Concept Reinforcement**  
    - Continuous random variables  
    - Relationship between distribution parameters and statistics

    **3. Real‑World Relevance**  
    - Time between machine failures  
    - Call‑center interarrival times

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np

    np.random.seed(0)                         # optional
    samples = np.random.exponential(scale=2, size=5000)

    print("Empirical Mean:   ", samples.mean())
    print("Empirical Variance:", samples.var())
    ```

    **5. Expected Outcomes & Interpretation**  
    - Mean ≈ 2  
    - Variance ≈ 4  
    Small deviations are normal due to randomness.

    **6. Extensions & Variations**  
    - Change `scale` (mean) parameter  
    - Plot histogram and overlay the theoretical PDF

    **7. Additional Notes & Tips**  
    - `scale` in NumPy’s exponential is \( 1/λ \) (i.e., the mean)  
    - Use `matplotlib` or `seaborn` for quick visual checks

???+ example "Exercise 5: Normal Distribution Sampling"
    **1. Overview & Purpose**  
    Draw 10,000 samples from a standard normal distribution (mean 0, σ = 1) and verify statistics.

    **2. Concept Reinforcement**  
    - Properties of the Gaussian distribution  
    - Central Limit Theorem (preview)

    **3. Real‑World Relevance**  
    - Measurement error modeling  
    - Standardized test scores / z‑scores

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np

    np.random.seed(0)                    # optional
    normals = np.random.randn(10000)     # mean=0, std=1

    print("Mean:", normals.mean())
    print("Variance:", normals.var())
    ```

    **5. Expected Outcomes & Interpretation**  
    - Mean ≈ 0, Variance ≈ 1 (allow small deviation)

    **6. Extensions & Variations**  
    - Use `np.random.normal(loc, scale, size)` for non‑standard normals  
    - Make a QQ plot vs. theoretical normal to check normality

    **7. Additional Notes & Tips**  
    - `plt.hist(normals, density=True)` to visualize the bell curve

???+ example "Exercise 6: Sampling Distribution of the Mean"
    **1. Overview & Purpose**  
    Run 1,000 “mini‑experiments.” Each experiment rolls a die 100 times, records the mean, and we plot the distribution of those means.

    **2. Concept Reinforcement**  
    - Law of Large Numbers  
    - Sampling variability decreases as sample size increases  
    - Sampling distribution & standard error

    **3. Real‑World Relevance**  
    - **Polling averages:** many small samples → distribution of means  
    - **Quality control:** batch averages instead of single measurements

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)  # optional
    means = [np.random.randint(1, 7, 100).mean() for _ in range(1000)]

    plt.hist(means, bins=20)
    plt.title('Sampling Distribution of Die Roll Means')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.show()
    ```

    **5. Expected Outcomes & Interpretation**  
    - Histogram looks roughly normal, centered near 3.5  
    - Spread is much narrower than individual die outcomes

    **6. Extensions & Variations**  
    - Change experiment size: n=10 vs. n=1000 → compare spreads  
    - Compute **standard error**: σ / √n (use σ ≈ 1.71 for a die)

    **7. Additional Notes & Tips**  
    - List comprehensions are fine here; for speed, you can vectorize with NumPy

???+ example "Exercise 7: Weighted Dice Simulation"
    **1. Overview & Purpose**  
    Simulate 1,000 rolls of a **biased** die where P(6) = 0.5 and the other faces share the remaining probability.

    **2. Concept Reinforcement**  
    - Custom discrete distributions  
    - How bias shifts mean and variance

    **3. Real‑World Relevance**  
    - Biased processes in manufacturing (defect more likely on one line)  
    - Skewed customer behavior (one product far more popular)

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np

    np.random.seed(0)  # optional
    faces = [1, 2, 3, 4, 5, 6]
    probs = [0.1]*5 + [0.5]     # 0.1 each for 1–5, 0.5 for 6
    rolls_biased = np.random.choice(faces, size=1000, p=probs)

    print("Mean:", rolls_biased.mean(), "Variance:", rolls_biased.var())
    ```

    **5. Expected Outcomes & Interpretation**  
    - Mean **> 3.5** due to heavy weight on 6  
    - Variance will differ from the fair‑die case

    **6. Extensions & Variations**  
    - Tweak `probs` for different biases  
    - Plot histograms for fair vs. biased dice side‑by‑side

    **7. Additional Notes & Tips**  
    - Ensure `sum(probs) == 1` or NumPy will error  
    - You can simulate many biased scenarios to stress‑test models

???+ example "Exercise 8: Vector Addition & Scaling"
    **1. Overview & Purpose**  
    Demonstrate vector addition and scalar multiplication using simple feature vectors.

    **2. Concept Reinforcement**  
    - Vector space operations (add, scale)  
    - Geometric interpretation (direction & length)

    **3. Real‑World Relevance**  
    - Combine feature effects (e.g., marketing channels)  
    - Scale normalized data or weights

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np

    v1 = np.array([2, 4, 6])
    v2 = np.array([1, 3, 5])

    sum_v   = v1 + v2        # vector addition
    scaled_v = 0.5 * v1      # scalar multiplication

    print("Sum:   ", sum_v)
    print("Scaled:", scaled_v)
    ```

    **5. Expected Outcomes & Interpretation**  
    - `Sum` = `[3, 7, 11]`  
    - `Scaled` = `[1, 2, 3]`

    **6. Extensions & Variations**  
    - Compute the **dot product**: `v1.dot(v2)`  
    - Visualize 2D/3D vectors (e.g., with matplotlib quiver)

    **7. Additional Notes & Tips**  
    - Ensure vectors have the same length for element‑wise ops  
    - Scalar multiplication stretches/shrinks the vector length

???+ example "Exercise 9: Matrix Multiplication Demonstration"
    **1. Overview & Purpose**  
    Multiply a 2×3 matrix by a 3×2 matrix to reinforce matrix‑multiplication rules and shape compatibility.

    **2. Concept Reinforcement**  
    - Shape rules (inner dimensions must match)  
    - Summation over the inner index (k)

    **3. Real‑World Relevance**  
    - Transforming feature spaces  
    - Chaining layers in neural networks (each layer = a matrix multiply)

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np

    A = np.array([[1, 2, 3],
                  [4, 5, 6]])          # shape (2, 3)

    B = np.array([[ 7,  8],
                  [ 9, 10],
                  [11, 12]])          # shape (3, 2)

    C = A.dot(B)            # or: C = A @ B in Python 3.5+
    print("Result:\n", C)
    ```

    **5. Expected Outcomes & Interpretation**  
    ```
    [[ 58  64]
     [139 154]]
    ```
    - You can verify: first row × first column → 1*7 + 2*9 + 3*11 = 58

    **6. Extensions & Variations**  
    - Try `B @ A` to see the shape error  
    - Use larger random matrices to test performance

    **7. Additional Notes & Tips**  
    - Check shapes with `A.shape`, `B.shape`  
    - `@` is shorthand for matrix multiply (`dot`) in NumPy/Python 3.5+

???+ example "Exercise 10: PCA on Toy Dataset"
    **1. Overview & Purpose**  
    Perform PCA on a tiny 2‑D dataset, reduce it to 1‑D, and see how much variance is captured.

    **2. Concept Reinforcement**  
    - Eigenvectors / eigenvalues  
    - Dimensionality reduction & variance explained

    **3. Real‑World Relevance**  
    - Compressing image or sensor data  
    - Feature extraction before clustering or modeling

    **4. Step-by-Step Instructions**
    ```python
    import numpy as np
    from sklearn.decomposition import PCA

    X = np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0]
    ])

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Projected data:\n", X_pca)
    ```

    **5. Expected Outcomes & Interpretation**  
    - First component should capture ~98% of variance for this toy set  
    - Projected 1‑D data preserves most “information” (spread)

    **6. Extensions & Variations**  
    - Plot original 2‑D vs. projected 1‑D points  
    - Try `n_components=2` (no reduction) and inspect components

    **7. Additional Notes & Tips**  
    - Requires `scikit-learn` (`pip install scikit-learn` if missing)  
    - PCA assumes linear structure; nonlinear data may need t‑SNE/UMAP

6. Summary of Week 1
Throughout this first week, you have:

Traced AI History: From the Dartmouth Workshop to Expert Systems (MYCIN) and the Deep Learning revolution (AlexNet).

Built Probability Skills: Learned random variables, expectation, variance—both by hand and in Python (die rolls, coin flips, exponential and normal sampling).

Visualized Data: Plotted histograms, demonstrated Central Limit Theorem.

Handled Bias: Modeled a weighted die to see effects on distribution.

Applied Linear Algebra: Performed vector ops, matrix multiplication, and PCA for dimensionality reduction.

These foundational concepts and hands‑on exercises prepare you for more advanced AI and ML topics.

7. Additional Resources
Probability Fundamentals: Khan Academy “Introduction to Probability”

Linear Algebra Visualizations: 3Blue1Brown “Essence of Linear Algebra” series

Python Tutorials: Official Python documentation at python.org

