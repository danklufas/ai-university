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

4. Tools Installation & Setup
Windows & Mac

A. Install Python & Anaconda
Navigate to: https://www.anaconda.com/products/distribution

Download the Python 3.x installer for your OS.

Run the installer, accept defaults.

Open Anaconda Navigator from your Start menu (Windows) or Applications folder (Mac).

B. Launch Jupyter Notebook
In Anaconda Navigator, click Launch under Jupyter Notebook.

A browser window opens showing your file system.

Click New → Python 3.

Rename the notebook to Week1_AI_Math.ipynb.

C. Install & Import NumPy & pandas
In a notebook cell, run:

bash
Copy
Edit
!conda install numpy pandas -y
Then, in the next cell:

python
Copy
Edit
import numpy as np
import pandas as pd
5. Step-by-Step Exercises
Exercise 1: Die Roll Simulation & Statistics (Template)
Exercise Overview & Purpose

What we’re doing: Simulate 1,000 rolls of a fair six‑sided die in Python to compute the empirical mean and variance.

Why: Reinforces theoretical vs. empirical probability, builds NumPy familiarity, and demonstrates sampling variability.

Concept Reinforcement

Probability & Random Variables

Expectation & Variance

Sampling Variability

Real World Relevance

Quality Control (defect rate simulation)

Risk Modeling (Monte Carlo portfolio variance)

Randomized Algorithms & Game Balancing

Step by Step Instructions

python
Copy
Edit
import numpy as np

# Simulate 1,000 die rolls
rolls = np.random.randint(1, 7, size=1000)

# Compute statistics
mean_rolls = rolls.mean()
var_rolls  = rolls.var()

print("Simulated Mean:    ", mean_rolls)
print("Simulated Variance:", var_rolls)
Notes:

np.random.randint(1, 7, size=1000): integers 1–6

.mean(), .var(): compute empirical mean & variance

Expected Outcomes & Interpretation

Mean ≈ 3.5, Variance ≈ 2.92 (± sampling noise)

Larger samples converge closer to theory

Extensions & Variations

Vary sample size (100, 10,000)

Simulate weighted/unfair die

Plot histogram with matplotlib

Additional Notes & Tips

Use np.random.seed(42) for reproducibility

Avoid Python loops; prefer NumPy vectorization

Exercise 2: Coin Flip Probability Estimation
Overview & Purpose
Simulate 10,000 coin flips to estimate the probability of heads and tails.

Concept Reinforcement

Discrete random variables

Empirical vs. theoretical probability

Real World Relevance

A/B testing conversion rates (success/failure)

Clinical trial outcomes

Step by Step Instructions

python
Copy
Edit
import numpy as np

np.random.seed(0)
flips = np.random.choice(['H','T'], size=10000)
p_heads = np.mean(flips == 'H')
p_tails = np.mean(flips == 'T')
print(f"P(heads): {p_heads:.3f}, P(tails): {p_tails:.3f}")
Expected Outcomes & Interpretation
~0.5 each, with fluctuations ~±0.01.

Extensions & Variations

Weighted coin (p=['H':0.3,'T':0.7])

Plot bar chart of counts

Additional Notes & Tips
Use np.random.seed(…) for reproducibility.

Exercise 3: Histogram of Die Rolls
Overview & Purpose
Visualize the distribution of the 1,000 die rolls from Exercise 1.

Concept Reinforcement

Frequency vs. probability

Data visualization basics

Real World Relevance

Sales distribution by category

Error rates by batch

Step by Step Instructions

python
Copy
Edit
import matplotlib.pyplot as plt

plt.hist(rolls, bins=range(1,8), align='left', rwidth=0.8)
plt.xlabel('Die Face')
plt.ylabel('Frequency')
plt.title('Histogram of 1,000 Die Rolls')
plt.show()
Expected Outcomes & Interpretation
Bars roughly equal height for faces 1–6.

Extensions & Variations

Normalized histogram (density=True)

Overlay theoretical PMF

Additional Notes & Tips
Ensure bins=range(1,8) to center on integer faces.

Exercise 4: Exponential Distribution Simulation
Overview & Purpose
Simulate 5,000 samples from an exponential distribution (mean = 2) and compute mean/variance.

Concept Reinforcement

Continuous random variables

Relationship between distribution parameters and statistics

Real World Relevance

Time between machine failures

Call-center interarrival times

Step by Step Instructions

python
Copy
Edit
samples = np.random.exponential(scale=2, size=5000)
print("Empirical Mean:", samples.mean())
print("Empirical Variance:", samples.var())
Expected Outcomes & Interpretation
Mean ≈ 2; variance ≈ 4.

Extensions & Variations

Change scale (mean) parameter

Plot histogram + theoretical PDF

Additional Notes & Tips
Use np.histogram or matplotlib for PDF overlay.

Exercise 5: Normal Distribution Sampling
Overview & Purpose
Draw 10,000 samples from a standard normal (mean 0, σ = 1) and verify statistics.

Concept Reinforcement

Properties of Gaussian distribution

Central Limit Theorem preview

Real World Relevance

Measurement errors

Standardized test score modeling

Step by Step Instructions

python
Copy
Edit
normals = np.random.randn(10000)
print("Mean:", normals.mean())
print("Variance:", normals.var())
Expected Outcomes & Interpretation
Mean ≈ 0, variance ≈ 1.

Extensions & Variations

Use np.random.normal(loc, scale, size)

QQ plot vs. theoretical normal

Additional Notes & Tips
Matplotlib’s plt.hist(..., density=True) for PDF shape.

Exercise 6: Sampling Distribution of the Mean
Overview & Purpose
Simulate 1,000 experiments, each of 100 die rolls, record sample means, and examine their distribution.

Concept Reinforcement

Law of Large Numbers

Sampling variability reduction

Real World Relevance

Polling averages

Quality metrics over batches

Step by Step Instructions

python
Copy
Edit
means = [np.random.randint(1,7,100).mean() for _ in range(1000)]
plt.hist(means, bins=20)
plt.title('Sampling Distribution of Die Roll Means')
plt.show()
Expected Outcomes & Interpretation
Histogram approximates normal around 3.5 with narrower spread.

Extensions & Variations

Vary experiment size (n=10, n=1000)

Compute standard error (σ/√n)

Additional Notes & Tips
List comprehensions vs. loops for clarity.

Exercise 7: Weighted Dice Simulation
Overview & Purpose
Simulate 1,000 rolls of a biased die with 
𝑃
(
6
)
=
0.5
P(6)=0.5, others equal.

Concept Reinforcement

Custom discrete distributions

Impact of bias on mean/variance

Real World Relevance

Biased processes in manufacturing

Skewed customer behavior models

Step by Step Instructions

python
Copy
Edit
faces = [1,2,3,4,5,6]
probs = [0.1]*5 + [0.5]
rolls_biased = np.random.choice(faces, size=1000, p=probs)
print("Mean:", rolls_biased.mean(), "Variance:", rolls_biased.var())
Expected Outcomes & Interpretation
Mean > 3.5, variance different from fair die.

Extensions & Variations

Tune probs for different biases

Compare histograms side by side

Additional Notes & Tips
Sum of probs must equal 1.

Exercise 8: Vector Addition & Scaling
Overview & Purpose
Demonstrate vector addition and scalar multiplication with feature vectors.

Concept Reinforcement

Vector space operations

Geometric interpretation

Real World Relevance

Combining feature influences

Scaling normalized data

Step by Step Instructions

python
Copy
Edit
v1 = np.array([2, 4, 6])
v2 = np.array([1, 3, 5])
sum_v   = v1 + v2      # vector addition
scaled_v = 0.5 * v1    # scalar multiplication
print("Sum:", sum_v)
print("Scaled:", scaled_v)
Expected Outcomes & Interpretation
Sum = [3, 7, 11]; scaled = [1, 2, 3].

Extensions & Variations

Compute dot product of sum_v and v2

Visualize vectors in 2D/3D

Additional Notes & Tips
Ensure consistent dimensions.

Exercise 9: Matrix Multiplication Demonstration
Overview & Purpose
Multiply a 2×3 matrix by a 3×2 matrix to reinforce matrix multiplication rules.

Concept Reinforcement

Shape compatibility

Summation over inner index

Real World Relevance

Transforming feature spaces

Composition of network layers

Step by Step Instructions

python
Copy
Edit
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[7,8],[9,10],[11,12]])
C = A.dot(B)
print("Result:\n", C)
Expected Outcomes & Interpretation
C = [[58, 64], [139, 154]].

Extensions & Variations

Reverse multiplication to show error

Use @ operator in Python 3.5+

Additional Notes & Tips
Check shapes via A.shape and B.shape.

Exercise 10: PCA on Toy Dataset
Overview & Purpose
Perform PCA on a small 2‑D dataset to reduce to 1‑D and visualize variance capture.

Concept Reinforcement

Eigenvectors/eigenvalues

Dimensionality reduction

Real World Relevance

Compressing image data

Feature extraction for clustering

Step by Step Instructions

python
Copy
Edit
from sklearn.decomposition import PCA
import numpy as np

X = np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0]])
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Projected data:\n", X_pca)
Expected Outcomes & Interpretation
Most variance captured in first component (≈98%).

Extensions & Variations

Plot original vs. projected points

Try n_components=2

Additional Notes & Tips
Requires scikit‑learn installation.

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

