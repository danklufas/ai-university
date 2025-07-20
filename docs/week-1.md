1. Lesson Overview
Learning Objectives:
By the end of this lesson, you will be able to:
•	Describe three pivotal AI milestones and their lasting impact.
•	Define Artificial Intelligence (AI), Machine Learning (ML), and Data Science with clear examples.
•	Understand probability fundamentals—random variables, expectation, variance—and compute them by hand and in Python.
•	Grasp key linear algebra concepts—vectors, matrices, dot products, matrix multiplication—and see how they underpin AI algorithms.
•	Install and launch the required tools (Python, Jupyter Notebook, NumPy, pandas) and execute basic code.
________________________________________
2. Core Definitions
Term	Definition & Example
Artificial Intelligence (AI)	“The science and engineering of making intelligent machines, especially intelligent computer programs.” — John McCarthy, 1956
Example: A chatbot that interprets questions and crafts human like responses.	
Machine Learning (ML)	Algorithms that improve performance on tasks by learning from data rather than explicit programming.
Example: A regression model that learns to predict steel prices from historical sales.	
Data Science	Interdisciplinary practice of using statistics, programming, and domain knowledge to extract insights from data.
Example: Cleaning and visualizing e commerce logs to uncover purchasing trends.	
________________________________________
3. Concept Sections
A. AI Milestones
Excerpt (Definition Box):
Artificial Intelligence (AI) – “The science and engineering of making intelligent machines, especially intelligent computer programs.” — John McCarthy, 1956
1.	The Dartmouth Workshop (1956)
In the summer of 1956, John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon convened at Dartmouth College to explore a bold question: “Can machines be made to simulate human intelligence?” They coined the term “Artificial Intelligence” and launched a two month study to investigate how machines might “learn from experience,” “make abstractions,” and “use language.”
o	Context & Significance:
Before Dartmouth, computers were viewed largely as number crunchers. This workshop reframed them as potential thinking machines, seeding optimism that a small team could tackle “every aspect of learning or any other feature of intelligence.”
o	First Programs:
	Logic Theorist (1955): Developed by Newell & Simon, it proved theorems in symbolic logic—demonstrating that “thinking” tasks could be mechanized.
	General Problem Solver (1957): An early attempt at a universal reasoning engine.
o	Why It Matters Today:
	Cycle of Hype & AI Winters: The booms and busts following Dartmouth teach us to balance ambition with realism when evaluating modern AI breakthroughs.
	Legacy in Modern Research: Symbolic reasoning and search algorithms from this era underpin today’s knowledge graphs and constraint solving systems.
2.	Expert Systems Era (1970s–1980s)
As symbolic AI matured, expert systems emerged—rule based programs encoding human expertise as “if–then” statements.
o	Core Idea:
Encode domain knowledge in production rules:
text
CopyEdit
IF symptom = fever AND symptom = rash  
THEN suggest = measles
o	Notable Example – MYCIN (1972–1980):
	Built at Stanford, MYCIN contained ~600 rules for diagnosing bacterial infections and recommending antibiotics.
	It queried patient data (age, symptoms), applied its rule base, and in blind tests matched or outperformed human experts.
o	Strengths & Limitations:
	Strength: Transparent logic—every recommendation traces back to specific rules.
	Limitation: Required hand crafting thousands of rules and handled uncertainty poorly (no probabilistic reasoning).
o	Modern Relevance:
	Rule based approaches inform decision support in finance and healthcare.
	Today’s hybrid systems combine rules with statistical ML (e.g., regulatory checks plus model based scoring).
3.	Deep Learning Boom (2010s–Present)
The field’s third surge harnessed large datasets and GPU acceleration to train deep neural networks—models with many layers that learn hierarchical features automatically.
o	Key Breakthrough – AlexNet (2012):
	An eight layer convolutional neural network (CNN) that halved error rates on the ImageNet challenge (1.2 million labeled images, 1,000 categories).
	Employed ReLU activations, dropout regularization, and GPU based training.
o	Why Deep Learning Emerged:
1.	Data: Massive labeled datasets (images, text, speech).
2.	Compute: GPUs excel at parallel matrix operations critical for neural nets.
3.	Algorithms: Innovations like batch normalization, architectural search, and optimized backpropagation.
o	Transformative Applications:
	Computer Vision: Object detection (self driving cars), medical imaging (tumor detection).
	Natural Language Processing: Language translation, text generation (GPT style models).
	Speech & Audio: Voice assistants, real time translation.
o	Why It Matters for You:
	Modern AI frameworks (TensorFlow, PyTorch) are built around neural networks.
	This era explains why subsequent terms focus on coding deep models and leveraging pretrained architectures for rapid deployment.
________________________________________
C. Probability Basics
Excerpt (Definition Box):
Probability Theory – “The mathematical framework for quantifying uncertainty and modeling random phenomena.”
C1. Introduction
1.	What Is Chance?
o	Everyday Analogy: Flipping a coin. You know there are two sides—heads or tails—but you can’t predict which will land face up.
o	Key Idea: Probability measures how likely something is to happen, on a scale from 0 (impossible) to 1 (certain).
	Example: A fair coin has probability 0.5 of landing heads.
2.	Simple Data & Averages
o	Real World Example: Your test scores this week: 80%, 90%, 70%, 100%, 60%.
o	Mean (Average): Add them up and divide by the number of tests:
80+90+70+100+605=80%. \frac{80 + 90 + 70 + 100 + 60}{5} = 80\%.580+90+70+100+60=80%. 
o	Why It Matters: The mean gives a sense of “typical” performance.
3.	Measuring Spread (Variance)
o	Analogy: If all scores are close to 80% (say 75%, 80%, 85%), that’s low spread; if they vary widely (60%, 100%, 70%), that’s high spread.
o	Step by Step (Scores Example):
1.	Compute each score’s difference from the mean (80): e.g., 60–80 = –20.
2.	Square these differences to make them positive: (–20)² = 400.
3.	Average the squared differences: if squares are [400,100,100,400,400], mean = 280.
4.	That average (280) is the variance; its square root (≈16.7) is the standard deviation.
o	Real World Uses:
	Weather Forecasts: “There’s a 30% chance of rain” guides umbrella choices.
	Quality Control: A factory measures weight of cereal boxes; variance tells if the filling machine is consistent.
C2. Formal Definitions & Deep Dive
1.	Understanding Random Variables
A random variable XXX formalizes outcomes of random processes by assigning numeric values.
o	Discrete RV: Takes countable values (e.g., die rolls, number of returned orders).
	Example: Rolling a six sided die → X∈{1,2,3,4,5,6}X \in \{1,2,3,4,5,6\}X∈{1,2,3,4,5,6} with P(X=k)=16P(X=k)=\tfrac{1}{6}P(X=k)=61.
o	Continuous RV: Takes any value in a continuum (e.g., time between machine failures).
	Example: Time (in minutes) between software crashes might follow an exponential distribution:
f(t)=λe−λt,t≥0. f(t)=\lambda e^{-\lambda t},\quad t\ge0.f(t)=λe−λt,t≥0. 
2.	Expectation (Mean)
The expectation E[X]E[X]E[X] is the long run average if the experiment repeats infinitely.
o	Formula (Discrete):
E[X]=∑ixi P(X=xi). E[X] = \sum_{i} x_i\,P(X=x_i).E[X]=i∑xiP(X=xi). 
o	Formula (Continuous):
E[X]=∫−∞∞x f(x) dx. E[X] = \int_{-\infty}^{\infty} x\,f(x)\,dx.E[X]=∫−∞∞xf(x)dx. 
o	Worked Example (Die):
E[X]=1+2+3+4+5+66=3.5. E[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5.E[X]=61+2+3+4+5+6=3.5. 
o	Relevance: Loss functions like mean squared error minimize expected error; understanding expectation clarifies why we average squared deviations.
3.	Variance & Standard Deviation
o	Variance:
Var(X)=E[(X−E[X])2]. \mathrm{Var}(X) = E\bigl[(X - E[X])^2\bigr].Var(X)=E[(X−E[X])2]. 
o	Standard Deviation:
σ=Var(X). \sigma = \sqrt{\mathrm{Var}(X)}.σ=Var(X). 
o	Worked Example (Die):
Var(X)=(1−3.5)2+⋯+(6−3.5)26=17.56≈2.92,σ≈1.71. \mathrm{Var}(X) = \frac{(1-3.5)^2 + \dots + (6-3.5)^2}{6} = \frac{17.5}{6} \approx 2.92,\quad \sigma \approx 1.71.Var(X)=6(1−3.5)2+⋯+(6−3.5)2=617.5≈2.92,σ≈1.71. 
o	Relevance: Guides feature scaling, sets confidence intervals, and underpins uncertainty quantification in finance or anomaly detection.
4.	Why These Concepts Matter in AI
o	Model Training: Loss functions (e.g., MSE) rely on expectation of squared errors.
o	Uncertainty Quantification: Variance informs risk metrics (VaR, confidence intervals).
o	Feature Engineering: Distribution shapes dictate transformations (e.g., log scaling skewed data).
________________________________________
D. Linear Algebra Basics
Excerpt (Definition Box):
Linear Algebra – “The branch of mathematics concerned with vectors, vector spaces, and linear transformations.”
D1. Introduction
1.	Vectors as Lists
o	Analogy: A grocery list: [2 bananas, 1 loaf bread, 500 g cheese].
o	Key Idea: A vector is just a list of numbers representing “features.”
2.	Matrices as Tables
o	Analogy: A seating chart in class: rows are table numbers, columns are seat positions.
text
CopyEdit
|    | S1 | S2 | S3 |
|----|----|----|----|
| T1 | A  | B  | C  |
| T2 | D  | E  | F  |
o	Key Idea: A matrix is multiple vectors “stacked” into rows or columns.
3.	Dot Product Intuition
o	Example (Bill Splitting): You and a friend order appetizers ([3, 2] plates) and drinks ([1, 2] each). To compute total cost if plates = $5, drinks = $2:
[3,2]⋅[5,2]=3×5+2×2=15+4=$19. [3,2]\cdot[5,2] = 3\times5 + 2\times2 = 15 + 4 = \$19.[3,2]⋅[5,2]=3×5+2×2=15+4=$19. 
o	Why It Matters: Combines quantities and prices; same math as a regression prediction.
4.	Real World Matrix Use
o	Recipe scaling: A 4 serving recipe’s ingredients in a matrix, multiply by 1.5 to get 6 servings.
o	School timetable: Days × hours grid for scheduling classes.
D2. Formal Definitions & Deep Dive
1.	Vectors & Their Interpretation
A vector x∈Rn\mathbf{x}\in\mathbb{R}^nx∈Rn is an ordered list of nnn numbers representing features or data points.
o	Example:
x=[age, monthly_spend, num_orders]=[45, 320.5, 12]. \mathbf{x} = [\text{age},\,\text{monthly\_spend},\,\text{num\_orders}] = [45,\,320.5,\,12].x=[age,monthly_spend,num_orders]=[45,320.5,12]. 
2.	Matrices & Batch Operations
A matrix X∈Rm×n\mathbf{X}\in\mathbb{R}^{m\times n}X∈Rm×n stacks mmm row vectors of dimension nnn.
o	Example:
X=[45320.51223150.05⋮⋮⋮]. \mathbf{X} = \begin{bmatrix} 45 & 320.5 & 12 \\ 23 & 150.0 & 5 \\ \vdots & \vdots & \vdots \end{bmatrix}.X=4523⋮320.5150.0⋮125⋮. 
3.	Dot Product & Linear Transformations
o	Dot Product:
a⋅b=∑i=1nai bi. \mathbf{a}\cdot\mathbf{b} = \sum_{i=1}^n a_i\,b_i.a⋅b=i=1∑naibi. 
Example: [1,2,3]⋅[4,5,6]=32.[1,2,3]\cdot[4,5,6] = 32.[1,2,3]⋅[4,5,6]=32.
o	Use in AI:
	Regression: y^=w⋅x+b\hat y = \mathbf{w}\cdot\mathbf{x} + by^=w⋅x+b.
	Neural Nets: Each neuron computes z=w⋅x+bz=\mathbf{w}\cdot\mathbf{x}+bz=w⋅x+b, then applies an activation.
4.	Matrix Multiplication
C=A×B,Cij=∑k=1nAik Bkj. \mathbf{C} = \mathbf{A}\times\mathbf{B},\quad C_{ij} = \sum_{k=1}^n A_{ik}\,B_{kj}.C=A×B,Cij=k=1∑nAikBkj. 
Example: Transforming feature spaces or chaining layers in a deep network.
5.	Relevance for AI Practitioners
o	Batch Processing: GPUs and NumPy rely on vectorized matrix operations.
o	Model Introspection: Weight matrices and activation maps in CNNs are built on these operations.
o	Dimensionality Reduction: PCA uses eigenvectors/eigenvalues of covariance matrices to compress data.
________________________________________
4. Tools Installation & Setup
Windows & Mac
A. Install Python & Anaconda
1.	Navigate to https://www.anaconda.com/products/distribution
2.	Download the Python 3.x installer for your OS.
3.	Run the installer, accept defaults.
4.	Open Anaconda Navigator from your Start menu (Windows) or Applications folder (Mac).
B. Launch Jupyter Notebook
1.	In Anaconda Navigator, click Launch under Jupyter Notebook.
2.	A browser window opens showing your file system.
3.	Click New → Python 3.
4.	Rename the notebook to Week1_AI_Math.ipynb.
C. Install & Import NumPy & pandas
1.	In a notebook cell, run:
bash
CopyEdit
!conda install numpy pandas -y
2.	In the next cell, import the libraries:
python
CopyEdit
import numpy as np
import pandas as pd
________________________________________
5.	Step by Step Exercises
Exercise 1: Die Roll Simulation & Statistics (Template)
1. Exercise Overview & Purpose
•	What we’re doing: Simulate 1,000 rolls of a fair six sided die in Python to compute the empirical mean and variance.
•	Why: Reinforces theoretical vs. empirical probability, builds NumPy familiarity, and demonstrates sampling variability.
2. Concept Reinforcement
•	Probability & Random Variables
•	Expectation & Variance
•	Sampling Variability
3. Real World Relevance
•	Quality Control (defect rate simulation)
•	Risk Modeling (Monte Carlo portfolio variance)
•	Randomized Algorithms & Game Balancing
4. Step by Step Instructions
python
CopyEdit
import numpy as np

# Simulate 1,000 die rolls
rolls = np.random.randint(1, 7, size=1000)

# Compute statistics
mean_rolls = rolls.mean()
var_rolls = rolls.var()

print("Simulated Mean:    ", mean_rolls)
print("Simulated Variance:", var_rolls)
•	Notes:
o	np.random.randint(1, 7, size=1000): integers 1–6
o	.mean(), .var(): compute empirical mean & variance
5. Expected Outcomes & Interpretation
•	Mean ≈ 3.5, Variance ≈ 2.92 (± sampling noise)
•	Larger samples converge closer to theory
6. Extensions & Variations
•	Vary sample size (100, 10,000)
•	Simulate weighted/unfair die
•	Plot histogram with matplotlib
7. Additional Notes & Tips
•	Use np.random.seed(42) for reproducibility
•	Avoid Python loops; prefer NumPy vectorization
Exercise 2: Coin Flip Probability Estimation
1. Overview & Purpose
Simulate 10,000 coin flips to estimate the probability of heads and tails.
2. Concept Reinforcement
•	Discrete random variables
•	Empirical vs. theoretical probability
3. Real World Relevance
•	A/B testing conversion rates (success/failure)
•	Clinical trial outcomes
4. Step by Step Instructions
python
CopyEdit
import numpy as np

np.random.seed(0)
flips = np.random.choice(['H','T'], size=10000)
p_heads = np.mean(flips == 'H')
p_tails = np.mean(flips == 'T')
print(f"P(heads): {p_heads:.3f}, P(tails): {p_tails:.3f}")
5. Expected Outcomes & Interpretation
~0.5 each, with fluctuations ~±0.01.
6. Extensions & Variations
•	Weighted coin (p=['H':0.3,'T':0.7])
•	Plot bar chart of counts
7. Additional Notes & Tips
Use np.random.seed(…) for reproducibility.
________________________________________
Exercise 3: Histogram of Die Rolls
1. Overview & Purpose
Visualize the distribution of the 1,000 die rolls from Exercise 1.
2. Concept Reinforcement
•	Frequency vs. probability
•	Data visualization basics
3. Real World Relevance
•	Sales distribution by category
•	Error rates by batch
4. Step by Step Instructions
python
CopyEdit
import matplotlib.pyplot as plt

plt.hist(rolls, bins=range(1,8), align='left', rwidth=0.8)
plt.xlabel('Die Face')
plt.ylabel('Frequency')
plt.title('Histogram of 1,000 Die Rolls')
plt.show()
5. Expected Outcomes & Interpretation
Bars roughly equal height for faces 1–6.
6. Extensions & Variations
•	Normalized histogram (density=True)
•	Overlay theoretical PMF
7. Additional Notes & Tips
Ensure bins=range(1,8) to center on integer faces.
________________________________________
Exercise 4: Exponential Distribution Simulation
1. Overview & Purpose
Simulate 5,000 samples from an exponential distribution (mean = 2) and compute mean/variance.
2. Concept Reinforcement
•	Continuous random variables
•	Relationship between distribution parameters and statistics
3. Real World Relevance
•	Time between machine failures
•	Call-center interarrival times
4. Step by Step Instructions
python
CopyEdit
samples = np.random.exponential(scale=2, size=5000)
print("Empirical Mean:", samples.mean())
print("Empirical Variance:", samples.var())
5. Expected Outcomes & Interpretation
Mean ≈ 2; variance ≈ 4.
6. Extensions & Variations
•	Change scale (mean) parameter
•	Plot histogram + theoretical PDF
7. Additional Notes & Tips
Use np.histogram or matplotlib for PDF overlay.
________________________________________
Exercise 5: Normal Distribution Sampling
1. Overview & Purpose
Draw 10,000 samples from a standard normal (mean 0, σ = 1) and verify statistics.
2. Concept Reinforcement
•	Properties of Gaussian distribution
•	Central Limit Theorem preview
3. Real World Relevance
•	Measurement errors
•	Standardized test score modeling
4. Step by Step Instructions
python
CopyEdit
normals = np.random.randn(10000)
print("Mean:", normals.mean())
print("Variance:", normals.var())
5. Expected Outcomes & Interpretation
Mean ≈ 0, variance ≈ 1.
6. Extensions & Variations
•	Use np.random.normal(loc, scale, size)
•	QQ plot vs. theoretical normal
7. Additional Notes & Tips
Matplotlib’s plt.hist(..., density=True) for PDF shape.
________________________________________
Exercise 6: Sampling Distribution of the Mean
1. Overview & Purpose
Simulate 1,000 experiments, each of 100 die rolls, record sample means, and examine their distribution.
2. Concept Reinforcement
•	Law of Large Numbers
•	Sampling variability reduction
3. Real World Relevance
•	Polling averages
•	Quality metrics over batches
4. Step by Step Instructions
python
CopyEdit
means = [np.random.randint(1,7,100).mean() for _ in range(1000)]
plt.hist(means, bins=20)
plt.title('Sampling Distribution of Die Roll Means')
plt.show()
5. Expected Outcomes & Interpretation
Histogram approximates normal around 3.5 with narrower spread.
6. Extensions & Variations
•	Vary experiment size (n=10, n=1000)
•	Compute standard error (σ/√n)
7. Additional Notes & Tips
List comprehensions vs. loops for clarity.
________________________________________
Exercise 7: Weighted Dice Simulation
1. Overview & Purpose
Simulate 1,000 rolls of a biased die with P(6)=0.5, others equal.
2. Concept Reinforcement
•	Custom discrete distributions
•	Impact of bias on mean/variance
3. Real World Relevance
•	Biased processes in manufacturing
•	Skewed customer behavior models
4. Step by Step Instructions
python
CopyEdit
faces = [1,2,3,4,5,6]
probs = [0.1]*5 + [0.5]
rolls_biased = np.random.choice(faces, size=1000, p=probs)
print("Mean:", rolls_biased.mean(), "Variance:", rolls_biased.var())
5. Expected Outcomes & Interpretation
Mean > 3.5, variance different from fair die.
6. Extensions & Variations
•	Tune probs for different biases
•	Compare histograms side by side
7. Additional Notes & Tips
Sum of probs must equal 1.
________________________________________
Exercise 8: Vector Addition & Scaling
1. Overview & Purpose
Demonstrate vector addition and scalar multiplication with feature vectors.
2. Concept Reinforcement
•	Vector space operations
•	Geometric interpretation
3. Real World Relevance
•	Combining feature influences
•	Scaling normalized data
4. Step by Step Instructions
python
CopyEdit
v1 = np.array([2, 4, 6])
v2 = np.array([1, 3, 5])
sum_v = v1 + v2              # vector addition
scaled_v = 0.5 * v1          # scalar multiplication
print("Sum:", sum_v)
print("Scaled:", scaled_v)
5. Expected Outcomes & Interpretation
Sum = [3,7,11]; scaled = [1,2,3].
6. Extensions & Variations
•	Compute dot product of sum_v and v2
•	Visualize vectors in 2D/3D
7. Additional Notes & Tips
Ensure consistent dimensions.
________________________________________
Exercise 9: Matrix Multiplication Demonstration
1. Overview & Purpose
Multiply a 2×3 matrix by a 3×2 matrix to reinforce matrix multiplication rules.
2. Concept Reinforcement
•	Shape compatibility
•	Summation over inner index
3. Real World Relevance
•	Transforming feature spaces
•	Composition of network layers
4. Step by Step Instructions
python
CopyEdit
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[7,8],[9,10],[11,12]])
C = A.dot(B)
print("Result:\n", C)
5. Expected Outcomes & Interpretation
C = [[58,64],[139,154]].
6. Extensions & Variations
•	Reverse multiplication to show error
•	Use @ operator in Python 3.5+
7. Additional Notes & Tips
Check shapes via A.shape and B.shape.
________________________________________
Exercise 10: PCA on Toy Dataset
1. Overview & Purpose
Perform PCA on a small 2 D dataset to reduce to 1 D and visualize variance capture.
2. Concept Reinforcement
•	Eigenvectors/eigenvalues
•	Dimensionality reduction
3. Real World Relevance
•	Compressing image data
•	Feature extraction for clustering
4. Step by Step Instructions
python
CopyEdit
from sklearn.decomposition import PCA

X = np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0]])
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Projected data:\n", X_pca)
5. Expected Outcomes & Interpretation
Most variance captured in first component (≈98%).
6. Extensions & Variations
•	Plot original vs. projected points
•	Try n_components=2
7. Additional Notes & Tips
Requires scikit-learn installation.
________________________________________
6. Summary of Week 1
Throughout this first week, you have:
•	Traced AI History: From the Dartmouth Workshop to Expert Systems (MYCIN) and the Deep Learning revolution (AlexNet).
•	Built Probability Skills: Learned random variables, expectation, variance—both by hand and in Python (die rolls, coin flips, exponential and normal sampling).
•	Visualized Data: Plotted histograms, demonstrated Central Limit Theorem.
•	Handled Bias: Modeled a weighted die to see effects on distribution.
•	Applied Linear Algebra: Performed vector ops, matrix multiplication, and PCA for dimensionality reduction.
These foundational concepts and hands on exercises prepare you for more advanced AI and ML topics.
________________________________________
7. Additional Resources
Probability Fundamentals: Khan Academy “Introduction to Probability”
•	Linear Algebra Visualizations: 3Blue1Brown “Essence of Linear Algebra” series
•	Python Tutorials: Official Python documentation at python.org

