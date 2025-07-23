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

> **Artificial Intelligence (AI)** – “The science and engineering of making intelligent machines, especially intelligent computer programs.” — John McCarthy, 1956

???+ info "The Dartmouth Workshop (1956)"
    **What happened (kept):**  
    In summer 1956, John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon met at Dartmouth College to ask: *“Can machines simulate human intelligence?”* They coined **“Artificial Intelligence”** and proposed studying how machines might “learn from experience,” “make abstractions,” and “use language.”
    
    **Context & significance (kept):**  
    - Pre‑1956, computers = number crunchers. Dartmouth reframed them as **potential thinking machines**.  
    - Sparked optimism (and funding) that small teams could crack “every aspect of learning.”

    **First programs (kept):**  
    - **Logic Theorist (1955)** – Newell & Simon proved logic theorems with a program.  
    - **General Problem Solver (1957)** – Early universal reasoning attempt.

    !!! note "Why this still matters (kept)"
        - Understanding the **hype → disappointment → AI winters** cycle helps you stay realistic about today’s claims.  
        - Symbolic reasoning/search ideas from this era live on in **knowledge graphs** and **constraint solvers**.

    ---
    **Deeper Context**  
    - **Computing state:** Transistors were new; programming languages just emerging (FORTRAN in 1957).  
    - **Intellectual backdrop:** Cybernetics & information theory showed feedback/communication could be formalized—why not “intelligence”?  
    - **People & ideas:** McCarthy later created LISP (1958); Minsky founded MIT AI Lab; Newell & Simon argued a “physical symbol system” can generate intelligence.

    **Core Technical Ideas (plain → precise)**  
    - Treat problems as **symbol strings**; solving = **searching** through legal symbol transformations.  
    - **Heuristics** prune search trees (A* algorithm later builds on this mindset).  
    ```text
    Problem → Encode as symbols
            → Define legal operations (rules)
            → Search for a sequence of operations to reach the goal
    ```

    **Business & Societal Impact (Then vs. Now)**  
    - *Then:* Government/academic funding—no commercial AI market yet.  
    - *Now:* Symbolic engines persist in compliance, tax, scheduling, and planning software.

    **Why It Matters to Dan (CEO/Entrepreneur)**  
    - Separate **vision** from **roadmap**. Dartmouth overpromised timelines—avoid that trap.  
    - Use **hybrid systems**: rules for constraints, ML for patterns (common in finance/ops).

    **Misconceptions & Lessons**  
    - Myth: “AI began with neural nets.” → It began symbolically.  
    - Lesson: Each wave leaves **usable tools**—don’t discard “old” tech.

    **Mini Timeline Callout**  
    | Year | Event | Why It Matters |
    |------|-------|----------------|
    | 1950 | Turing’s “Imitation Game” paper | First formal test of “machine intelligence” |
    | 1955 | Logic Theorist | Proved math theorems—machines can “reason” |
    | 1956 | Dartmouth Workshop | AI term coined; field launched |
    | 1957 | General Problem Solver | Shows limits of “universal” reasoning |
    | 1958 | LISP created | Dominant AI language for decades |

    **See Also**  
    - Week 4 (Neural Network Basics) for a contrast with symbolic approaches.  
    - Week 15 (Anomaly Detection/QoE) where rules can wrap around ML for governance.

---

???+ info "Expert Systems Era (1970s–1980s)"
    **Core idea (kept):** Encode expert knowledge as **IF–THEN rules**.
    
    ```text
    IF symptom = fever AND symptom = rash
    THEN suggest = measles
    ```

    **MYCIN (1972–1980) (kept):**  
    - ~600 rules to diagnose bacterial infections & suggest antibiotics  
    - Matched/surpassed human experts in blind tests

    **Strengths vs. limits (kept):**  
    - ✅ Transparent logic (traceable to specific rules)  
    - ❌ Hard to scale (thousands of hand‑written rules), weak with uncertainty

    !!! tip "Modern relevance (kept)"
        - Rule‑based logic still used in finance/healthcare compliance.  
        - Today’s **hybrid systems**: rules for regulation + ML models for scoring.

    ---
    **Deeper Context**  
    - **Hardware shift:** Minicomputers/workstations made corporate AI feasible; dedicated LISP machines sold.  
    - **Commercialization:** DEC’s XCON configured VAX computers—saved ~$25M/year; oil, med, and manufacturing sectors experimented widely.

    **Core Technical Ideas**  
    - **Production rules + Inference engine:**  
      ```text
      Knowledge Base (rules)
      + Working Memory (facts)
      + Inference Engine (forward/backward chaining)
      = Conclusion
      ```  
      - Forward chaining: facts → conclusions  
      - Backward chaining: goal → supporting rules  
    - **Certainty factors:** MYCIN’s workaround for uncertainty (e.g., 0.6 confidence).

    **Business & Societal Impact**  
    - *Then:* Big early ROI stories, then maintenance bottlenecks → **AI winter** (late ’80s).  
    - *Now:* Business rules engines (Drools, BRMS) enforce policies & compliance.

    **Why It Matters to Dan**  
    - **Maintenance cost lesson:** Knowledge capture & upkeep is expensive—design for continuous update (MLOps mindset).  
    - **Explainability:** In regulated spaces, traceable logic is critical; combine rules + ML for auditability.

    **Misconceptions & Lessons**  
    - Myth: “Expert systems are dead.” → They evolved into modern decision engines.  
    - Lesson: Balance **transparency vs. flexibility**—all‑rules = rigid, all‑ML = opaque.

    **Mini Timeline Callout**  
    | Year | Event | Why It Matters |
    |------|-------|----------------|
    | 1972 | MYCIN begins | First impactful medical expert system |
    | 1979 | XCON deployed at DEC | Massive real-world cost savings |
    | 1984 | Japan’s 5th Gen Project | National bet on symbolic AI |
    | 1987 | AI Winter hits | Funding collapses after hype |
    | 1990 | Probabilistic models rise | Bayes nets mark shift to statistical AI |

    **See Also**  
    - Week 5 (Ethics/Bias): rules help enforce fairness policies.  
    - Deployment weeks: wrap ML predictions with business rules.

---

???+ info "Deep Learning Boom (2010s–Present)"
    **Key breakthrough – AlexNet (2012) (kept):**  
    - 8‑layer CNN, cut ImageNet error rate in half (1.2M images, 1,000 classes)  
    - Used ReLU, dropout, and **GPU training**.

    **Why deep learning emerged (kept):**  
    1. **Data:** Huge labeled datasets (images, text, speech)  
    2. **Compute:** GPUs = fast parallel matrix ops  
    3. **Algorithms:** Batch norm, better backprop, new architectures

    **Transformative apps (kept):**  
    - Computer vision: self‑driving cars, medical imaging  
    - NLP: translation, GPT‑style generation  
    - Speech: voice assistants, real‑time translation

    !!! success "Why this matters for you (kept)"
        - Modern frameworks (TensorFlow, PyTorch) are built around neural nets.  
        - Explains why later terms focus on coding deep models & leveraging **pretrained architectures** quickly.

    ---
    **Deeper Context**  
    - **Before 2012:** Neural nets unfashionable; SVMs/ensembles dominated. ImageNet (2009) provided the benchmark that changed that.  
    - **GPU/CUDA era:** NVIDIA CUDA (2007) let researchers use gaming GPUs for matrix math cheaply.  
    - **Open-source wave:** Theano/Torch → TensorFlow (2015) → PyTorch (2016) democratized DL.

    **Core Technical Ideas**  
    - **Representation learning:** Networks learn features automatically vs. manual engineering.  
    - **Backprop + Optimizers:** Adam/RMSProp etc. speed convergence.  
    - **Regularization:** Dropout, data augmentation reduce overfitting.  
    - **Transfer learning:** Fine‑tune large pretrained models for your niche.  
      ```python
      # Pseudocode: fine-tune a pretrained CNN
      base = load_pretrained_model("resnet50", weights="imagenet")
      freeze_layers(base, up_to="layer3")
      new_head = Dense(1, activation="sigmoid")(base.output)
      model = Model(inputs=base.input, outputs=new_head)
      model.compile(optimizer="adam", loss="binary_crossentropy")
      model.fit(my_data, my_labels)
      ```

    **Business & Societal Impact**  
    - *Then:* Explosion of startups (vision, NLP, speech).  
    - *Now:* Foundation models/LLMs power copilots, automations, creative tools.  
    - *Risks:* Opaqueness, bias, IP, energy costs → regulation & governance needed.

    **Why It Matters to Dan**  
    - **Speed to value:** APIs + pretrained nets = rapid prototypes.  
    - **Data moat:** High-quality labeled data becomes a competitive advantage.  
    - **Build vs. Buy:** Decide between API use (fast, less control) and in-house models (costly, differentiated).

    **Misconceptions & Lessons**  
    - Myth: “Deep learning is always best.” → For tabular biz data, simpler models often win.  
    - Lesson: Start simple; escalate complexity when ROI demands it.

    **Mini Timeline Callout**  
    | Year | Event | Why It Matters |
    |------|-------|----------------|
    | 2006 | “Deep Learning” term revived (Hinton et al.) | Layer-wise pretraining shows deep nets can work |
    | 2009 | ImageNet released | Benchmark catalyzes vision progress |
    | 2012 | AlexNet wins ImageNet | GPU CNN breakthrough |
    | 2015 | TensorFlow open-sourced | Industrial-grade DL tooling |
    | 2017 | Transformer paper | Base of modern LLMs (GPT, etc.) |
    | 2020+| GPT‑3, Stable Diffusion | Generative AI mainstreams |

    **See Also**  
    - Week 4 (Neural Net Basics) & Week 10 (TensorFlow/Keras) for hands-on DL.  
    - Term 3 apps (marketing, inventory, QoE) where transfer learning/LLMs drive ROI.

---

!!! tip "Pattern to Remember"
    **Symbolic → Rule‑Based → Deep Learning**  
    1. Each wave adds new capabilities.  
    2. Each brings new limitations.  
    3. Each leaves tools you can still exploit.  
    Your edge = knowing **when to mix them** for real business value.


### B. Probability Basics

!!! abstract "Definition"
    **Probability Theory** – “The mathematical framework for quantifying uncertainty and modeling random phenomena.”

#### B1. Gentle Introduction 

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

#### B2. Formal Definitions & Deep Dive

???+ info "Random Variables (Discrete & Continuous)"
    **Plain-English idea:** A *random variable* (RV) is just a number that comes from a random process.  
    - **Discrete RV:** Jumps between separate values you can list or count. *Example:* A die roll → {1,2,3,4,5,6}.  
    - **Continuous RV:** Can take any value in a range. *Example:* Time between website sign-ups (could be 3.1 min, 3.11 min, …).

    **Formal bits:**  
    - Discrete RV has a **Probability Mass Function (PMF)**: `P(X = x_i)` gives the probability of each value.  
    - Continuous RV has a **Probability Density Function (PDF)**: `f(x)` where probabilities come from areas: `P(a ≤ X ≤ b) = ∫_a^b f(x) dx`.  
    - **CDF (Cumulative Distribution Function):** `F(x) = P(X ≤ x)` (works for both types).

    **Quick table:**

    | Type       | Notation example | How you get probability                  | Typical example              |
    |------------|-------------------|-------------------------------------------|------------------------------|
    | Discrete   | `P(X = k)`        | Direct from PMF                           | Die face, number of returns  |
    | Continuous | `P(a ≤ X ≤ b)`    | Integrate PDF from `a` to `b`             | Time until failure, height   |

    **Worked examples:**
    - *Discrete:* Fair die ⇒ `P(X = k) = 1/6` for k = 1…6.  
    - *Continuous:* Exponential with rate λ=0.5 (`mean = 2`) ⇒ `f(t)=0.5 e^{-0.5 t}`, t ≥ 0.

    ```python
    # Simulate both kinds quickly
    import numpy as np

    # Discrete die
    rolls = np.random.randint(1, 7, size=10_000)

    # Continuous exponential (mean=2 -> scale=2)
    exp_samples = np.random.exponential(scale=2, size=10_000)
    ```

    !!! tip "Why you care (AI/business)"
        - Choosing **loss functions** or **evaluation metrics** often assumes a distribution (Gaussian errors, Poisson counts, etc.).  
        - In SaaS ops, “time between failures” or “days between churn events” are continuous RVs; “number of support tickets” is discrete.

---

???+ info "Expectation (Mean)"
    **Plain-English idea:** The *expectation* is the long-run average you’d get if you could repeat the random experiment forever.

    **Formulas:**  
    - **Discrete:** `E[X] = Σ x_i · P(X = x_i)`  
    - **Continuous:** `E[X] = ∫_{-∞}^{∞} x · f(x) dx`

    **Linearity of expectation:** `E[aX + bY] = aE[X] + bE[Y]` (no independence needed). This is why averages of many variables are easy to reason about.

    **Worked example (discrete die):**  
    `E[X] = (1+2+3+4+5+6)/6 = 3.5`

    ```python
    import numpy as np
    rolls = np.random.randint(1, 7, size=1000)
    empirical_mean = rolls.mean()
    print(empirical_mean)  # ~3.5
    ```

    !!! note "Where it shows up in ML"
        - **Mean Squared Error (MSE)** = expected squared difference between prediction and truth.  
        - **Baseline models**: predicting the average is often the simplest benchmark.

    !!! example "Business angle"
        - Average order value, average time-to-resolution, expected downtime—these are all expectations that guide KPIs and decisions.

---

???+ info "Variance & Standard Deviation"
    **Plain-English idea:** Variance measures how *spread out* values are around the mean. Standard deviation (σ) is the square root of variance—same units as the data.

    **Formulas:**  
    - `Var(X) = E[(X - E[X])^2]`  
    - `σ = √Var(X)`

    **Shortcut (discrete):** `Var(X) = E[X^2] - (E[X])^2`  
    (Often easier to compute.)

    **Worked example (die):**  
    1. `E[X] = 3.5`  
    2. `E[X^2] = (1^2+2^2+...+6^2)/6 = 91/6 ≈ 15.17`  
    3. `Var(X) = 15.17 - (3.5)^2 = 15.17 - 12.25 ≈ 2.92`  
       `σ ≈ 1.71`

    ```python
    import numpy as np
    rolls = np.random.randint(1, 7, size=1000)
    var_emp = rolls.var()          # population variance by default
    std_emp = rolls.std()
    print(var_emp, std_emp)
    ```

    !!! warning "Population vs. sample formulas"
        - NumPy’s default `var()` uses the population formula (divide by N).  
        - For *sample* variance, use `ddof=1`: `rolls.var(ddof=1)` (divide by N-1).

    !!! tip "Why you care (AI/business)"
        - **Risk & uncertainty:** Higher variance = higher risk (finance, ops).  
        - **Feature engineering:** Highly skewed or high-variance features often need scaling or transformation (log, z-score).  
        - **Model confidence:** Prediction intervals depend on estimated variance.

---



### C. Linear Algebra Basics

> **Linear Algebra** – “The branch of mathematics concerned with vectors, vector spaces, and linear transformations.”

#### D1. Gentle Introduction

???+ info "Vectors as Lists (Plain-Language Start)"
    **Think grocery list:** `[2 bananas, 1 loaf bread, 500 g cheese]` → a vector is just an ordered list of numbers.  
    In AI, those numbers are usually **features** about something (a customer, a product, a pixel).

    ```text
    Customer vector x = [age, monthly_spend, num_orders] = [45, 320.5, 12]
    ```

    **Why it matters:** Every row in your dataset (spreadsheet) is a vector. Models read and manipulate these vectors constantly.

---

???+ info "Matrices as Tables"
    **Think spreadsheet or seating chart:** Rows × Columns.

    |    | S1 | S2 | S3 |
    |----|----|----|----|
    | T1 | A  | B  | C  |
    | T2 | D  | E  | F  |

    A **matrix** stacks many vectors. If you have 100 customers, each with 3 features, you can store them as a 100×3 matrix.

    ```python
    import numpy as np
    X = np.array([
        [45, 320.5, 12],
        [23, 150.0,  5],
        # ...
    ])
    print(X.shape)  # (100, 3) for 100 rows, 3 columns
    ```

    **Why it matters:** Most ML libraries assume your training data is in one big matrix `X` (rows = samples, columns = features).

---

???+ info "Dot Product Intuition"
    **Bill splitting analogy:**  
    Quantities `[3 appetizers, 2 drinks]` • Prices `[5, 2]` → Total cost = `3*5 + 2*2 = 19`.

    That’s the **dot product**.

    ```text
    [3, 2] · [5, 2] = 15 + 4 = 19
    ```

    **AI link:** A linear regression prediction is a dot product of weights and features:  
    \[
    \hat{y} = \mathbf{w} \cdot \mathbf{x} + b
    \]

---

???+ info "Real-World Matrix Use"
    - **Recipe scaling:** Multiply ingredient matrix by 1.5 to feed more people.  
    - **Scheduling grid:** Matrix of days × hours helps visualize time slots.  
    - **Image data:** A color image is height × width × 3 (RGB channels)—essentially a stack of matrices.

---

#### D2. Formal Definitions & Deep Dive

##### D2.1 From Lists to Arrows (Extra-Gentle Bridge)

???+ info "Vectors as arrows you can move around"
    - **Picture it:** A vector is an **arrow** — it has a length (how big) and a direction (which way).  
      You can slide it around the page without changing it (only length + direction matter).
    - **Add arrows = add effects:** Put one arrow’s tail on the other’s head: the new arrow is the sum.  
      This is the same as adding two feature lists in code.
    - **Stretch/Shrink arrows:** Multiply by 2 makes it twice as long (stronger effect); multiply by −1 flips direction (opposite effect).

    ```python
    import numpy as np
    a = np.array([2, 1])     # arrow 1
    b = np.array([1, 3])     # arrow 2
    print(a + b)             # [3 4]
    print(-1 * a)            # [-2 -1]
    ```

    **Real-world feelers:**
    - **Marketing metrics combo:** “Email opens” + “Ad clicks” vectors → one bigger “engagement” vector.  
    - **Portfolio weights:** A weight vector `[0.5, 0.3, 0.2]` says “50% stock A, 30% stock B, 20% stock C.”

---

##### D2.2 Formal Definitions (Yes, This Gets Mathy)

!!! note "Heads up!"
    Don’t stress if this feels heavy. You **do not** need to memorize every axiom.  
    The goal: recognize the terminology when you see it in books/docs and know roughly what it means.

???+ info "Vector space (over the real numbers)"
    A **vector space** \(V\) over \(\mathbb{R}\) is a set of “things” (vectors) where you can:
    1. **Add** any two vectors and stay in the set.
    2. **Multiply** any vector by a real number (scalar) and stay in the set.

    More formally, it satisfies a bunch of rules (axioms) like:  
    - Addition is commutative & associative.  
    - There’s a **zero vector** (acts like 0).  
    - Every vector has an additive inverse (acts like −v).  
    - Scalar multiplication distributes over vector addition, etc.

    **Basis & dimension:**  
    - A **basis** is a minimal set of vectors that can build every other vector (by addition & scaling).  
    - The **dimension** is how many vectors are in a basis.  
      Example: In ordinary 3D space, any vector can be made from `[1,0,0]`, `[0,1,0]`, `[0,0,1]`, so dimension = 3.

    **Subspace:**  
    A smaller vector space inside a bigger one.  
    Example: All 2D points of the form `[x, 2x]` is a line through the origin — it’s closed under add/scale ⇒ a subspace.

    | Concept      | Plain Words                           | Why It Shows Up in ML/AI                           |
    |--------------|---------------------------------------|-----------------------------------------------------|
    | Vector space | A safe sandbox for add/scale          | We add gradients, scale loss terms, etc.            |
    | Basis        | Minimal “building blocks”              | PCA finds a new basis to compress data              |
    | Dimension    | How many numbers you need to describe | “High-dimensional data” = many features             |
    | Subspace     | Smaller sandbox inside the big one     | Feature constraints (“sum to 1” weights) form one   |

---

##### D2.3 How ML & Business Actually Use This (No Scary Math)

???+ info "Spreadsheet view → Matrix view"
    - Your CSV file with rows = customers and columns = features is a **matrix** `X`.  
    - A column (one feature across all customers) is a **vector**.  
    - A model’s weights are another vector `w`. Prediction is basically:  
      \[
        \text{prediction} = X \cdot w
      \]  
      (matrix × vector = new vector of predictions)

    ```python
    import numpy as np
    X = np.array([[45, 320.5, 12],
                  [23, 150.0,  5]])     # 2 customers × 3 features
    w = np.array([0.02, 0.001, 0.5])    # weights
    preds = X.dot(w)                    # predicted score/value
    print(preds)
    ```

    **Real-world examples:**
    - **Lead scoring:** Combine features (opens, clicks, revenue) with weights to get a score.  
    - **Inventory forecasting:** Multiply a matrix of past sales by learned coefficients to get tomorrow’s demand.  
    - **Chatbot embeddings:** Words/sentences live as long vectors; comparing them (dot product) tells how similar meanings are.

    !!! tip "GPU magic = fast matrix math"
        Training deep nets is mostly giant matrix multiplications. GPUs are great at doing many of these in parallel.

---

##### D2.M Matrices, Shape & Transpose

???+ info "From Spreadsheet to Shape (Extra‑Gentle Bridge)"
    - **Matrix = big spreadsheet of numbers.** Rows = things/people/items. Columns = features about them.  
    - Shape tells you “how many rows by how many columns”: `(rows, columns)`.  
    - **Transpose** just flips rows ↔ columns. Think of turning the sheet on its side.

    ```python
    import numpy as np
    X = np.array([[45, 320.5, 12],
                  [23, 150.0,  5]])
    print(X.shape)   # (2, 3): 2 rows, 3 columns
    print(X.T.shape) # (3, 2): transposed
    ```

    **Real‑world angle:**  
    - A customer table (rows=customers, cols=metrics) is your `X`.  
    - Transpose shows each feature as a vector of length “number of customers”.

!!! note "Formal corner (OK to skim!)"
    - A matrix \( \mathbf{X} \in \mathbb{R}^{m \times n} \) is an array with \(m\) rows and \(n\) columns.  
    - The **transpose** \( \mathbf{X}^T \in \mathbb{R}^{n \times m} \) satisfies \( (\mathbf{X}^T)_{ij} = \mathbf{X}_{ji} \).  
    - Rows are often denoted \( \mathbf{x}_i^T \) (row vectors), columns \( \mathbf{x}^{(j)} \) (column vectors).

???+ info "How ML/Business Uses Shapes"
    - Libraries assume: `X.shape = (num_samples, num_features)`.  
    - If shapes don’t align, your code errors (or silently gives wrong math).  
    - **ETL sanity check:** a sudden shape change can mean broken data ingestion.
---

##### D2.D Dot Product, Norms & Geometric View

???+ info "Extra‑Gentle: ‘How aligned are two arrows?’"
    - **Dot product** measures how much two vectors “point the same way.”  
      If they point in opposite directions, the dot is negative; if perpendicular, ~0.

    ```python
    import numpy as np
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(a.dot(b))              # 32
    print(np.linalg.norm(a))     # length of a
    ```

    - **Norm** = length of the arrow (vector). Like measuring the “size” of an effect.

    **Plain examples:**  
    - Product pricing: quantities · prices = total bill (dot product).  
    - “Similarity” score between two customer profiles = dot product of their feature vectors.

!!! note "Formal corner (breathe… you’re fine)"
    - Dot product: \( \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i \).  
    - Norm (Euclidean): \( \|\mathbf{a}\| = \sqrt{\mathbf{a} \cdot \mathbf{a}} \).  
    - Angle relation: \( \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta \).

???+ info "ML/AI Tie‑ins (Simple View)"
    - **Regression prediction:** \( \hat{y} = \mathbf{w} \cdot \mathbf{x} + b \).  
    - **Embedding similarity (NLP, Recommenders):** cosine similarity = normalized dot product.  
    - **Gradient steps:** length (norm) of gradient controls learning rate effect.
---

##### D2.MM Matrix Multiplication (What Really Happens)

???+ info "Extra‑Gentle: Combine many dots at once"
    - Matrix multiplication is just doing **lots of dot products**.  
      Each output cell = dot(row from A, column from B).

    ```python
    import numpy as np
    A = np.array([[1,2,3],
                  [4,5,6]])        # 2x3
    B = np.array([[7,8],
                  [9,10],
                  [11,12]])        # 3x2
    C = A.dot(B)                    # 2x2 result
    print(C)
    # [[ 58  64]
    #  [139 154]]
    ```

    - **Shape rule:** (m×n) · (n×p) → (m×p). The inside numbers must match.

!!! note "Formal corner (notation time)"
    - \( C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj} \)  
    - It’s associative: \( (AB)C = A(BC) \).  
    - Not commutative: \( AB \neq BA \) in general.

???+ info "Why ML cares (plain English)"
    - A neural network layer is: `output = input @ weights + bias`.  
    - GPUs are optimized to do **huge matrix multiplications** fast → deep learning is feasible.  
    - Batch predictions: one matrix multiply gives predictions for thousands of rows at once.
---

##### D2.I Identity, Inverse & Rank (Quick Glimpse)

???+ info "Gentle Pass"
    - **Identity matrix \(I\):** like multiplying a number by 1 — it leaves things unchanged.  
    - **Inverse \(A^{-1}\):** the matrix that “undoes” \(A\) (if it exists).  
    - **Rank:** how many truly independent columns/rows you have (redundant columns lower rank).

!!! note "Formal corner (skim ok!)"
    - \( AI = IA = A \).  
    - \( A^{-1} A = AA^{-1} = I \) (only for invertible/square matrices).  
    - Rank is the dimension of the column space (or row space). Full rank ⇒ columns are linearly independent.

???+ info "Simple Business/ML Uses"
    - **Identity** appears in regularization terms (e.g., \( \lambda I \) in ridge regression).  
    - **Non-invertible (singular) matrix:** means features are perfectly correlated → model can’t find unique weights.  
    - **Rank checks** help detect multicollinearity before training.


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

## 6. Week 1 Summary & What You Can Now Do

!!! success "You can now…"
    - **Explain key AI milestones**: Dartmouth (1956), Expert Systems (1970s–80s), Deep Learning boom (2010s–present) and why each wave mattered.  
    - **Use probability concepts** (random variables, mean, variance) and verify them empirically in Python.  
    - **Work with basic linear algebra objects** (vectors, matrices, dot products, matrix multiplication) and see how they power ML models.  
    - **Install and run core tools** (Anaconda, Jupyter, NumPy, pandas) to explore data and math interactively.

### A. AI History in One Breath
- **Dartmouth Workshop (1956):** coined “AI”; optimism about simulating intelligence.  
  *Lesson:* Ambition vs. realism—avoid hype traps.  
- **Expert Systems (’70s–’80s):** rule-based IF–THEN logic (e.g., MYCIN).  
  *Lesson:* Transparency is great, but brittle without probabilities.  
- **Deep Learning (2010s–):** data + GPUs + better algorithms (AlexNet etc.) → breakthroughs.  
  *Lesson:* Modern toolkits (TensorFlow/PyTorch) center on neural nets.

### B. Probability: From Intuition to Math
- **Random Variables:** discrete (die, coin), continuous (time-to-failure).  
- **Expectation & Variance:** long-run average and spread—computed by hand and via NumPy.  
- **Why it matters:** Loss functions, risk estimation, feature engineering all use these ideas.  
- **Real world ties:** A/B tests, forecasting demand spikes, Monte Carlo risk simulations.

!!! note "Anchor formulas"
    - Discrete mean: \\(E[X] = \sum x_i P(X=x_i)\\)  
    - Variance: \\(\mathrm{Var}(X) = E[(X - E[X])^2]\\)

### C. Linear Algebra: The Language of ML
- **Vectors:** feature lists (e.g., `[age, spend, orders]`).  
- **Matrices:** batches of vectors; all your data at once.  
- **Dot Product:** core of regression and neuron activations.  
- **Matrix Multiplication:** chaining transformations (layers) in neural nets.  
- **Why it matters:** Speed (GPU vectorization), interpretability (weights are matrices), compression (PCA).

### D. Tools & Workflow Locked In
- **Anaconda & Jupyter:** you spun up a notebook and ran code.  
- **NumPy/pandas:** you handled arrays, stats, and basic data manipulation.  
- **Workflow habits:** preview locally (`mkdocs serve`), commit/push, deploy (`gh-deploy`).

### E. Exercises Recap (What each taught you)
| Exercise | Core Idea | Concept Reinforced | Real‑World Parallel |
|---|---|---|---|
| 1. Die roll stats | Empirical vs. theoretical | Mean, variance, sampling noise | QC sampling, Monte Carlo |
| 2. Coin flips | Bernoulli trials | Discrete RVs, proportions | A/B tests, pass/fail outcomes |
| 3. Histogram | Visual distributions | Frequency vs. prob., plotting | Category sales distributions |
| 4. Exponential sim | Time-to-event model | Continuous RVs, λ & scale | Failure rates, wait times |
| 5. Normal samples | Gaussian basics | CLT preview, z-scores | Measurement error, scores |
| 6. Sampling means | Distribution of means | Law of Large Numbers | Polling averages, batch metrics |
| 7. Weighted die | Biased distributions | Shifted mean/var, custom PMFs | Skewed demand, unfair odds |
| 8. Vector ops | Add/scale vectors | Vector spaces | Feature scaling, weight tuning |
| 9. Matrix multiply | Shapes & transforms | Linear maps, dot sums | NN layers, feature transforms |
|10. PCA toy data | Dimensionality reduction | Eigenvectors/variance explained | Data compression, preprocessing |



## 7. Additional Resources

!!! tip "Probability & Statistics"
- **Khan Academy – Introduction to Probability**  
  Beginner‑friendly videos and practice problems.  
  <https://www.khanacademy.org/math/statistics-probability/probability-library>
- **Seeing Theory (Brown University)**  
  Interactive visual explanations of probability concepts.  
  <https://seeing-theory.brown.edu/>

!!! tip "Linear Algebra"
- **3Blue1Brown – *Essence of Linear Algebra*** (YouTube series)  
  Beautiful visuals for vectors, matrices, dot products, eigenvectors.  
  <https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr>
- **Khan Academy – Linear Algebra**  
  Step‑by‑step lessons with exercises.  
  <https://www.khanacademy.org/math/linear-algebra>

!!! tip "Python, NumPy & pandas"
- **Official Python Docs** – Syntax, stdlib, tutorials.  
  <https://docs.python.org/3/>
- **NumPy User Guide** – Arrays, broadcasting, linear algebra.  
  <https://numpy.org/doc/stable/user/>
- **pandas Getting Started** – DataFrames, cleaning, transforms.  
  <https://pandas.pydata.org/docs/getting_started/index.html>

!!! note "AI History & Overviews"
- **“Competing in the Age of AI” (Iansiti & Lakhani) – Ch. 1–3** *(already on your list)*  
- **“Deep Learning” (Goodfellow, Bengio, Courville) – Intro & Ch. 6–7** (free online)  
  <https://www.deeplearningbook.org/>

!!! info "Visualization & Math Intuition"
- **Matplotlib Gallery** – Quick plot recipes.  
  <https://matplotlib.org/stable/gallery/index.html>
- **Desmos Graphing Calculator** – Fast function plots and geometry.  
  <https://www.desmos.com/calculator>

!!! success "Bonus Cheat Sheets"
- **Markdown Syntax Cheat Sheet** (for editing your site)  
  <https://www.markdownguide.org/cheat-sheet/>
- **NumPy/Pandas one‑pagers** (various printable PDFs)  
  <https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf>  
  <https://www.dataquest.io/blog/numpy-cheat-sheet/>


