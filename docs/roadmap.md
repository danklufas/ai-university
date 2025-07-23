# Curriculum Roadmap

> A high‑level view of all 24 weeks. Click into each Week page for full details.

---

## Term 1 — AIU 101: Introduction to AI & ML  *(Weeks 1–6)*

???+ summary "Week 1 – History of AI & Math Foundations"
    **Module Title:** History of AI & Math Foundations  
    **Focus:** AI milestones, core definitions (AI/ML/Data Science), probability (mean/variance), linear algebra (vectors/matrices), tool setup.  
    **Outputs:** Working Python/Jupyter env, 10 math/probability exercises.  
    **Link:** [Week 1 details](week-1.md)

??? summary "Week 2 – Supervised Learning: Regression & Classification"
    - Linear vs. logistic regression (MSE, cross‑entropy)  
    - Metrics: R², accuracy, precision/recall, F1  
    - Overfitting vs. underfitting  
    **Tools:** scikit‑learn, matplotlib  
    **Deliverable:** Train & evaluate a regression + classification model

??? summary "Week 3 – Unsupervised Learning: Clustering & PCA"
    - k‑means, elbow method, hierarchical clustering  
    - PCA for dimensionality reduction, variance explained  
    - Customer segmentation case study  
    **Tools:** scikit‑learn, seaborn  
    **Deliverable:** Segment a dataset & visualize clusters

??? summary "Week 4 – Neural Network Basics"
    - Perceptron, activations, MLP architecture  
    - Backpropagation & gradient descent  
    - Regularization: dropout, weight decay  
    **Tools:** TensorFlow/Keras or PyTorch  
    **Deliverable:** Build a simple MLP on a tabular dataset

??? summary "Week 5 – Ethics, Bias & Responsible AI"
    - Bias sources, fairness metrics (demographic parity, equal opportunity)  
    - Privacy (GDPR), governance frameworks  
    **Tools:** pandas for subgroup analysis  
    **Deliverable:** Bias audit on a sample dataset

??? summary "Week 6 – Mini Project: Regression Pipeline"
    - Data loading → cleaning → splitting  
    - Hyperparameter tuning & reporting  
    **Tools:** scikit‑learn Pipelines, matplotlib  
    **Deliverable:** End‑to‑end regression report

---

## Term 2 — AIU 201: Hands‑On with Python AI Frameworks  *(Weeks 7–12)*

??? summary "Week 7 – Environment Setup & Version Control"
    - Conda vs. venv, install TF/PyTorch/scikit‑learn  
    - Jupyter best practices, Git basics  
    **Tools:** Anaconda, Git/GitHub  
    **Deliverable:** Reproducible env + first committed notebook

??? summary "Week 8 – Data Wrangling & Pipelines"
    - pandas filtering, groupby, merge  
    - Missing data/outliers, feature scaling  
    - sklearn `Pipeline`  
    **Tools:** pandas, scikit‑learn  
    **Deliverable:** Reusable data-cleaning pipeline

??? summary "Week 9 – Model Building in scikit‑learn"
    - Estimator overview, GridSearchCV  
    - Model evaluation pipeline, joblib export  
    **Tools:** scikit‑learn, joblib  
    **Deliverable:** Tuned model saved & reloaded

??? summary "Week 10 – Deep Learning with TensorFlow/Keras"
    - Sequential vs. Functional API  
    - Dense/Conv/LSTM layers, callbacks (EarlyStopping)  
    - TensorBoard monitoring  
    **Tools:** TensorFlow/Keras, TensorBoard  
    **Deliverable:** Trained DL model with tracked metrics

??? summary "Week 11 – Custom Architectures in PyTorch"
    - `nn.Module`, custom layers  
    - `Dataset`/`DataLoader`, training loop (forward/backward/step)  
    - Saving/loading `state_dict`  
    **Tools:** PyTorch  
    **Deliverable:** Custom PyTorch model & clean training script

??? summary "Week 12 – Deploying & Serving Models"
    - Export formats (SavedModel, TorchScript)  
    - Flask/FastAPI basics, Docker containerization  
    - REST endpoint design  
    **Tools:** Flask/FastAPI, Docker  
    **Deliverable:** Local API serving a model

---

## Term 3 — AIU 301: AI for Business & Personal Productivity  *(Weeks 13–18)*

??? summary "Week 13 – Marketing Automation & Predictive Lead Scoring"
    - Feature engineering (opens, clicks, demographics)  
    - Logistic vs. tree models, lift charts, ROI  
    - CRM integration  
    **Tools:** OpenAI API, LangChain, Streamlit  
    **Deliverable:** Lead scoring prototype & dashboard

??? summary "Week 14 – Inventory Forecasting"
    - SARIMA vs. Prophet vs. LSTM  
    - STL decomposition, error metrics (MAPE, MASE)  
    - Auto‑reorder alerts  
    **Tools:** Prophet, pandas, matplotlib  
    **Deliverable:** Forecast report & alert script

??? summary "Week 15 – Quality of Earnings (QoE) with Anomaly Detection"
    - QoE definition, IsolationForest, One‑Class SVM  
    - Visualizing anomalies on financial time series  
    **Tools:** scikit‑learn, Plotly  
    **Deliverable:** Anomaly report for a sample financial dataset

??? summary "Week 16 – Customer Training & Chatbot Design"
    - Instructional design for bots, intents/entities  
    - OpenAI GPT integration, fallback flows, metrics  
    **Tools:** Landbot/Chatfuel, OpenAI API  
    **Deliverable:** Prototype customer‑support chatbot

??? summary "Week 17 – Personal Productivity with AI"
    - Smart schedulers, AI writing/summarization, auto‑notes  
    - Ethics of personal data use  
    **Tools:** Microsoft Copilot, Otter.ai  
    **Deliverable:** Personal AI productivity stack configured

??? summary "Week 18 – No‑Code AI Workflows"
    - Zapier/Make triggers & actions  
    - Error handling, monitoring  
    - API connections sans code  
    **Tools:** Zapier, Make, Bubble  
    **Deliverable:** Automated workflow for a repetitive business task

---

## Term 4 — AIU 401: Capstone & Advanced Ops  *(Weeks 19–24)*

??? summary "Week 19 – Capstone Proposal & Scoping"
    - Project charter, scope, deliverables  
    - Stakeholders & success metrics  
    - Data requirements & milestones  
    **Tools:** GitHub Projects, Trello/Gantt tool  
    **Deliverable:** Approved capstone charter

??? summary "Week 20 – Data Collection & Pipelines"
    - ETL vs. ELT, REST ingestion, batch jobs  
    - Data quality checks (schema, null ratios)  
    - Airflow basics  
    **Tools:** Airflow, Python scripts  
    **Deliverable:** Automated data pipeline draft

??? summary "Week 21 – Model Development & Iteration"
    - Experiment tracking (MLflow), Optuna tuning  
    - Dataset/model versioning, CV strategies  
    **Tools:** MLflow, Optuna, sklearn/TF  
    **Deliverable:** Logged experiments & best model artifacts

??? summary "Week 22 – Deployment Architecture & Monitoring"
    - Dockerizing services, Kubernetes basics  
    - Monitoring (Prometheus/Grafana), logging best practices  
    **Tools:** Docker, k3s/minikube, Prometheus/Grafana  
    **Deliverable:** Containerized model with metrics dashboard

??? summary "Week 23 – Testing, Evaluation & ROI"
    - Unit/integration tests for ML  
    - A/B testing frameworks  
    - Measuring AI ROI  
    **Tools:** pytest, Streamlit/BI dashboards  
    **Deliverable:** Test suite + ROI report

??? summary "Week 24 – Final Presentation & Ethics Wrap‑Up"
    - Storytelling & visualization of impact  
    - Ethical reflection: bias, misuse  
    - Lessons learned & next steps  
    **Tools:** PowerPoint/Google Slides, Plotly/Dash  
    **Deliverable:** Final presentation & ethics checklist

