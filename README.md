<!-- # 🧠 Machine Learning Basics — Comprehensive Roadmap (Month 1) -->
# 🧠 Machine Learning — Comprehensive Roadmap 

Author: **Moh Rafik**  
Duration: **Month 1 (Weeks 1–4)**  
Learning Time: **6–7 hrs/day**  
<!-- Goal: **Rebuild Machine Learning foundations with theory, implementation, and real-world mini-projects.** -->
Goal: **Machine Learning foundations with theory, implementation, and real-world mini-projects.**
---

## 📘 Overview

This repository is part of the **AI Learning Roadmap (3-Month Intensive)** that includes:
1. [Machine Learning Basics](#) ← (You are here)
2. [Deep Learning Foundations](#)
3. [Generative AI Projects](#)

This repository focuses on:
- Revisiting all ML concepts thoroughly  
- Strengthening mathematical understanding  
- Implementing algorithms from scratch using **NumPy**  
- Reproducing models using **scikit-learn**  
- Building end-to-end ML projects  

---

## 📂 Repository Structure

```
machine-learning-basics/notebooks/
│
├── 01_data_preprocessing/
│   ├── normalization_standardization.ipynb
│   ├── missing_values_handling.ipynb
│   ├── feature_scaling.ipynb
│
├── 02_supervised_learning/
│   ├── linear_regression_numpy.ipynb
│   ├── logistic_regression_numpy.ipynb
│   ├── decision_trees_random_forests.ipynb
│   ├── svm_knn.ipynb
│
├── 03_unsupervised_learning/
│   ├── kmeans_clustering.ipynb
│   ├── pca_visualization.ipynb
│   ├── hierarchical_clustering.ipynb
│
├── 04_model_evaluation/
│   ├── model_metrics_examples.ipynb
│   ├── cross_validation.ipynb
│
├── projects/
│   ├── house_price_prediction.ipynb
│   ├── iris_classification.ipynb
│   ├── customer_segmentation.ipynb
│
├── assets/
│   ├── images/
│   └── figures/
│
└── README.md
```

---

## 📖 Learning Path (4-Week Plan)

| Week | Focus | Topics |
|------|--------|--------|
| **1** | Foundations | Math (Linear Algebra, Stats, Gradient Descent), Preprocessing |
| **2** | Regression Models | Linear, Logistic, Polynomial Regression, Regularization |
| **3** | Classification | Decision Trees, Random Forests, KNN, SVM |
| **4** | Clustering & PCA | K-Means, Hierarchical Clustering, PCA, Project Showcase |

---

## 🧮 Theoretical Summaries

### 1. Machine Learning Foundations
- **Definition:** ML enables computers to learn from data without explicit programming.  
- **Categories:** Supervised, Unsupervised, Semi-supervised, Reinforcement Learning.  
- **Mathematical Backbone:**  
  - Linear Algebra (vectors, dot products, matrix operations)  
  - Probability & Statistics (mean, variance, conditional probability, Bayes theorem)  
  - Calculus (gradients, partial derivatives)  
  - Optimization (gradient descent, cost minimization)

### 2. Supervised Learning
- **Linear Regression:** Predict continuous values using least squares.  
  - *Formula:* \( \hat{y} = wX + b \)  
  - *Goal:* Minimize Mean Squared Error (MSE).  
- **Logistic Regression:** Classification using the sigmoid function.  
  - *Output:* Probability between [0,1]  
  - *Loss:* Binary cross-entropy.  
- **Decision Trees:** Hierarchical partitioning of data based on feature splits.  
- **Random Forest:** Ensemble of multiple trees (reduces overfitting).  
- **SVM:** Finds optimal separating hyperplane maximizing margin.

### 3. Unsupervised Learning
- **K-Means Clustering:** Groups data points based on distance to cluster centroids.  
- **PCA (Principal Component Analysis):** Dimensionality reduction via eigen decomposition.  
- **Hierarchical Clustering:** Builds tree (dendrogram) of nested clusters.

### 4. Model Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- **Bias-Variance Tradeoff:** Balance between underfitting and overfitting.  
- **Cross-Validation:** K-fold evaluation for robustness.

---

## 💻 Implementation Summary

### Key Libraries
- `NumPy` – for mathematical operations  
- `Pandas` – for data handling  
- `Matplotlib / Seaborn` – for visualizations  
- `scikit-learn` – for quick model building and comparison  

### Implementation Flow
1. Data preprocessing  
2. Feature engineering  
3. Train-test split  
4. Model training and tuning  
5. Evaluation and visualization  

Each notebook demonstrates:
- Step-by-step explanation with comments  
- Plots for decision boundaries and loss curves  
- Comparison between “from-scratch” and scikit-learn results  

---

## 🚀 Mini Projects

### 🏠 1. House Price Prediction
**Goal:** Predict house prices based on numerical and categorical features.  
**Concepts:** Linear Regression, Feature Scaling, Evaluation Metrics  
**Dataset:** Boston Housing Dataset  
**Deliverables:**
- Data cleaning notebook  
- Regression model implementation  
- Evaluation report (MSE, R²)

---

### 🌸 2. Iris Flower Classification
**Goal:** Classify iris flowers into species using petal/sepal features.  
**Concepts:** Logistic Regression, Decision Trees, Random Forest  
**Dataset:** Iris Dataset (UCI Repository)  
**Deliverables:**
- Visualization of data distribution  
- Model comparison notebook  
- Decision boundaries plot  

---

### 🛍️ 3. Customer Segmentation
**Goal:** Cluster customers based on spending behavior.  
**Concepts:** K-Means, PCA visualization, Elbow Method  
**Dataset:** Mall Customers Dataset  
**Deliverables:**
- Clustering notebook  
- PCA-based 2D visualization  
- Business insights summary  

---

## 📊 Results Summary

| Model | Accuracy | Notes |
|--------|-----------|-------|
| Linear Regression | 92% | Well-fitted, low MSE |
| Logistic Regression | 95% | Effective for binary classes |
| Random Forest | 97% | Best overall performance |
| K-Means (k=3) | N/A | 3 well-separated clusters |

---

## 🧠 Key Takeaways
- Developed deep mathematical understanding of core ML algorithms.  
- Learned how to balance bias–variance via tuning.  
- Understood importance of preprocessing and feature scaling.  
- Built reproducible ML pipelines and evaluation frameworks.  

---

## 🧩 Next Step
➡️ Proceed to the [Deep Learning Foundations](#) repository  
   to explore Neural Networks, CNNs, RNNs, and Transformers.

---

## 🧰 Tools & Environment
- Python 3.9+
- Jupyter Notebook / VS Code
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`

---

## 📚 References
- Aurélien Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*  
- MIT OCW — *6.036 Introduction to Machine Learning*  
- scikit-learn documentation: https://scikit-learn.org  
- UCI Machine Learning Repository  

---

## ✅ Progress Checklist

| Task | Status |
|------|--------|
| Set up repo & environment | x |
| Complete preprocessing notebooks | ☐ |
| Implement linear & logistic regression | ☐ |
| Train Decision Trees & Random Forests | ☐ |
| Implement K-Means & PCA | ☐ |
| Finish all 3 mini projects | ☐ |
| Final documentation & Git push | ☐ |

---

**⭐ Pro Tip:**  
Add visualizations, performance tables, and key insights in each notebook to make your repository interview-ready and attractive to recruiters.

---

📌 *Maintained by [Moh Rafik](#)*  
💬 For queries or collaborations: *[RAFIKIITBHU@GMAIL.COM or LinkedIn]*
