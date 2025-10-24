# Iris Flower Classification using K-Nearest Neighbors (KNN)

This project demonstrates how to build, train, and evaluate a **K-Nearest Neighbors (KNN)** classifier on the classic **Iris dataset** using **Python** and **scikit-learn**. It also includes model training, evaluation, and visualization of accuracy for different values of *k* (number of neighbors).

---

## Dataset

- **Name:** Iris Dataset *(from scikit-learn)*  
- **Features:**  
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- **Target (Labels):**  
  - Setosa  
  - Versicolor  
  - Virginica  
- **Size:** 150 samples *(50 per species)*

---

## Dependencies

- numpy  
- matplotlib  
- scikit-learn  

Install all dependencies using:

pip install numpy matplotlib scikit-learn

## Results

### Model Performance Metrics

**Overall Accuracy:** 0.9777 (≈98%)

### Key Findings

- **Best k Value:** 5 (from hyperparameter analysis)
- **Model Robustness:** Excellent performance across all three classes
- **Generalization:** Model shows strong predictive capability on unseen data
- **Class Separation:** Clear feature boundaries, particularly for Setosa vs other species
