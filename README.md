# 🚀 InsureNet: Deep Learning for Insurance Churn Prediction

## 📌 Project Overview
**InsureNet** is a predictive analytics project that uses **Artificial Neural Networks (ANN)** to forecast customer churn in the insurance sector. The objective is to proactively identify customers at risk of discontinuing their policies and provide insurers with data-driven insights to improve retention strategies.

## 🛠️ Technologies Used
- **Python** (3.8+)
- **TensorFlow** & **Keras** – Deep learning model development
- **Pandas**, **NumPy** – Data manipulation and processing
- **Scikit-learn** – Model evaluation and preprocessing
- **Seaborn**, **Matplotlib** – Data visualization

## 🧠 Model Architecture
The model is built using a **Multi-Layer Perceptron (MLP)** with:
- LeakyReLU activation functions
- Dropout regularization
- Binary Focal Loss to address class imbalance
- Optimized using Adam with a low learning rate for stability

The training incorporates both original and **SMOTE-resampled datasets** to compare model performance on balanced and imbalanced data.

## ✅ Key Features
- Data cleaning, transformation, and feature engineering
- Normalization using `MinMaxScaler` and `StandardScaler`
- Resampling using SMOTE for balanced classification
- Model evaluation using confusion matrix, classification report, ROC curve
- Visualization of churn trends and correlations across key features

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Saket-Maganti/InsureNet.git
   cd InsureNet
