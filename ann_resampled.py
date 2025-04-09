#%%
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from helper_model_evaluation import *  # Utility functions for evaluation (confusion matrix, ROC, etc.)

#%%
# 🚀 Step 1: Load and Filter Dataset
df = pd.read_csv("autoinsurance_cleaned_group2.csv")

# Limit analysis to individuals aged 65 or below
df = df[df["age_in_years"] <= 65]

#%%
# 🧾 Step 2: Feature Selection
features = [
    "curr_ann_amt", 
    "age_in_years", 
    "home_market_value_mid", 
    "income", 
    "has_children", 
    "length_of_residence", 
    "marital_status",
    "home_owner", 
    "college_degree", 
    "good_credit",
]

X = df[features]
y = df["Churn"]

#%%
# 🔀 Step 3: Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

#%%
# 🔧 Step 4: Normalize Numerical Features using MinMaxScaler
scaler = MinMaxScaler()
cols_to_scale = [
    "curr_ann_amt", 
    "age_in_years", 
    "home_market_value_mid", 
    "income", 
    "has_children", 
    "length_of_residence", 
]

preprocessor = ColumnTransformer(
    transformers=[("minmax", scaler, cols_to_scale)],
    remainder="passthrough"
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

#%%
# 🧠 Step 5: Load Pre-trained ANN Model (trained with SMOTE)
with open("churn_fnn_0.85_0.2_ep100_resampled_0.58.pkl", "rb") as file:  
    ann_model = pickle.load(file)

#%%
# 🔁 Step 6: Apply SMOTE to Training Set
smote = SMOTE(random_state=123)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Make predictions on both sets
y_train_pred = ann_model.predict(X_train_resampled)
y_test_pred = ann_model.predict(X_test_scaled)

#%%
# 📈 Step 7: Evaluate Model Performance

# Visualize predicted probabilities
sns.displot(pd.DataFrame(y_test_pred), kde=True)
plt.title("Predicted Probabilities on Test Set")
plt.xlabel("Churn Probability")
plt.ylabel("Density")
plt.show()

# Calculate threshold based on class imbalance
churn_ratio = y_test.sum() / len(y_test)
threshold = np.percentile(y_test_pred, (1 - churn_ratio) * 100)
print(f"Suggested percentile threshold: {threshold:.3f}")

# Confusion matrix at predefined threshold
plot_confusion(y_test, y_test_pred, threshold=0.58)

# ROC curves for both training (resampled) and test sets
plot_roc("Train (Resampled)", y_train_resampled, y_train_pred, color="blue")
plot_roc("Test", y_test, y_test_pred, color="red", linestyle='--')
plt.title("ROC Curve - Train vs Test")
plt.legend(loc="lower right")
plt.show()
