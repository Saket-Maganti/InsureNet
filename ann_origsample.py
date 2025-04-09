#%%
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from helper_model_evaluation import *  # Custom evaluation utilities (confusion matrix, ROC, etc.)

#%%
# 🚀 Step 1: Load the Cleaned Dataset
df = pd.read_csv("autoinsurance_cleaned_group2.csv")

# Filter dataset to include only customers aged 65 or below
df = df[df["age_in_years"] <= 65]

#%%
# 🧾 Step 2: Define Features and Target
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
# 🔀 Step 3: Split Data into Training and Testing Sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

#%%
# 📊 Step 4: Feature Scaling with MinMaxScaler
mms = MinMaxScaler()
cols_to_scale = [
    "curr_ann_amt", 
    "age_in_years", 
    "home_market_value_mid", 
    "income", 
    "has_children", 
    "length_of_residence", 
]

ct = ColumnTransformer(
    transformers=[("minmax_scaler", mms, cols_to_scale)],
    remainder="passthrough"
)

X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

#%%
# 🧠 Step 5: Load Pre-trained ANN Model
with open("churn_fnn_0.85_0.2_ep250_age65_0.395.pkl", "rb") as file:
    fnn_model = pickle.load(file)

# Make predictions on both training and testing sets
y_train_pred = fnn_model.predict(X_train_scaled)
y_test_pred = fnn_model.predict(X_test_scaled)

#%%
# 📈 Step 6: Evaluate Model Performance

# Plot distribution of predicted probabilities
sns.displot(pd.DataFrame(y_test_pred), kde=True)
plt.title("Distribution of Predicted Churn Probabilities")
plt.xlabel("Probability")
plt.ylabel("Density")
plt.show()

# Calculate percentile threshold for classifying churn
class_1_ratio = y_test.sum() / len(y_test)
threshold_percentile = np.percentile(y_test_pred, (1 - class_1_ratio) * 100)
print(f"Suggested percentile threshold: {threshold_percentile:.3f}")

# Plot confusion matrix using a manually selected threshold
plot_confusion(y_test, y_test_pred, threshold=0.395)

# Plot ROC curves for both training and testing sets
plot_roc("Train", y_train, y_train_pred, color="blue")
plot_roc("Test", y_test, y_test_pred, color="red", linestyle='--')
plt.title("ROC Curve - Train vs Test")
plt.legend(loc="lower right")
plt.show()
