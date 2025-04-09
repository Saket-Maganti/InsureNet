#%%
# 📦 Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from helper_model_evaluation import *  # Custom plot_confusion function

#%%
# 📥 Step 1: Load Cleaned Dataset
df = pd.read_csv("autoinsurance_cleaned_group2.csv")

# Filter out individuals older than 65
df = df[df["age_in_years"] <= 65]

#%%
# 🧾 Step 2: Define Features and Target
features = [
    'curr_ann_amt',
    'age_in_years',
    'home_market_value_mid',
    'income',
    'has_children',
    'length_of_residence',
    'marital_status',
    'home_owner',
    'college_degree',
    'good_credit',
]

X = df[features]
y = df["Churn"]

#%%
# 🔀 Step 3: Split into Train/Test Sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

#%%
# ⚖️ Step 4: Balance Training Data with SMOTE
smote = SMOTE(random_state=123)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#%%
# 🌲 Step 5: Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=123)
dt_model.fit(X_train_resampled, y_train_resampled)

#%%
# 🔍 Step 6: Predict on Test Set
y_pred = dt_model.predict(X_test)

#%%
# 📊 Step 7: Evaluate the Model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plot_confusion(y_test, y_pred)
