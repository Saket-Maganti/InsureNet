#%%
# 📦 Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from helper_model_evaluation import *

#%%
# 📥 Load Cleaned Dataset
df = pd.read_csv("autoinsurance_cleaned_group2.csv")

# Focus on individuals aged 65 or younger
df = df[df["age_in_years"] <= 65]

#%%
# 🧾 Select Features and Target
features = [
    'curr_ann_amt', 'age_in_years', 'home_market_value_mid', 'income',
    'has_children', 'length_of_residence', 'marital_status',
    'home_owner', 'college_degree', 'good_credit'
]

X = df[features]
y = df["Churn"]

#%%
# 🔀 Step 1: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%%
# 📊 Step 2: Normalize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
# ⚖️ Step 3: Apply SMOTE to Balance Training Set
smote = SMOTE(random_state=123)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#%%
# 🧠 Step 4: Train Logistic Regression Model
logreg_model = LogisticRegression(random_state=123)
logreg_model.fit(X_train_resampled, y_train_resampled)

#%%
# 🔍 Step 5: Make Predictions
y_pred = logreg_model.predict(X_test_scaled)

#%%
# 📈 Step 6: Evaluate the Model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
plot_confusion(y_test, y_pred)

#%%
# 📊 Step 7: Visualize Feature Distributions

# Categorical Feature Distribution by Churn
categorical_features = ['income', 'good_credit', 'has_children', 'marital_status', 'college_degree', 'home_owner']
for col in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='Churn', data=df)
    plt.title(f'Distribution of {col} by Churn')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Churn')
    plt.tight_layout()
    plt.show()

# Numerical Feature Boxplots by Churn
numerical_features = ['curr_ann_amt', 'length_of_residence', 'home_market_value_mid', 'age_in_years']
for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f'{col} by Churn')
    plt.xlabel('Churn')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

#%%
# 🧮 Step 8: Correlation Matrix

# Renaming for better labels in heatmap
correlation_cols_raw = [
    'curr_ann_amt', 'age_in_years', 'home_market_value_mid', 'income',
    'has_children', 'length_of_residence', 'marital_status',
    'home_owner', 'college_degree', 'good_credit', 'Churn'
]

correlation_cols_named = [
    'Annual Premium', 'Age', 'Home Market Value', 'Income',
    'Has Children', 'Length of Residence', 'Marital Status',
    'Home Owner', 'College Degree', 'Good Credit', 'Churn'
]

df_corr = df[correlation_cols_raw].copy()
df_corr.columns = correlation_cols_named

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
