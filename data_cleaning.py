#%%
import pandas as pd
import seaborn as sns

#%%
# 📥 Load Raw Dataset
df = pd.read_csv("autoinsurance_group2.csv")
df.info()

#%%
# 🧹 Initial Cleaning

# Remove known outlier income
df = df[df["income"] != 80372.176]

# Filter age to <= 100
df = df[df["age_in_years"] <= 100]

# Remove records with invalid birthdate
df = df[df["date_of_birth"] != "1967-07-07"]

#%%
# 🏠 Process HOME_MARKET_VALUE into numerical features

# Replace "1000000 Plus" and drop NaNs
df["home_market_value"] = df["home_market_value"].replace("1000000 Plus", "1000000 - 1000000")
hmv = df["home_market_value"].dropna()

# Split into min, max, and mid columns
df_hmv = hmv.str.split(" - ", expand=True)
df_hmv[0] = df_hmv[0].astype(int)
df_hmv[1] = df_hmv[1].astype(int) + 1
df_hmv[2] = df_hmv.mean(axis=1).astype(int)

# Insert new columns into original DataFrame
df.insert(12, "home_market_value_min", df_hmv[0])
df.insert(13, "home_market_value_max", df_hmv[1])
df.insert(14, "home_market_value_mid", df_hmv[2])

# Fill any missing mid values with the average
df["home_market_value_mid"].fillna(df["home_market_value_mid"].mean(), inplace=True)

#%%
# 💍 Encode Marital Status
df["marital_status"].replace({"Married": 1, "Single": 0}, inplace=True)

#%%
# 📊 Reorder Columns for Consistency
desired_order = [
    'individual_id', 'address_id', 'curr_ann_amt', 'days_tenure', 'cust_orig_date',
    'age_in_years', 'date_of_birth', 'latitude', 'longitude', 'city', 'state', 'county',
    'home_market_value', 'home_market_value_min', 'home_market_value_max', 'home_market_value_mid',
    'income', 'has_children', 'length_of_residence', 'marital_status',
    'home_owner', 'college_degree', 'good_credit', 'acct_suspd_date', 'Churn'
]

df = df[desired_order].reset_index(drop=True)
df.info()

#%%
# 💾 Save Cleaned Dataset
# Uncomment the line below to save the final cleaned file
# df.to_csv("autoinsurance_cleaned_group2.csv", index=False)
