This project includes a variety of Python scripts and assets, each serving a specific purpose in the pipeline of preparing data, training models, evaluating results, and interpreting outcomes.

requirements.txt – A list of all Python packages needed to run this project. You can install them by running pip install -r requirements.txt in your terminal.

regression.py – Trains and evaluates a Logistic Regression model using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

decisiontree.py – Similar to the regression script but uses a Decision Tree classifier, also incorporating SMOTE for better handling of imbalanced data.

ann_origsample.py – A script for training and evaluating an Artificial Neural Network (ANN) model without applying SMOTE. Useful for understanding model behavior on the original, imbalanced dataset.

ann_resampled.py – Trains and evaluates the ANN model with SMOTE applied. This version generally provides better generalization for churn prediction.

churn_fnn_0.85_0.2_ep250_age65_0.395.pkl – A pre-trained ANN model saved using Python's pickle library. This version was trained without SMOTE, with specific hyperparameters (e.g., 250 epochs and age filtering at 65).

churn_fnn_0.85_0.2_ep100_resampled_0.58.pkl – Another trained ANN model, this time using SMOTE and fewer epochs (100). Ready to be loaded and used for predictions or evaluation.

helper_model_evaluation.py – Contains helper functions that make it easier to visualize model performance, including confusion matrix and ROC curve plotting.

data_cleaning.py – A preprocessing script that handles raw data cleaning, missing value treatment, and preparation of features for modeling.

autoinsurance.csv – The raw dataset used in this project. It contains customer demographic, behavioral, and financial information. Sourced from this Kaggle dataset.
https://www.kaggle.com/datasets/merishnasuwal/auto-insurance-churn-analysis-dataset

Presentation_Capstone.pptx – A PowerPoint deck that outlines the business problem, solution approach, and key insights — suitable for showcasing in interviews or demos.




summary of the key columns in the dataset:

INDIVIDUAL_ID: Unique ID for each insurance customer.

ADDRESS_ID: ID representing the customer's primary address.

CURR_ANN_AMT: The actual amount paid by the customer in the previous year (not the policy amount).

DAYS_TENURE: Number of days the customer has been with the insurance company.

CUST_ORIG_DATE: The date the customer first joined.

AGE_IN_YEARS: Age of the individual.

LATITUDE / LONGITUDE: Geographic coordinates of the customer’s residence.

CITY, STATE, COUNTY: Location information.

HOME_MARKET_VALUE, HOME_MARKET_VALUE_MIN, HOME_MARKET_VALUE_MAX, HOME_MARKET_VALUE_MID: Estimated value of the customer’s home.

INCOME: Estimated annual household income.

HAS_CHILDREN: Boolean flag (1 = has children, 0 = no children).

LENGTH_OF_RESIDENCE: How long the customer has lived in their current home (years).

MARITAL_STATUS: Either "Married" or "Single".

HOME_OWNER: 1 if the individual owns their home, 0 otherwise.

COLLEGE_DEGREE: 1 if the customer has a college degree or higher.

GOOD_CREDIT: 1 if the customer has a FICO score greater than 630.

ACCT_SUSPD_DATE: The date when the customer account was suspended or canceled.

CHURN: Target variable — 1 indicates the customer has churned, 0 otherwise.


inspired from: https://github.com/Weidsn/Predictive_Analysis_SAIT_Capstone_Project