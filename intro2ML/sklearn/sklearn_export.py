import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes

# 1. California Housing
cal_data = fetch_california_housing()
cal_df = pd.DataFrame(cal_data.data, columns=cal_data.feature_names)
cal_df["MedHouseVal"] = cal_data.target

cal_df.to_csv("california_housing.csv", index=False)

# 2. Diabetes
diab_data = load_diabetes()
diab_df = pd.DataFrame(diab_data.data, columns=diab_data.feature_names)
diab_df["DiseaseProgression"] = diab_data.target

diab_df.to_csv("diabetes.csv", index=False)

print("Datasets exported as 'california_housing.csv' and 'diabetes.csv'.")
