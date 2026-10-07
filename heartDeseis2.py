import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('datasetForHaeratDesies.csv')

# Separate features (X) and target variable (y)
X = data.drop(columns=['target'])  # Features (all columns except 'target')
y = data['target']  # Target variable ('heart disease present or not')

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save the split datasets to CSV files
X_train.to_csv('Train_Set/features.csv', index=False)
y_train.to_csv('Train_Set/target.csv', index=False)
X_test.to_csv('Test_Set/features.csv', index=False)
y_test.to_csv('Test_Set/target.csv', index=False)

print("Data has been successfully split into training and testing sets.")
