import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import os

# Load data
data = pd.read_excel('Heart_disease_cleveland_new.xlsx')

# Check if 'target' column exists
if 'target' not in data.columns:
    raise ValueError("The 'target' column is missing in the dataset. Please ensure the column exists.")

# Fill missing values with column means
data.fillna(data.mean(), inplace=True)

# Features and target separation
x = data.drop(columns=['target'])
y = data['target']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Ensure the directories for saving data exist
os.makedirs('Train_data', exist_ok=True)
os.makedirs('Test_data', exist_ok=True)

# Save train and test data with target
train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1)

train_data.to_excel('Train_data/Train_Set.xlsx', index=False)
test_data.to_excel('Test_data/Test_Set.xlsx', index=False)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize the models
log_reg_model = LogisticRegression(max_iter=1100)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True, random_state=42)

# List of models and their names
models = [log_reg_model, rf_model, svm_model]
model_names = ['Logistic Regression', 'Random Forest', 'Support Vector Machine']

# Check number of unique classes
num_classes = len(y.unique())
if num_classes > 2:
    average = 'weighted'  # For multi-class problems
else:
    average = 'binary'  # For binary classification

# Evaluate each model
for model, name in zip(models, model_names):
    print(f"Training {name}...")
    model.fit(x_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(x_test_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    f1 = f1_score(y_test, y_pred, average=average)

    # Calculate ROC-AUC if model supports it
    if hasattr(model, 'predict_proba'):
        roc_auc = roc_auc_score(y_test, model.predict_proba(x_test_scaled)[:, 1])
    else:
        roc_auc = None

    # Print evaluation results
    print(f" Results for {name}:")
    print(f" Accuracy:  {accuracy:.2f}")
    print(f" Precision: {precision:.2f}")
    print(f" Recall: {recall:.2f}")
    print(f" F1-Score: {f1:.2f}")
    if roc_auc is not None:
        print(f" ROC-AUC: {roc_auc:.2f}")
    print("-" * 40)
