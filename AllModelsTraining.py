import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# files saving of models and scaler
Model_LR = 'logistic_regression_model.pkl'
Model_RF = 'random_forest_model.pkl'
Model_SVM = 'svm_model.pkl'
Scaler_Path = 'scaler.pkl'

#model existence chaking
if not os.path.exists(Model_RF) or not os.path.exists(Model_LR) or not os.path.exists(Model_SVM):
    print("Training models....")

    #load training and testing datasets
    train_data = pd.read_excel('Train_Set.xlsx')
    test_data = pd.read_excel('Test_Set.xlsx')

    # X_Features and  y_targit variable
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']

    #Features scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #save the scaler
    joblib.dump(scaler, Scaler_Path)

    #Train Logistic Regression
    LR_model = LogisticRegression(max_iter=1000, random_state = 42)
    LR_model.fit(X_train_scaled, y_train)
    joblib.dump(LR_model, Model_LR)

    #train Random forest
    RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
    RF_model.fit(X_train_scaled, y_train)
    joblib.dump(RF_model, Model_RF)

    #train support vector machine
    SVM_model = SVC(probability=True, random_state=42)
    SVM_model.fit(X_train_scaled, y_train)
    joblib.dump(SVM_model, Model_SVM)

    #Models Evaluation
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        return accuracy, precision, recall, f1, roc_auc
    #Logistc regression evaluation
    LR_matrics = evaluate_model(LR_model, X_test_scaled,y_test)
    print(f"Logistc Resgression Metrics = Accuracy:{LR_matrics[0]:.2f}, Precision: {LR_matrics[1]:.2f}, Recall: {LR_matrics[2]:.2f}, F1: {LR_matrics[3]:.2f}, ROC_AUC: {LR_matrics[4]:.2f}")

    # Random Forest evaluation
    RF_matrics = evaluate_model(RF_model, X_test_scaled, y_test)
    print(
        f"Random Forest Metrics = Accuracy:{RF_matrics[0]:.2f}, Precision: {RF_matrics[1]:.2f}, Recall: {RF_matrics[2]:.2f}, F1: {RF_matrics[3]:.2f}, ROC_AUC: {RF_matrics[4]:.2f}")

    # Logistc regression evaluation
    SVM_matrics = evaluate_model(SVM_model, X_test_scaled, y_test)
    print(
        f"Logistc Resgression metrix = Accuracy:{SVM_matrics[0]:.2f}, Precision: {SVM_matrics[1]:.2f}, Recall: {SVM_matrics[2]:.2f}, F1: {SVM_matrics[3]:.2f}, ROC_AUC: {SVM_matrics[4]:.2f}")
else:
    print("Models already trained. Loading pre-trained models...")
    LR_model = joblib.load(Model_LR)
    RF_model = joblib.load(Model_RF)
    SVM_model = joblib.load(Model_SVM)
    scaler = joblib.load(Scaler_Path)


