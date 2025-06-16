# Danny Clemens
#
# train.py

''' A script that trains my Alzheimer's AI Agent '''

import pdb
import pandas as pd
import os
import xgboost as xgb # Gradient Tree (main training algorithm)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'Data')
MAINCSV = os.path.join(DATA, 'alzheimers_disease_data.csv')

def main():
    
    xtrain, xtest, ytrain, ytest = pre_process_data()
    
    model = build_model()
    
    trained_model = train_model(model, xtrain, xtest, ytrain, ytest)
    
    plot_feature_importance(trained_model)

def pre_process_data():
    ''' Prepare the CSV file for the algorithm '''
    
    df = pd.read_csv(MAINCSV)
    df.drop(columns=['PatientID', 'DoctorInCharge'], inplace=True) # Irellevant columns
    
    # x: predictors (input)
    # y: diagnosis (output)
    x = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    
    # Create the train/test split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=78)
    
    return xtrain, xtest, ytrain, ytest


def build_model():
    
    # Model: Boosted Gradient Tree
    # A decision tree of decision trees, each based off the last
    # this allows the model to learn from its mistakes efficiently
    # also allows model to handle large number of data fields
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=78)
    
    return model


def train_model(model, xtrain, xtest, ytrain, ytest):
    model.fit(xtrain, ytrain)
    preds = model.predict(xtest)
    acc = accuracy_score(ytest, preds)

    print(f"Test Accuracy: {100 * acc:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(ytest, preds))
    print("Classification Report:")
    print(classification_report(ytest, preds))

    return model


def plot_feature_importance(model):
    xgb.plot_importance(model, importance_type='gain', show_values=False)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()