import torch
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
import xgboost as xgb
from HINT.dataloader import csv_three_feature_2_dataloader
from HINT.molecule_encode import MPNN
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import HINTModel 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, precision_score, recall_score, auc, roc_curve
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


torch.manual_seed(0) 
warnings.filterwarnings("ignore")

device = torch.device("cpu")

if not os.path.exists("figure"):
    os.makedirs("figure")


DATA_FOLDER = "data"

def get_filepaths(base_name, datafolder=DATA_FOLDER):
    return {split: os.path.join(datafolder, f"{base_name}_{split}.csv") for split in ["train", "valid", "test"]}

def initialize_models(device):
    mpnn_model = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)
    icdcode2ancestor_dict = build_icdcode2ancestor_dict()
    gram_model = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
    protocol_model = Protocol_Embedding(output_dim=50, highway_num=3, device=device)
    return mpnn_model, gram_model, protocol_model

def get_embed(dataloader, molecule_encoder, disease_encoder, protocol_encoder):
    data_lists = [[], [], [], [], []]
    for data in dataloader:
        for i, item in enumerate(data):
            if i == 1:  # label
                data_lists[i].extend([j.item() for j in item])
            else:
                data_lists[i].extend(item)

    molecule_embed = molecule_encoder.forward_smiles_lst_lst(data_lists[2])
    icd_embed = disease_encoder.forward_code_lst3(data_lists[3])
    protocol_embed = protocol_encoder.forward(data_lists[4])

    return molecule_embed, icd_embed, protocol_embed

def preprocess(file, loader, molecule_encoder, disease_encoder, protocol_encoder):
    df = pd.read_csv(file)
    columns_to_drop = ['phase', 'why_stop', 'icdcodes', 'smiless', 'criteria', 'diseases', 'drugs']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    molecule_embed, icd_embed, protocol_embed = get_embed(loader, molecule_encoder, disease_encoder, protocol_encoder)

    embeddings = [molecule_embed, icd_embed, protocol_embed]
    feature_names = ['molecule', 'icd', 'protocol']
    for embed, name in zip(embeddings, feature_names):
        embed_df = pd.DataFrame(embed.detach().numpy(), columns=[f'{name}_feature_{i}' for i in range(len(embed[0]))])
        df = pd.concat([df, embed_df], axis=1)

    return df

base_name = 'phase_II'
filepaths = get_filepaths(base_name)

train_loader = csv_three_feature_2_dataloader(filepaths['train'], shuffle=True, batch_size=32)
valid_loader = csv_three_feature_2_dataloader(filepaths['valid'], shuffle=False, batch_size=32)
test_loader = csv_three_feature_2_dataloader(filepaths['test'], shuffle=False, batch_size=32)

molecule_encoder, disease_encoder, protocol_encoder = initialize_models(device)

train_df = preprocess(filepaths['train'], train_loader, molecule_encoder, disease_encoder, protocol_encoder)
valid_df = preprocess(filepaths['valid'], valid_loader, molecule_encoder, disease_encoder, protocol_encoder)
test_df = preprocess(filepaths['test'], test_loader, molecule_encoder, disease_encoder, protocol_encoder)

combined_df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
combined_df = pd.get_dummies(combined_df, columns=['status'], drop_first=True)

train_idx = range(len(train_df))
valid_idx = range(len(train_df), len(train_df) + len(valid_df))
test_idx = range(len(train_df) + len(valid_df), len(train_df) + len(valid_df) + len(test_df))

X_train, y_train = combined_df.loc[train_idx].drop(['nctid', 'label'], axis=1), combined_df.loc[train_idx]['label']
X_valid, y_valid = combined_df.loc[valid_idx].drop(['nctid', 'label'], axis=1), combined_df.loc[valid_idx]['label']
X_test, y_test = combined_df.loc[test_idx].drop(['nctid', 'label'], axis=1), combined_df.loc[test_idx]['label']

print(X_train.shape, X_valid.shape, X_test.shape)
print("\nMissing data in X_test:", X_test.isnull().sum().sort_values(ascending=False))

np.random.seed(42) 

def xgb():
    # Fit the classifier

    # Initialize XGBoost classifier with new parameters
    xgb_clf = XGBClassifier(
        objective='binary:logistic', 
        eta=0.01, # Reduced learning rate
        max_depth=3, # Reduced depth of trees
        min_child_weight=5, # Increased minimum child weight
        gamma=0.5, # Minimum loss reduction to make a further partition
        alpha=0.1, # L1 regularization
        n_estimators=1000, # Large number of boosting rounds (will rely on early stopping)
        random_state=42
    )
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

    # Extract XGBoost training results
    results = xgb_clf.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # Plot XGBoost log loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, results['validation_0']['logloss'], label='XGBoost Train')
    plt.plot(x_axis, results['validation_1']['logloss'], label='XGBoost Test') # Added validation/test results
    plt.title('XGBoost Log Loss')
    plt.legend()

    return xgb_clf


# Function to evaluate the models
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # Probability for the positive class

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    fpr, tpr, _ = roc_curve(y, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

print("\nXGBoost Evaluation:")
model = xgb() 
evaluate_model(model, X_test, y_test) 


#Set random seed for reproducibility
np.random.seed(42)

# Define the hyperparameters and their possible values
param_dist = {
    'n_estimators': [10, 50, 100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None] + list(np.arange(10, 110, 10)),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'bootstrap': [True, False]
}

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, verbose=1)

# Set up the RandomizedSearchCV
rf_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=100, scoring='accuracy', 
    verbose=2, random_state=42, n_jobs=-1
)

# Fit the model using training data
rf_search.fit(X_train, y_train)

# Use the best model from RandomizedSearchCV
best_rf = rf_search.best_estimator_

# Using the function to evaluate on the test set
print("Evaluation on Test Set:")
evaluate_model(best_rf, X_test, y_test)

# Feature Importance
feature_importance = best_rf.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Plot top 10 feature importances
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10]) 
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()  
plt.show()

# Plot confusion matrix using the predictions
y_pred = best_rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()