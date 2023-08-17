import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm

# Load data function
def load_data(phase):
    train = pd.read_csv(f'phase_{phase}_train.csv')
    valid = pd.read_csv(f'phase_{phase}_valid.csv')
    test = pd.read_csv(f'phase_{phase}_test.csv')
    return train, valid, test

# Handling missing data function
def handle_missing(df):
    imputer = SimpleImputer(strategy='most_frequent')
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_filled

# Encoding categorical variables function
def encode_categorical(df):
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return df, label_encoders

# Train and evaluate function
def train_evaluate(phase):
    train, valid, test = load_data(phase)
    
    # Handle missing values and encode categorical variables
    train = handle_missing(train)
    valid = handle_missing(valid)
    test = handle_missing(test)

    train, label_encoders = encode_categorical(train)
    valid, _ = encode_categorical(valid)
    test, _ = encode_categorical(test)

    X_train, y_train = train.drop('label', axis=1), train['label']
    X_valid, y_valid = valid.drop('label', axis=1), valid['label']
    X_test, y_test = test.drop('label', axis=1), test['label']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Use tqdm to show progress bar for training
    for _ in tqdm(range(100), desc=f"Training Phase {phase}", ncols=100):  
        clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    y_pred_proba_test = clf.predict_proba(X_test)[:, 1]  # Get the probability of the positive class
    
    print(f'Phase {phase} Metrics:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred_test):.2f}')
    print(f'Precision: {precision_score(y_test, y_pred_test):.2f}')
    print(f'Recall: {recall_score(y_test, y_pred_test):.2f}')
    print(f'F1 Score: {f1_score(y_test, y_pred_test):.2f}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_proba_test):.2f}')
    
    # PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
    print(f'PR AUC: {auc(recall, precision):.2f}')
    print('----------------------------------')
    
    # Save the trained model
    joblib.dump(clf, f'random_forest_model_phase_{phase}.pkl')

    return clf, label_encoders

# Training for each phase
for phase in ['I', 'II', 'III']:
    train_evaluate(phase)

# Note: This assumes your label is binary. If not, some modifications will be needed.
