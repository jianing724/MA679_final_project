from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score)
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def get_M_training_data() -> pd.DataFrame:
    m18 = pd.read_csv('M18.csv')
    m19 = pd.read_csv('M19.csv')

    # m18 and m19 are training data
    train = pd.concat([m18, m19], axis=0)
    return train


def get_M_test_data() -> pd.DataFrame:
    m20 = pd.read_csv('M20.csv')
    test = m20
    return test


def get_T_training_data() -> pd.DataFrame:
    t18 = pd.read_csv('T18.csv')
    t19 = pd.read_csv('T19.csv')

    # t18 and t19 are training data
    train = pd.concat([t18, t19], axis=0)
    return train


def get_T_test_data() -> pd.DataFrame:
    t20 = pd.read_csv('T20.csv')
    test = t20
    return test


def rf(X_train, y_train, X_test):
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred


def gbc(X_train, y_train, X_test):
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    return gbc, y_pred


def run_M():
    M_train = get_M_training_data()
    M_test = get_M_test_data()

    X_train = M_train.drop(['readmit_flag'], axis=1)
    y_train = M_train['readmit_flag']
    X_test = M_test.drop(['readmit_flag'], axis=1)
    X_test = X_test[X_train.columns]
    y_test = M_test['readmit_flag']

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    rf_model, y_pred = gbc(X_train_resampled, y_train_resampled, X_test)
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)  # Changed metric to accuracy
    print(f'Accuracy: {accuracy}')  # Changed print statement

    conf_mat = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix: {conf_mat}')

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_probs)
    print("ROC AUC Score:", roc_auc)


def run_T():
    T_train = get_T_training_data()
    T_test = get_T_test_data()

    X_train = T_train.drop(['readmit_flag'], axis=1)
    y_train = T_train['readmit_flag']
    X_test = T_test.drop(['readmit_flag'], axis=1)
    X_test = X_test[X_train.columns]
    y_test = T_test['readmit_flag']

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    rf_model, y_pred = gbc(X_train_resampled, y_train_resampled, X_test)
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)  # Changed metric to accuracy
    print(f'Accuracy: {accuracy}')  # Changed print statement

    conf_mat = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix: {conf_mat}')

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_probs)
    print("ROC AUC Score:", roc_auc)


def main():
    run_M()
    run_T()


if __name__ == '__main__':
    main()
