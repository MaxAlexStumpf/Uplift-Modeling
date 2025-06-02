import optuna
from xgboost import XGBClassifier
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def calculate_qini(y_true, uplift, treatment):
    """
    Qini calculation (did this manually, since there was a bug with the function from causalml)
    """
    df = pd.DataFrame({
        'y': y_true,
        'uplift': uplift, 
        'treatment': treatment
    }).sort_values('uplift', ascending=False).reset_index(drop=True)
    
    n = len(df)
    n_treatment = (df['treatment'] == 1).sum()
    n_control = (df['treatment'] == 0).sum()
    
    cum_n_treatment = np.cumsum(df['treatment'])
    cum_n_control = np.arange(1, n + 1) - cum_n_treatment
    
    cum_resp_treatment = np.cumsum(df['y'] * df['treatment'])
    cum_resp_control = np.cumsum(df['y'] * (1 - df['treatment']))
    
    uplift_curve = (cum_resp_treatment / n_treatment - 
                    cum_resp_control / n_control) * (cum_n_treatment + cum_n_control) / n
    
    random_curve = (cum_n_treatment / n_treatment - 
                    cum_n_control / n_control) * (cum_n_treatment + cum_n_control) / n
    
    return (uplift_curve - random_curve).mean()


def optimize_two_model(X_train_treated, y_train_treated, X_train_control, y_train_control, n_trials=50):
    """
    Optimize T - Learner
    """
    X_t_train, X_t_val, y_t_train, y_t_val = train_test_split(
        X_train_treated, y_train_treated, test_size=0.2, random_state=42
    )
    X_c_train, X_c_val, y_c_train, y_c_val = train_test_split(
        X_train_control, y_train_control, test_size=0.2, random_state=42
    )
    
    X_val = pd.concat([X_t_val, X_c_val])
    y_val = pd.concat([y_t_val, y_c_val])
    val_treatments = np.concatenate([np.ones(len(X_t_val)), np.zeros(len(X_c_val))])
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        model_treated = XGBClassifier(**params)
        model_control = XGBClassifier(**params)
        
        model_treated.fit(X_t_train, y_t_train)
        model_control.fit(X_c_train, y_c_train)
        
        prob_treated = model_treated.predict_proba(X_val)[:, 1]
        prob_control = model_control.predict_proba(X_val)[:, 1]
        uplift = prob_treated - prob_control
        
        return calculate_qini(y_val, uplift, val_treatments)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_transformation(X_train_features, Z_train, y_train, treatment_train, n_trials=50):
    indices = np.arange(len(X_train_features))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_t_train = X_train_features[train_idx]
    X_t_val = X_train_features[val_idx]
    Z_t_train = Z_train[train_idx]
    y_val = y_train[val_idx]
    treatment_val = treatment_train[val_idx]
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        model = XGBClassifier(**params)
        model.fit(X_t_train, Z_t_train)
        
        prob_z = model.predict_proba(X_t_val)[:, 1]
        uplift = 2 * prob_z - 1
        
        return calculate_qini(y_val, uplift, treatment_val)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params


def optimize_tree(X_train_features, treatment_train, y_train_array, n_trials=50):
    """
    Optimize Decision Tree
    """
    indices = np.arange(len(X_train_features))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_t_train = X_train_features[train_idx]
    X_t_val = X_train_features[val_idx]
    treatment_t_train = treatment_train[train_idx]
    treatment_t_val = treatment_train[val_idx]
    y_t_train = y_train_array[train_idx]
    y_t_val = y_train_array[val_idx]
    
    val_treatments = np.where(treatment_t_val == "treatment", 1, 0)
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 50, 170),
            'min_samples_treatment': trial.suggest_int('min_samples_treatment', 50, 150),
            'control_name': 'control'
        }
        
        tree = UpliftTreeClassifier(**params)
        tree.fit(X=X_t_train, treatment=treatment_t_train, y=y_t_train)
        
        pred = tree.predict(X_t_val)
        uplift = pred[:, 1] - pred[:, 0]
        
        return calculate_qini(y_t_val, uplift, val_treatments)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def optimize_rf(X_train_features, treatment_train, y_train_array, n_trials=50):
    """
    Optimize Random Forest
    """
    indices = np.arange(len(X_train_features))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_t_train = X_train_features[train_idx]
    X_t_val = X_train_features[val_idx]
    treatment_t_train = treatment_train[train_idx]
    treatment_t_val = treatment_train[val_idx]
    y_t_train = y_train_array[train_idx]
    y_t_val = y_train_array[val_idx]
    
    val_treatments = np.where(treatment_t_val == "treatment", 1, 0)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 50, 170),
            'min_samples_treatment': trial.suggest_int('min_samples_treatment', 50, 170),
            'control_name': 'control',
            'random_state': 42
        }
        
        rf = UpliftRandomForestClassifier(**params)
        rf.fit(X=X_t_train, treatment=treatment_t_train, y=y_t_train)
        
        uplift = rf.predict(X_t_val)
        if hasattr(uplift, 'ravel'):
            uplift = uplift.ravel()
        
        return calculate_qini(y_t_val, uplift, val_treatments)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value