import optuna
from xgboost import XGBClassifier
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
import numpy as np
import pandas as pd

def calculate_qini(y_true, uplift, treatment):
    df = pd.DataFrame({
        "y": y_true,
        "uplift": uplift, 
        "treatment": treatment
    }).sort_values("uplift", ascending=False).reset_index(drop=True)
    
    n = len(df)
    n_treatment = (df["treatment"] == 1).sum()
    n_control = (df["treatment"] == 0).sum()
    
    cum_n_treatment = np.cumsum(df["treatment"])
    cum_n_control = np.arange(1, n + 1) - cum_n_treatment
    
    cum_resp_treatment = np.cumsum(df["y"] * df["treatment"])
    cum_resp_control = np.cumsum(df["y"] * (1 - df["treatment"]))
    
    uplift_curve = (cum_resp_treatment / n_treatment - 
                    cum_resp_control / n_control) * (cum_n_treatment + cum_n_control) / n
    
    random_curve = (cum_n_treatment / n_treatment - 
                    cum_n_control / n_control) * (cum_n_treatment + cum_n_control) / n
    
    return (uplift_curve - random_curve).mean()

def optimize_two_model(X_train_treated, y_train_treated, X_train_control, y_train_control, 
                       X_test_features, y_test, test_treatments, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "eval_metric": "logloss",
            "random_state": 42
        }
        
        model_treated = XGBClassifier(**params)
        model_control = XGBClassifier(**params)
        
        model_treated.fit(X_train_treated, y_train_treated)
        model_control.fit(X_train_control, y_train_control)
        
        prob_treated = model_treated.predict_proba(X_test_features)[:, 1]
        prob_control = model_control.predict_proba(X_test_features)[:, 1]
        uplift = prob_treated - prob_control
        
        return calculate_qini(y_test, uplift, test_treatments)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def optimize_transformation(X_train_features, Z_train, X_test_features, y_test, test_treatments, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "eval_metric": "logloss",
            "random_state": 42
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train_features, Z_train)
        
        prob_z = model.predict_proba(X_test_features)[:, 1]
        uplift = 2 * prob_z - 1
        
        return calculate_qini(y_test, uplift, test_treatments)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def optimize_tree(X_train_features, treatment_train, y_train_array, 
                  X_test_features, y_test_array, test_treatments, n_trials=50):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "min_samples_treatment": trial.suggest_int("min_samples_treatment", 20, 200),
            "control_name": "control"
        }
        
        tree = UpliftTreeClassifier(**params)
        tree.fit(X=X_train_features, treatment=treatment_train, y=y_train_array)
        
        pred = tree.predict(X_test_features)
        uplift = pred[:, 1] - pred[:, 0]
        
        return calculate_qini(y_test_array, uplift, test_treatments)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

def optimize_rf(X_train_df, treatment_train, y_train_array, 
                X_test_df, y_test_array, test_treatments, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "min_samples_treatment": trial.suggest_int("min_samples_treatment", 20, 200),
            "control_name": "control",
            "random_state": 42
        }
        
        rf = UpliftRandomForestClassifier(**params)
        rf.fit(X=X_train_df, treatment=treatment_train, y=y_train_array)
        
        uplift = rf.predict(X_test_df)
        if hasattr(uplift, "ravel"):
            uplift = uplift.ravel()
        
        return calculate_qini(y_test_array, uplift, test_treatments)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value