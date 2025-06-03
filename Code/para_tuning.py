import optuna
from xgboost import XGBClassifier
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


def calculate_uplift_auc(y_true, uplift, treatment):
    df = pd.DataFrame({'y': y_true, 'uplift': uplift, 'treatment': treatment})
    df = df.sort_values('uplift', ascending=False).reset_index(drop=True)
    n = len(df)
    n_t = (df['treatment'] == 1).sum()
    n_c = n - n_t
    cum_t = np.cumsum(df['treatment'])
    cum_c = np.arange(1, n + 1) - cum_t
    cum_resp_t = np.cumsum(df['y'] * df['treatment'])
    cum_resp_c = np.cumsum(df['y'] * (1 - df['treatment']))
    uplift_curve = (cum_resp_t / n_t - cum_resp_c / n_c) * (cum_t + cum_c) / n
    random_curve = (cum_t / n_t - cum_c / n_c) * (cum_t + cum_c) / n
    diff = uplift_curve - random_curve
    return np.trapz(diff, dx=1 / n)


def _joint_stratum(t, y):
    return t.astype(int) * 2 + y.astype(int)


def _to_numeric_treatment(t):
    if np.issubdtype(t.dtype, np.number):
        return t.astype(int)
    return (t == 'treatment').astype(int)


sampler = optuna.samplers.TPESampler(seed=42)



def optimize_two_model(X_train_treated, y_train_treated, X_train_control, y_train_control, n_trials=50):
    X_full = pd.concat([X_train_treated, X_train_control])
    y_full = pd.concat([y_train_treated, y_train_control])
    t_full = np.concatenate([np.ones(len(X_train_treated)), np.zeros(len(X_train_control))])
    joint = _joint_stratum(t_full, y_full.values)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_full, joint):
            X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
            y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
            t_tr, t_val = t_full[tr_idx], t_full[val_idx]
            mask = t_tr == 1
            m_t = XGBClassifier(**params)
            m_c = XGBClassifier(**params)
            m_t.fit(X_tr[mask], y_tr[mask])
            m_c.fit(X_tr[~mask], y_tr[~mask])
            uplift = m_t.predict_proba(X_val)[:, 1] - m_c.predict_proba(X_val)[:, 1]
            scores.append(calculate_uplift_auc(y_val.values, uplift, t_val))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params



def optimize_transformation(X_train_features, Z_train, y_train, treatment_train, n_trials=50):
    joint = _joint_stratum(treatment_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_train_features, joint):
            X_tr, X_val = X_train_features[tr_idx], X_train_features[val_idx]
            Z_tr = Z_train[tr_idx]
            y_val = y_train[val_idx]
            t_val = treatment_train[val_idx]
            model = XGBClassifier(**params)
            model.fit(X_tr, Z_tr)
            uplift = 2 * model.predict_proba(X_val)[:, 1] - 1
            scores.append(calculate_uplift_auc(y_val, uplift, t_val))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params



def optimize_tree(X_train_features, treatment_train, y_train_array, n_trials=50):
    t_num = _to_numeric_treatment(np.asarray(treatment_train))
    joint = _joint_stratum(t_num, y_train_array)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 50, 170),
            'min_samples_treatment': trial.suggest_int('min_samples_treatment', 50, 150),
            'control_name': 'control'
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_train_features, joint):
            X_tr, X_val = X_train_features[tr_idx], X_train_features[val_idx]
            y_tr, y_val = y_train_array[tr_idx], y_train_array[val_idx]
            t_tr, t_val = treatment_train[tr_idx], treatment_train[val_idx]
            tree = UpliftTreeClassifier(**params)
            tree.fit(X_tr, t_tr, y_tr)
            preds = tree.predict(X_val)
            uplift = preds[:, 1] - preds[:, 0]
            scores.append(calculate_uplift_auc(y_val, uplift, _to_numeric_treatment(np.asarray(t_val))))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def optimize_rf(X_train_features, treatment_train, y_train_array, n_trials=50):
    t_num = _to_numeric_treatment(np.asarray(treatment_train))
    joint = _joint_stratum(t_num, y_train_array)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 50, 170),
            'min_samples_treatment': trial.suggest_int('min_samples_treatment', 50, 170),
            'control_name': 'control',
            'random_state': 42,
            'n_jobs': -1
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_train_features, joint):
            X_tr, X_val = X_train_features[tr_idx], X_train_features[val_idx]
            y_tr, y_val = y_train_array[tr_idx], y_train_array[val_idx]
            t_tr, t_val = treatment_train[tr_idx], treatment_train[val_idx]
            rf = UpliftRandomForestClassifier(**params)
            rf.fit(X_tr, t_tr, y_tr)
            uplift = rf.predict(X_val)
            if hasattr(uplift, 'ravel'):
                uplift = uplift.ravel()
            scores.append(calculate_uplift_auc(y_val, uplift, _to_numeric_treatment(np.asarray(t_val))))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value