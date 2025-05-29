import pandas as pd

def uplift_by_decile(
    df: pd.DataFrame,
    uplift_col: str,
    treat_col: str,
    outcome_col: str,
    n_bins: int = 10
) -> pd.DataFrame:
    
    df = df.copy()
    df['__decile_tmp__'] = pd.qcut(df[uplift_col], q=n_bins, labels=False, duplicates='drop')
    df['decile'] = (df['__decile_tmp__'].max() - df['__decile_tmp__']).astype(int)
    df.drop(columns='__decile_tmp__', inplace=True)
    
    unique_treats = df[treat_col].unique()
    if 'treatment' in unique_treats and 'control' in unique_treats:
        treat_val, control_val = 'treatment', 'control'
    elif 1 in unique_treats and 0 in unique_treats:
        treat_val, control_val = 1, 0
    else:
        raise ValueError(f"Unexpected treatment values: {unique_treats}")
    
    records = []
    for d in sorted(df['decile'].unique()):
        group = df[df['decile'] == d]
        treated = group[group[treat_col] == treat_val]
        control = group[group[treat_col] == control_val]
        
        treated_rate = treated[outcome_col].mean() if len(treated) > 0 else float('nan')
        control_rate = control[outcome_col].mean() if len(control) > 0 else float('nan')
        
        records.append({
            'decile': d,
            'n_customers': len(group),
            'avg_predicted_uplift': group[uplift_col].mean(),
            'actual_uplift': treated_rate - control_rate,
            'treated_purchase_rate': treated_rate,
            'control_purchase_rate': control_rate
        })
    
    return pd.DataFrame(records).sort_values('decile')