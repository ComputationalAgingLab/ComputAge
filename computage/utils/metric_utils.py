from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

def print_metrics(y_true, y_pred, return_result=False):
    r2 = r2_score(y_true, y_pred)
    r, pval_r = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE = {mae:.3f}")
    print(f"R2 = {r2:.3f}")
    print(f"r = {r:.3f}")
    if return_result:
        return (mae, r2, r, pval_r)