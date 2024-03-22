from deage.base import DeAgeBaseEstimator
import pandas as pd

class AutoModel(DeAgeBaseEstimator):
    """
    The Great AutoDeAge Estimator!
    This automatically decides which of estimators is better 
    suitable for your data. 
    
    You do not need any parameters to define for it!
    """

    def __init__(self, verbose=0):
        super().__init__()

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series):
        pass

    def predict(self, 
                X: pd.DataFrame, 
                y: pd.Series):
        pass
