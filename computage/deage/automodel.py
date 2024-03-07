from deage.base import DeAgeBaseEstimator

class AutoModel(DeAgeBaseEstimator):
    """
    The Great AutoDeAge Estimator!
    This automatically decides which of estimators is better 
    suitable for your data. 
    
    You do not need any parameters to define for it!
    """

    def __init__(self, verbose=0):
        super().__init__()
        
