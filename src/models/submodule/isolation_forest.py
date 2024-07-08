from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
                
from submodule import CONTAMINATION, SCALAR, PCA_DEFAULT

class IsolationForest (SklearnIsolationForest) :
    def __init__ (self, contamination = CONTAMINATION, **kwargs) :
        super().__init__(contamination = contamination, **kwargs)

    def fit (self, X, y = None, sample_weight = None) :
        return super().fit(X, y, sample_weight)
        
    def predict (self, X, **kwargs) :
        return super().predict(X, **kwargs)

class IsolationForestPipeline :
    def __init__ (self, isolation_forest : IsolationForest, is_pca = False, **kwargs) :
        self.isolation_forest = isolation_forest
        steps = [
            ("scaler", SCALAR),
            ("classifier", self.isolation_forest),
        ]

        if is_pca :
            steps.append(("pca", PCA_DEFAULT))

        self.pipeline = Pipeline(steps)


if __name__ == "__main__" :
    print(CONTAMINATION)