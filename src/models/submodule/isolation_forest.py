from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
                
from submodule import CONTAMINATION, SCALAR, PCA_DEFAULT

class IsolationForest (SklearnIsolationForest) :
    '''
    sklearn.ensemble.IsolationForest 를 상속받아, contamination 을 기본값으로 설정한 IsolationForest 클래스입니다.
    '''
    def __init__ (self, contamination = CONTAMINATION, **kwargs) :
        super().__init__(contamination = contamination, **kwargs)

    def fit (self, X, y = None, sample_weight = None) :
        return super().fit(X, y, sample_weight)
        
    def predict (self, X, **kwargs) :
        return super().predict(X, **kwargs)

class IsolationForestPipeline :
    '''
    sklearn.pipeline.Pipeline 을 상속받아, IsolationForest 를 적용한 pipeline 을 정의합니다.
    '''
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
    isolationforest = IsolationForest()
    isolationforest_pipeline = IsolationForestPipeline(isolationforest)
    print(isolationforest_pipeline)