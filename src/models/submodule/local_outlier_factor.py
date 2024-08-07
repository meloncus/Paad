from sklearn.neighbors import LocalOutlierFactor as SklearnLocalOutlierFactor
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))    

from submodule import NOVELTY, CONTAMINATION, SCALAR, PCA_DEFAULT

class LocalOutlierFactor (SklearnLocalOutlierFactor) :
    '''
    sklearn.neighbors.LocalOutlierFactor 를 상속받아, novelty 와 contamination 을 기본값으로 설정한 LocalOutlierFactor 클래스입니다.
    '''
    def __init__ (self, novelty = NOVELTY, contamination = CONTAMINATION, **kwargs) :
        super().__init__(novelty = novelty, contamination = contamination, **kwargs)

    def fit (self, X, y = None, sample_weight = None) :
        return super().fit(X, y, sample_weight)
        
    def predict (self, X, **kwargs) :
        return super().predict(X, **kwargs)
    
class LocalOutlierFactorPipeline :
    '''
    sklearn.pipeline.Pipeline 을 상속받아, LocalOutlierFactor 를 적용한 pipeline 을 정의합니다.
    '''
    def __init__ (self, local_outlier_factor : LocalOutlierFactor, is_pca = False, **kwargs) :
        self.local_outlier_factor = local_outlier_factor
        steps = [
            ("scaler", SCALAR),
            ("classifier", self.local_outlier_factor),
        ]

        if is_pca :
            steps.append(("pca", PCA_DEFAULT))

        self.pipeline = Pipeline(steps)

    def fit (self, X, y = None, sample_weight = None) :
        return self.pipeline.fit(X, y, sample_weight)


if __name__ == "__main__" :
    localOutlierFactor = LocalOutlierFactor()