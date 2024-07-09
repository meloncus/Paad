from submodule.autoencoder import Autoencoder
from submodule.isolation_forest import IsolationForest, IsolationForestPipeline
from submodule.lof import LocalOutlierFactor, LocalOutlierFactorPipeline


def get_models () :
    '''
    get statistic models for anomaly detection
    '''
    isolation_forest = IsolationForest()
    isolation_forest_pipeline = IsolationForestPipeline(isolation_forest)
    isolation_forest_with_pca_pipeline = IsolationForestPipeline(isolation_forest, is_pca = True)

    local_outlier_factor = LocalOutlierFactor()
    local_outlier_factor_pipeline = LocalOutlierFactorPipeline(local_outlier_factor)
    local_outlier_factor_with_pca_pipeline = LocalOutlierFactorPipeline(local_outlier_factor, is_pca = True)

    models = {
        "Isolation Forest" : isolation_forest_pipeline,
        "Isolation Forest with PCA" : isolation_forest_with_pca_pipeline,
        "Local Outlier Factor" : local_outlier_factor_pipeline,
        "Local Outlier Factor with PCA" : local_outlier_factor_with_pca_pipeline,
    }

    return models