# paad/src/model/__init__.py

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

VERSION = "0.1.0"
SCALAR = MinMaxScaler((-5, 5))
PCA_DEFAULT = PCA(n_components = 0.9)
CONTAMINATION = 0.0001

# only for lof
NOVELTY = True