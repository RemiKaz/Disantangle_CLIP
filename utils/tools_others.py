import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def pca(vector, N):

    # Standardize the data
    scaler = StandardScaler()
    vector_standardized = scaler.fit_transform(vector)

    # Apply PCA
    pca = PCA()
    pca.fit(vector_standardized)

    # Eliminate the N most important components
    # To eliminate the N most important components, we reconstruct the data
    # using all components except the top N
    pca.n_components = max(1, pca.n_components_ - N)
    vector = pca.inverse_transform(pca.transform(vector_standardized))

    return vector
