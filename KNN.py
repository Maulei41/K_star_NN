import numpy as np
from scipy.spatial.distance import cdist


class KNN:
    """
    Standard k-NN regressor
    """
    def __init__(self, k):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Calculate distances to all training points
            distances = cdist([x], self.X_train, metric='euclidean').flatten()
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            # Average the labels of k nearest neighbors
            pred = np.mean(self.y_train[nearest_indices])
            predictions.append(pred)
        return np.array(predictions)