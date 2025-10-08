import numpy as np
from scipy.spatial.distance import cdist


class NWerror:
    """
    Nadaraya-Watson estimator with Gaussian kernel
    """
    def __init__(self, sigma):
        self.sigma = sigma
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Calculate distances to all training points
            distances = cdist([x], self.X_train, metric='euclidean').flatten()
            # Calculate Gaussian kernel weights
            weights = np.exp(-0.5 * (distances / self.sigma)**2) / self.sigma
            
            # Weighted prediction
            numerator = np.sum(weights * self.y_train)
            denominator = np.sum(weights)
            
            if denominator != 0:
                pred = numerator / denominator
            else:
                # If all weights are zero, return mean of labels
                pred = np.mean(self.y_train)
            
            predictions.append(pred)
        
        return np.array(predictions)