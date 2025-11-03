import numpy as np
from scipy.spatial.distance import cdist


class KStarNN:
    """
    k*-NN algorithm implementation based on the paper
    """
    def __init__(self, L_C_ratio=1.0):
        self.L_C_ratio = L_C_ratio  # Lipschitz constant to noise ratio
        self.k_values_used = []  # Track the range of k values used
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        predictions = []
        self.k_values_used = []  # Reset tracking for each prediction
        
        for x in X_test:
            # Calculate distances to all training points
            distances = cdist([x], self.X_train, metric='euclidean').flatten()
            # Sort distances and corresponding labels
            sorted_indices = np.argsort(distances)
            sorted_distances = distances[sorted_indices]
            sorted_labels = self.y_train[sorted_indices]
            
            # Compute beta values as per paper: beta_i = L * d(x_i, x_0) / C
            beta = self.L_C_ratio * sorted_distances
            
            # Algorithm implementation using beta values
            lambda_val, k = self._calculate_optimal_k(beta)
            
            # Track the k value used for this prediction
            self.k_values_used.append(k)
            
            # Calculate weights for the prediction using sorted_distances for final calculation
            # The lambda_val was calculated using scaled beta values, but weights are calculated accordingly
            weights = self._calculate_weights(beta, lambda_val)
            
            # Weighted prediction
            pred = np.sum(weights * sorted_labels)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_k_range(self):
        """
        Get the range of k values used by the algorithm
        """
        if self.k_values_used:
            return f"{min(self.k_values_used)}-{max(self.k_values_used)}"
        else:
            return "1-1"  # Default if no predictions made yet
    
    def _calculate_optimal_k(self, beta):
        """
        Implements the algorithm from the paper:
        - beta: vector of ordered distances scaled by L/C
        """
        n = len(beta)
        if n == 0:
            return 0.0, 0
        
        # Start with k=1 since we need at least one neighbor
        k = 1
        sum_beta = beta[0]
        sum_beta_sq = beta[0]**2
        discriminant = k + sum_beta**2 - k * sum_beta_sq
        discriminant = max(0, discriminant)
        sqrt_discriminant = np.sqrt(discriminant)
        lambda_k = (sum_beta + sqrt_discriminant) / k
        
        # While we can consider more neighbors and the condition is satisfied
        while k < n and lambda_k > beta[k]:
            k += 1
            sum_beta = np.sum(beta[:k])
            sum_beta_sq = np.sum(beta[:k]**2)
            discriminant = k + sum_beta**2 - k * sum_beta_sq
            if discriminant < 0:
                if discriminant < -1e-10:
                    break
                discriminant = 0.0
            sqrt_discriminant = np.sqrt(discriminant)
            lambda_k = (sum_beta + sqrt_discriminant) / k
        
        return lambda_k, k
    
    def _calculate_weights(self, beta, lambda_k):
        """
        Calculate weight vector alpha based on the paper formula:
        alpha_i = (lambda_k - beta_i) * 1_{beta_i < lambda_k} / sum_j (lambda_k - beta_j) * 1_{beta_j < lambda_k}
        """
        mask = beta < lambda_k
        weights = np.zeros_like(beta, dtype=np.float64)
        
        if np.sum(mask) == 0:
            # If no beta_i < lambda_k, return uniform weights
            return np.ones(len(beta)) / len(beta)
        
        weights[mask] = lambda_k - beta[mask]
        weights_sum = np.sum(weights)
        
        if weights_sum > 0:
            weights = weights / weights_sum
        else:
            # If weights sum is 0, return uniform weights
            weights = np.ones(len(beta)) / len(beta)
        
        return weights