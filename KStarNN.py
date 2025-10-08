import numpy as np
from scipy.spatial.distance import cdist


class KStarNN:
    """
    k*-NN algorithm implementation based on the paper
    """
    def __init__(self):
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
            
            # Algorithm implementation
            lambda_val, k = self._calculate_optimal_k(sorted_distances)
            
            # Track the k value used for this prediction
            self.k_values_used.append(k)
            
            # Calculate weights for the prediction
            weights = self._calculate_weights(sorted_distances, lambda_val)
            
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
        - beta: vector of ordered distances
        """
        n = len(beta)
        lambda_k = beta[0] + 1  # lambda_0 = beta_1 + 1 (using 0-indexing)
        k = 0
        
        while k < n - 1 and lambda_k > beta[k + 1]:  # λ_k > β_{k+1}, and k ≤ n-1
            k += 1
            sum_beta = np.sum(beta[:k])
            sum_beta_sq = np.sum(beta[:k]**2)
            
            # Calculate lambda_k using the formula
            sqrt_term = k + sum_beta**2 - k * sum_beta_sq
            if sqrt_term < 0:
                # Handle numerical issues
                sqrt_term = max(0, sqrt_term)  # Use max to handle small numerical errors
            
            # Avoid division by zero if k is 0 (though it shouldn't happen in the while loop)
            if k > 0:
                lambda_k = (1.0 / k) * (sum_beta + np.sqrt(max(0, sqrt_term)))
        
        # Ensure k is at least 1 if possible
        if k == 0 and n > 0:
            k = 1
        
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