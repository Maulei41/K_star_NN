import numpy as np


class KStarNNRegressor:
    """
    k*-Nearest Neighbors regressor implementation based on Anava & Levy (2017).
    
    Adaptively chooses k* and weights alpha per query point by minimizing
    a bound on prediction error, with weights decaying linearly with distance.
    For regression, predicts target value via weighted average of neighbors.
    """
    
    def __init__(self, gamma=1.0):
        """
        Initialize the k*NN regressor.
        
        Parameters:
        gamma (float): Ratio L/C (Lipschitz constant / noise bound)
        """
        self.gamma = gamma
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store the training data.
        
        Parameters:
        X (array-like): Training data features, shape (n_samples, n_features)
        y (array-like): Training data targets, shape (n_samples,) or (n_samples, 1)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _euclidean_distance(self, x, X):
        return np.sqrt(np.sum((x - X)**2, axis=-1))
    
    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters:
        X (array-like): Input data features, shape (n_samples, n_features)
        
        Returns:
        array: Predicted target values, shape (n_samples,)
        """
        X = np.atleast_2d(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """
        Predict the target value for a single sample.
        
        Parameters:
        x (array-like): Single input sample
        
        Returns:
        Predicted target value
        """
        dists = self._euclidean_distance(x, self.X_train)
        sort_idx = np.argsort(dists)
        beta = self.gamma * dists[sort_idx]
        
        if len(beta) == 0:
            return np.mean(self.y_train)
        
        k = 0
        lambda_k = beta[0] + 1 if len(beta) > 0 else 0
        while k < len(beta) - 1 and lambda_k > beta[k + 1]:
            k += 1
            sum_b = np.sum(beta[:k])
            sum_b2 = np.sum(beta[:k]**2)
            disc = k + sum_b**2 - k * sum_b2
            sqrt_disc = np.sqrt(max(disc, 0))
            lambda_k = (1 / k) * (sum_b + sqrt_disc)
        
        alpha = np.maximum(0, lambda_k - beta[:k])
        sum_alpha = np.sum(alpha)
        
        if sum_alpha > 0:
            alpha /= sum_alpha
            prediction = np.dot(alpha, self.y_train[sort_idx[:k]])
            return prediction
        else:
            # Fallback: return mean of targets if weights are all zero
            return np.mean(self.y_train[sort_idx[:min(k+1, len(self.y_train))]])
    
    def mean_squared_error(self, X_test, y_test):
        """
        Calculate the mean squared error of the model.
        
        Parameters:
        X_test (array-like): Test data features
        y_test (array-like): True targets for test data
        
        Returns:
        float: Mean squared error
        """
        predictions = self.predict(X_test)
        return np.mean((predictions - y_test) ** 2)
    
    def r2_score(self, X_test, y_test):
        """
        Calculate the R² score (coefficient of determination) of the model.
        
        Parameters:
        X_test (array-like): Test data features
        y_test (array-like): True targets for test data
        
        Returns:
        float: R² score
        """
        predictions = self.predict(X_test)
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (ss_res / ss_tot)


# Example usage and testing
if __name__ == "__main__":
    # Sample dataset for regression: points with features and continuous targets
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]], dtype=float)
    y_train = np.array([1.2, 1.8, 1.1, 5.2, 6.8, 6.1])  # Continuous target values
    
    # Create KStarNN regressor with gamma=1.0
    kstarnn_reg = KStarNNRegressor(gamma=1.0)
    
    # Fit the model with training data
    kstarnn_reg.fit(X_train, y_train)
    
    # Test data points
    X_test = np.array([[2, 2], [7, 6]], dtype=float)
    
    # Make predictions
    predictions = kstarnn_reg.predict(X_test)
    
    print("K*-Nearest Neighbors Regressor (KStarNNRegressor) Example - Anava & Levy (2017) Implementation:")
    print("="*85)
    print("Training data:")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print("\nTest data:")
    print("X_test:", X_test)
    print("KStarNNRegressor Predictions:", predictions)
    
    # Compare with standard KNN regressor for reference
    from KNNRegressor import KNNRegressor
    knn_reg = KNNRegressor(k=3)
    knn_reg.fit(X_train, y_train)
    knn_predictions = knn_reg.predict(X_test)
    
    print("\nComparison with Standard KNN Regressor (k=3):")
    print("KNNRegressor Predictions:     ", knn_predictions)
    print("KStarNNRegressor Predictions: ", predictions)
    
    # Test with more samples to show regression capabilities
    X_test_extended = np.array([[1.5, 2.5], [2.5, 1.5], [6.5, 6.5], [7.5, 5.5]], dtype=float)
    y_test_extended = np.array([1.4, 1.3, 6.0, 6.3])  # True targets for extended test
    
    kstar_extended_predictions = kstarnn_reg.predict(X_test_extended)
    kstar_mse = kstarnn_reg.mean_squared_error(X_test_extended, y_test_extended)
    kstar_r2 = kstarnn_reg.r2_score(X_test_extended, y_test_extended)
    
    knn_extended_predictions = knn_reg.predict(X_test_extended)
    knn_mse = knn_reg.mean_squared_error(X_test_extended, y_test_extended)
    knn_r2 = knn_reg.r2_score(X_test_extended, y_test_extended)
    
    print(f"\nExtended test with {len(X_test_extended)} samples:")
    print("X_test_extended:", X_test_extended)
    print("Expected targets: ", y_test_extended)
    print("KNN predictions:  ", knn_extended_predictions)
    print("KStarNN predict.: ", kstar_extended_predictions)
    print(f"KNN MSE:          {knn_mse:.4f}")
    print(f"KStarNN MSE:      {kstar_mse:.4f}")
    print(f"KNN R2 Score:     {knn_r2:.4f}")
    print(f"KStarNN R2 Score: {kstar_r2:.4f}")
    
    # Show how different gamma values affect prediction
    print("\nEffect of different gamma values on predictions:")
    test_point = np.array([[4.5, 4.0]], dtype=float)
    
    for gamma_val in [0.5, 1.0, 2.0, 5.0]:
        kstar_temp = KStarNNRegressor(gamma=gamma_val)
        kstar_temp.fit(X_train, y_train)
        pred = kstar_temp.predict(test_point)
        # Get the number of neighbors used for this prediction
        dists = kstar_temp._euclidean_distance(test_point[0], kstar_temp.X_train)
        sort_idx = np.argsort(dists)
        beta = gamma_val * dists[sort_idx]
        
        k = 0
        lambda_k = beta[0] + 1 if len(beta) > 0 else 0
        while k < len(beta) - 1 and lambda_k > beta[k + 1]:
            k += 1
            sum_b = np.sum(beta[:k])
            sum_b2 = np.sum(beta[:k]**2)
            disc = k + sum_b**2 - k * sum_b2
            sqrt_disc = np.sqrt(max(disc, 0))
            lambda_k = (1 / k) * (sum_b + sqrt_disc)
        
        print(f"gamma={gamma_val}: prediction for {test_point[0]} is {pred[0]:.3f} (using k*={k} neighbors)")