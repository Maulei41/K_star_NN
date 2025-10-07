import numpy as np


class KNNRegressor:
    """
    K-Nearest Neighbors regressor implementation.
    
    This implementation uses the Euclidean distance to find the k nearest neighbors
    and predicts the target value as the average of the target values of the k 
    nearest neighbors.
    """
    
    def __init__(self, k=3):
        """
        Initialize the KNN regressor.
        
        Parameters:
        k (int): Number of nearest neighbors to consider for regression
        """
        self.k = k
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
    
    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.
        
        Parameters:
        point1 (array-like): First point
        point2 (array-like): Second point
        
        Returns:
        float: Euclidean distance between the two points
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters:
        X (array-like): Input data features, shape (n_samples, n_features)
        
        Returns:
        array: Predicted target values, shape (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """
        Predict the target value for a single sample.
        
        Parameters:
        x (array-like): Single input sample
        
        Returns:
        Predicted target value (float or similar)
        """
        # Calculate distances between x and all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_targets = [self.y_train[i] for i in k_indices]
        
        # Return the average of the target values of the k nearest neighbors
        return np.mean(k_nearest_targets)
    
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
    
    # Create KNN regressor with k=3
    knn_reg = KNNRegressor(k=3)
    
    # Fit the model with training data
    knn_reg.fit(X_train, y_train)
    
    # Test data points
    X_test = np.array([[2, 2], [7, 6]], dtype=float)
    
    # Make predictions
    predictions = knn_reg.predict(X_test)
    
    print("K-Nearest Neighbors Regressor (KNNRegressor) Example:")
    print("="*55)
    print("Training data:")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print("\nTest data:")
    print("X_test:", X_test)
    print("KNNRegressor Predictions:", predictions)
    
    # Test with more samples to show regression capabilities
    X_test_extended = np.array([[1.5, 2.5], [2.5, 1.5], [6.5, 6.5], [7.5, 5.5]], dtype=float)
    y_test_extended = np.array([1.4, 1.3, 6.0, 6.3])  # True targets for extended test
    
    extended_predictions = knn_reg.predict(X_test_extended)
    mse = knn_reg.mean_squared_error(X_test_extended, y_test_extended)
    r2 = knn_reg.r2_score(X_test_extended, y_test_extended)
    
    print(f"\nExtended test with {len(X_test_extended)} samples:")
    print("X_test_extended:", X_test_extended)
    print("Expected targets: ", y_test_extended)
    print("Predictions:      ", extended_predictions)
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Show how different k values affect prediction
    print("\nEffect of different k values on predictions:")
    test_point = np.array([[4.5, 4.0]], dtype=float)
    
    for k_val in [1, 3, 5]:
        knn_temp = KNNRegressor(k=k_val)
        knn_temp.fit(X_train, y_train)
        pred = knn_temp.predict(test_point)
        print(f"k={k_val}: prediction for {test_point[0]} is {pred[0]:.3f}")