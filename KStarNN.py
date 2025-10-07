import numpy as np
from collections import Counter


class KStarNN:
    """
    k*-Nearest Neighbors classifier implementation based on Anava & Levy (2017).
    
    Adaptively chooses k* and weights alpha per query point by minimizing
    a bound on prediction error, with weights decaying linearly with distance.
    For binary classification, estimates P(class=1) via weighted average,
    then rounds to nearest class.
    """
    
    def __init__(self, gamma=1.0):
        """
        Initialize the k*NN classifier.
        
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
        y (array-like): Training data labels, shape (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _euclidean_distance(self, x, X):
        return np.sqrt(np.sum((x - X)**2, axis=-1))
    
    def predict(self, X):
        """
        Predict the class labels for the input data.
        
        Parameters:
        X (array-like): Input data features, shape (n_samples, n_features)
        
        Returns:
        array: Predicted class labels, shape (n_samples,)
        """
        X = np.atleast_2d(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """
        Predict the class label for a single sample.
        
        Parameters:
        x (array-like): Single input sample
        
        Returns:
        Predicted class label
        """
        dists = self._euclidean_distance(x, self.X_train)
        sort_idx = np.argsort(dists)
        beta = self.gamma * dists[sort_idx]
        
        if len(beta) == 0:
            unique, counts = np.unique(self.y_train, return_counts=True)
            return unique[np.argmax(counts)]
        
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
            prob = np.dot(alpha, self.y_train[sort_idx[:k]])
            return np.round(prob)  # For binary; round to nearest class
        else:
            # Fallback: return most common class if weights are all zero
            unique, counts = np.unique(self.y_train[sort_idx[:min(k+1, len(self.y_train))]], return_counts=True)
            return unique[np.argmax(counts)]
    
    def accuracy(self, X_test, y_test):
        """
        Calculate the accuracy of the model.
        
        Parameters:
        X_test (array-like): Test data features
        y_test (array-like): True labels for test data
        
        Returns:
        float: Accuracy score
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)


# Example usage and testing
if __name__ == "__main__":
    # Sample dataset: points with features and binary labels
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Create KStarNN classifier with gamma=1.0
    kstarnn = KStarNN(gamma=1.0)
    
    # Fit the model with training data
    kstarnn.fit(X_train, y_train)
    
    # Test data points
    X_test = np.array([[2, 2], [7, 6]])
    
    # Make predictions
    predictions = kstarnn.predict(X_test)
    
    print("K*-Nearest Neighbors (KStarNN) Example - Anava & Levy (2017) Implementation:")
    print("="*70)
    print("Training data:")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print("\nTest data:")
    print("X_test:", X_test)
    print("KStarNN Predictions:", predictions)
    
    # Compare with standard KNN for reference
    from KNN import KNN
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    
    print("\nComparison with Standard KNN (k=3):")
    print("KNN Predictions:      ", knn_predictions)
    print("KStarNN Predictions:  ", predictions)
    
    # Test with more samples to show KStarNN capabilities
    X_test_extended = np.array([[1.5, 2.5], [2.5, 1.5], [6.5, 6.5], [7.5, 5.5]])
    y_test_extended = np.array([0, 0, 1, 1])  # True labels for extended test
    
    kstar_extended_predictions = kstarnn.predict(X_test_extended)
    kstar_accuracy = kstarnn.accuracy(X_test_extended, y_test_extended)
    
    knn_extended_predictions = knn.predict(X_test_extended)
    knn_accuracy = knn.accuracy(X_test_extended, y_test_extended)
    
    print(f"\nExtended test with {len(X_test_extended)} samples:")
    print("X_test_extended:", X_test_extended)
    print("Expected labels:  ", y_test_extended)
    print("KNN predictions:  ", knn_extended_predictions)
    print("KStarNN predict.: ", kstar_extended_predictions)
    print(f"KNN accuracy:     {knn_accuracy:.2f}")
    print(f"KStarNN accuracy: {kstar_accuracy:.2f}")
    
    # Show how different gamma values affect prediction
    print("\nEffect of different gamma values on predictions:")
    test_point = np.array([[4.5, 4.0]])  # A point roughly in the middle
    
    for gamma_val in [0.5, 1.0, 2.0, 5.0]:
        kstar_temp = KStarNN(gamma=gamma_val)
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
        
        print(f"gamma={gamma_val}: prediction for {test_point[0]} is class {int(pred[0])} (using k*={k} neighbors)")