import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbors classifier implementation.
    
    This implementation uses the Euclidean distance to find the k nearest neighbors
    and predicts the class based on majority vote among neighbors.
    """
    
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.
        
        Parameters:
        k (int): Number of nearest neighbors to consider for classification
        """
        self.k = k
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
        Predict the class labels for the input data.
        
        Parameters:
        X (array-like): Input data features, shape (n_samples, n_features)
        
        Returns:
        array: Predicted class labels, shape (n_samples,)
        """
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
        # Calculate distances between x and all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
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
    # Sample dataset: points with features and labels
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Create KNN classifier with k=3
    knn = KNN(k=3)
    
    # Fit the model with training data
    knn.fit(X_train, y_train)
    
    # Test data points
    X_test = np.array([[2, 2], [7, 6]])
    
    # Make predictions
    predictions = knn.predict(X_test)
    
    print("K-Nearest Neighbors (KNN) Example:")
    print("="*40)
    print("Training data:")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print("\nTest data:")
    print("X_test:", X_test)
    print("KNN Predictions:", predictions)
    
    # Expected output: [0, 1] since [2,2] is closer to class 0 and [7,6] is closer to class 1
    
    # Test with more samples to show KNN capabilities
    X_test_extended = np.array([[1.5, 2.5], [2.5, 1.5], [6.5, 6.5], [7.5, 5.5]])
    y_test_extended = np.array([0, 0, 1, 1])  # True labels for extended test
    
    extended_predictions = knn.predict(X_test_extended)
    accuracy = knn.accuracy(X_test_extended, y_test_extended)
    
    print(f"\nExtended test with {len(X_test_extended)} samples:")
    print("X_test_extended:", X_test_extended)
    print("Expected labels: ", y_test_extended)
    print("Predictions:     ", extended_predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Show how different k values affect prediction
    print("\nEffect of different k values on predictions:")
    test_point = np.array([[4.5, 4.0]])  # A point roughly in the middle
    
    for k_val in [1, 3, 5]:
        knn_temp = KNN(k=k_val)
        knn_temp.fit(X_train, y_train)
        pred = knn_temp.predict(test_point)
        print(f"k={k_val}: prediction for {test_point[0]} is class {pred[0]}")