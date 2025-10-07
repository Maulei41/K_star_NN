"""
Main script to run all KNN and KStarNN implementations.

This script provides a unified interface to run:
- KNN for classification
- KStarNN for classification
- KNNRegressor for regression
- KStarNNRegressor for regression

It includes example usage with sample datasets demonstrating both 
classification and regression capabilities.
"""

import numpy as np
from KNN import KNN
from KStarNN import KStarNN
from KNNRegressor import KNNRegressor
from KStarNNRegressor import KStarNNRegressor


def run_classification_example():
    """
    Run example with classification algorithms on binary-labeled datasets.
    """
    print("="*60)
    print("CLASSIFICATION EXAMPLE")
    print("="*60)
    
    # Sample dataset for classification: points with features and binary labels
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Create and train KNN classifier
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # Create and train KStarNN classifier
    kstarnn = KStarNN(gamma=1.0)
    kstarnn.fit(X_train, y_train)
    
    # Test data
    X_test = np.array([[2, 2], [7, 6]])
    
    # Make predictions
    knn_predictions = knn.predict(X_test)
    kstarnn_predictions = kstarnn.predict(X_test)
    
    print("Training data:")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print()
    print("Test data:")
    print("X_test:", X_test)
    print()
    print("KNN Predictions:", knn_predictions)
    print("KStarNN Predictions:", kstarnn_predictions)
    
    # Extended testing
    X_test_extended = np.array([[1.5, 2.5], [2.5, 1.5], [6.5, 6.5], [7.5, 5.5]])
    y_test_extended = np.array([0, 0, 1, 1])
    
    knn_extended_predictions = knn.predict(X_test_extended)
    kstarnn_extended_predictions = kstarnn.predict(X_test_extended)
    
    knn_accuracy = knn.accuracy(X_test_extended, y_test_extended)
    kstarnn_accuracy = kstarnn.accuracy(X_test_extended, y_test_extended)
    
    print(f"\nExtended test with {len(X_test_extended)} samples:")
    print("X_test_extended:", X_test_extended)
    print("Expected labels:   ", y_test_extended)
    print("KNN predictions:   ", knn_extended_predictions)
    print("KStarNN predictions:", kstarnn_extended_predictions)
    print(f"KNN accuracy:      {knn_accuracy:.4f}")
    print(f"KStarNN accuracy:  {kstarnn_accuracy:.4f}")
    

def run_regression_example():
    """
    Run example with regression algorithms on real-valued labeled datasets.
    """
    print("\n" + "="*60)
    print("REGRESSION EXAMPLE")
    print("="*60)
    
    # Sample dataset for regression: points with features and continuous targets
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]], dtype=float)
    y_train = np.array([1.2, 1.8, 1.1, 5.2, 6.8, 6.1])
    
    # Create and train KNN regressor
    knn_reg = KNNRegressor(k=3)
    knn_reg.fit(X_train, y_train)
    
    # Create and train KStarNN regressor
    kstarnn_reg = KStarNNRegressor(gamma=1.0)
    kstarnn_reg.fit(X_train, y_train)
    
    # Test data
    X_test = np.array([[2, 2], [7, 6]], dtype=float)
    
    # Make predictions
    knn_reg_predictions = knn_reg.predict(X_test)
    kstarnn_reg_predictions = kstarnn_reg.predict(X_test)
    
    print("Training data:")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print()
    print("Test data:")
    print("X_test:", X_test)
    print()
    print("KNNRegressor Predictions:     ", knn_reg_predictions)
    print("KStarNNRegressor Predictions: ", kstarnn_reg_predictions)
    
    # Extended testing
    X_test_extended = np.array([[1.5, 2.5], [2.5, 1.5], [6.5, 6.5], [7.5, 5.5]], dtype=float)
    y_test_extended = np.array([1.4, 1.3, 6.0, 6.3])
    
    knn_reg_extended_predictions = knn_reg.predict(X_test_extended)
    kstarnn_reg_extended_predictions = kstarnn_reg.predict(X_test_extended)
    
    knn_reg_mse = knn_reg.mean_squared_error(X_test_extended, y_test_extended)
    kstarnn_reg_mse = kstarnn_reg.mean_squared_error(X_test_extended, y_test_extended)
    
    knn_reg_r2 = knn_reg.r2_score(X_test_extended, y_test_extended)
    kstarnn_reg_r2 = kstarnn_reg.r2_score(X_test_extended, y_test_extended)
    
    print(f"\nExtended test with {len(X_test_extended)} samples:")
    print("X_test_extended:", X_test_extended)
    print("Expected targets:  ", y_test_extended)
    print("KNNRegressor pred:    ", knn_reg_extended_predictions)
    print("KStarNNRegressor pred:", kstarnn_reg_extended_predictions)
    print(f"KNNRegressor MSE:     {knn_reg_mse:.4f}")
    print(f"KStarNNRegressor MSE: {kstarnn_reg_mse:.4f}")
    print(f"KNNRegressor R2 Score:     {knn_reg_r2:.4f}")
    print(f"KStarNNRegressor R2 Score: {kstarnn_reg_r2:.4f}")
    

def run_parameter_analysis():
    """
    Run parameter analysis for both classification and regression algorithms.
    """
    print("\n" + "="*60)
    print("PARAMETER ANALYSIS")
    print("="*60)
    
    # Sample datasets
    X_train_class = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
    y_train_class = np.array([0, 0, 0, 1, 1, 1])
    
    X_train_reg = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]], dtype=float)
    y_train_reg = np.array([1.2, 1.8, 1.1, 5.2, 6.8, 6.1])
    
    test_point = np.array([[4.5, 4.0]])
    
    print("Effect of different k values in KNN (classification):")
    for k_val in [1, 3, 5]:
        knn = KNN(k=k_val)
        knn.fit(X_train_class, y_train_class)
        pred = knn.predict(test_point)
        print(f"  k={k_val}: prediction for {test_point[0]} is class {pred[0]}")
    
    print("\nEffect of different k values in KNN (regression):")
    for k_val in [1, 3, 5]:
        knn_reg = KNNRegressor(k=k_val)
        knn_reg.fit(X_train_reg, y_train_reg)
        pred = knn_reg.predict(test_point)
        print(f"  k={k_val}: prediction for {test_point[0]} is {pred[0]:.3f}")
        
    print("\nEffect of different gamma values in KStarNN (classification):")
    for gamma_val in [0.5, 1.0, 2.0, 5.0]:
        kstarnn = KStarNN(gamma=gamma_val)
        kstarnn.fit(X_train_class, y_train_class)
        pred = kstarnn.predict(test_point)
        # Count how many neighbors are used for this prediction (using implementation-specific method)
        dists = kstarnn._euclidean_distance(test_point[0], kstarnn.X_train)
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
        print(f"  gamma={gamma_val}: prediction for {test_point[0]} is class {int(pred[0])} (using k*={k} neighbors)")
        
    print("\nEffect of different gamma values in KStarNN (regression):")
    for gamma_val in [0.5, 1.0, 2.0, 5.0]:
        kstarnn_reg = KStarNNRegressor(gamma=gamma_val)
        kstarnn_reg.fit(X_train_reg, y_train_reg)
        pred = kstarnn_reg.predict(test_point)
        # Count how many neighbors are used for this prediction (using implementation-specific method)
        dists = kstarnn_reg._euclidean_distance(test_point[0], kstarnn_reg.X_train)
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
        print(f"  gamma={gamma_val}: prediction for {test_point[0]} is {pred[0]:.3f} (using k*={k} neighbors)")


def main():
    """
    Main function to run all examples.
    """
    print("K-Nearest Neighbors and K*-Nearest Neighbors Algorithms")
    print("Unified Interface for Classification and Regression")
    
    # Run classification example
    run_classification_example()
    
    # Run regression example
    run_regression_example()
    
    # Run parameter analysis
    run_parameter_analysis()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()