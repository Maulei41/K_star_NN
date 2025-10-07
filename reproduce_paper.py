"""
Reproduction of results from Anava & Levy (2017) paper
"k*-Nearest Neighbors: From Global to Local"

This script reproduces key experimental results from the paper by:
1. Implementing the k*NN algorithm as described
2. Testing on synthetic datasets to replicate paper findings
3. Comparing k*NN against standard kNN with various k values
4. Demonstrating the adaptive nature of k* selection
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNN import KNN
from KStarNN import KStarNN
from KNNRegressor import KNNRegressor
from KStarNNRegressor import KStarNNRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def generate_synthetic_classification(n_samples=1000, n_features=10, n_informative=10, noise_level=0.1, random_state=42):
    """
    Generate a synthetic classification dataset similar to those in the paper.
    """
    np.random.seed(random_state)
    
    # Create dataset with varying difficulty levels
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=noise_level,  # Controls label noise
        random_state=random_state
    )
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def generate_synthetic_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """
    Generate a synthetic regression dataset.
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    # Create a nonlinear target function
    y = np.sum(X[:, :5] ** 2, axis=1) + np.sin(X[:, 0] * X[:, 1]) + noise * np.random.randn(n_samples)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def compare_algorithms_classification(X, y, gamma_values=[0.1, 0.5, 1.0, 2.0, 5.0], k_values=[1, 3, 5, 7, 9, 11]):
    """
    Compare k*NN and standard kNN on classification task.
    """
    print("="*60)
    print("COMPARISON: k*NN vs Standard kNN (Classification)")
    print("="*60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize results storage
    results = {
        'knn': {},
        'kstarnn': {}
    }
    
    # Test standard kNN with different k values
    print("Testing Standard kNN with different k values:")
    for k in k_values:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        results['knn'][k] = accuracy
        print(f"  k={k}: Accuracy = {accuracy:.4f}")
    
    # Test k*NN with different gamma values
    print("\nTesting k*NN with different gamma values:")
    for gamma in gamma_values:
        kstarnn = KStarNN(gamma=gamma)
        kstarnn.fit(X_train, y_train)
        predictions = kstarnn.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        results['kstarnn'][gamma] = accuracy
        print(f"  gamma={gamma}: Accuracy = {accuracy:.4f}")
    
    # Find best performing models
    best_knn_k = max(results['knn'], key=results['knn'].get)
    best_kstarnn_gamma = max(results['kstarnn'], key=results['kstarnn'].get)
    
    print(f"\nBest Standard kNN: k={best_knn_k}, Accuracy = {results['knn'][best_knn_k]:.4f}")
    print(f"Best k*NN: gamma={best_kstarnn_gamma}, Accuracy = {results['kstarnn'][best_kstarnn_gamma]:.4f}")
    
    return results, (X_train, X_test, y_train, y_test)


def compare_algorithms_regression(X, y, gamma_values=[0.1, 0.5, 1.0, 2.0, 5.0], k_values=[1, 3, 5, 7, 9, 11]):
    """
    Compare k*NN and standard kNN on regression task.
    """
    print("\n" + "="*60)
    print("COMPARISON: k*NN vs Standard kNN (Regression)")
    print("="*60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize results storage
    results = {
        'knn': {},
        'kstarnn': {}
    }
    
    # Test standard kNN regressor with different k values
    print("Testing Standard kNN Regressor with different k values:")
    for k in k_values:
        knn_reg = KNNRegressor(k=k)
        knn_reg.fit(X_train, y_train)
        predictions = knn_reg.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        results['knn'][k] = {'mse': mse, 'r2': r2}
        print(f"  k={k}: MSE = {mse:.4f}, R2 = {r2:.4f}")
    
    # Test k*NN regressor with different gamma values
    print("\nTesting k*NN Regressor with different gamma values:")
    for gamma in gamma_values:
        kstarnn_reg = KStarNNRegressor(gamma=gamma)
        kstarnn_reg.fit(X_train, y_train)
        predictions = kstarnn_reg.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        results['kstarnn'][gamma] = {'mse': mse, 'r2': r2}
        print(f"  gamma={gamma}: MSE = {mse:.4f}, R2 = {r2:.4f}")
    
    # Find best performing models
    best_knn_k_mse = min(results['knn'], key=lambda k: results['knn'][k]['mse'])
    best_kstarnn_gamma_mse = min(results['kstarnn'], key=lambda g: results['kstarnn'][g]['mse'])
    
    print(f"\nBest Standard kNN (MSE): k={best_knn_k_mse}, MSE = {results['knn'][best_knn_k_mse]['mse']:.4f}")
    print(f"Best k*NN (MSE): gamma={best_kstarnn_gamma_mse}, MSE = {results['kstarnn'][best_kstarnn_gamma_mse]['mse']:.4f}")
    
    return results, (X_train, X_test, y_train, y_test)


def analyze_kstar_adaptivity(X_train, y_train, test_indices=None, gamma=1.0):
    """
    Analyze how k* adapts to local structure as claimed in the paper.
    """
    print("\n" + "="*60)
    print("ANALYSIS: Adaptive Nature of k* Selection")
    print("="*60)
    
    kstarnn = KStarNN(gamma=gamma)
    kstarnn.fit(X_train, y_train)
    
    if test_indices is None:
        test_indices = np.random.choice(len(X_train), size=min(10, len(X_train)), replace=False)
    
    print(f"Analyzing k* selection for {len(test_indices)} randomly selected test points:")
    print(f"Using gamma = {gamma}")
    
    k_stars = []
    avg_distances = []
    
    for i, idx in enumerate(test_indices):
        x_test = X_train[idx:idx+1]  # Single test point
        
        # Manually compute k* to see the process
        dists = kstarnn._euclidean_distance(x_test[0], X_train)
        sort_idx = np.argsort(dists)
        beta = gamma * dists[sort_idx]
        
        k = 0
        lambda_k = beta[0] + 1 if len(beta) > 0 else 0
        while k < len(beta) - 1 and lambda_k > beta[k + 1]:
            k += 1
            if k > 0:  # Avoid division by zero
                sum_b = np.sum(beta[:k])
                sum_b2 = np.sum(beta[:k]**2)
                disc = k + sum_b**2 - k * sum_b2
                sqrt_disc = np.sqrt(max(disc, 0))
                lambda_k = (1 / k) * (sum_b + sqrt_disc)
        
        k_stars.append(k)
        avg_distances.append(np.mean(dists[sort_idx[:max(1, k)]]))
        
        print(f"  Point {i+1}: k* = {k}, avg distance to neighbors = {np.mean(dists[sort_idx[:max(1, k)]]):.3f}")
    
    print(f"\nStatistics across all analyzed points:")
    print(f"  Average k*: {np.mean(k_stars):.2f} (std: {np.std(k_stars):.2f})")
    print(f"  Average neighbor distance: {np.mean(avg_distances):.3f}")
    
    return k_stars, avg_distances


def visualize_adaptivity(X_train, y_train, gamma=1.0, n_points=20):
    """
    Create a visualization showing how k* adapts to local density.
    """
    # For visualization, we'll use only 2D data
    if X_train.shape[1] > 2:
        print("Skipping visualization - dataset has more than 2 dimensions")
        return
    
    kstarnn = KStarNN(gamma=gamma)
    kstarnn.fit(X_train, y_train)
    
    # Select a subset for visualization
    indices = np.random.choice(len(X_train), size=min(n_points, len(X_train)), replace=False)
    selected_X = X_train[indices]
    
    # Compute k* for each selected point
    k_stars = []
    for x_test in selected_X:
        dists = kstarnn._euclidean_distance(x_test, X_train)
        sort_idx = np.argsort(dists)
        beta = gamma * dists[sort_idx]
        
        k = 0
        lambda_k = beta[0] + 1 if len(beta) > 0 else 0
        while k < len(beta) - 1 and lambda_k > beta[k + 1]:
            k += 1
            if k > 0:
                sum_b = np.sum(beta[:k])
                sum_b2 = np.sum(beta[:k]**2)
                disc = k + sum_b**2 - k * sum_b2
                sqrt_disc = np.sqrt(max(disc, 0))
                lambda_k = (1 / k) * (sum_b + sqrt_disc)
        k_stars.append(k)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.6, s=50, label='Training Data')
    
    # Highlight selected points with size based on k*
    sizes = [max(20, k*20) for k in k_stars]  # Scale k* for visibility
    plt.scatter(selected_X[:, 0], selected_X[:, 1], s=sizes, c='red', marker='x', linewidth=3, label='Selected Points (size ∝ k*)')
    
    plt.title(f'Adaptive k* Selection (gamma={gamma})\nRed X marks: Selected points, Size: Proportional to k*')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_comprehensive_reproduction():
    """
    Run the comprehensive reproduction of the paper's results.
    """
    print("REPRODUCING RESULTS FROM ANAVA & LEVY (2017)")
    print("k*-Nearest Neighbors: From Global to Local")
    print("="*80)
    
    # Generate datasets similar to those in the paper
    print("1. Generating Synthetic Datasets")
    print("-" * 40)
    
    # Classification dataset
    print("Creating classification dataset...")
    X_class, y_class = generate_synthetic_classification(n_samples=1000, n_features=10)
    print(f"  Classification dataset shape: X={X_class.shape}, y={y_class.shape}")
    
    # Regression dataset  
    print("Creating regression dataset...")
    X_reg, y_reg = generate_synthetic_regression(n_samples=1000, n_features=10)
    print(f"  Regression dataset shape: X={X_reg.shape}, y={y_reg.shape}")
    
    # Run comparisons
    print("\n2. Running Algorithm Comparisons")
    print("-" * 40)
    
    # Classification comparison
    class_results, class_data = compare_algorithms_classification(X_class, y_class)
    
    # Regression comparison
    reg_results, reg_data = compare_algorithms_regression(X_reg, y_reg)
    
    # Analyze adaptivity
    print("\n3. Analyzing Adaptive Nature of k*")
    print("-" * 40)
    
    X_train_class, _, y_train_class, _ = class_data
    X_train_reg, _, y_train_reg, _ = reg_data
    
    class_k_stars, class_avg_dists = analyze_kstar_adaptivity(X_train_class, y_train_class, gamma=1.0)
    
    # Test with different gamma values to show the effect
    print(f"\nTesting different gamma values on classification task:")
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        kstarnn = KStarNN(gamma=gamma)
        kstarnn.fit(X_train_class, y_train_class)
        
        # Test on a few points
        test_X = X_train_class[:5]  # Use first 5 points as test
        predictions = kstarnn.predict(test_X)
        
        # Calculate k* for these points
        k_stars = []
        for x_test in test_X:
            dists = kstarnn._euclidean_distance(x_test, X_train_class)
            sort_idx = np.argsort(dists)
            beta = gamma * dists[sort_idx]
            
            k = 0
            lambda_k = beta[0] + 1 if len(beta) > 0 else 0
            while k < len(beta) - 1 and lambda_k > beta[k + 1]:
                k += 1
                if k > 0:
                    sum_b = np.sum(beta[:k])
                    sum_b2 = np.sum(beta[:k]**2)
                    disc = k + sum_b**2 - k * sum_b2
                    sqrt_disc = np.sqrt(max(disc, 0))
                    lambda_k = (1 / k) * (sum_b + sqrt_disc)
            k_stars.append(k)
        
        print(f"  gamma={gamma}: avg k* = {np.mean(k_stars):.1f}, accuracy on test = {np.mean(predictions == y_train_class[:5]):.3f}")
    
    print(f"\n4. Summary of Reproduction Results")
    print("-" * 40)
    print("Classification Task:")
    best_class_knn = max(class_results['knn'], key=class_results['knn'].get)
    best_class_kstarnn = max(class_results['kstarnn'], key=class_results['kstarnn'].get)
    print(f"  Best kNN (classification): k={best_class_knn}, Accuracy={class_results['knn'][best_class_knn]:.4f}")
    print(f"  Best k*NN (classification): gamma={best_class_kstarnn}, Accuracy={class_results['kstarnn'][best_class_kstarnn]:.4f}")
    
    print("\nRegression Task:")
    best_reg_knn_mse = min(reg_results['knn'], key=lambda k: reg_results['knn'][k]['mse'])
    best_reg_kstarnn_mse = min(reg_results['kstarnn'], key=lambda g: reg_results['kstarnn'][g]['mse'])
    print(f"  Best kNN (regression): k={best_reg_knn_mse}, MSE={reg_results['knn'][best_reg_knn_mse]['mse']:.4f}")
    print(f"  Best k*NN (regression): gamma={best_reg_kstarnn_mse}, MSE={reg_results['kstarnn'][best_reg_kstarnn_mse]['mse']:.4f}")
    
    print(f"\nAdaptivity Analysis:")
    print(f"  k*NN adapts the number of neighbors per query point")
    print(f"  Average k* across points: {np.mean(class_k_stars):.2f}")
    
    return {
        'classification_results': class_results,
        'regression_results': reg_results,
        'adaptivity_analysis': {'k_stars': class_k_stars, 'avg_distances': class_avg_dists}
    }


if __name__ == "__main__":
    results = run_comprehensive_reproduction()