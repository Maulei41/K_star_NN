import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import urllib.request
import os
import pandas as pd
from scipy.io import arff
import zipfile
import warnings

warnings.filterwarnings('ignore')

from KNN import KNN
from KStarNN import KStarNN
from NWerror import NWerror


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model using mean absolute error
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae


def cross_validation(model_class, X, y, param_range, cv_folds=5):
    """
    Perform cross-validation to find the best parameter
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=456)
    scores = {}

    for param in param_range:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if model_class.__name__ == 'KNN':
                model = model_class(k=param)
            elif model_class.__name__ == 'NWerror':
                model = model_class(sigma=param)
            else:
                model = model_class()

            score = evaluate_model(model, X_train, y_train, X_val, y_val)
            fold_scores.append(score)

        scores[param] = np.mean(fold_scores)

    best_param = min(scores, key=scores.get)
    return best_param, scores[best_param]


def load_uci_dataset(name, url, delimiter=',', has_header=False, is_arff=False):
    """
    Load a UCI dataset by name and URL
    """
    if not os.path.exists('data'):
        os.makedirs('data')

    local_filename = f'data/{name}.download'
    if not os.path.exists(local_filename):
        print(f"Downloading {name} dataset...")
        try:
            urllib.request.urlretrieve(url, local_filename)
        except Exception as e:
            print(f"Error downloading {name} dataset: {e}")
            return None, None

    extracted_file = f'data/{name}_extracted.data'
    if zipfile.is_zipfile(local_filename):
        print(f"Extracting ZIP file for {name} dataset...")
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            data_filename = next((f for f in file_list if
                                  f.endswith('.arff') or f.endswith('.csv') or f.endswith('.data') or f.endswith(
                                      '.txt')), None)
            if data_filename:
                with zip_ref.open(data_filename) as file:
                    with open(extracted_file, 'wb') as output:
                        output.write(file.read())
                data_file = extracted_file
            else:
                print(f"No data file found in {local_filename}")
                return None, None
    else:
        data_file = local_filename

    try:
        if is_arff:
            data, meta = arff.loadarff(data_file)
            df = pd.DataFrame(data)
            # Convert object columns (e.g., byte strings) to appropriate types
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.decode('utf-8') if df[col].dtype.name == 'object' else df[col]
        else:
            df = pd.read_csv(data_file, header=0 if has_header else None, delimiter=delimiter, engine='python')

        # Convert to numeric, handling non-numeric values
        converted_cols = []
        for col in df.columns:
            col_data = pd.to_numeric(df[col], errors='coerce')
            if col_data.isna().sum() > len(col_data) * 0.5:
                converted_cols.append(df[col])
            else:
                converted_cols.append(col_data)
        df = pd.concat(converted_cols, axis=1)

    except Exception as e:
        print(f"Could not read {data_file}: {e}")
        return None, None

    if df is None or df.empty:
        print(f"Could not read {data_file} properly")
        return None, None

    data = df.values
    if data is None or len(data.shape) == 1 or data.shape[0] == 0 or data.shape[1] < 2:
        print(f"Could not load {name} properly - insufficient data")
        return None, None

    X_raw = data[:, :-1]
    y_raw = data[:, -1]

    X = np.zeros(X_raw.shape, dtype=np.float64)
    for i in range(X_raw.shape[1]):
        X[:, i] = pd.to_numeric(X_raw[:, i], errors='coerce')

    y = pd.to_numeric(pd.Series(y_raw), errors='coerce').values
    if np.all(np.isnan(y)):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))

    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) == 0:
        print(f"All data in {name} was NaN, cannot process")
        return None, None

    print(f"Loaded {name} dataset with shape X: {X.shape}, y: {y.shape}")
    return X, y


def is_float(s):
    """Helper function to check if a value can be converted to float"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def main():
    """
    Main function to reproduce the experimental results
    """
    print("Reproducing k*-NN Experimental Results")
    print("=" * 50)

    datasets_info = {
        'Diabetes': {
            'name': 'Diabeties',
            'loader': None,
            'url': 'https://archive.ics.uci.edu/static/public/329/diabetic+retinopathy+debrecen.zip',
            'has_header': False,
            'delimiter': ',',
            'is_arff': True,
            'target_col': -1
        },
        'QSAR': {
            'name': 'QSAR',
            'loader': None,
            'url': 'https://archive.ics.uci.edu/static/public/254/qsar+biodegradation.zip',
            'has_header': True,
            'delimiter': ';',
            'is_arff': False,
            'target_col': -1
        },
        'Sonar': {
            'name': 'Sonar',
            'loader': None,
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data',
            'has_header': False,
            'delimiter': ',',
            'is_arff': False,
            'target_col': -1
        },
        'Ionosphere': {
            'name': 'Ionosphere',
            'loader': None,
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data',
            'has_header': False,
            'delimiter': ',',
            'is_arff': False,
            'target_col': -1
        },
        'Fertility': {
            'name': 'Fertility',
            'loader': None,
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt',
            'has_header': False,
            'delimiter': ',',
            'is_arff': False,
            'target_col': -1
        },
        'Slump': {
            'name': 'Slump',
            'loader': None,
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data',
            'has_header': True,
            'delimiter': ',',
            'is_arff': False,
            'target_col': -1
        },
        'Yacht': {
            'name': 'Yacht',
            'loader': None,
            'url': 'https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip',
            'has_header': False,
            'delimiter': r'\s+',
            'is_arff': False,
            'target_col': -1
        }
    }

    k_values = list(range(1, 11))
    sigma_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    results = []

    for key, info in datasets_info.items():
        print(f"\nProcessing {info['name']} dataset...")

        X, y = None, None
        if info['loader'] is not None:
            data = info['loader']()
            X, y = data.data, data.target
        else:
            X, y = load_uci_dataset(
                info['name'],
                info['url'],
                delimiter=info['delimiter'],
                has_header=info['has_header'],
                is_arff=info['is_arff']
            )

        if X is not None and y is not None and len(X) > 0 and len(y) > 0:
            n_samples, n_features = X.shape
            dataset_name = f"{info['name']} ({n_samples},{n_features})"
            print(f"  Dataset shape: {X.shape}")

            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                y = y.astype(float)

            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]

            if len(X) == 0:
                print(f"  No valid samples after cleaning, skipping {info['name']}")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_val, X_test, y_val, y_test = train_test_split(
                X_scaled, y, test_size=0.5, random_state=77
            )

            print("  Finding best parameters via 5-fold cross-validation on validation set...")
            best_k, _ = cross_validation(KNN, X_val, y_val, k_values)
            print(f"  Best k for k-NN: {best_k}")

            best_sigma, _ = cross_validation(NWerror, X_val, y_val, sigma_values)
            print(f"  Best sigma for NWerror: {best_sigma}")

            print("  Evaluating models on test set...")
            knn_model = KNN(k=best_k)
            knn_error = evaluate_model(knn_model, X_val, y_val, X_test, y_test)

            nw_model = NWerror(sigma=best_sigma)
            nw_error = evaluate_model(nw_model, X_val, y_val, X_test, y_test)

            kstar_model = KStarNN()
            kstar_model.fit(X_val, y_val)
            kstar_pred = kstar_model.predict(X_test)
            kstar_error = np.mean(np.abs(y_test - kstar_pred))

            k_range_kstar = kstar_model.get_k_range()

            results.append({
                'Dataset': dataset_name,
                'k_NN_Error': knn_error,
                'k_NN_k': best_k,
                'NW_Error': nw_error,
                'NW_Sigma': best_sigma,
                'KStar_Error': kstar_error,
                'KStar_Range': k_range_kstar
            })

            print(f"  k-NN Error: {knn_error:.4f} (k={best_k})")
            print(f"  NW Error: {nw_error:.4f} (sigma={best_sigma})")
            print(f"  k*-NN Error: {kstar_error:.4f} (k range: {k_range_kstar})")
        else:
            print(f"  Failed to load {info['name']}, skipping...")

    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS")
    print("=" * 80)
    if results:
        print(
            f"{'Dataset':<20} {'k-NN Error':<12} {'k':<6} {'NW Error':<12} {'Sigma':<8} {'k*-NN Error':<12} {'Range of k':<10}")
        print("-" * 80)
        for result in results:
            print(f"{result['Dataset']:<20} {result['k_NN_Error']:<12.4f} {result['k_NN_k']:<6} "
                  f"{result['NW_Error']:<12.4f} {result['NW_Sigma']:<8.3f} {result['KStar_Error']:<12.4f} {result['KStar_Range']:<10}")
    else:
        print("No results to display - all datasets failed to load")


if __name__ == "__main__":
    main()