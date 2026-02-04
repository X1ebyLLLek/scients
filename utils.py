import requests
import pandas as pd
from io import StringIO
import numpy as np

def download_and_load_csv(url: str, file_description: str) -> pd.DataFrame:
    """
    Downloads or loads a CSV file either from URL or local path.
    """
    try:
        print(f"Loading {file_description} from {url}...")
        if url.startswith("http"):
            # Online download
            response = requests.get(url, stream=True, timeout=1200)
            response.raise_for_status()
            data = StringIO(response.text)
            df = pd.read_csv(data)
        else:
            # Local file read
            df = pd.read_csv(url)
        print(f"{file_description} loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        # Re-raise exception to allow fallback handling in caller
        raise e

def robust_train_test_split(session_df, test_size=0.2, min_anomalies_in_test=1, random_state=42):
    """
    Разделяет данные, гарантируя, что в тестовой выборке окажется
    значимое количество аномалий.
    """
    import pandas as _pd

    anomalies = session_df[session_df['Label'] == 1]
    normals = session_df[session_df['Label'] == 0]

    if len(anomalies) == 0:
        # Attempt to add at least one anomaly to test if possible, otherwise raise error
        if len(normals) > 0:
            print(
                "Warning: No anomalies found for splitting. Assigning a few normal sessions as 'synthetic' anomalies for testing purposes.")
            # This is a fallback for cases with no actual anomalies, not ideal for real evaluation
            num_to_assign_as_anomaly = min(min_anomalies_in_test, len(normals))
            synthetic_test_anomalies = normals.sample(n=num_to_assign_as_anomaly, random_state=random_state)
            synthetic_test_anomalies['Label'] = 1
            normals = normals.drop(synthetic_test_anomalies.index)
            anomalies = synthetic_test_anomalies
        else:
            raise ValueError("В данных нет данных для разделения (ни аномалий, ни нормальных сессий).")

    # 1. Разделяем аномалии. В тест пойдет test_size от их числа, но не меньше min_anomalies_in_test.
    # Это гарантирует, что у нас будет достаточно примеров для оценки.
    n_anomalies_test = max(min_anomalies_in_test, int(round(len(anomalies) * test_size)))
    # Убедимся, что в трейне тоже останутся аномалии (если их больше одной)
    if len(anomalies) > 1:
        n_anomalies_test = min(n_anomalies_test, len(anomalies) - 1)
    elif len(anomalies) == 1:
        # If only one anomaly, put it in test set if min_anomalies_in_test > 0
        if min_anomalies_in_test > 0:
            n_anomalies_test = 1
            train_anomalies = _pd.DataFrame(columns=anomalies.columns)  # empty dataframe
        else:  # If min_anomalies_in_test is 0 or less, put it in train
            n_anomalies_test = 0
            train_anomalies = anomalies
        test_anomalies = anomalies.drop(train_anomalies.index)  # This handles the single anomaly case

    else:  # len(anomalies) == 0, handled by the initial check, but defensive programming
        n_anomalies_test = 0
        test_anomalies = _pd.DataFrame(columns=anomalies.columns)
        train_anomalies = _pd.DataFrame(columns=anomalies.columns)

    if n_anomalies_test > 0:
        # Ensure we don't sample more anomalies than available
        n_anomalies_test = min(n_anomalies_test, len(anomalies))
        test_anomalies = anomalies.sample(n=n_anomalies_test, random_state=random_state)
        train_anomalies = anomalies.drop(test_anomalies.index)
    else:
        test_anomalies = _pd.DataFrame(columns=anomalies.columns)
        train_anomalies = anomalies

    # 2. Разделяем нормальные сессии, чтобы сохранить общую пропорцию test_size
    n_total_test = int(round(len(session_df) * test_size))
    n_normals_test = max(0, n_total_test - len(test_anomalies))

    # Проверяем, достаточно ли нормальных сессий для выборки
    if n_normals_test > len(normals):
        print(
            f"Warning: Not enough normal samples to meet test_size ({n_normals_test} requested, {len(normals)} available). Using all available normals for test.")
        n_normals_test = len(normals)

    test_normals = normals.sample(n=n_normals_test, random_state=random_state)
    train_normals = normals.drop(test_normals.index)

    # 3. Собираем и перемешиваем итоговые датафреймы
    train_df = _pd.concat([train_normals, train_anomalies]).sample(frac=1, random_state=random_state).reset_index(
        drop=True)
    test_df = _pd.concat([test_normals, test_anomalies]).sample(frac=1, random_state=random_state).reset_index(
        drop=True)

    # Диагностика
    print("robust_train_test_split diagnostics:")
    print("  overall label counts:", session_df['Label'].value_counts().to_dict())
    print("  train label counts:", train_df['Label'].value_counts().to_dict())
    print("  test label counts:", test_df['Label'].value_counts().to_dict())

    return train_df, test_df

def get_risk_category(score: float, threshold: float) -> str:
    """Categorizes the anomaly score into risk levels."""
    if score < threshold:
        return "Low"
    elif score < threshold * 1.5:
        return "Medium"
    return "High"

def apply_contextual_weighting(score: float, user: str = 'user', resource: str = 'standard') -> float:
    """
    Placeholder function to adjust score based on context.
    A real system would integrate with identity and asset management systems.
    """
    weight = 1.0
    if user == 'admin': weight *= 1.5
    if resource == 'critical': weight *= 2.0
    return score * weight
