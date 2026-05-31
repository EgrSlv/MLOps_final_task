import numpy as np


def compute_psi(expected: np.ndarray, actual: np.ndarray,
                bins: int = 10) -> float:
    expected_hist, _ = np.histogram(expected, bins=bins, range=(0, 1))
    actual_hist, _ = np.histogram(actual, bins=bins, range=(0, 1))
    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)
    expected_pct = np.clip(expected_pct, 0.001, None)
    actual_pct = np.clip(actual_pct, 0.001, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

def detect_drift(reference_probs: np.ndarray, current_probs: np.ndarray,
                 threshold: float = 0.1,) -> tuple[bool, float]:
    psi = compute_psi(reference_probs, current_probs)
    drifted = psi > threshold
    return drifted, psi
