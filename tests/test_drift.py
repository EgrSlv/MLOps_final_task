import numpy as np
import pytest

from src.monitoring.drift import compute_psi, detect_drift


def test_compute_psi_identical():
    data = np.random.normal(0.5, 0.1, 1000)
    psi = compute_psi(data, data)
    assert psi < 0.01

def test_compute_psi_different():
    ref = np.random.normal(0.3, 0.1, 1000)
    cur = np.random.normal(0.7, 0.1, 1000)
    psi = compute_psi(ref, cur)
    assert psi > 0.1

def test_detect_drift_no_drift():
    data = np.random.normal(0.5, 0.1, 1000)
    drifted, psi = detect_drift(data, data, threshold=0.1)
    assert not drifted
    assert psi < 0.1

def test_detect_drift_with_drift():
    ref = np.random.normal(0.3, 0.1, 1000)
    cur = np.random.normal(0.7, 0.1, 1000)
    drifted, psi = detect_drift(ref, cur, threshold=0.1)
    assert drifted
    assert psi > 0.1
