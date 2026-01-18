"""
MCCV Data Module

Contains data loading, preprocessing, and synthetic data generation utilities.
"""

def get_synthetic_generator():
    """
    Prefer lite generator (pure python) to avoid environments where numpy/pandas
    are unavailable or unstable.
    """
    from mccv.lite.synthetic_generator import MedicareSyntheticGeneratorLite
    return MedicareSyntheticGeneratorLite

__all__ = ["get_synthetic_generator"]
