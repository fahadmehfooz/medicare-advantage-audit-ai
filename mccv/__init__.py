"""
MCCV: Multimodal Clinical Coherence Validation

An open-source AI system to detect unsupported diagnosis codes in Medicare Advantage
claims through multimodal clinical coherence validation using heterogeneous graph
neural networks.

Author: Fahad Mehfooz
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Fahad Mehfooz"

def get_synthetic_generator():
    """
    Return the best available synthetic generator.

    NOTE: In some environments, `numpy` may be broken and segfault on import.
    The lite generator avoids numpy/pandas and is safe for prototype runs.
    """
    # Prefer lite to avoid hard crashes from numpy/pandas imports.
    from mccv.lite.synthetic_generator import MedicareSyntheticGeneratorLite

    return MedicareSyntheticGeneratorLite


def get_knowledge_graph():
    """
    Return the best available clinical knowledge graph.

    Lite variant is pure-Python (safe when numpy/pandas are unavailable).
    """
    from mccv.lite.knowledge_graph import ClinicalKnowledgeGraphLite

    return ClinicalKnowledgeGraphLite

def get_mccv_model():
    """
    Optional import for the deep learning model.

    MCCV's GNN-based model depends on `torch_geometric` (and friends). In some
    environments those aren't installed (e.g., lightweight prototype runs).
    Use this accessor to get a helpful error instead of failing on `import mccv`.
    """
    try:
        from mccv.models.mccv_model import MCCVModel  # pylint: disable=import-error
        return MCCVModel
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MCCVModel is an optional dependency and requires torch + torch_geometric.\n"
            "Install requirements.txt (or at least torch-geometric) to use the GNN model."
        ) from e

__all__ = [
    "get_mccv_model",
    "get_synthetic_generator",
    "get_knowledge_graph",
]
