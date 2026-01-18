"""
MCCV Models Module

Contains the core MCCV model architecture and GNN layers.
"""

try:
    # Optional dependency: requires torch_geometric
    from mccv.models.mccv_model import MCCVModel, MCCVLoss, CrossModalAttention, CoherenceScorer
except Exception:  # pragma: no cover
    MCCVModel = None
    MCCVLoss = None
    CrossModalAttention = None
    CoherenceScorer = None

__all__ = [
    "MCCVModel",
    "MCCVLoss", 
    "CrossModalAttention",
    "CoherenceScorer",
]
