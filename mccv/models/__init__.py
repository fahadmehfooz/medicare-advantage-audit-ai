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

try:
    # Simple GNN for proof-of-concept (requires only torch, not torch_geometric)
    from mccv.models.simple_gnn import SimpleHeteroGNN, MCCVGNNTrainer, build_heterogeneous_graph_from_data
except Exception:  # pragma: no cover
    SimpleHeteroGNN = None
    MCCVGNNTrainer = None
    build_heterogeneous_graph_from_data = None

__all__ = [
    "MCCVModel",
    "MCCVLoss", 
    "CrossModalAttention",
    "CoherenceScorer",
    "SimpleHeteroGNN",
    "MCCVGNNTrainer",
    "build_heterogeneous_graph_from_data",
]
