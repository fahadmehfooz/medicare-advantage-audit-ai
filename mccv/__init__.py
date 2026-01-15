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

from mccv.models.mccv_model import MCCVModel
from mccv.data.synthetic_generator import MedicareSyntheticGenerator
from mccv.data.knowledge_graph import ClinicalKnowledgeGraph

__all__ = [
    "MCCVModel",
    "MedicareSyntheticGenerator", 
    "ClinicalKnowledgeGraph",
]
