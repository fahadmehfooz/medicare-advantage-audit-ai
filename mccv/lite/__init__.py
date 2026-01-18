"""
MCCV Lite (pure-Python prototype)

This subpackage avoids numpy/pandas/torch so the end-to-end prototype can run in
minimal environments (or environments where numpy is unstable).
"""

from mccv.lite.synthetic_generator import MedicareSyntheticGeneratorLite
from mccv.lite.knowledge_graph import ClinicalKnowledgeGraphLite
from mccv.lite.rule_based_scorer import RuleBasedCoherenceScorerLite
from mccv.lite.audit_report import AuditReportGeneratorLite

__all__ = [
    "MedicareSyntheticGeneratorLite",
    "ClinicalKnowledgeGraphLite",
    "RuleBasedCoherenceScorerLite",
    "AuditReportGeneratorLite",
]

