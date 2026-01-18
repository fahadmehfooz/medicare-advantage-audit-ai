"""
MCCV Inference Module

Contains scoring and report generation utilities.
"""

def get_audit_report_generator():
    """Lite report generator (pure python)."""
    from mccv.lite.audit_report import AuditReportGeneratorLite
    return AuditReportGeneratorLite

def get_coherence_scorer():
    """Lite scorer (pure python)."""
    from mccv.lite.rule_based_scorer import RuleBasedCoherenceScorerLite
    return RuleBasedCoherenceScorerLite

__all__ = ["get_audit_report_generator", "get_coherence_scorer"]
