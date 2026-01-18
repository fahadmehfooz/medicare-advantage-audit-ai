"""
Pure-Python Clinical Knowledge Graph (lite).

Stores expected treatments per HCC using simple dictionaries. This is sufficient
for a working prototype: expected evidence + scoring + audit report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TreatmentExpectationLite:
    treatment_code: str
    treatment_name: str
    treatment_type: str  # medication, lab, specialist, procedure
    importance_weight: float = 1.0
    guideline_source: str = "Lite"


@dataclass
class DiagnosisNodeLite:
    hcc_code: str
    name: str
    description: str
    category: str
    annual_payment_impact: float
    expected_treatments: List[TreatmentExpectationLite] = field(default_factory=list)


class ClinicalKnowledgeGraphLite:
    """
    Lite knowledge graph: expected treatments derived from the same HCC treatment
    lists used by the synthetic generator.
    """

    def __init__(self, hcc_definitions: Optional[Dict] = None):
        self.diagnoses: Dict[str, DiagnosisNodeLite] = {}
        self.loaded_guidelines = {"LITE"}
        if hcc_definitions:
            self.load_from_hcc_definitions(hcc_definitions)

    def load_from_hcc_definitions(self, hcc_definitions: Dict):
        for hcc, d in hcc_definitions.items():
            node = DiagnosisNodeLite(
                hcc_code=hcc,
                name=d["name"],
                description=d.get("description", ""),
                category=d.get("category", "Unknown"),
                annual_payment_impact=float(d.get("annual_payment_impact", 0.0)),
            )

            for med in d.get("expected_medications", []):
                node.expected_treatments.append(
                    TreatmentExpectationLite(
                        treatment_code=str(med),
                        treatment_name=str(med).replace("_", " ").title(),
                        treatment_type="medication",
                        importance_weight=float(d.get("pharmacy_weight", 0.8)),
                        guideline_source="Lite (HCC expected meds)",
                    )
                )
            for lab in d.get("expected_labs", []):
                node.expected_treatments.append(
                    TreatmentExpectationLite(
                        treatment_code=str(lab),
                        treatment_name=str(lab).replace("_", " ").upper(),
                        treatment_type="lab",
                        importance_weight=float(d.get("lab_weight", 0.8)),
                        guideline_source="Lite (HCC expected labs)",
                    )
                )
            for spec in d.get("expected_specialists", []):
                node.expected_treatments.append(
                    TreatmentExpectationLite(
                        treatment_code=str(spec),
                        treatment_name=str(spec).replace("_", " ").title(),
                        treatment_type="specialist",
                        importance_weight=float(d.get("specialist_weight", 0.7)),
                        guideline_source="Lite (HCC expected specialists)",
                    )
                )
            for proc in d.get("expected_procedures", []):
                node.expected_treatments.append(
                    TreatmentExpectationLite(
                        treatment_code=str(proc),
                        treatment_name=str(proc).replace("_", " ").title(),
                        treatment_type="procedure",
                        importance_weight=float(d.get("procedure_weight", 0.7)),
                        guideline_source="Lite (HCC expected procedures)",
                    )
                )

            self.diagnoses[hcc] = node

    def get_expected_treatments(self, hcc_code: str, treatment_type: Optional[str] = None) -> List[TreatmentExpectationLite]:
        node = self.diagnoses.get(hcc_code)
        if not node:
            return []
        treatments = node.expected_treatments
        if treatment_type:
            treatments = [t for t in treatments if t.treatment_type == treatment_type]
        return sorted(treatments, key=lambda t: t.importance_weight, reverse=True)

    def get_coherence_weights(self, hcc_code: str) -> Dict[str, float]:
        # Default equal weights (config file can override in scorer)
        return {"medication": 0.25, "lab": 0.25, "specialist": 0.25, "procedure": 0.25}

