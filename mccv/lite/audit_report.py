"""
Pure-Python audit report generator (lite).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from mccv.lite.knowledge_graph import ClinicalKnowledgeGraphLite


@dataclass
class EvidenceAnalysisLite:
    modality: str
    expected: List[str]
    found: List[str]
    missing: List[str]
    modality_score: float
    modality_weight: float
    contribution: float


@dataclass
class AuditReportLite:
    beneficiary_id: str
    hcc_code: str
    hcc_name: str
    payment_impact: float
    coherence_score: float
    risk_level: str
    evidence_analyses: List[EvidenceAnalysisLite]
    diagnosis_origin: Dict
    recommendation: str
    generated_at: datetime
    guideline_sources: List[str]


class AuditReportGeneratorLite:
    def __init__(self, knowledge_graph: ClinicalKnowledgeGraphLite):
        self.kg = knowledge_graph

    @staticmethod
    def _in_window(service_date: str, measurement_start_date: Optional[str], measurement_end_date: Optional[str]) -> bool:
        if not measurement_start_date or not measurement_end_date:
            return True
        # Dates are ISO YYYY-MM-DD so lexicographic compare works.
        return measurement_start_date <= service_date <= measurement_end_date

    @staticmethod
    def _risk(score: float) -> str:
        if score < 0.3:
            return "HIGH RISK"
        if score < 0.6:
            return "MEDIUM RISK"
        return "LOW RISK"

    @staticmethod
    def _recommend(risk_level: str) -> str:
        if risk_level == "HIGH RISK":
            return "FLAG FOR RADV AUDIT REVIEW - Diagnosis lacks clinical coherence"
        if risk_level == "MEDIUM RISK":
            return "RECOMMEND ADDITIONAL DOCUMENTATION REVIEW"
        return "NO ACTION REQUIRED - Sufficient treatment evidence"

    def _analyze(
        self,
        modality: str,
        hcc_code: str,
        claims: List[Dict],
        modality_weight: float,
    ) -> EvidenceAnalysisLite:
        expected = [t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, modality)]
        expected_disp = [t.treatment_name for t in self.kg.get_expected_treatments(hcc_code, modality)]

        if modality == "medication":
            found = sorted({c.get("medication_name", "") for c in claims if c.get("claim_type") == "pharmacy"} - {""})
        elif modality == "lab":
            found = sorted({c.get("lab_name", "") for c in claims if c.get("claim_type") == "laboratory"} - {""})
        elif modality == "specialist":
            found = sorted({c.get("specialty_name", "") for c in claims if c.get("claim_type") == "specialist_visit"} - {""})
        elif modality == "procedure":
            found = sorted({c.get("procedure_name", "") for c in claims if c.get("claim_type") == "procedure"} - {""})
        else:
            found = []

        found_l = {f.lower() for f in found}
        missing = [disp for disp, code in zip(expected_disp, expected) if str(code).lower() not in found_l]

        if expected:
            modality_score = 1.0 - (len(missing) / max(1, len(expected)))
        else:
            modality_score = 1.0

        contribution = float(modality_weight) * float(max(0.0, min(1.0, modality_score)))

        return EvidenceAnalysisLite(
            modality=modality,
            expected=expected_disp,
            found=found,
            missing=missing,
            modality_score=float(max(0.0, min(1.0, modality_score))),
            modality_weight=float(modality_weight),
            contribution=contribution,
        )

    def generate_report(
        self,
        beneficiary_id: str,
        hcc_code: str,
        coherence_score: float,
        all_claims: List[Dict],
        diagnosis_origin: Optional[Dict] = None,
        weights: Optional[Dict[str, float]] = None,
        measurement_start_date: Optional[str] = None,
        measurement_end_date: Optional[str] = None,
    ) -> AuditReportLite:
        node = self.kg.diagnoses.get(hcc_code)
        hcc_name = node.name if node else f"Unknown HCC ({hcc_code})"
        impact = float(node.annual_payment_impact) if node else 0.0

        risk = self._risk(coherence_score)
        rec = self._recommend(risk)

        # Keep report evidence consistent with scoring: filter to the measurement window.
        all_for_pair = [c for c in all_claims if c.get("beneficiary_id") == beneficiary_id and c.get("related_hcc") == hcc_code]
        claims = [
            c
            for c in all_for_pair
            if self._in_window(str(c.get("service_date", "")), measurement_start_date, measurement_end_date)
        ]
        out_of_window_count = max(0, len(all_for_pair) - len(claims))
        in_window_count = len(claims)

        w = weights or {"medication": 0.25, "lab": 0.25, "specialist": 0.25, "procedure": 0.25}
        analyses = [
            self._analyze("medication", hcc_code, claims, w.get("medication", 0.25)),
            self._analyze("lab", hcc_code, claims, w.get("lab", 0.25)),
            self._analyze("specialist", hcc_code, claims, w.get("specialist", 0.25)),
            self._analyze("procedure", hcc_code, claims, w.get("procedure", 0.25)),
        ]

        if diagnosis_origin is None:
            diagnosis_origin = {"source": "Unknown", "date": "Unknown", "provider": "Unknown"}

        guideline_sources = sorted({a.modality for a in analyses})

        if measurement_start_date and measurement_end_date:
            diagnosis_origin = dict(diagnosis_origin)
            diagnosis_origin["measurement_window"] = f"{measurement_start_date} to {measurement_end_date}"
            if out_of_window_count > 0:
                diagnosis_origin["note"] = (
                    f"{out_of_window_count} related claims exist outside the measurement window "
                    f"(possible coding lag / inactive condition)."
                )
            diagnosis_origin["claims_in_window"] = in_window_count
            diagnosis_origin["claims_out_of_window"] = out_of_window_count

        # Simple explainable category inference (heuristic)
        # Priority: coding lag (old evidence) > HRA-only > paper diagnosis > upcoding/partial evidence
        likely = None
        src = str(diagnosis_origin.get("source", "")).upper()
        if out_of_window_count > 0 and in_window_count == 0:
            likely = "coding_lag (inactive/old evidence)"
        elif src == "HRA" and in_window_count == 0:
            likely = "hra_only (documented via HRA with no follow-up evidence)"
        elif out_of_window_count == 0 and in_window_count == 0:
            likely = "paper_diagnosis (complete absence of evidence)"
        else:
            # evidence exists but incomplete
            if coherence_score < 0.6:
                likely = "upcoding_or_partial (severity mismatch / partial evidence)"
            else:
                likely = "coherent (evidence supports diagnosis)"
        diagnosis_origin = dict(diagnosis_origin)
        diagnosis_origin["likely_category"] = likely

        return AuditReportLite(
            beneficiary_id=beneficiary_id,
            hcc_code=hcc_code,
            hcc_name=hcc_name,
            payment_impact=impact,
            coherence_score=float(coherence_score),
            risk_level=risk,
            evidence_analyses=analyses,
            diagnosis_origin=diagnosis_origin,
            recommendation=rec,
            generated_at=datetime.now(),
            guideline_sources=guideline_sources,
        )

    @staticmethod
    def format_report(report: AuditReportLite) -> str:
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("MCCV AUDIT REPORT (LITE)")
        lines.append("=" * 80)
        lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(f"Beneficiary ID: {report.beneficiary_id}")
        lines.append(f"HCC Code: {report.hcc_code} - {report.hcc_name}")
        lines.append(f"Expected Annual Payment Impact: ${report.payment_impact:,.2f}")
        lines.append("")
        bar = "█" * int(report.coherence_score * 20) + "░" * (20 - int(report.coherence_score * 20))
        lines.append(f"CLINICAL COHERENCE SCORE: {report.coherence_score:.2f} / 1.00 [{report.risk_level}]")
        lines.append(f"[{bar}]")
        lines.append("")
        lines.append("EVIDENCE ANALYSIS:")
        lines.append("-" * 80)
        for a in report.evidence_analyses:
            lines.append(f"\n{a.modality.upper()} (Weight: {a.modality_weight:.2f})")
            lines.append(f"  Expected: {', '.join(a.expected[:5])}{' ...' if len(a.expected) > 5 else ''}" if a.expected else "  Expected: (none)")
            lines.append(f"  Found: {', '.join(a.found[:5])}{' ...' if len(a.found) > 5 else ''}" if a.found else "  Found: NONE")
            if a.missing:
                lines.append(f"  Missing: {', '.join(a.missing[:5])}{' ...' if len(a.missing) > 5 else ''}")
            lines.append(f"  Modality Score: {a.modality_score:.2f} | Contribution: {a.contribution:.2f}")
        lines.append("")
        lines.append("DIAGNOSIS ORIGIN:")
        lines.append("-" * 80)
        lines.append(f"  Source: {report.diagnosis_origin.get('source', 'Unknown')}")
        lines.append(f"  Date: {report.diagnosis_origin.get('date', 'Unknown')}")
        lines.append(f"  Provider: {report.diagnosis_origin.get('provider', 'Unknown')}")
        if report.diagnosis_origin.get("likely_category"):
            lines.append(f"  Likely Category: {report.diagnosis_origin.get('likely_category')}")
        if report.diagnosis_origin.get("measurement_window"):
            lines.append(f"  Measurement Window: {report.diagnosis_origin.get('measurement_window')}")
        if report.diagnosis_origin.get("note"):
            lines.append(f"  Note: {report.diagnosis_origin.get('note')}")
        if report.diagnosis_origin.get("claims_in_window") is not None:
            lines.append(f"  Claims (in window): {report.diagnosis_origin.get('claims_in_window')}")
        if report.diagnosis_origin.get("claims_out_of_window") is not None:
            lines.append(f"  Claims (out of window): {report.diagnosis_origin.get('claims_out_of_window')}")
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"RECOMMENDATION: {report.recommendation}")
        lines.append("=" * 80)
        return "\n".join(lines)

