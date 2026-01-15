"""
Audit Report Generator for MCCV

Generates human-readable audit reports explaining why a diagnosis was flagged.
These reports are designed to be used by CMS auditors for RADV reviews.

The reports include:
- Coherence score and risk level
- Evidence analysis by modality (pharmacy, lab, specialist, procedure)
- Specific missing treatments based on clinical guidelines
- Diagnosis origin information
- Actionable recommendations
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from mccv.data.knowledge_graph import ClinicalKnowledgeGraph, TreatmentExpectation


@dataclass
class EvidenceAnalysis:
    """Analysis of treatment evidence for a single modality."""
    modality: str
    expected_treatments: List[str]
    found_treatments: List[str]
    missing_treatments: List[str]
    modality_score: float
    modality_weight: float


@dataclass
class AuditReport:
    """Complete audit report for a beneficiary-diagnosis pair."""
    beneficiary_id: str
    hcc_code: str
    hcc_name: str
    payment_impact: float
    coherence_score: float
    risk_level: str
    evidence_analyses: List[EvidenceAnalysis]
    diagnosis_origin: Dict
    recommendation: str
    generated_at: datetime
    guideline_sources: List[str]


class AuditReportGenerator:
    """
    Generates explainable audit reports for MCCV predictions.
    
    These reports translate the model's coherence scores into
    actionable insights for CMS auditors.
    
    Parameters
    ----------
    knowledge_graph : ClinicalKnowledgeGraph
        Clinical knowledge graph with expected treatments
    """
    
    def __init__(self, knowledge_graph: Optional[ClinicalKnowledgeGraph] = None):
        if knowledge_graph is None:
            self.kg = ClinicalKnowledgeGraph()
            self.kg.load_guidelines("ADA")
            self.kg.load_guidelines("ACC_AHA")
            self.kg.load_guidelines("KDIGO")
            self.kg.load_guidelines("GOLD")
            self.kg.load_guidelines("NCCN")
        else:
            self.kg = knowledge_graph
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from coherence score."""
        if score < 0.3:
            return "HIGH RISK"
        elif score < 0.6:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    def _get_recommendation(self, score: float, risk_level: str) -> str:
        """Generate recommendation based on risk level."""
        if risk_level == "HIGH RISK":
            return "FLAG FOR RADV AUDIT REVIEW - Diagnosis lacks clinical coherence"
        elif risk_level == "MEDIUM RISK":
            return "RECOMMEND ADDITIONAL DOCUMENTATION REVIEW"
        else:
            return "NO ACTION REQUIRED - Sufficient treatment evidence"
    
    def _analyze_modality(
        self,
        modality: str,
        hcc_code: str,
        beneficiary_claims: pd.DataFrame,
        modality_weight: float
    ) -> EvidenceAnalysis:
        """Analyze treatment evidence for a single modality."""
        # Get expected treatments from knowledge graph
        expected = self.kg.get_expected_treatments(hcc_code, treatment_type=modality)
        expected_names = [t.treatment_name for t in expected]
        
        # Find actual treatments in claims
        if modality == "medication":
            found = beneficiary_claims[
                beneficiary_claims["claim_type"] == "pharmacy"
            ]["medication_name"].unique().tolist() if len(beneficiary_claims) > 0 else []
        elif modality == "lab":
            found = beneficiary_claims[
                beneficiary_claims["claim_type"] == "laboratory"
            ]["lab_name"].unique().tolist() if len(beneficiary_claims) > 0 else []
        elif modality == "specialist":
            found = beneficiary_claims[
                beneficiary_claims["claim_type"] == "specialist_visit"
            ]["specialty_name"].unique().tolist() if len(beneficiary_claims) > 0 else []
        elif modality == "procedure":
            found = beneficiary_claims[
                beneficiary_claims["claim_type"] == "procedure"
            ]["procedure_name"].unique().tolist() if len(beneficiary_claims) > 0 else []
        else:
            found = []
        
        # Calculate missing treatments
        missing = [t for t in expected_names if t.lower() not in [f.lower() for f in found]]
        
        # Calculate modality score
        if len(expected_names) > 0:
            modality_score = len(found) / (len(found) + len(missing))
        else:
            modality_score = 1.0  # No expected treatments = no penalty
        
        return EvidenceAnalysis(
            modality=modality,
            expected_treatments=expected_names,
            found_treatments=found,
            missing_treatments=missing,
            modality_score=modality_score,
            modality_weight=modality_weight
        )
    
    def generate_report(
        self,
        beneficiary_id: str,
        hcc_code: str,
        coherence_score: float,
        beneficiary_claims: Optional[pd.DataFrame] = None,
        diagnosis_origin: Optional[Dict] = None
    ) -> AuditReport:
        """
        Generate a complete audit report.
        
        Parameters
        ----------
        beneficiary_id : str
            Beneficiary identifier
        hcc_code : str
            HCC code being evaluated
        coherence_score : float
            Model's coherence score (0-1)
        beneficiary_claims : pd.DataFrame, optional
            All claims for this beneficiary
        diagnosis_origin : Dict, optional
            Information about how the diagnosis was documented
        
        Returns
        -------
        AuditReport
            Complete audit report
        """
        # Get HCC info
        if hcc_code in self.kg.diagnoses:
            hcc_info = self.kg.diagnoses[hcc_code]
            hcc_name = hcc_info.name
            payment_impact = hcc_info.annual_payment_impact
        else:
            hcc_name = f"Unknown HCC ({hcc_code})"
            payment_impact = 0.0
        
        # Determine risk level and recommendation
        risk_level = self._get_risk_level(coherence_score)
        recommendation = self._get_recommendation(coherence_score, risk_level)
        
        # Get coherence weights
        weights = self.kg.get_coherence_weights(hcc_code)
        
        # Analyze each modality
        modalities = ["medication", "lab", "specialist", "procedure"]
        evidence_analyses = []
        
        if beneficiary_claims is None:
            beneficiary_claims = pd.DataFrame()
        
        for modality in modalities:
            analysis = self._analyze_modality(
                modality=modality,
                hcc_code=hcc_code,
                beneficiary_claims=beneficiary_claims,
                modality_weight=weights.get(modality, 0.25)
            )
            evidence_analyses.append(analysis)
        
        # Default diagnosis origin if not provided
        if diagnosis_origin is None:
            diagnosis_origin = {
                "source": "Unknown",
                "date": "Unknown",
                "provider": "Unknown"
            }
        
        # Get guideline sources
        expected_treatments = self.kg.get_expected_treatments(hcc_code)
        guideline_sources = list(set([t.guideline_source for t in expected_treatments]))
        
        return AuditReport(
            beneficiary_id=beneficiary_id,
            hcc_code=hcc_code,
            hcc_name=hcc_name,
            payment_impact=payment_impact,
            coherence_score=coherence_score,
            risk_level=risk_level,
            evidence_analyses=evidence_analyses,
            diagnosis_origin=diagnosis_origin,
            recommendation=recommendation,
            generated_at=datetime.now(),
            guideline_sources=guideline_sources
        )
    
    def format_report(self, report: AuditReport) -> str:
        """
        Format audit report as human-readable text.
        
        Parameters
        ----------
        report : AuditReport
            The audit report to format
        
        Returns
        -------
        str
            Formatted report text
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("MCCV AUDIT REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Beneficiary and diagnosis info
        lines.append(f"Beneficiary ID: {report.beneficiary_id}")
        lines.append(f"HCC Code: {report.hcc_code} - {report.hcc_name}")
        lines.append(f"Expected Annual Payment Impact: ${report.payment_impact:,.2f}")
        lines.append("")
        
        # Coherence score with visual indicator
        score_bar = "█" * int(report.coherence_score * 20) + "░" * (20 - int(report.coherence_score * 20))
        lines.append(f"CLINICAL COHERENCE SCORE: {report.coherence_score:.2f} / 1.00 [{report.risk_level}]")
        lines.append(f"[{score_bar}]")
        lines.append("")
        
        # Evidence analysis
        lines.append("EVIDENCE ANALYSIS:")
        lines.append("-" * 80)
        
        modality_display = {
            "medication": "PHARMACY EVIDENCE",
            "lab": "LABORATORY EVIDENCE",
            "specialist": "SPECIALIST EVIDENCE",
            "procedure": "PROCEDURE EVIDENCE"
        }
        
        for analysis in report.evidence_analyses:
            lines.append(f"\n{modality_display.get(analysis.modality, analysis.modality.upper())} "
                        f"(Weight: {analysis.modality_weight:.2f})")
            
            if analysis.expected_treatments:
                lines.append(f"  Expected: {', '.join(analysis.expected_treatments[:5])}")
                if len(analysis.expected_treatments) > 5:
                    lines.append(f"            ...and {len(analysis.expected_treatments) - 5} more")
            else:
                lines.append("  Expected: (none specified in guidelines)")
            
            if analysis.found_treatments:
                lines.append(f"  Found: {', '.join(analysis.found_treatments[:5])}")
                if len(analysis.found_treatments) > 5:
                    lines.append(f"         ...and {len(analysis.found_treatments) - 5} more")
            else:
                lines.append("  Found: NONE")
            
            if analysis.missing_treatments:
                lines.append(f"  Missing: {', '.join(analysis.missing_treatments[:5])}")
                if len(analysis.missing_treatments) > 5:
                    lines.append(f"           ...and {len(analysis.missing_treatments) - 5} more")
            
            lines.append(f"  Modality Score: {analysis.modality_score:.2f}")
        
        # Diagnosis origin
        lines.append("")
        lines.append("DIAGNOSIS ORIGIN:")
        lines.append("-" * 80)
        lines.append(f"  Source: {report.diagnosis_origin.get('source', 'Unknown')}")
        lines.append(f"  Date: {report.diagnosis_origin.get('date', 'Unknown')}")
        lines.append(f"  Provider: {report.diagnosis_origin.get('provider', 'Unknown')}")
        
        if report.diagnosis_origin.get('source', '').lower() in ['hra', 'health risk assessment']:
            lines.append("")
            lines.append("  ⚠️  WARNING: Diagnosis documented only via Health Risk Assessment")
            lines.append("     No follow-up treatment claims found.")
        
        # Guideline sources
        lines.append("")
        lines.append("CLINICAL GUIDELINE SOURCES:")
        lines.append("-" * 80)
        for source in report.guideline_sources:
            lines.append(f"  • {source}")
        
        # Recommendation
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"RECOMMENDATION: {report.recommendation}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_batch_report(
        self,
        predictions: pd.DataFrame,
        claims_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate summary report for batch predictions.
        
        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame with beneficiary_id, hcc_code, coherence_score
        claims_data : Dict[str, pd.DataFrame]
            Claims data by beneficiary_id
        
        Returns
        -------
        pd.DataFrame
            Summary report with all beneficiaries
        """
        reports = []
        
        for _, row in predictions.iterrows():
            bene_id = row["beneficiary_id"]
            hcc_code = row["hcc_code"]
            score = row["coherence_score"]
            
            bene_claims = claims_data.get(bene_id, pd.DataFrame())
            
            report = self.generate_report(
                beneficiary_id=bene_id,
                hcc_code=hcc_code,
                coherence_score=score,
                beneficiary_claims=bene_claims
            )
            
            reports.append({
                "beneficiary_id": report.beneficiary_id,
                "hcc_code": report.hcc_code,
                "hcc_name": report.hcc_name,
                "payment_impact": report.payment_impact,
                "coherence_score": report.coherence_score,
                "risk_level": report.risk_level,
                "recommendation": report.recommendation,
                "pharmacy_score": next(
                    (a.modality_score for a in report.evidence_analyses if a.modality == "medication"),
                    0.0
                ),
                "lab_score": next(
                    (a.modality_score for a in report.evidence_analyses if a.modality == "lab"),
                    0.0
                ),
                "specialist_score": next(
                    (a.modality_score for a in report.evidence_analyses if a.modality == "specialist"),
                    0.0
                ),
                "procedure_score": next(
                    (a.modality_score for a in report.evidence_analyses if a.modality == "procedure"),
                    0.0
                ),
            })
        
        return pd.DataFrame(reports)


if __name__ == "__main__":
    # Demo the audit report generator
    print("Testing Audit Report Generator...")
    
    # Create generator with knowledge graph
    generator = AuditReportGenerator()
    
    # Generate a sample report for a HIGH RISK case
    report = generator.generate_report(
        beneficiary_id="BENE_ABC123",
        hcc_code="HCC18",
        coherence_score=0.12,
        beneficiary_claims=pd.DataFrame(),  # No claims = suspicious
        diagnosis_origin={
            "source": "Health Risk Assessment",
            "date": "2024-03-15",
            "provider": "ABC Assessment Services (NPI: 1234567890)"
        }
    )
    
    # Format and print
    formatted = generator.format_report(report)
    print(formatted)
    
    print("\n" + "=" * 80)
    print("Testing LOW RISK case...")
    print("=" * 80 + "\n")
    
    # Generate a sample report for a LOW RISK case
    # Create some fake claims
    fake_claims = pd.DataFrame([
        {"claim_type": "pharmacy", "medication_name": "metformin"},
        {"claim_type": "pharmacy", "medication_name": "insulin"},
        {"claim_type": "laboratory", "lab_name": "hba1c"},
        {"claim_type": "laboratory", "lab_name": "lipid_panel"},
        {"claim_type": "specialist_visit", "specialty_name": "endocrinology"},
        {"claim_type": "procedure", "procedure_name": "diabetic_eye_exam"},
    ])
    
    report_low = generator.generate_report(
        beneficiary_id="BENE_XYZ789",
        hcc_code="HCC18",
        coherence_score=0.87,
        beneficiary_claims=fake_claims,
        diagnosis_origin={
            "source": "Outpatient Visit",
            "date": "2024-01-10",
            "provider": "Dr. Smith, Internal Medicine"
        }
    )
    
    formatted_low = generator.format_report(report_low)
    print(formatted_low)
