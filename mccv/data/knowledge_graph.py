"""
Clinical Knowledge Graph for MCCV

This module builds the clinical knowledge graph that encodes expected
relationships between diagnoses and treatments based on clinical practice
guidelines from ADA, ACC/AHA, KDIGO, GOLD, and other authoritative sources.

The knowledge graph defines:
- Which medications are expected for each diagnosis
- Which lab tests should be monitored
- Which specialists should be involved
- Which procedures are standard of care

This enables MCCV to validate whether submitted diagnoses have appropriate
treatment evidence, rather than relying solely on statistical patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import json


@dataclass
class TreatmentExpectation:
    """Expected treatment for a diagnosis based on clinical guidelines."""
    treatment_code: str
    treatment_name: str
    treatment_type: str  # medication, lab, specialist, procedure
    expected_frequency: str  # daily, monthly, quarterly, annually, once
    importance_weight: float  # 0.0 to 1.0
    guideline_source: str
    guideline_citation: str


@dataclass  
class DiagnosisNode:
    """Node representing a diagnosis in the clinical knowledge graph."""
    hcc_code: str
    icd10_codes: List[str]
    name: str
    description: str
    category: str
    annual_payment_impact: float
    expected_treatments: List[TreatmentExpectation] = field(default_factory=list)
    related_diagnoses: List[str] = field(default_factory=list)


class ClinicalKnowledgeGraph:
    """
    Clinical Knowledge Graph for validating diagnosis-treatment coherence.
    
    This graph encodes expert medical knowledge about what treatments
    are expected for each diagnosis. It is built from clinical practice
    guidelines published by medical associations.
    
    Supported Guidelines:
    - ADA: American Diabetes Association (diabetes)
    - ACC/AHA: American College of Cardiology/American Heart Association (cardiovascular)
    - KDIGO: Kidney Disease Improving Global Outcomes (kidney disease)
    - GOLD: Global Initiative for Chronic Obstructive Lung Disease (COPD)
    - NCCN: National Comprehensive Cancer Network (oncology)
    
    Usage:
    >>> kg = ClinicalKnowledgeGraph()
    >>> kg.load_guidelines("ADA")
    >>> kg.load_guidelines("ACC_AHA")
    >>> expected = kg.get_expected_treatments("HCC18")
    """
    
    def __init__(self):
        self.diagnoses: Dict[str, DiagnosisNode] = {}
        self.treatments: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, str, float]] = []  # (diagnosis, treatment, type, weight)
        self.loaded_guidelines: Set[str] = set()
        
        # Initialize with base HCC definitions
        self._init_hcc_codes()
        
    def _init_hcc_codes(self):
        """Initialize base HCC code definitions from CMS model."""
        hcc_base = {
            "HCC18": {
                "name": "Diabetes with Chronic Complications",
                "icd10_codes": ["E11.21", "E11.22", "E11.29", "E11.31", "E11.32", 
                               "E11.33", "E11.34", "E11.35", "E11.36", "E11.40",
                               "E11.41", "E11.42", "E11.43", "E11.44", "E11.51",
                               "E11.52", "E11.59", "E11.610", "E11.618", "E11.620",
                               "E11.621", "E11.622", "E11.628", "E11.630", "E11.638",
                               "E11.641", "E11.649", "E11.65", "E11.69"],
                "category": "Diabetes",
                "payment_impact": 2500.0,
                "description": "Type 2 diabetes mellitus with diabetic chronic kidney disease, retinopathy, neuropathy, or other chronic complications"
            },
            "HCC19": {
                "name": "Diabetes without Complication", 
                "icd10_codes": ["E11.8", "E11.9", "E11.00", "E11.01"],
                "category": "Diabetes",
                "payment_impact": 600.0,
                "description": "Type 2 diabetes mellitus without documented complications"
            },
            "HCC85": {
                "name": "Congestive Heart Failure",
                "icd10_codes": ["I50.1", "I50.20", "I50.21", "I50.22", "I50.23",
                               "I50.30", "I50.31", "I50.32", "I50.33", "I50.40",
                               "I50.41", "I50.42", "I50.43", "I50.810", "I50.811",
                               "I50.812", "I50.813", "I50.814", "I50.82", "I50.83",
                               "I50.84", "I50.89", "I50.9"],
                "category": "Heart Disease",
                "payment_impact": 3200.0,
                "description": "Heart failure with reduced or preserved ejection fraction"
            },
            "HCC86": {
                "name": "Acute Myocardial Infarction",
                "icd10_codes": ["I21.01", "I21.02", "I21.09", "I21.11", "I21.19",
                               "I21.21", "I21.29", "I21.3", "I21.4", "I21.9",
                               "I21.A1", "I21.A9"],
                "category": "Heart Disease",
                "payment_impact": 4500.0,
                "description": "Recent heart attack requiring intensive management"
            },
            "HCC111": {
                "name": "Chronic Obstructive Pulmonary Disease",
                "icd10_codes": ["J44.0", "J44.1", "J44.9"],
                "category": "Respiratory",
                "payment_impact": 1800.0,
                "description": "COPD with chronic respiratory symptoms and airflow limitation"
            },
            "HCC135": {
                "name": "Acute Renal Failure",
                "icd10_codes": ["N17.0", "N17.1", "N17.2", "N17.8", "N17.9"],
                "category": "Kidney Disease",
                "payment_impact": 3800.0,
                "description": "Acute kidney injury requiring nephrology management"
            },
            "HCC136": {
                "name": "Chronic Kidney Disease Stage 5",
                "icd10_codes": ["N18.5", "N18.6"],
                "category": "Kidney Disease",
                "payment_impact": 6200.0,
                "description": "End-stage renal disease requiring dialysis or transplant"
            },
            "HCC96": {
                "name": "Specified Heart Arrhythmias",
                "icd10_codes": ["I48.0", "I48.1", "I48.2", "I48.20", "I48.21",
                               "I48.91", "I49.01", "I49.02"],
                "category": "Heart Disease", 
                "payment_impact": 1500.0,
                "description": "Atrial fibrillation, flutter, or other significant arrhythmias"
            },
            "HCC22": {
                "name": "Morbid Obesity",
                "icd10_codes": ["E66.01", "E66.2"],
                "category": "Metabolic",
                "payment_impact": 800.0,
                "description": "BMI >= 40 or BMI >= 35 with obesity-related comorbidities"
            },
            "HCC12": {
                "name": "Breast, Prostate, and Other Cancers",
                "icd10_codes": ["C50.011", "C50.012", "C50.019", "C61"],
                "category": "Oncology",
                "payment_impact": 4200.0,
                "description": "Active malignancy requiring treatment"
            },
        }
        
        for hcc_code, info in hcc_base.items():
            self.diagnoses[hcc_code] = DiagnosisNode(
                hcc_code=hcc_code,
                icd10_codes=info["icd10_codes"],
                name=info["name"],
                description=info["description"],
                category=info["category"],
                annual_payment_impact=info["payment_impact"]
            )
    
    def load_guidelines(self, guideline: str):
        """
        Load clinical practice guidelines into the knowledge graph.
        
        Parameters
        ----------
        guideline : str
            Guideline identifier: 'ADA', 'ACC_AHA', 'KDIGO', 'GOLD', 'NCCN'
        """
        if guideline in self.loaded_guidelines:
            print(f"Guideline {guideline} already loaded")
            return
            
        loader_map = {
            "ADA": self._load_ada_guidelines,
            "ACC_AHA": self._load_acc_aha_guidelines,
            "KDIGO": self._load_kdigo_guidelines,
            "GOLD": self._load_gold_guidelines,
            "NCCN": self._load_nccn_guidelines,
        }
        
        if guideline not in loader_map:
            raise ValueError(f"Unknown guideline: {guideline}. Available: {list(loader_map.keys())}")
        
        loader_map[guideline]()
        self.loaded_guidelines.add(guideline)
        print(f"Loaded {guideline} guidelines")
    
    def _load_ada_guidelines(self):
        """Load American Diabetes Association guidelines."""
        # ADA Standards of Medical Care in Diabetes - 2024
        # https://diabetesjournals.org/care/issue/47/Supplement_1
        
        diabetes_treatments = [
            # Medications
            TreatmentExpectation("metformin", "Metformin", "medication", "daily", 0.95,
                                "ADA", "Standards of Care 2024, Section 9"),
            TreatmentExpectation("sglt2i", "SGLT2 Inhibitor", "medication", "daily", 0.85,
                                "ADA", "Standards of Care 2024, Section 9"),
            TreatmentExpectation("glp1ra", "GLP-1 Receptor Agonist", "medication", "daily/weekly", 0.80,
                                "ADA", "Standards of Care 2024, Section 9"),
            TreatmentExpectation("insulin", "Insulin (any type)", "medication", "daily", 0.75,
                                "ADA", "Standards of Care 2024, Section 9"),
            TreatmentExpectation("statin", "Statin Therapy", "medication", "daily", 0.90,
                                "ADA", "Standards of Care 2024, Section 10"),
            TreatmentExpectation("ace_arb", "ACE Inhibitor or ARB", "medication", "daily", 0.85,
                                "ADA", "Standards of Care 2024, Section 11"),
            
            # Labs
            TreatmentExpectation("hba1c", "HbA1c Test", "lab", "quarterly", 0.98,
                                "ADA", "Standards of Care 2024, Section 6"),
            TreatmentExpectation("lipid_panel", "Lipid Panel", "lab", "annually", 0.85,
                                "ADA", "Standards of Care 2024, Section 10"),
            TreatmentExpectation("egfr", "eGFR/Creatinine", "lab", "annually", 0.90,
                                "ADA", "Standards of Care 2024, Section 11"),
            TreatmentExpectation("uacr", "Urine Albumin-Creatinine Ratio", "lab", "annually", 0.85,
                                "ADA", "Standards of Care 2024, Section 11"),
            
            # Specialists
            TreatmentExpectation("ophthalmology", "Eye Exam", "specialist", "annually", 0.70,
                                "ADA", "Standards of Care 2024, Section 12"),
            TreatmentExpectation("podiatry", "Foot Exam", "specialist", "annually", 0.65,
                                "ADA", "Standards of Care 2024, Section 12"),
            TreatmentExpectation("endocrinology", "Endocrinology Referral", "specialist", "as_needed", 0.50,
                                "ADA", "Standards of Care 2024"),
            
            # Procedures
            TreatmentExpectation("dilated_eye_exam", "Dilated Eye Exam", "procedure", "annually", 0.70,
                                "ADA", "Standards of Care 2024, Section 12"),
            TreatmentExpectation("foot_exam", "Comprehensive Foot Exam", "procedure", "annually", 0.65,
                                "ADA", "Standards of Care 2024, Section 12"),
        ]
        
        # Add to HCC18 and HCC19
        for hcc in ["HCC18", "HCC19"]:
            if hcc in self.diagnoses:
                self.diagnoses[hcc].expected_treatments.extend(diabetes_treatments)
                for treatment in diabetes_treatments:
                    self.edges.append((hcc, treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
    
    def _load_acc_aha_guidelines(self):
        """Load ACC/AHA cardiovascular guidelines."""
        # ACC/AHA Guidelines for Heart Failure, Afib, ACS
        
        hf_treatments = [
            # Guideline-directed medical therapy (GDMT) for HFrEF
            TreatmentExpectation("ace_arb_arni", "ACEi/ARB/ARNI", "medication", "daily", 0.98,
                                "ACC/AHA", "2022 HF Guideline"),
            TreatmentExpectation("beta_blocker", "Beta Blocker", "medication", "daily", 0.95,
                                "ACC/AHA", "2022 HF Guideline"),
            TreatmentExpectation("mra", "Mineralocorticoid Receptor Antagonist", "medication", "daily", 0.90,
                                "ACC/AHA", "2022 HF Guideline"),
            TreatmentExpectation("sglt2i_hf", "SGLT2 Inhibitor", "medication", "daily", 0.92,
                                "ACC/AHA", "2022 HF Guideline"),
            TreatmentExpectation("diuretic", "Loop Diuretic", "medication", "daily", 0.85,
                                "ACC/AHA", "2022 HF Guideline"),
            
            # Labs
            TreatmentExpectation("bnp_ntprobnp", "BNP or NT-proBNP", "lab", "as_needed", 0.95,
                                "ACC/AHA", "2022 HF Guideline"),
            TreatmentExpectation("cmp", "Comprehensive Metabolic Panel", "lab", "quarterly", 0.90,
                                "ACC/AHA", "2022 HF Guideline"),
            TreatmentExpectation("cbc", "Complete Blood Count", "lab", "annually", 0.75,
                                "ACC/AHA", "2022 HF Guideline"),
            
            # Specialists & Procedures
            TreatmentExpectation("cardiology", "Cardiology Follow-up", "specialist", "quarterly", 0.85,
                                "ACC/AHA", "2022 HF Guideline"),
            TreatmentExpectation("echo", "Echocardiogram", "procedure", "annually", 0.90,
                                "ACC/AHA", "2022 HF Guideline"),
        ]
        
        if "HCC85" in self.diagnoses:
            self.diagnoses["HCC85"].expected_treatments.extend(hf_treatments)
            for treatment in hf_treatments:
                self.edges.append(("HCC85", treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
        
        # AMI treatments
        ami_treatments = [
            TreatmentExpectation("dual_antiplatelet", "DAPT (Aspirin + P2Y12)", "medication", "daily", 0.99,
                                "ACC/AHA", "2021 ACS Guideline"),
            TreatmentExpectation("high_intensity_statin", "High-intensity Statin", "medication", "daily", 0.98,
                                "ACC/AHA", "2021 ACS Guideline"),
            TreatmentExpectation("beta_blocker_ami", "Beta Blocker", "medication", "daily", 0.95,
                                "ACC/AHA", "2021 ACS Guideline"),
            TreatmentExpectation("ace_arb_ami", "ACE Inhibitor or ARB", "medication", "daily", 0.90,
                                "ACC/AHA", "2021 ACS Guideline"),
            TreatmentExpectation("troponin", "Troponin", "lab", "serial", 0.99,
                                "ACC/AHA", "2021 ACS Guideline"),
            TreatmentExpectation("cardiac_cath", "Cardiac Catheterization", "procedure", "once", 0.95,
                                "ACC/AHA", "2021 ACS Guideline"),
            TreatmentExpectation("interventional_cardiology", "Interventional Cardiology", "specialist", "once", 0.95,
                                "ACC/AHA", "2021 ACS Guideline"),
        ]
        
        if "HCC86" in self.diagnoses:
            self.diagnoses["HCC86"].expected_treatments.extend(ami_treatments)
            for treatment in ami_treatments:
                self.edges.append(("HCC86", treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
        
        # Afib treatments
        afib_treatments = [
            TreatmentExpectation("anticoagulant", "Oral Anticoagulant", "medication", "daily", 0.95,
                                "ACC/AHA", "2023 Afib Guideline"),
            TreatmentExpectation("rate_control", "Rate Control Agent", "medication", "daily", 0.85,
                                "ACC/AHA", "2023 Afib Guideline"),
            TreatmentExpectation("inr", "INR Monitoring (if warfarin)", "lab", "monthly", 0.80,
                                "ACC/AHA", "2023 Afib Guideline"),
            TreatmentExpectation("electrophysiology", "EP Evaluation", "specialist", "as_needed", 0.70,
                                "ACC/AHA", "2023 Afib Guideline"),
        ]
        
        if "HCC96" in self.diagnoses:
            self.diagnoses["HCC96"].expected_treatments.extend(afib_treatments)
            for treatment in afib_treatments:
                self.edges.append(("HCC96", treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
    
    def _load_kdigo_guidelines(self):
        """Load KDIGO kidney disease guidelines."""
        # KDIGO Clinical Practice Guidelines for CKD
        
        ckd_treatments = [
            TreatmentExpectation("ace_arb_ckd", "ACEi or ARB", "medication", "daily", 0.95,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("sglt2i_ckd", "SGLT2 Inhibitor", "medication", "daily", 0.90,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("phosphate_binder", "Phosphate Binder", "medication", "daily", 0.80,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("esa", "Erythropoiesis-Stimulating Agent", "medication", "weekly", 0.75,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("egfr_creatinine", "eGFR/Creatinine", "lab", "quarterly", 0.98,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("uacr_ckd", "Urine ACR", "lab", "annually", 0.90,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("pth", "PTH Level", "lab", "quarterly", 0.85,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("nephrology", "Nephrology Follow-up", "specialist", "quarterly", 0.95,
                                "KDIGO", "2021 CKD Guideline"),
        ]
        
        esrd_treatments = [
            TreatmentExpectation("dialysis", "Dialysis", "procedure", "thrice_weekly", 0.99,
                                "KDIGO", "2021 CKD Guideline"),
            TreatmentExpectation("vascular_access", "Vascular Access Care", "procedure", "ongoing", 0.95,
                                "KDIGO", "2021 CKD Guideline"),
        ]
        
        if "HCC135" in self.diagnoses:
            self.diagnoses["HCC135"].expected_treatments.extend(ckd_treatments)
            for treatment in ckd_treatments:
                self.edges.append(("HCC135", treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
        
        if "HCC136" in self.diagnoses:
            self.diagnoses["HCC136"].expected_treatments.extend(ckd_treatments + esrd_treatments)
            for treatment in ckd_treatments + esrd_treatments:
                self.edges.append(("HCC136", treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
    
    def _load_gold_guidelines(self):
        """Load GOLD COPD guidelines."""
        # GOLD 2024 Report
        
        copd_treatments = [
            TreatmentExpectation("lama", "Long-acting Muscarinic Antagonist", "medication", "daily", 0.90,
                                "GOLD", "2024 Report"),
            TreatmentExpectation("laba", "Long-acting Beta Agonist", "medication", "daily", 0.85,
                                "GOLD", "2024 Report"),
            TreatmentExpectation("ics_copd", "Inhaled Corticosteroid", "medication", "daily", 0.70,
                                "GOLD", "2024 Report"),
            TreatmentExpectation("saba", "Short-acting Beta Agonist (rescue)", "medication", "as_needed", 0.80,
                                "GOLD", "2024 Report"),
            TreatmentExpectation("spirometry", "Spirometry", "procedure", "annually", 0.85,
                                "GOLD", "2024 Report"),
            TreatmentExpectation("pulmonology", "Pulmonology Follow-up", "specialist", "annually", 0.75,
                                "GOLD", "2024 Report"),
        ]
        
        if "HCC111" in self.diagnoses:
            self.diagnoses["HCC111"].expected_treatments.extend(copd_treatments)
            for treatment in copd_treatments:
                self.edges.append(("HCC111", treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
    
    def _load_nccn_guidelines(self):
        """Load NCCN oncology guidelines."""
        # NCCN Clinical Practice Guidelines in Oncology
        
        cancer_treatments = [
            TreatmentExpectation("oncology_visit", "Medical Oncology", "specialist", "monthly", 0.98,
                                "NCCN", "Various Cancer Guidelines"),
            TreatmentExpectation("tumor_markers", "Tumor Markers", "lab", "quarterly", 0.85,
                                "NCCN", "Various Cancer Guidelines"),
            TreatmentExpectation("imaging", "Surveillance Imaging", "procedure", "quarterly", 0.90,
                                "NCCN", "Various Cancer Guidelines"),
            TreatmentExpectation("chemotherapy", "Chemotherapy", "procedure", "per_protocol", 0.80,
                                "NCCN", "Various Cancer Guidelines"),
        ]
        
        if "HCC12" in self.diagnoses:
            self.diagnoses["HCC12"].expected_treatments.extend(cancer_treatments)
            for treatment in cancer_treatments:
                self.edges.append(("HCC12", treatment.treatment_code, treatment.treatment_type, treatment.importance_weight))
    
    def get_expected_treatments(
        self,
        hcc_code: str,
        treatment_type: Optional[str] = None
    ) -> List[TreatmentExpectation]:
        """
        Get expected treatments for a diagnosis.
        
        Parameters
        ----------
        hcc_code : str
            HCC code (e.g., 'HCC18')
        treatment_type : str, optional
            Filter by type: 'medication', 'lab', 'specialist', 'procedure'
        
        Returns
        -------
        List[TreatmentExpectation]
            List of expected treatments
        """
        if hcc_code not in self.diagnoses:
            return []
        
        treatments = self.diagnoses[hcc_code].expected_treatments
        
        if treatment_type:
            treatments = [t for t in treatments if t.treatment_type == treatment_type]
        
        return sorted(treatments, key=lambda x: x.importance_weight, reverse=True)
    
    def get_coherence_weights(self, hcc_code: str) -> Dict[str, float]:
        """
        Get modality weights for coherence scoring.
        
        These weights determine how much each treatment modality contributes
        to the overall coherence score for a diagnosis.
        
        Parameters
        ----------
        hcc_code : str
            HCC code
        
        Returns
        -------
        Dict[str, float]
            Weights by modality type
        """
        if hcc_code not in self.diagnoses:
            return {"medication": 0.25, "lab": 0.25, "specialist": 0.25, "procedure": 0.25}
        
        treatments = self.diagnoses[hcc_code].expected_treatments
        
        # Aggregate weights by type
        weights = {"medication": 0.0, "lab": 0.0, "specialist": 0.0, "procedure": 0.0}
        counts = {"medication": 0, "lab": 0, "specialist": 0, "procedure": 0}
        
        for t in treatments:
            weights[t.treatment_type] += t.importance_weight
            counts[t.treatment_type] += 1
        
        # Average and normalize
        for t_type in weights:
            if counts[t_type] > 0:
                weights[t_type] /= counts[t_type]
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def to_graph_data(self) -> Dict:
        """
        Export knowledge graph for GNN consumption.
        
        Returns
        -------
        Dict
            Graph data with nodes, edges, and features
        """
        return {
            "diagnoses": {
                hcc: {
                    "name": node.name,
                    "category": node.category,
                    "payment_impact": node.annual_payment_impact,
                    "n_treatments": len(node.expected_treatments)
                }
                for hcc, node in self.diagnoses.items()
            },
            "edges": self.edges,
            "loaded_guidelines": list(self.loaded_guidelines)
        }
    
    def save(self, filepath: str):
        """Save knowledge graph to JSON."""
        data = {
            "diagnoses": {
                hcc: {
                    "hcc_code": node.hcc_code,
                    "name": node.name,
                    "description": node.description,
                    "category": node.category,
                    "payment_impact": node.annual_payment_impact,
                    "icd10_codes": node.icd10_codes,
                    "expected_treatments": [
                        {
                            "code": t.treatment_code,
                            "name": t.treatment_name,
                            "type": t.treatment_type,
                            "frequency": t.expected_frequency,
                            "weight": t.importance_weight,
                            "source": t.guideline_source,
                            "citation": t.guideline_citation
                        }
                        for t in node.expected_treatments
                    ]
                }
                for hcc, node in self.diagnoses.items()
            },
            "loaded_guidelines": list(self.loaded_guidelines)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved knowledge graph to {filepath}")


if __name__ == "__main__":
    # Test the knowledge graph
    print("Building Clinical Knowledge Graph...")
    
    kg = ClinicalKnowledgeGraph()
    
    # Load all guidelines
    kg.load_guidelines("ADA")
    kg.load_guidelines("ACC_AHA")
    kg.load_guidelines("KDIGO")
    kg.load_guidelines("GOLD")
    kg.load_guidelines("NCCN")
    
    # Test queries
    print("\n" + "="*60)
    print("Expected treatments for HCC18 (Diabetes with Complications):")
    print("="*60)
    
    for treatment in kg.get_expected_treatments("HCC18"):
        print(f"  [{treatment.treatment_type}] {treatment.treatment_name} "
              f"(weight: {treatment.importance_weight:.2f}) - {treatment.guideline_source}")
    
    print("\n" + "="*60)
    print("Coherence weights for HCC85 (Heart Failure):")
    print("="*60)
    weights = kg.get_coherence_weights("HCC85")
    for modality, weight in weights.items():
        print(f"  {modality}: {weight:.3f}")
    
    # Save to file
    kg.save("clinical_knowledge_graph.json")
    
    print("\nKnowledge graph built successfully!")
