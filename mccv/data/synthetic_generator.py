"""
Synthetic Medicare Data Generator for MCCV Development and Testing

This module generates realistic but entirely synthetic Medicare Advantage claims data
for model development. It creates correlated patterns between diagnoses and treatments
that mirror real-world clinical coherence patterns, with configurable fraud injection.

NO REAL PATIENT DATA IS USED OR GENERATED.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import hashlib


@dataclass
class HCCDefinition:
    """Hierarchical Condition Category definition with expected treatments."""
    code: str
    name: str
    description: str
    annual_payment_impact: float
    expected_medications: List[str]
    expected_labs: List[str]
    expected_specialists: List[str]
    expected_procedures: List[str]
    pharmacy_weight: float = 0.90
    lab_weight: float = 0.85
    specialist_weight: float = 0.70
    procedure_weight: float = 0.65


# Clinical knowledge base: HCC codes with expected treatment patterns
# Based on ADA, ACC/AHA, KDIGO, and GOLD clinical practice guidelines
HCC_DEFINITIONS = {
    "HCC18": HCCDefinition(
        code="HCC18",
        name="Diabetes with Chronic Complications",
        description="Type 2 diabetes mellitus with diabetic chronic kidney disease, retinopathy, or neuropathy",
        annual_payment_impact=2500.0,
        expected_medications=["metformin", "insulin_glargine", "insulin_lispro", "empagliflozin", 
                             "semaglutide", "lisinopril", "atorvastatin"],
        expected_labs=["hba1c", "fasting_glucose", "creatinine", "egfr", "urine_albumin", 
                      "lipid_panel", "comprehensive_metabolic"],
        expected_specialists=["endocrinology", "nephrology", "ophthalmology", "podiatry"],
        expected_procedures=["diabetic_eye_exam", "diabetic_foot_exam", "cgm_placement",
                            "monofilament_test", "dilated_fundus_exam"],
        pharmacy_weight=0.95,
        lab_weight=0.98,
        specialist_weight=0.70,
        procedure_weight=0.65
    ),
    "HCC19": HCCDefinition(
        code="HCC19", 
        name="Diabetes without Complication",
        description="Type 2 diabetes mellitus without documented complications",
        annual_payment_impact=600.0,
        expected_medications=["metformin", "glipizide", "sitagliptin"],
        expected_labs=["hba1c", "fasting_glucose", "lipid_panel"],
        expected_specialists=["primary_care", "endocrinology"],
        expected_procedures=["diabetic_eye_exam", "diabetic_foot_exam"],
        pharmacy_weight=0.90,
        lab_weight=0.95,
        specialist_weight=0.40,
        procedure_weight=0.50
    ),
    "HCC85": HCCDefinition(
        code="HCC85",
        name="Congestive Heart Failure",
        description="Heart failure with reduced or preserved ejection fraction",
        annual_payment_impact=3200.0,
        expected_medications=["lisinopril", "carvedilol", "metoprolol", "furosemide",
                             "spironolactone", "sacubitril_valsartan", "empagliflozin"],
        expected_labs=["bnp", "nt_probnp", "comprehensive_metabolic", "cbc", "tsh",
                      "iron_studies", "lipid_panel"],
        expected_specialists=["cardiology", "heart_failure_clinic"],
        expected_procedures=["echocardiogram", "cardiac_cath", "stress_test", 
                            "cardiac_mri", "chest_xray"],
        pharmacy_weight=0.98,
        lab_weight=0.95,
        specialist_weight=0.85,
        procedure_weight=0.90
    ),
    "HCC86": HCCDefinition(
        code="HCC86",
        name="Acute Myocardial Infarction",
        description="Recent heart attack requiring intensive management",
        annual_payment_impact=4500.0,
        expected_medications=["aspirin", "clopidogrel", "ticagrelor", "atorvastatin",
                             "metoprolol", "lisinopril", "nitroglycerin"],
        expected_labs=["troponin", "bnp", "comprehensive_metabolic", "cbc", "lipid_panel", "pt_inr"],
        expected_specialists=["cardiology", "interventional_cardiology", "cardiac_rehab"],
        expected_procedures=["cardiac_cath", "pci_stent", "echocardiogram", "stress_test",
                            "cardiac_rehab_sessions"],
        pharmacy_weight=0.99,
        lab_weight=0.98,
        specialist_weight=0.95,
        procedure_weight=0.95
    ),
    "HCC111": HCCDefinition(
        code="HCC111",
        name="Chronic Obstructive Pulmonary Disease",
        description="COPD with chronic respiratory symptoms and airflow limitation",
        annual_payment_impact=1800.0,
        expected_medications=["albuterol", "ipratropium", "tiotropium", "fluticasone_salmeterol",
                             "budesonide_formoterol", "prednisone", "azithromycin"],
        expected_labs=["cbc", "comprehensive_metabolic", "abg", "alpha1_antitrypsin"],
        expected_specialists=["pulmonology"],
        expected_procedures=["spirometry", "pft", "chest_ct", "chest_xray", "6mwt",
                            "oxygen_assessment"],
        pharmacy_weight=0.95,
        lab_weight=0.70,
        specialist_weight=0.75,
        procedure_weight=0.85
    ),
    "HCC135": HCCDefinition(
        code="HCC135",
        name="Acute Renal Failure",
        description="Acute kidney injury requiring nephrology management",
        annual_payment_impact=3800.0,
        expected_medications=["sodium_bicarbonate", "kayexalate", "sevelamer", 
                             "epoetin", "calcitriol"],
        expected_labs=["creatinine", "bun", "egfr", "comprehensive_metabolic", "cbc",
                      "urinalysis", "urine_protein", "renal_ultrasound"],
        expected_specialists=["nephrology"],
        expected_procedures=["dialysis", "renal_biopsy", "av_fistula_creation",
                            "renal_ultrasound"],
        pharmacy_weight=0.85,
        lab_weight=0.99,
        specialist_weight=0.95,
        procedure_weight=0.80
    ),
    "HCC136": HCCDefinition(
        code="HCC136",
        name="Chronic Kidney Disease Stage 5",
        description="End-stage renal disease requiring dialysis or transplant evaluation",
        annual_payment_impact=6200.0,
        expected_medications=["sevelamer", "cinacalcite", "epoetin", "darbepoetin",
                             "calcitriol", "iron_sucrose"],
        expected_labs=["creatinine", "bun", "egfr", "comprehensive_metabolic", "cbc",
                      "pth", "phosphorus", "calcium", "iron_studies", "albumin"],
        expected_specialists=["nephrology", "vascular_surgery", "transplant_surgery"],
        expected_procedures=["hemodialysis", "peritoneal_dialysis", "av_fistula_creation",
                            "transplant_evaluation", "kt_v_measurement"],
        pharmacy_weight=0.90,
        lab_weight=0.99,
        specialist_weight=0.99,
        procedure_weight=0.95
    ),
    "HCC96": HCCDefinition(
        code="HCC96",
        name="Specified Heart Arrhythmias",
        description="Atrial fibrillation, flutter, or other significant arrhythmias",
        annual_payment_impact=1500.0,
        expected_medications=["warfarin", "apixaban", "rivaroxaban", "metoprolol",
                             "diltiazem", "amiodarone", "digoxin", "flecainide"],
        expected_labs=["pt_inr", "comprehensive_metabolic", "cbc", "tsh", "bnp"],
        expected_specialists=["cardiology", "electrophysiology"],
        expected_procedures=["ecg", "holter_monitor", "echocardiogram", "ablation",
                            "cardioversion", "loop_recorder"],
        pharmacy_weight=0.95,
        lab_weight=0.85,
        specialist_weight=0.80,
        procedure_weight=0.75
    ),
    "HCC22": HCCDefinition(
        code="HCC22",
        name="Morbid Obesity",
        description="BMI >= 40 or BMI >= 35 with obesity-related comorbidities",
        annual_payment_impact=800.0,
        expected_medications=["orlistat", "phentermine_topiramate", "liraglutide",
                             "semaglutide", "metformin"],
        expected_labs=["lipid_panel", "hba1c", "comprehensive_metabolic", "tsh",
                      "liver_function"],
        expected_specialists=["bariatric_surgery", "endocrinology", "nutrition"],
        expected_procedures=["gastric_bypass", "sleeve_gastrectomy", "lap_band",
                            "nutrition_counseling", "behavioral_therapy"],
        pharmacy_weight=0.60,
        lab_weight=0.85,
        specialist_weight=0.50,
        procedure_weight=0.40
    ),
    "HCC12": HCCDefinition(
        code="HCC12",
        name="Breast, Prostate, and Other Cancers",
        description="Active malignancy requiring treatment",
        annual_payment_impact=4200.0,
        expected_medications=["tamoxifen", "anastrozole", "leuprolide", "enzalutamide",
                             "pembrolizumab", "chemotherapy_agents", "ondansetron"],
        expected_labs=["cbc", "comprehensive_metabolic", "tumor_markers", "psa",
                      "ca125", "cea", "liver_function"],
        expected_specialists=["oncology", "radiation_oncology", "surgical_oncology"],
        expected_procedures=["chemotherapy", "radiation_therapy", "tumor_biopsy",
                            "pet_scan", "ct_scan", "bone_scan"],
        pharmacy_weight=0.90,
        lab_weight=0.95,
        specialist_weight=0.98,
        procedure_weight=0.95
    ),
}

# Medication NDC code mappings (simplified - real NDCs are 11 digits)
MEDICATION_NDC_MAPPING = {
    "metformin": ["00093-7212", "00093-7214", "68462-0102"],
    "insulin_glargine": ["00088-2220", "00024-5016"],
    "insulin_lispro": ["00002-7510", "00002-7516"],
    "empagliflozin": ["00597-0152", "00597-0154"],
    "semaglutide": ["00169-4132", "00169-4774"],
    "lisinopril": ["00093-7205", "65862-0542"],
    "atorvastatin": ["00071-0157", "00378-3952"],
    "carvedilol": ["00007-4140", "65862-0565"],
    "metoprolol": ["00378-0181", "00781-5071"],
    "furosemide": ["00054-8299", "00591-5510"],
    "spironolactone": ["00093-0835", "00378-0025"],
    "albuterol": ["00487-9801", "00173-0682"],
    "tiotropium": ["00597-0075", "00597-0076"],
    "warfarin": ["00056-0172", "00603-2211"],
    "apixaban": ["00003-0894", "00003-0893"],
    "aspirin": ["00904-2013", "00113-0274"],
}

# Lab LOINC code mappings
LAB_LOINC_MAPPING = {
    "hba1c": ["4548-4", "17856-6"],
    "fasting_glucose": ["1558-6", "14771-0"],
    "creatinine": ["2160-0", "38483-4"],
    "egfr": ["33914-3", "48642-3"],
    "lipid_panel": ["24331-1", "57698-3"],
    "bnp": ["30934-4", "33762-6"],
    "nt_probnp": ["33762-6", "83107-3"],
    "troponin": ["6598-7", "89579-7"],
    "cbc": ["58410-2", "57021-8"],
    "comprehensive_metabolic": ["24323-8", "51990-0"],
    "tsh": ["3016-3", "11580-8"],
    "pt_inr": ["5902-2", "34714-6"],
}

# Specialist taxonomy codes
SPECIALIST_TAXONOMY_MAPPING = {
    "endocrinology": "207RE0101X",
    "nephrology": "207RN0300X",
    "cardiology": "207RC0000X",
    "pulmonology": "207RP1001X",
    "ophthalmology": "207W00000X",
    "podiatry": "213E00000X",
    "oncology": "207RX0202X",
    "primary_care": "208D00000X",
}

# Procedure CPT/HCPCS codes
PROCEDURE_CPT_MAPPING = {
    "diabetic_eye_exam": ["92002", "92004", "92012", "92014", "2022F"],
    "diabetic_foot_exam": ["2028F", "G0245", "G0246"],
    "echocardiogram": ["93306", "93307", "93308"],
    "cardiac_cath": ["93452", "93453", "93454"],
    "spirometry": ["94010", "94060"],
    "hemodialysis": ["90935", "90937", "90940"],
    "chemotherapy": ["96401", "96402", "96409", "96413"],
}


@dataclass
class SyntheticBeneficiary:
    """Represents a synthetic Medicare beneficiary with claims history."""
    beneficiary_id: str
    age: int
    sex: str
    state: str
    hcc_codes: List[str]
    is_fraudulent: bool
    fraud_type: Optional[str]
    pharmacy_claims: List[Dict]
    lab_claims: List[Dict]
    specialist_visits: List[Dict]
    procedure_claims: List[Dict]
    hra_records: List[Dict]


class MedicareSyntheticGenerator:
    """
    Generates synthetic Medicare Advantage claims data for MCCV development.
    
    This generator creates realistic but entirely artificial claims data that
    mirrors the patterns seen in real Medicare data. It generates:
    - Beneficiary demographics
    - HCC diagnosis codes
    - Pharmacy claims (Part D)
    - Laboratory claims (Part B)
    - Specialist visits (Part B)
    - Procedure claims (Part A/B)
    - Health Risk Assessment records
    
    The generator injects configurable fraud patterns including:
    - Paper diagnoses (HCC codes with no treatment evidence)
    - Upcoding (severity level exceeds treatment intensity)
    - HRA-only diagnoses (codes appearing only on assessments)
    
    Parameters
    ----------
    n_beneficiaries : int
        Number of synthetic beneficiaries to generate
    fraud_rate : float
        Proportion of beneficiaries with fraudulent claims (0.0 to 1.0)
    seed : int
        Random seed for reproducibility
    start_date : datetime
        Start of the claims period
    end_date : datetime
        End of the claims period
    """
    
    def __init__(
        self,
        n_beneficiaries: int = 10000,
        fraud_rate: float = 0.15,
        seed: int = 42,
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2024, 12, 31)
    ):
        self.n_beneficiaries = n_beneficiaries
        self.fraud_rate = fraud_rate
        self.seed = seed
        self.start_date = start_date
        self.end_date = end_date
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.hcc_definitions = HCC_DEFINITIONS
        self.generated_data = None
        
    def _generate_beneficiary_id(self, index: int) -> str:
        """Generate a unique beneficiary ID."""
        hash_input = f"BENE_{index}_{self.seed}"
        return f"BENE_{hashlib.md5(hash_input.encode()).hexdigest()[:10].upper()}"
    
    def _generate_demographics(self, n: int) -> pd.DataFrame:
        """Generate synthetic beneficiary demographics."""
        ages = np.random.normal(72, 10, n).clip(65, 100).astype(int)
        sexes = np.random.choice(["M", "F"], n, p=[0.45, 0.55])
        states = np.random.choice(
            ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
             "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"],
            n,
            p=[0.12, 0.10, 0.11, 0.09, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04,
               0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02]
        )
        
        return pd.DataFrame({
            "beneficiary_id": [self._generate_beneficiary_id(i) for i in range(n)],
            "age": ages,
            "sex": sexes,
            "state": states,
        })
    
    def _assign_hcc_codes(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """
        Assign HCC codes based on age and realistic prevalence rates.
        Older patients and those with comorbidities have higher HCC counts.
        """
        hcc_assignments = []
        
        for _, row in demographics.iterrows():
            age = row["age"]
            base_hcc_count = max(1, int(np.random.poisson(lam=(age - 65) / 15 + 1)))
            
            available_hccs = list(self.hcc_definitions.keys())
            weights = self._get_hcc_prevalence_weights(age)
            
            n_hccs = min(base_hcc_count, len(available_hccs))
            selected_hccs = np.random.choice(
                available_hccs,
                size=n_hccs,
                replace=False,
                p=weights
            )
            
            hcc_assignments.append(list(selected_hccs))
        
        demographics["hcc_codes"] = hcc_assignments
        return demographics
    
    def _get_hcc_prevalence_weights(self, age: int) -> np.ndarray:
        """Calculate age-adjusted prevalence weights for HCC codes."""
        base_weights = {
            "HCC18": 0.15,  # Diabetes with complications
            "HCC19": 0.20,  # Diabetes without complications
            "HCC85": 0.10,  # CHF
            "HCC86": 0.03,  # AMI
            "HCC111": 0.12, # COPD
            "HCC135": 0.02, # Acute renal failure
            "HCC136": 0.03, # CKD Stage 5
            "HCC96": 0.08,  # Arrhythmias
            "HCC22": 0.07,  # Morbid obesity
            "HCC12": 0.05,  # Cancers
        }
        
        weights = np.array([base_weights[hcc] for hcc in self.hcc_definitions.keys()])
        
        # Age adjustments: older patients have higher rates of chronic conditions
        if age >= 80:
            weights[list(self.hcc_definitions.keys()).index("HCC85")] *= 1.5  # CHF
            weights[list(self.hcc_definitions.keys()).index("HCC96")] *= 1.4  # Arrhythmias
        elif age >= 75:
            weights[list(self.hcc_definitions.keys()).index("HCC18")] *= 1.3  # Diabetes
            weights[list(self.hcc_definitions.keys()).index("HCC111")] *= 1.2  # COPD
        
        return weights / weights.sum()
    
    def _generate_random_date(self) -> datetime:
        """Generate a random date within the claims period."""
        delta = self.end_date - self.start_date
        random_days = random.randint(0, delta.days)
        return self.start_date + timedelta(days=random_days)
    
    def _generate_pharmacy_claims(
        self,
        beneficiary_id: str,
        hcc_codes: List[str],
        is_fraudulent: bool,
        fraud_type: Optional[str]
    ) -> List[Dict]:
        """
        Generate Part D pharmacy claims for a beneficiary.
        
        Fraudulent beneficiaries may have missing or reduced medications
        for their documented diagnoses.
        """
        claims = []
        
        for hcc in hcc_codes:
            hcc_def = self.hcc_definitions.get(hcc)
            if not hcc_def:
                continue
            
            # Determine if we should generate pharmacy evidence
            if is_fraudulent and fraud_type == "paper_diagnosis":
                # Paper diagnosis: NO pharmacy evidence
                continue
            elif is_fraudulent and fraud_type == "upcoding":
                # Upcoding: reduced pharmacy evidence (only 1-2 basic meds)
                n_meds = random.randint(1, 2)
                medications = random.sample(hcc_def.expected_medications[:3], 
                                          min(n_meds, len(hcc_def.expected_medications[:3])))
            else:
                # Legitimate: full pharmacy evidence
                n_meds = random.randint(2, len(hcc_def.expected_medications))
                medications = random.sample(hcc_def.expected_medications, n_meds)
            
            for med in medications:
                # Generate multiple fills throughout the year
                n_fills = random.randint(6, 12)  # Monthly or quarterly fills
                for _ in range(n_fills):
                    ndc_codes = MEDICATION_NDC_MAPPING.get(med, ["00000-0000"])
                    claims.append({
                        "beneficiary_id": beneficiary_id,
                        "claim_type": "pharmacy",
                        "service_date": self._generate_random_date().strftime("%Y-%m-%d"),
                        "ndc_code": random.choice(ndc_codes),
                        "medication_name": med,
                        "days_supply": random.choice([30, 90]),
                        "quantity": random.randint(30, 180),
                        "related_hcc": hcc,
                    })
        
        return claims
    
    def _generate_lab_claims(
        self,
        beneficiary_id: str,
        hcc_codes: List[str],
        is_fraudulent: bool,
        fraud_type: Optional[str]
    ) -> List[Dict]:
        """
        Generate laboratory claims for a beneficiary.
        
        Fraudulent beneficiaries may have missing or reduced lab monitoring.
        """
        claims = []
        
        for hcc in hcc_codes:
            hcc_def = self.hcc_definitions.get(hcc)
            if not hcc_def:
                continue
            
            if is_fraudulent and fraud_type == "paper_diagnosis":
                # Paper diagnosis: NO lab evidence
                continue
            elif is_fraudulent and fraud_type == "upcoding":
                # Upcoding: minimal lab evidence
                n_labs = random.randint(1, 2)
                labs = random.sample(hcc_def.expected_labs[:3],
                                    min(n_labs, len(hcc_def.expected_labs[:3])))
            else:
                # Legitimate: appropriate lab monitoring
                n_labs = random.randint(3, len(hcc_def.expected_labs))
                labs = random.sample(hcc_def.expected_labs, n_labs)
            
            for lab in labs:
                # Labs typically done 2-4 times per year for chronic conditions
                n_tests = random.randint(2, 4)
                for _ in range(n_tests):
                    loinc_codes = LAB_LOINC_MAPPING.get(lab, ["00000-0"])
                    claims.append({
                        "beneficiary_id": beneficiary_id,
                        "claim_type": "laboratory",
                        "service_date": self._generate_random_date().strftime("%Y-%m-%d"),
                        "loinc_code": random.choice(loinc_codes),
                        "lab_name": lab,
                        "result_value": self._generate_lab_result(lab),
                        "result_unit": self._get_lab_unit(lab),
                        "related_hcc": hcc,
                    })
        
        return claims
    
    def _generate_lab_result(self, lab_name: str) -> float:
        """Generate a realistic lab result value."""
        lab_ranges = {
            "hba1c": (5.0, 12.0),
            "fasting_glucose": (70, 300),
            "creatinine": (0.6, 4.0),
            "egfr": (15, 120),
            "bnp": (50, 2000),
            "troponin": (0.01, 5.0),
        }
        low, high = lab_ranges.get(lab_name, (0, 100))
        return round(random.uniform(low, high), 2)
    
    def _get_lab_unit(self, lab_name: str) -> str:
        """Get the unit for a lab test."""
        lab_units = {
            "hba1c": "%",
            "fasting_glucose": "mg/dL",
            "creatinine": "mg/dL",
            "egfr": "mL/min/1.73m2",
            "bnp": "pg/mL",
            "troponin": "ng/mL",
        }
        return lab_units.get(lab_name, "units")
    
    def _generate_specialist_visits(
        self,
        beneficiary_id: str,
        hcc_codes: List[str],
        is_fraudulent: bool,
        fraud_type: Optional[str]
    ) -> List[Dict]:
        """Generate specialist visit claims."""
        claims = []
        
        for hcc in hcc_codes:
            hcc_def = self.hcc_definitions.get(hcc)
            if not hcc_def:
                continue
            
            if is_fraudulent and fraud_type in ["paper_diagnosis", "hra_only"]:
                # No specialist visits for fraudulent diagnoses
                continue
            elif is_fraudulent and fraud_type == "upcoding":
                # Minimal specialist involvement
                specialists = random.sample(hcc_def.expected_specialists[:1], 1) \
                    if hcc_def.expected_specialists else []
            else:
                # Appropriate specialist care
                n_specialists = random.randint(1, len(hcc_def.expected_specialists))
                specialists = random.sample(hcc_def.expected_specialists, n_specialists)
            
            for specialist in specialists:
                n_visits = random.randint(2, 6)
                taxonomy = SPECIALIST_TAXONOMY_MAPPING.get(specialist, "000000000X")
                
                for _ in range(n_visits):
                    claims.append({
                        "beneficiary_id": beneficiary_id,
                        "claim_type": "specialist_visit",
                        "service_date": self._generate_random_date().strftime("%Y-%m-%d"),
                        "provider_taxonomy": taxonomy,
                        "specialty_name": specialist,
                        "visit_type": random.choice(["office_visit", "follow_up", "consultation"]),
                        "related_hcc": hcc,
                    })
        
        return claims
    
    def _generate_procedure_claims(
        self,
        beneficiary_id: str,
        hcc_codes: List[str],
        is_fraudulent: bool,
        fraud_type: Optional[str]
    ) -> List[Dict]:
        """Generate procedure claims."""
        claims = []
        
        for hcc in hcc_codes:
            hcc_def = self.hcc_definitions.get(hcc)
            if not hcc_def:
                continue
            
            if is_fraudulent and fraud_type in ["paper_diagnosis", "hra_only"]:
                continue
            elif is_fraudulent and fraud_type == "upcoding":
                procedures = random.sample(hcc_def.expected_procedures[:1], 1) \
                    if hcc_def.expected_procedures else []
            else:
                n_procedures = random.randint(1, len(hcc_def.expected_procedures))
                procedures = random.sample(hcc_def.expected_procedures, n_procedures)
            
            for procedure in procedures:
                cpt_codes = PROCEDURE_CPT_MAPPING.get(procedure, ["00000"])
                claims.append({
                    "beneficiary_id": beneficiary_id,
                    "claim_type": "procedure",
                    "service_date": self._generate_random_date().strftime("%Y-%m-%d"),
                    "cpt_code": random.choice(cpt_codes),
                    "procedure_name": procedure,
                    "related_hcc": hcc,
                })
        
        return claims
    
    def _generate_hra_records(
        self,
        beneficiary_id: str,
        hcc_codes: List[str],
        is_fraudulent: bool,
        fraud_type: Optional[str]
    ) -> List[Dict]:
        """
        Generate Health Risk Assessment records.
        
        HRA-only diagnoses are a key fraud pattern: conditions documented
        during in-home assessments with no follow-up treatment.
        """
        records = []
        
        # All beneficiaries may have HRA records
        # But fraudulent "hra_only" cases have diagnoses ONLY from HRAs
        
        for hcc in hcc_codes:
            hcc_def = self.hcc_definitions.get(hcc)
            if not hcc_def:
                continue
            
            # Generate HRA record
            records.append({
                "beneficiary_id": beneficiary_id,
                "record_type": "hra",
                "assessment_date": self._generate_random_date().strftime("%Y-%m-%d"),
                "hcc_code": hcc,
                "hcc_name": hcc_def.name,
                "assessor_npi": f"{random.randint(1000000000, 9999999999)}",
                "assessment_company": random.choice([
                    "ABC Assessment Services",
                    "HealthCheck Assessors Inc",
                    "In-Home Medical Evaluations",
                    "Senior Care Assessments LLC"
                ]),
                "is_hra_only": is_fraudulent and fraud_type == "hra_only",
            })
        
        return records
    
    def generate(self) -> Dict[str, pd.DataFrame]:
        """
        Generate the complete synthetic dataset.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing:
            - 'beneficiaries': Beneficiary demographics and HCC assignments
            - 'pharmacy_claims': Part D pharmacy claims
            - 'lab_claims': Laboratory claims
            - 'specialist_visits': Specialist visit claims
            - 'procedure_claims': Procedure claims
            - 'hra_records': Health Risk Assessment records
            - 'labels': Fraud labels for model training
        """
        print(f"Generating synthetic data for {self.n_beneficiaries} beneficiaries...")
        
        # Generate demographics
        demographics = self._generate_demographics(self.n_beneficiaries)
        demographics = self._assign_hcc_codes(demographics)
        
        # Assign fraud labels
        n_fraudulent = int(self.n_beneficiaries * self.fraud_rate)
        fraud_indices = np.random.choice(
            self.n_beneficiaries, 
            size=n_fraudulent, 
            replace=False
        )
        
        demographics["is_fraudulent"] = False
        demographics.loc[fraud_indices, "is_fraudulent"] = True
        
        # Assign fraud types
        fraud_types = ["paper_diagnosis", "upcoding", "hra_only"]
        demographics["fraud_type"] = None
        demographics.loc[fraud_indices, "fraud_type"] = np.random.choice(
            fraud_types,
            size=n_fraudulent,
            p=[0.4, 0.35, 0.25]  # Distribution of fraud types
        )
        
        # Generate claims for each beneficiary
        all_pharmacy = []
        all_labs = []
        all_specialists = []
        all_procedures = []
        all_hra = []
        
        for _, row in demographics.iterrows():
            bene_id = row["beneficiary_id"]
            hcc_codes = row["hcc_codes"]
            is_fraud = row["is_fraudulent"]
            fraud_type = row["fraud_type"]
            
            all_pharmacy.extend(
                self._generate_pharmacy_claims(bene_id, hcc_codes, is_fraud, fraud_type)
            )
            all_labs.extend(
                self._generate_lab_claims(bene_id, hcc_codes, is_fraud, fraud_type)
            )
            all_specialists.extend(
                self._generate_specialist_visits(bene_id, hcc_codes, is_fraud, fraud_type)
            )
            all_procedures.extend(
                self._generate_procedure_claims(bene_id, hcc_codes, is_fraud, fraud_type)
            )
            all_hra.extend(
                self._generate_hra_records(bene_id, hcc_codes, is_fraud, fraud_type)
            )
        
        # Create DataFrames
        self.generated_data = {
            "beneficiaries": demographics,
            "pharmacy_claims": pd.DataFrame(all_pharmacy),
            "lab_claims": pd.DataFrame(all_labs),
            "specialist_visits": pd.DataFrame(all_specialists),
            "procedure_claims": pd.DataFrame(all_procedures),
            "hra_records": pd.DataFrame(all_hra),
        }
        
        # Generate coherence labels for training
        labels = self._generate_coherence_labels()
        self.generated_data["labels"] = labels
        
        print(f"Generated {len(demographics)} beneficiaries")
        print(f"  - Fraudulent: {demographics['is_fraudulent'].sum()} ({self.fraud_rate*100:.1f}%)")
        print(f"  - Pharmacy claims: {len(all_pharmacy)}")
        print(f"  - Lab claims: {len(all_labs)}")
        print(f"  - Specialist visits: {len(all_specialists)}")
        print(f"  - Procedure claims: {len(all_procedures)}")
        print(f"  - HRA records: {len(all_hra)}")
        
        return self.generated_data
    
    def _generate_coherence_labels(self) -> pd.DataFrame:
        """
        Generate ground truth coherence labels for each beneficiary-HCC pair.
        
        This creates the training labels for the MCCV model.
        """
        labels = []
        
        for _, row in self.generated_data["beneficiaries"].iterrows():
            bene_id = row["beneficiary_id"]
            hcc_codes = row["hcc_codes"]
            is_fraud = row["is_fraudulent"]
            fraud_type = row["fraud_type"]
            
            for hcc in hcc_codes:
                if is_fraud:
                    if fraud_type == "paper_diagnosis":
                        coherence_score = np.random.uniform(0.0, 0.15)
                    elif fraud_type == "upcoding":
                        coherence_score = np.random.uniform(0.20, 0.45)
                    elif fraud_type == "hra_only":
                        coherence_score = np.random.uniform(0.05, 0.25)
                    else:
                        coherence_score = np.random.uniform(0.0, 0.30)
                else:
                    coherence_score = np.random.uniform(0.70, 0.99)
                
                labels.append({
                    "beneficiary_id": bene_id,
                    "hcc_code": hcc,
                    "coherence_score": round(coherence_score, 3),
                    "is_fraudulent": is_fraud,
                    "fraud_type": fraud_type,
                })
        
        return pd.DataFrame(labels)
    
    def save(self, output_dir: str):
        """Save generated data to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.generated_data is None:
            raise ValueError("No data generated yet. Call generate() first.")
        
        for name, df in self.generated_data.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {name} to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get summary statistics of generated data."""
        if self.generated_data is None:
            raise ValueError("No data generated yet. Call generate() first.")
        
        bene_df = self.generated_data["beneficiaries"]
        labels_df = self.generated_data["labels"]
        
        return {
            "n_beneficiaries": len(bene_df),
            "n_fraudulent": bene_df["is_fraudulent"].sum(),
            "fraud_rate": bene_df["is_fraudulent"].mean(),
            "mean_hcc_count": bene_df["hcc_codes"].apply(len).mean(),
            "mean_age": bene_df["age"].mean(),
            "fraud_type_distribution": bene_df[bene_df["is_fraudulent"]]["fraud_type"].value_counts().to_dict(),
            "mean_coherence_fraudulent": labels_df[labels_df["is_fraudulent"]]["coherence_score"].mean(),
            "mean_coherence_legitimate": labels_df[~labels_df["is_fraudulent"]]["coherence_score"].mean(),
        }


if __name__ == "__main__":
    # Example usage
    generator = MedicareSyntheticGenerator(
        n_beneficiaries=1000,
        fraud_rate=0.15,
        seed=42
    )
    
    data = generator.generate()
    stats = generator.get_statistics()
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save to disk
    generator.save("./data/synthetic/")
