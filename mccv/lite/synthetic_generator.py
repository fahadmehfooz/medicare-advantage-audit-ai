"""
Pure-Python synthetic Medicare Advantage data generator (lite).

This avoids numpy/pandas entirely (they can be unavailable or unstable in some
environments). Output is a dict of lists-of-dicts, and can be saved to CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import csv
import hashlib
import json
import math
import random
import os


@dataclass(frozen=True)
class HCCDefinitionLite:
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
    category: str = "Unknown"


# Minimal HCC knowledge base (pure python). Mirrors the main generator’s intent.
HCC_DEFINITIONS_LITE: Dict[str, HCCDefinitionLite] = {
    "HCC18": HCCDefinitionLite(
        code="HCC18",
        name="Diabetes with Chronic Complications",
        description="Type 2 diabetes with chronic complications",
        annual_payment_impact=2500.0,
        expected_medications=["metformin", "insulin_glargine", "insulin_lispro", "empagliflozin", "semaglutide", "lisinopril", "atorvastatin"],
        expected_labs=["hba1c", "fasting_glucose", "creatinine", "egfr", "urine_albumin", "lipid_panel", "comprehensive_metabolic"],
        expected_specialists=["endocrinology", "nephrology", "ophthalmology", "podiatry"],
        expected_procedures=["diabetic_eye_exam", "diabetic_foot_exam", "cgm_placement", "monofilament_test", "dilated_fundus_exam"],
        pharmacy_weight=0.95,
        lab_weight=0.98,
        specialist_weight=0.70,
        procedure_weight=0.65,
        category="Diabetes",
    ),
    "HCC19": HCCDefinitionLite(
        code="HCC19",
        name="Diabetes without Complication",
        description="Type 2 diabetes without documented complications",
        annual_payment_impact=600.0,
        expected_medications=["metformin", "glipizide", "sitagliptin"],
        expected_labs=["hba1c", "fasting_glucose", "lipid_panel"],
        expected_specialists=["primary_care", "endocrinology"],
        expected_procedures=["diabetic_eye_exam", "diabetic_foot_exam"],
        pharmacy_weight=0.90,
        lab_weight=0.95,
        specialist_weight=0.40,
        procedure_weight=0.50,
        category="Diabetes",
    ),
    "HCC85": HCCDefinitionLite(
        code="HCC85",
        name="Congestive Heart Failure",
        description="Heart failure",
        annual_payment_impact=3200.0,
        expected_medications=["lisinopril", "carvedilol", "metoprolol", "furosemide", "spironolactone", "sacubitril_valsartan", "empagliflozin"],
        expected_labs=["bnp", "nt_probnp", "comprehensive_metabolic", "cbc", "tsh", "iron_studies", "lipid_panel"],
        expected_specialists=["cardiology", "heart_failure_clinic"],
        expected_procedures=["echocardiogram", "cardiac_cath", "stress_test", "cardiac_mri", "chest_xray"],
        pharmacy_weight=0.98,
        lab_weight=0.95,
        specialist_weight=0.85,
        procedure_weight=0.90,
        category="Heart Disease",
    ),
    "HCC111": HCCDefinitionLite(
        code="HCC111",
        name="Chronic Obstructive Pulmonary Disease",
        description="COPD",
        annual_payment_impact=1800.0,
        expected_medications=["albuterol", "ipratropium", "tiotropium", "fluticasone_salmeterol", "budesonide_formoterol", "prednisone", "azithromycin"],
        expected_labs=["cbc", "comprehensive_metabolic", "abg", "alpha1_antitrypsin"],
        expected_specialists=["pulmonology"],
        expected_procedures=["spirometry", "pft", "chest_ct", "chest_xray", "6mwt", "oxygen_assessment"],
        pharmacy_weight=0.95,
        lab_weight=0.70,
        specialist_weight=0.75,
        procedure_weight=0.85,
        category="Respiratory",
    ),
}


def _poisson(lam: float) -> int:
    # Knuth algorithm (good enough for small lam)
    if lam <= 0:
        return 0
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= random.random()
    return k - 1


def _weighted_sample_without_replacement(items: List[str], weights: List[float], k: int) -> List[str]:
    items = list(items)
    weights = list(weights)
    chosen: List[str] = []
    for _ in range(min(k, len(items))):
        pick = random.choices(items, weights=weights, k=1)[0]
        idx = items.index(pick)
        chosen.append(pick)
        items.pop(idx)
        weights.pop(idx)
    return chosen


class MedicareSyntheticGeneratorLite:
    def __init__(
        self,
        n_beneficiaries: int = 1000,
        fraud_rate: float = 0.15,
        seed: int = 42,
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2024, 12, 31),
        hcc_definitions: Optional[Dict[str, HCCDefinitionLite]] = None,
    ):
        self.n_beneficiaries = int(n_beneficiaries)
        self.fraud_rate = float(fraud_rate)
        self.seed = int(seed)
        self.start_date = start_date
        self.end_date = end_date
        random.seed(self.seed)

        self.hcc_definitions = hcc_definitions or HCC_DEFINITIONS_LITE
        self.generated_data: Optional[Dict[str, List[Dict]]] = None

    def _beneficiary_id(self, index: int) -> str:
        h = hashlib.md5(f"BENE_{index}_{self.seed}".encode()).hexdigest()[:10].upper()
        return f"BENE_{h}"

    def _random_date(self) -> datetime:
        delta = (self.end_date - self.start_date).days
        return self.start_date + timedelta(days=random.randint(0, max(0, delta)))

    def _random_date_inactive(self) -> datetime:
        """
        Generate a date OUTSIDE the measurement window, simulating an inactive condition
        (coding lag): old evidence exists, but not in the current measurement period.
        """
        # Default: 2-4 years before the measurement window
        years_back = random.randint(2, 4)
        base = self.start_date - timedelta(days=365 * years_back)
        return base + timedelta(days=random.randint(0, 365))

    def _generate_demographics(self) -> List[Dict]:
        states = [
            "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
            "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
        ]
        state_w = [
            0.12, 0.10, 0.11, 0.09, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04,
            0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
        ]

        out = []
        for i in range(self.n_beneficiaries):
            age = int(round(random.gauss(72, 10)))
            age = min(100, max(65, age))
            sex = random.choices(["M", "F"], weights=[0.45, 0.55], k=1)[0]
            state = random.choices(states, weights=state_w, k=1)[0]
            out.append(
                {
                    "beneficiary_id": self._beneficiary_id(i),
                    "age": age,
                    "sex": sex,
                    "state": state,
                }
            )
        return out

    def _hcc_prevalence_weights(self, age: int) -> Dict[str, float]:
        base = {
            "HCC18": 0.15,
            "HCC19": 0.20,
            "HCC85": 0.10,
            "HCC111": 0.12,
        }
        # age bumps
        if age >= 80 and "HCC85" in base:
            base["HCC85"] *= 1.5
        if age >= 75 and "HCC18" in base:
            base["HCC18"] *= 1.3
        if age >= 75 and "HCC111" in base:
            base["HCC111"] *= 1.2

        # normalize
        total = sum(base.values()) or 1.0
        return {k: v / total for k, v in base.items()}

    def _pick_hcc_codes_for_age(self, age: int) -> List[str]:
        hccs = list(self.hcc_definitions.keys())
        lam = (age - 65) / 15 + 1
        base_count = max(1, _poisson(lam))
        w_map = self._hcc_prevalence_weights(age)
        weights = [w_map.get(h, 0.01) for h in hccs]
        return _weighted_sample_without_replacement(hccs, weights, k=min(base_count, len(hccs)))

    def _assign_hccs(self, demographics: List[Dict]) -> None:
        hccs = list(self.hcc_definitions.keys())
        for row in demographics:
            age = int(row["age"])
            lam = (age - 65) / 15 + 1
            base_count = max(1, _poisson(lam))
            w_map = self._hcc_prevalence_weights(age)
            weights = [w_map.get(h, 0.01) for h in hccs]
            picked = _weighted_sample_without_replacement(hccs, weights, k=min(base_count, len(hccs)))
            row["hcc_codes"] = picked

    def _assign_fraud(self, demographics: List[Dict]) -> None:
        n_fraud = int(self.n_beneficiaries * self.fraud_rate)
        fraud_idx = set(random.sample(range(self.n_beneficiaries), k=min(n_fraud, self.n_beneficiaries)))
        # Align with NIW RFE writeup:
        # - paper_diagnosis: complete absence of evidence
        # - upcoding: severity mismatch (evidence exists, but insufficient for severity)
        # - hra_only: documented only via HRA with no follow-up
        # - coding_lag: inactive/old condition (evidence exists historically, not in measurement window)
        fraud_types = ["paper_diagnosis", "upcoding", "hra_only", "coding_lag"]
        fraud_w = [0.35, 0.30, 0.20, 0.15]

        for i, row in enumerate(demographics):
            is_fraud = i in fraud_idx
            row["is_fraudulent"] = is_fraud
            row["fraud_type"] = random.choices(fraud_types, weights=fraud_w, k=1)[0] if is_fraud else ""

    def _gen_pharmacy(self, bene_id: str, hcc_codes: List[str], is_fraud: bool, fraud_type: str) -> List[Dict]:
        claims: List[Dict] = []
        for hcc in hcc_codes:
            d = self.hcc_definitions[hcc]
            if is_fraud and fraud_type == "paper_diagnosis":
                continue
            if is_fraud and fraud_type == "upcoding":
                # Severity mismatch: generate "basic" evidence (e.g., only metformin)
                meds = d.expected_medications[: min(1, len(d.expected_medications))]
            else:
                meds = random.sample(d.expected_medications, k=random.randint(1, max(1, len(d.expected_medications))))
            for med in meds:
                for _ in range(random.randint(3, 6)):
                    if is_fraud and fraud_type == "coding_lag":
                        service_date = self._random_date_inactive()
                    else:
                        service_date = self._random_date()
                    claims.append(
                        {
                            "beneficiary_id": bene_id,
                            "claim_type": "pharmacy",
                            "service_date": service_date.strftime("%Y-%m-%d"),
                            "medication_name": med,
                            "days_supply": random.choice([30, 90]),
                            "related_hcc": hcc,
                        }
                    )
        return claims

    def _gen_labs(self, bene_id: str, hcc_codes: List[str], is_fraud: bool, fraud_type: str) -> List[Dict]:
        claims: List[Dict] = []
        for hcc in hcc_codes:
            d = self.hcc_definitions[hcc]
            if is_fraud and fraud_type == "paper_diagnosis":
                continue
            if is_fraud and fraud_type == "upcoding":
                labs = d.expected_labs[: min(2, len(d.expected_labs))]
            else:
                labs = random.sample(d.expected_labs, k=random.randint(1, max(1, len(d.expected_labs))))
            for lab in labs:
                for _ in range(random.randint(1, 3)):
                    if is_fraud and fraud_type == "coding_lag":
                        service_date = self._random_date_inactive()
                    else:
                        service_date = self._random_date()
                    claims.append(
                        {
                            "beneficiary_id": bene_id,
                            "claim_type": "laboratory",
                            "service_date": service_date.strftime("%Y-%m-%d"),
                            "lab_name": lab,
                            "related_hcc": hcc,
                        }
                    )
        return claims

    def _gen_specialists(self, bene_id: str, hcc_codes: List[str], is_fraud: bool, fraud_type: str) -> List[Dict]:
        claims: List[Dict] = []
        for hcc in hcc_codes:
            d = self.hcc_definitions[hcc]
            if is_fraud and fraud_type in ["paper_diagnosis", "hra_only"]:
                continue
            specs = d.expected_specialists[:1] if (is_fraud and fraud_type == "upcoding") else d.expected_specialists
            for spec in specs:
                for _ in range(random.randint(1, 3)):
                    if is_fraud and fraud_type == "coding_lag":
                        service_date = self._random_date_inactive()
                    else:
                        service_date = self._random_date()
                    claims.append(
                        {
                            "beneficiary_id": bene_id,
                            "claim_type": "specialist_visit",
                            "service_date": service_date.strftime("%Y-%m-%d"),
                            "specialty_name": spec,
                            "related_hcc": hcc,
                        }
                    )
        return claims

    def _gen_procedures(self, bene_id: str, hcc_codes: List[str], is_fraud: bool, fraud_type: str) -> List[Dict]:
        claims: List[Dict] = []
        for hcc in hcc_codes:
            d = self.hcc_definitions[hcc]
            if is_fraud and fraud_type in ["paper_diagnosis", "hra_only"]:
                continue
            procs = d.expected_procedures[:1] if (is_fraud and fraud_type == "upcoding") else d.expected_procedures
            for proc in procs:
                if is_fraud and fraud_type == "coding_lag":
                    service_date = self._random_date_inactive()
                else:
                    service_date = self._random_date()
                claims.append(
                    {
                        "beneficiary_id": bene_id,
                        "claim_type": "procedure",
                        "service_date": service_date.strftime("%Y-%m-%d"),
                        "procedure_name": proc,
                        "related_hcc": hcc,
                    }
                )
        return claims

    def _gen_hra(self, bene_id: str, hcc_codes: List[str], is_fraud: bool, fraud_type: str) -> List[Dict]:
        out = []
        for hcc in hcc_codes:
            d = self.hcc_definitions[hcc]
            out.append(
                {
                    "beneficiary_id": bene_id,
                    "record_type": "hra",
                    "assessment_date": self._random_date().strftime("%Y-%m-%d"),
                    "hcc_code": hcc,
                    "hcc_name": d.name,
                    "assessment_company": random.choice(
                        ["ABC Assessment Services", "HealthCheck Assessors Inc", "In-Home Medical Evaluations", "Senior Care Assessments LLC"]
                    ),
                    "is_hra_only": bool(is_fraud and fraud_type == "hra_only"),
                }
            )
        return out

    @staticmethod
    def _rand_npi() -> str:
        # synthetic 10-digit NPI-like string
        return "".join(str(random.randint(0, 9)) for _ in range(10))

    def _gen_diagnosis_records(self, bene_id: str, hcc_codes: List[str], is_fraud: bool, fraud_type: str) -> List[Dict]:
        """
        Diagnosis/HCC records aligned with NIW + architecture doc:
        beneficiary_id, hcc_code, diagnosis_date, source_type, provider_npi, claim_id.
        """
        rows: List[Dict] = []
        for hcc in hcc_codes:
            if is_fraud and fraud_type == "hra_only":
                source_type = "HRA"
            elif is_fraud and fraud_type == "paper_diagnosis":
                source_type = random.choice(["RAPS", "EDS"])
            else:
                source_type = random.choice(["Encounter", "EDS"])

            if is_fraud and fraud_type == "coding_lag":
                diag_date = self._random_date_inactive()
            else:
                diag_date = self._random_date()

            rows.append(
                {
                    "beneficiary_id": bene_id,
                    "hcc_code": hcc,
                    "diagnosis_date": diag_date.strftime("%Y-%m-%d"),
                    "source_type": source_type,  # RAPS, EDS, Encounter, HRA
                    "provider_npi": self._rand_npi(),
                    "claim_id": f"CLM_{hashlib.md5(f'{bene_id}_{hcc}_{diag_date}'.encode()).hexdigest()[:12].upper()}",
                }
            )
        return rows

    def generate(self) -> Dict[str, List[Dict]]:
        demographics = self._generate_demographics()
        self._assign_hccs(demographics)
        self._assign_fraud(demographics)

        pharmacy, labs, specs, procs, hra, dx, labels = [], [], [], [], [], [], []

        for row in demographics:
            bene_id = row["beneficiary_id"]
            hcc_codes = row["hcc_codes"]
            is_fraud = bool(row["is_fraudulent"])
            fraud_type = row.get("fraud_type", "")

            dx.extend(self._gen_diagnosis_records(bene_id, hcc_codes, is_fraud, fraud_type))
            pharmacy.extend(self._gen_pharmacy(bene_id, hcc_codes, is_fraud, fraud_type))
            labs.extend(self._gen_labs(bene_id, hcc_codes, is_fraud, fraud_type))
            specs.extend(self._gen_specialists(bene_id, hcc_codes, is_fraud, fraud_type))
            procs.extend(self._gen_procedures(bene_id, hcc_codes, is_fraud, fraud_type))
            hra.extend(self._gen_hra(bene_id, hcc_codes, is_fraud, fraud_type))

            # ground-truth coherence label (for evaluation)
            for hcc in hcc_codes:
                if is_fraud:
                    if fraud_type == "paper_diagnosis":
                        score = random.uniform(0.0, 0.15)
                    elif fraud_type == "upcoding":
                        score = random.uniform(0.20, 0.45)
                    elif fraud_type == "hra_only":
                        score = random.uniform(0.05, 0.25)
                    else:
                        score = random.uniform(0.0, 0.30)
                else:
                    score = random.uniform(0.70, 0.99)

                labels.append(
                    {
                        "beneficiary_id": bene_id,
                        "hcc_code": hcc,
                        "coherence_score": round(score, 3),
                        "is_fraudulent": is_fraud,
                        "fraud_type": fraud_type,
                    }
                )

        self.generated_data = {
            "meta": [
                {
                    "measurement_start_date": self.start_date.strftime("%Y-%m-%d"),
                    "measurement_end_date": self.end_date.strftime("%Y-%m-%d"),
                    "seed": self.seed,
                }
            ],
            "beneficiaries": demographics,
            "diagnosis_records": dx,
            "pharmacy_claims": pharmacy,
            "lab_claims": labs,
            "specialist_visits": specs,
            "procedure_claims": procs,
            "hra_records": hra,
            "labels": labels,
        }
        return self.generated_data

    def generate_to_disk(
        self,
        output_dir: str,
        keep_in_memory: bool = False,
        progress_every: int = 5000,
    ) -> Optional[Dict[str, List[Dict]]]:
        """
        Stream-generate a large synthetic dataset directly to CSV files.

        This is the recommended approach for *very large* n_beneficiaries since it avoids
        holding all claims in memory.
        """
        os.makedirs(output_dir, exist_ok=True)

        # precompute fraud indices for reproducibility
        n_fraud = int(self.n_beneficiaries * self.fraud_rate)
        fraud_idx = set(random.sample(range(self.n_beneficiaries), k=min(n_fraud, self.n_beneficiaries)))
        fraud_types = ["paper_diagnosis", "upcoding", "hra_only", "coding_lag"]
        fraud_w = [0.35, 0.30, 0.20, 0.15]

        def open_writer(name: str, fieldnames: List[str]):
            f = open(os.path.join(output_dir, f"{name}.csv"), "w", newline="")
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            return f, w

        files = []
        try:
            meta_f, meta_w = open_writer("meta", ["measurement_start_date", "measurement_end_date", "seed"])
            files.append(meta_f)
            meta_w.writerow(
                {
                    "measurement_start_date": self.start_date.strftime("%Y-%m-%d"),
                    "measurement_end_date": self.end_date.strftime("%Y-%m-%d"),
                    "seed": self.seed,
                }
            )

            bene_f, bene_w = open_writer(
                "beneficiaries",
                ["beneficiary_id", "age", "sex", "state", "hcc_codes", "is_fraudulent", "fraud_type"],
            )
            dx_f, dx_w = open_writer(
                "diagnosis_records",
                ["beneficiary_id", "hcc_code", "diagnosis_date", "source_type", "provider_npi", "claim_id"],
            )
            ph_f, ph_w = open_writer(
                "pharmacy_claims",
                ["beneficiary_id", "claim_type", "service_date", "medication_name", "days_supply", "related_hcc"],
            )
            lab_f, lab_w = open_writer(
                "lab_claims",
                ["beneficiary_id", "claim_type", "service_date", "lab_name", "related_hcc"],
            )
            spec_f, spec_w = open_writer(
                "specialist_visits",
                ["beneficiary_id", "claim_type", "service_date", "specialty_name", "related_hcc"],
            )
            proc_f, proc_w = open_writer(
                "procedure_claims",
                ["beneficiary_id", "claim_type", "service_date", "procedure_name", "related_hcc"],
            )
            hra_f, hra_w = open_writer(
                "hra_records",
                ["beneficiary_id", "record_type", "assessment_date", "hcc_code", "hcc_name", "assessment_company", "is_hra_only"],
            )
            labels_f, labels_w = open_writer(
                "labels",
                ["beneficiary_id", "hcc_code", "coherence_score", "is_fraudulent", "fraud_type"],
            )
            files.extend([bene_f, dx_f, ph_f, lab_f, spec_f, proc_f, hra_f, labels_f])

            # in-memory accumulation (optional)
            mem = {
                "meta": [
                    {
                        "measurement_start_date": self.start_date.strftime("%Y-%m-%d"),
                        "measurement_end_date": self.end_date.strftime("%Y-%m-%d"),
                        "seed": self.seed,
                    }
                ],
                "beneficiaries": [],
                "diagnosis_records": [],
                "pharmacy_claims": [],
                "lab_claims": [],
                "specialist_visits": [],
                "procedure_claims": [],
                "hra_records": [],
                "labels": [],
            } if keep_in_memory else None

            # demographics generation, row-by-row
            states = [
                "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
                "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
            ]
            state_w = [
                0.12, 0.10, 0.11, 0.09, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04,
                0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02,
            ]

            for i in range(self.n_beneficiaries):
                bene_id = self._beneficiary_id(i)
                age = int(round(random.gauss(72, 10)))
                age = min(100, max(65, age))
                sex = random.choices(["M", "F"], weights=[0.45, 0.55], k=1)[0]
                state = random.choices(states, weights=state_w, k=1)[0]
                hcc_codes = self._pick_hcc_codes_for_age(age)

                is_fraud = i in fraud_idx
                fraud_type = random.choices(fraud_types, weights=fraud_w, k=1)[0] if is_fraud else ""

                bene_row = {
                    "beneficiary_id": bene_id,
                    "age": age,
                    "sex": sex,
                    "state": state,
                    "hcc_codes": json.dumps(hcc_codes),
                    "is_fraudulent": bool(is_fraud),
                    "fraud_type": fraud_type,
                }
                bene_w.writerow(bene_row)
                if mem is not None:
                    mem["beneficiaries"].append({**bene_row, "hcc_codes": hcc_codes})

                # diagnosis records
                dx_rows = self._gen_diagnosis_records(bene_id, hcc_codes, bool(is_fraud), fraud_type)
                for r in dx_rows:
                    dx_w.writerow(r)
                if mem is not None:
                    mem["diagnosis_records"].extend(dx_rows)

                # claims
                ph_rows = self._gen_pharmacy(bene_id, hcc_codes, bool(is_fraud), fraud_type)
                for r in ph_rows:
                    ph_w.writerow(r)
                lab_rows = self._gen_labs(bene_id, hcc_codes, bool(is_fraud), fraud_type)
                for r in lab_rows:
                    lab_w.writerow(r)
                spec_rows = self._gen_specialists(bene_id, hcc_codes, bool(is_fraud), fraud_type)
                for r in spec_rows:
                    spec_w.writerow(r)
                proc_rows = self._gen_procedures(bene_id, hcc_codes, bool(is_fraud), fraud_type)
                for r in proc_rows:
                    proc_w.writerow(r)

                if mem is not None:
                    mem["pharmacy_claims"].extend(ph_rows)
                    mem["lab_claims"].extend(lab_rows)
                    mem["specialist_visits"].extend(spec_rows)
                    mem["procedure_claims"].extend(proc_rows)

                hra_rows = self._gen_hra(bene_id, hcc_codes, bool(is_fraud), fraud_type)
                for r in hra_rows:
                    hra_w.writerow(r)
                if mem is not None:
                    mem["hra_records"].extend(hra_rows)

                # labels
                for hcc in hcc_codes:
                    if is_fraud:
                        if fraud_type == "paper_diagnosis":
                            score = random.uniform(0.0, 0.15)
                        elif fraud_type == "upcoding":
                            score = random.uniform(0.20, 0.45)
                        elif fraud_type == "hra_only":
                            score = random.uniform(0.05, 0.25)
                        elif fraud_type == "coding_lag":
                            score = random.uniform(0.0, 0.20)
                        else:
                            score = random.uniform(0.0, 0.30)
                    else:
                        score = random.uniform(0.70, 0.99)
                    label = {
                        "beneficiary_id": bene_id,
                        "hcc_code": hcc,
                        "coherence_score": round(score, 3),
                        "is_fraudulent": bool(is_fraud),
                        "fraud_type": fraud_type,
                    }
                    labels_w.writerow(label)
                    if mem is not None:
                        mem["labels"].append(label)

                if progress_every and (i + 1) % progress_every == 0:
                    print(f"Generated {i+1:,}/{self.n_beneficiaries:,} beneficiaries → {output_dir}")

            if mem is not None:
                self.generated_data = mem
                return mem
            self.generated_data = None
            return None
        finally:
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass

    def save(self, output_dir: str) -> None:
        if self.generated_data is None:
            raise ValueError("No data generated yet. Call generate() first.")
        os.makedirs(output_dir, exist_ok=True)
        for name, rows in self.generated_data.items():
            path = os.path.join(output_dir, f"{name}.csv")
            self._write_csv(path, rows)

    @staticmethod
    def _write_csv(path: str, rows: List[Dict]) -> None:
        if not rows:
            with open(path, "w", newline="") as f:
                f.write("")
            return

        # Collect all keys to keep schema consistent
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                out = {}
                for k in fieldnames:
                    v = r.get(k, "")
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v)
                    out[k] = v
                w.writerow(out)

