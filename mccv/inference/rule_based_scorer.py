"""
Rule-based clinical coherence scoring (prototype-friendly).

This scorer provides an end-to-end runnable MCCV prototype without requiring
torch_geometric/GNN dependencies. It combines:
- ClinicalKnowledgeGraph expected treatments (codes)
- Per-HCC modality weights from configs/hcc_weights.yaml
- Simple evidence coverage scoring by modality

It outputs coherence scores in [0, 1] for each (beneficiary_id, hcc_code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

from mccv.data.knowledge_graph import ClinicalKnowledgeGraph


def _lower_set(values) -> set[str]:
    return {str(v).strip().lower() for v in values if pd.notna(v) and str(v).strip() != ""}


@dataclass(frozen=True)
class ModalityBreakdown:
    pharmacy_score: float
    laboratory_score: float
    specialist_score: float
    procedure_score: float


class RuleBasedCoherenceScorer:
    """
    Computes coherence by measuring how much expected evidence is present.

    Score = weighted average of modality coverage ratios:
      coverage(modality) = |found ∩ expected| / |expected|
      if |expected| == 0 => coverage = 1.0
    """

    def __init__(
        self,
        knowledge_graph: Optional[ClinicalKnowledgeGraph] = None,
        weights_config_path: Optional[str] = None,
    ):
        self.kg = knowledge_graph or ClinicalKnowledgeGraph()
        # Ensure guidelines are loaded (so expected_treatments are populated)
        if not self.kg.loaded_guidelines:
            for g in ["ADA", "ACC_AHA", "KDIGO", "GOLD", "NCCN"]:
                self.kg.load_guidelines(g)

        self.weights = {}
        self.default_weights = {"pharmacy": 0.80, "laboratory": 0.80, "specialist": 0.70, "procedure": 0.70}
        self.high_risk_threshold = 0.30
        self.medium_risk_threshold = 0.60

        if weights_config_path:
            with open(weights_config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            self.weights = (cfg.get("hcc_weights") or {})
            scoring = cfg.get("scoring") or {}
            self.default_weights = scoring.get("default_weights") or self.default_weights
            self.high_risk_threshold = float(scoring.get("high_risk_threshold", self.high_risk_threshold))
            self.medium_risk_threshold = float(scoring.get("medium_risk_threshold", self.medium_risk_threshold))

    def _get_hcc_weights(self, hcc_code: str) -> Dict[str, float]:
        entry = self.weights.get(hcc_code, {})
        w = (entry.get("weights") or {}) if isinstance(entry, dict) else {}
        merged = dict(self.default_weights)
        merged.update({k: float(v) for k, v in w.items()})
        # normalize to sum to 1 (if all zeros, fallback)
        total = sum(merged.values())
        if total <= 0:
            return {"pharmacy": 0.25, "laboratory": 0.25, "specialist": 0.25, "procedure": 0.25}
        return {k: v / total for k, v in merged.items()}

    def _expected_codes(self, hcc_code: str) -> Dict[str, set[str]]:
        expected = self.kg.get_expected_treatments(hcc_code)
        expected_by_type = {"pharmacy": set(), "laboratory": set(), "specialist": set(), "procedure": set()}
        for t in expected:
            code = str(t.treatment_code).strip().lower()
            if not code:
                continue
            if t.treatment_type == "medication":
                expected_by_type["pharmacy"].add(code)
            elif t.treatment_type == "lab":
                expected_by_type["laboratory"].add(code)
            elif t.treatment_type == "specialist":
                expected_by_type["specialist"].add(code)
            elif t.treatment_type == "procedure":
                expected_by_type["procedure"].add(code)
        return expected_by_type

    def _found_codes(self, beneficiary_claims: pd.DataFrame, hcc_code: str) -> Dict[str, set[str]]:
        if beneficiary_claims is None or len(beneficiary_claims) == 0:
            return {"pharmacy": set(), "laboratory": set(), "specialist": set(), "procedure": set()}

        claims = beneficiary_claims
        if "related_hcc" in claims.columns:
            claims = claims[claims["related_hcc"] == hcc_code]

        found = {"pharmacy": set(), "laboratory": set(), "specialist": set(), "procedure": set()}
        if len(claims) == 0:
            return found

        found["pharmacy"] = _lower_set(claims.loc[claims["claim_type"] == "pharmacy", "medication_name"])
        found["laboratory"] = _lower_set(claims.loc[claims["claim_type"] == "laboratory", "lab_name"])
        found["specialist"] = _lower_set(claims.loc[claims["claim_type"] == "specialist_visit", "specialty_name"])
        found["procedure"] = _lower_set(claims.loc[claims["claim_type"] == "procedure", "procedure_name"])
        return found

    @staticmethod
    def _coverage(found: set[str], expected: set[str]) -> float:
        if not expected:
            return 1.0
        return len(found.intersection(expected)) / max(1, len(expected))

    def score_beneficiary_hcc(
        self,
        beneficiary_id: str,
        hcc_code: str,
        beneficiary_claims: pd.DataFrame,
    ) -> Tuple[float, ModalityBreakdown]:
        expected = self._expected_codes(hcc_code)
        found = self._found_codes(beneficiary_claims, hcc_code)
        weights = self._get_hcc_weights(hcc_code)

        s_ph = self._coverage(found["pharmacy"], expected["pharmacy"])
        s_lab = self._coverage(found["laboratory"], expected["laboratory"])
        s_spec = self._coverage(found["specialist"], expected["specialist"])
        s_proc = self._coverage(found["procedure"], expected["procedure"])

        score = (
            weights["pharmacy"] * s_ph
            + weights["laboratory"] * s_lab
            + weights["specialist"] * s_spec
            + weights["procedure"] * s_proc
        )
        score = max(0.0, min(1.0, float(score)))
        return score, ModalityBreakdown(s_ph, s_lab, s_spec, s_proc)

    def score_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Score all (beneficiary_id, hcc_code) pairs present in data['labels'] or data['beneficiaries'].
        """
        beneficiaries = data.get("beneficiaries")
        if beneficiaries is None or len(beneficiaries) == 0:
            raise ValueError("data must include a non-empty 'beneficiaries' DataFrame")

        # Build one combined claims table for easy slicing
        claims_tables = []
        for key in ["pharmacy_claims", "lab_claims", "specialist_visits", "procedure_claims"]:
            df = data.get(key)
            if df is not None and len(df) > 0:
                claims_tables.append(df)
        all_claims = pd.concat(claims_tables, axis=0, ignore_index=True) if claims_tables else pd.DataFrame()

        rows = []
        for _, b in beneficiaries.iterrows():
            bene_id = b["beneficiary_id"]
            hcc_list = b.get("hcc_codes", []) or []
            bene_claims = all_claims[all_claims["beneficiary_id"] == bene_id] if len(all_claims) > 0 else pd.DataFrame()

            for hcc in hcc_list:
                score, breakdown = self.score_beneficiary_hcc(bene_id, hcc, bene_claims)
                rows.append(
                    {
                        "beneficiary_id": bene_id,
                        "hcc_code": hcc,
                        "coherence_score": score,
                        "pharmacy_score": breakdown.pharmacy_score,
                        "lab_score": breakdown.laboratory_score,
                        "specialist_score": breakdown.specialist_score,
                        "procedure_score": breakdown.procedure_score,
                    }
                )

        preds = pd.DataFrame(rows)
        return preds

