"""
Pure-Python rule-based coherence scorer (lite).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from mccv.lite.knowledge_graph import ClinicalKnowledgeGraphLite


def _lower_set(values: List[str]) -> set[str]:
    out = set()
    for v in values or []:
        s = str(v).strip().lower()
        if s:
            out.add(s)
    return out


@dataclass(frozen=True)
class ModalityBreakdownLite:
    pharmacy_score: float
    laboratory_score: float
    specialist_score: float
    procedure_score: float


class RuleBasedCoherenceScorerLite:
    def __init__(
        self,
        knowledge_graph: ClinicalKnowledgeGraphLite,
        weights_config_path: Optional[str] = None,
        measurement_start_date: Optional[str] = None,
        measurement_end_date: Optional[str] = None,
    ):
        self.kg = knowledge_graph
        self.weights = {}
        self.default_weights = {"pharmacy": 0.80, "laboratory": 0.80, "specialist": 0.70, "procedure": 0.70}
        self.high_risk_threshold = 0.30
        self.medium_risk_threshold = 0.60
        self.measurement_start_date = measurement_start_date
        self.measurement_end_date = measurement_end_date

        if weights_config_path:
            with open(weights_config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            self.weights = cfg.get("hcc_weights") or {}
            scoring = cfg.get("scoring") or {}
            self.default_weights = scoring.get("default_weights") or self.default_weights
            self.high_risk_threshold = float(scoring.get("high_risk_threshold", self.high_risk_threshold))
            self.medium_risk_threshold = float(scoring.get("medium_risk_threshold", self.medium_risk_threshold))

    def _get_hcc_weights(self, hcc_code: str) -> Dict[str, float]:
        entry = self.weights.get(hcc_code, {})
        w = (entry.get("weights") or {}) if isinstance(entry, dict) else {}
        merged = dict(self.default_weights)
        merged.update({k: float(v) for k, v in w.items()})
        total = sum(merged.values())
        if total <= 0:
            return {"pharmacy": 0.25, "laboratory": 0.25, "specialist": 0.25, "procedure": 0.25}
        return {k: v / total for k, v in merged.items()}

    @staticmethod
    def _coverage(found: set[str], expected: set[str]) -> float:
        if not expected:
            return 1.0
        return len(found.intersection(expected)) / max(1, len(expected))

    def get_weights_for_report(self, hcc_code: str) -> Dict[str, float]:
        """
        Return normalized weights keyed by report modality names.
        """
        w = self._get_hcc_weights(hcc_code)
        return {"medication": w["pharmacy"], "lab": w["laboratory"], "specialist": w["specialist"], "procedure": w["procedure"]}

    def _in_window(self, service_date: str) -> bool:
        if not self.measurement_start_date or not self.measurement_end_date:
            return True
        return self.measurement_start_date <= service_date <= self.measurement_end_date

    def score_one(self, beneficiary_id: str, hcc_code: str, all_claims: List[Dict]) -> tuple[float, ModalityBreakdownLite]:
        claims = [
            c
            for c in all_claims
            if c.get("beneficiary_id") == beneficiary_id
            and c.get("related_hcc") == hcc_code
            and self._in_window(str(c.get("service_date", "")))
        ]

        found_ph = _lower_set([c.get("medication_name", "") for c in claims if c.get("claim_type") == "pharmacy"])
        found_lab = _lower_set([c.get("lab_name", "") for c in claims if c.get("claim_type") == "laboratory"])
        found_spec = _lower_set([c.get("specialty_name", "") for c in claims if c.get("claim_type") == "specialist_visit"])
        found_proc = _lower_set([c.get("procedure_name", "") for c in claims if c.get("claim_type") == "procedure"])

        exp_ph = _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "medication")])
        exp_lab = _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "lab")])
        exp_spec = _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "specialist")])
        exp_proc = _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "procedure")])

        s_ph = self._coverage(found_ph, exp_ph)
        s_lab = self._coverage(found_lab, exp_lab)
        s_spec = self._coverage(found_spec, exp_spec)
        s_proc = self._coverage(found_proc, exp_proc)

        w = self._get_hcc_weights(hcc_code)
        score = w["pharmacy"] * s_ph + w["laboratory"] * s_lab + w["specialist"] * s_spec + w["procedure"] * s_proc
        score = max(0.0, min(1.0, float(score)))

        return score, ModalityBreakdownLite(s_ph, s_lab, s_spec, s_proc)

    def _build_claim_index(self, all_claims: List[Dict]) -> Dict[tuple[str, str], List[Dict]]:
        """
        Build an index once so scoring scales to large datasets.
        Keyed by (beneficiary_id, related_hcc) and includes only in-window claims.
        """
        idx: Dict[tuple[str, str], List[Dict]] = {}
        for c in all_claims:
            sd = str(c.get("service_date", ""))
            if not self._in_window(sd):
                continue
            key = (str(c.get("beneficiary_id", "")), str(c.get("related_hcc", "")))
            if key[0] == "" or key[1] == "":
                continue
            idx.setdefault(key, []).append(c)
        return idx

    def score_dataset(self, data: Dict[str, List[Dict]]) -> List[Dict]:
        beneficiaries = data.get("beneficiaries") or []
        all_claims = []
        for key in ["pharmacy_claims", "lab_claims", "specialist_visits", "procedure_claims"]:
            all_claims.extend(data.get(key) or [])

        claim_idx = self._build_claim_index(all_claims)

        # Cache expected sets per HCC (avoids recomputing for each beneficiary)
        exp_cache: Dict[str, Dict[str, set[str]]] = {}
        def expected_sets(hcc_code: str) -> Dict[str, set[str]]:
            if hcc_code in exp_cache:
                return exp_cache[hcc_code]
            exp_cache[hcc_code] = {
                "pharmacy": _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "medication")]),
                "laboratory": _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "lab")]),
                "specialist": _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "specialist")]),
                "procedure": _lower_set([t.treatment_code for t in self.kg.get_expected_treatments(hcc_code, "procedure")]),
            }
            return exp_cache[hcc_code]

        preds: List[Dict] = []
        for b in beneficiaries:
            bene_id = b["beneficiary_id"]
            for hcc in b.get("hcc_codes") or []:
                claims = claim_idx.get((bene_id, hcc), [])

                found_ph = _lower_set([c.get("medication_name", "") for c in claims if c.get("claim_type") == "pharmacy"])
                found_lab = _lower_set([c.get("lab_name", "") for c in claims if c.get("claim_type") == "laboratory"])
                found_spec = _lower_set([c.get("specialty_name", "") for c in claims if c.get("claim_type") == "specialist_visit"])
                found_proc = _lower_set([c.get("procedure_name", "") for c in claims if c.get("claim_type") == "procedure"])

                exp = expected_sets(hcc)
                s_ph = self._coverage(found_ph, exp["pharmacy"])
                s_lab = self._coverage(found_lab, exp["laboratory"])
                s_spec = self._coverage(found_spec, exp["specialist"])
                s_proc = self._coverage(found_proc, exp["procedure"])

                w = self._get_hcc_weights(hcc)
                score = w["pharmacy"] * s_ph + w["laboratory"] * s_lab + w["specialist"] * s_spec + w["procedure"] * s_proc
                score = max(0.0, min(1.0, float(score)))
                br = ModalityBreakdownLite(s_ph, s_lab, s_spec, s_proc)
                preds.append(
                    {
                        "beneficiary_id": bene_id,
                        "hcc_code": hcc,
                        "coherence_score": score,
                        "pharmacy_score": br.pharmacy_score,
                        "lab_score": br.laboratory_score,
                        "specialist_score": br.specialist_score,
                        "procedure_score": br.procedure_score,
                    }
                )
        return preds

