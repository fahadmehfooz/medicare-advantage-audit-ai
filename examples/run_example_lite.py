#!/usr/bin/env python3
"""
MCCV Lite Example (no numpy/pandas/torch)

Workflow:
1) Generate synthetic data
2) Build lite knowledge graph
3) Score coherence (rule-based)
4) Print an audit report for one high-risk case
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mccv.lite.synthetic_generator import MedicareSyntheticGeneratorLite, HCC_DEFINITIONS_LITE
from mccv.lite.knowledge_graph import ClinicalKnowledgeGraphLite
from mccv.lite.rule_based_scorer import RuleBasedCoherenceScorerLite
from mccv.lite.audit_report import AuditReportGeneratorLite


def main():
    print("=" * 80)
    print("MCCV LITE: Multimodal Clinical Coherence Validation (Prototype)")
    print("=" * 80)

    print("\n[Step 1] Generating synthetic data (lite)...")
    gen = MedicareSyntheticGeneratorLite(n_beneficiaries=300, fraud_rate=0.15, seed=42)
    data = gen.generate()

    print("\n[Step 2] Building lite knowledge graph...")
    kg = ClinicalKnowledgeGraphLite(hcc_definitions={k: v.__dict__ for k, v in HCC_DEFINITIONS_LITE.items()})

    print("\n[Step 3] Scoring coherence (rule-based lite)...")
    weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mccv", "configs", "hcc_weights.yaml")
    meta = (data.get("meta") or [{}])[0]
    scorer = RuleBasedCoherenceScorerLite(
        knowledge_graph=kg,
        weights_config_path=weights_path,
        measurement_start_date=meta.get("measurement_start_date"),
        measurement_end_date=meta.get("measurement_end_date"),
    )
    preds = scorer.score_dataset(data)
    high_risk = [p for p in preds if p["coherence_score"] < scorer.high_risk_threshold]
    print(f"High-risk (score < {scorer.high_risk_threshold:.2f}): {len(high_risk)} / {len(preds)}")
    # Align with NIW: show which category/fraud type dominates flagged cases (ground truth from generator)
    labels = data.get("labels") or []
    by_key = {(l["beneficiary_id"], l["hcc_code"]): l for l in labels}
    fraud_counts = {}
    for p in high_risk:
        l = by_key.get((p["beneficiary_id"], p["hcc_code"]))
        if not l or not l.get("is_fraudulent"):
            continue
        ft = l.get("fraud_type") or "unknown"
        fraud_counts[ft] = fraud_counts.get(ft, 0) + 1
    if fraud_counts:
        print("High-risk breakdown (ground truth fraud_type):")
        for k in sorted(fraud_counts.keys()):
            print(f"  - {k}: {fraud_counts[k]}")

    print("\n[Step 4] Generating audit report for one high-risk example...")
    all_claims = []
    for k in ["pharmacy_claims", "lab_claims", "specialist_visits", "procedure_claims"]:
        all_claims.extend(data.get(k, []))

    if high_risk:
        sample = high_risk[0]
        report_gen = AuditReportGeneratorLite(kg)
        weights_for_report = scorer.get_weights_for_report(sample["hcc_code"])
        report = report_gen.generate_report(
            beneficiary_id=sample["beneficiary_id"],
            hcc_code=sample["hcc_code"],
            coherence_score=sample["coherence_score"],
            all_claims=all_claims,
            diagnosis_origin={"source": "Health Risk Assessment", "date": "2024-03-15", "provider": "ABC Assessment Services"},
            weights=weights_for_report,
        )
        print("\n" + report_gen.format_report(report))
    else:
        print("No high-risk cases found (try increasing fraud_rate).")


if __name__ == "__main__":
    main()

