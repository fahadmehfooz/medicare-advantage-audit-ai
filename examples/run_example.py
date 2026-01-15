#!/usr/bin/env python3
"""
MCCV Example: Generate Synthetic Data and Run Coherence Validation

This script demonstrates the complete MCCV workflow:
1. Generate synthetic Medicare Advantage claims data
2. Build the clinical knowledge graph
3. Score diagnoses for clinical coherence
4. Generate audit reports for flagged cases

Author: Fahad Mehfooz
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mccv.data.synthetic_generator import MedicareSyntheticGenerator
from mccv.data.knowledge_graph import ClinicalKnowledgeGraph
from mccv.inference.audit_report import AuditReportGenerator
import pandas as pd


def main():
    print("=" * 80)
    print("MCCV: Multimodal Clinical Coherence Validation")
    print("Example Workflow")
    print("=" * 80)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic Medicare claims data...")
    generator = MedicareSyntheticGenerator(
        n_beneficiaries=1000,
        fraud_rate=0.15,
        seed=42
    )
    data = generator.generate()
    
    # Print statistics
    stats = generator.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total beneficiaries: {stats['n_beneficiaries']}")
    print(f"  Fraudulent cases: {stats['n_fraudulent']} ({stats['fraud_rate']*100:.1f}%)")
    print(f"  Mean HCC codes per beneficiary: {stats['mean_hcc_count']:.2f}")
    print(f"  Mean coherence (legitimate): {stats['mean_coherence_legitimate']:.3f}")
    print(f"  Mean coherence (fraudulent): {stats['mean_coherence_fraudulent']:.3f}")
    
    # Step 2: Build knowledge graph
    print("\n[Step 2] Loading clinical knowledge graph...")
    kg = ClinicalKnowledgeGraph()
    kg.load_guidelines("ADA")
    kg.load_guidelines("ACC_AHA")
    kg.load_guidelines("KDIGO")
    kg.load_guidelines("GOLD")
    kg.load_guidelines("NCCN")
    
    print(f"  Loaded guidelines: {', '.join(kg.loaded_guidelines)}")
    print(f"  Diagnosis nodes: {len(kg.diagnoses)}")
    print(f"  Treatment edges: {len(kg.edges)}")
    
    # Step 3: Identify high-risk cases
    print("\n[Step 3] Identifying high-risk diagnoses...")
    labels = data["labels"]
    high_risk = labels[labels["coherence_score"] < 0.3].copy()
    
    print(f"  High-risk diagnosis codes: {len(high_risk)}")
    print(f"  By fraud type:")
    if "fraud_type" in high_risk.columns:
        fraud_counts = high_risk[high_risk["is_fraudulent"]]["fraud_type"].value_counts()
        for fraud_type, count in fraud_counts.items():
            print(f"    - {fraud_type}: {count}")
    
    # Step 4: Generate sample audit report
    print("\n[Step 4] Generating sample audit report...")
    report_generator = AuditReportGenerator(kg)
    
    if len(high_risk) > 0:
        sample = high_risk.iloc[0]
        report = report_generator.generate_report(
            beneficiary_id=sample["beneficiary_id"],
            hcc_code=sample["hcc_code"],
            coherence_score=sample["coherence_score"],
            diagnosis_origin={
                "source": "Health Risk Assessment",
                "date": "2024-03-15",
                "provider": "ABC Assessment Services"
            }
        )
        
        print("\n" + report_generator.format_report(report))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"High-Risk Diagnoses: {len(high_risk)}")
    print(f"Estimated Annual Overpayment: ${len(high_risk) * 2000:,.2f}")
    
    return data, high_risk


if __name__ == "__main__":
    data, high_risk = main()
