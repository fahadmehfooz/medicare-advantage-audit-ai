#!/usr/bin/env python
"""Simple test to isolate import issues"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

print("Step 1: Import torch...")
import torch
print(f"  ✓ Torch {torch.__version__}")

print("Step 2: Import sklearn...")
from sklearn.metrics import roc_auc_score
print(f"  ✓ sklearn metrics")

print("Step 3: Import numpy...")
import numpy as np
print(f"  ✓ numpy {np.__version__}")

print("Step 4: Import mccv.lite...")
from mccv.lite.synthetic_generator import MedicareSyntheticGeneratorLite
print(f"  ✓ mccv.lite generator")

print("Step 5: Generate small dataset...")
gen = MedicareSyntheticGeneratorLite(n_beneficiaries=100, fraud_rate=0.15, seed=42)
data = gen.generate()
print(f"  ✓ Generated {len(data['beneficiaries'])} beneficiaries")

print("Step 6: Import simple_gnn...")
from mccv.models.simple_gnn import SimpleHeteroGNN, build_heterogeneous_graph_from_data
print(f"  ✓ simple_gnn module")

print("Step 7: Build graph...")
graph_data, labels, diagnosis_ids, _ = build_heterogeneous_graph_from_data(data, max_samples=50)
print(f"  ✓ Built graph with {len(labels)} diagnoses")

print("Step 8: Create model...")
model = SimpleHeteroGNN(hidden_dim=32, num_layers=1)
print(f"  ✓ Model created")

print("Step 9: Forward pass...")
with torch.no_grad():
    preds = model(graph_data)
print(f"  ✓ Forward pass produced {len(preds)} predictions")

print("\n✓ All tests passed!")
