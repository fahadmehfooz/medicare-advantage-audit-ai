# MCCV: Multimodal Clinical Coherence Validation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**An open-source AI system to detect unsupported diagnosis codes in Medicare Advantage claims through multimodal clinical coherence validation.**

## Problem Statement

Medicare Advantage plans receive higher payments for sicker patients through Hierarchical Condition Category (HCC) codes. This creates financial incentives to document diagnoses that may not reflect actual patient health status. The Medicare Payment Advisory Commission (MedPAC) estimates **$84 billion in annual overpayments** due to coding intensity differences.

Current audit approaches review only medical record documentation. MCCV takes a fundamentally different approach: **validating whether submitted diagnoses have corresponding treatment evidence across multiple data modalities**.

## How MCCV Works

MCCV asks a simple question that no existing tool asks:

> *"If this patient has [DIAGNOSIS], where is the evidence of treatment?"*

### The Four-Modality Approach

| Modality | Data Source | Example for Diabetes (HCC 18) |
|----------|-------------|-------------------------------|
| **Pharmacy** | Part D NDC codes | Metformin, Insulin, SGLT2 inhibitors |
| **Laboratory** | LOINC codes | HbA1c, fasting glucose, creatinine |
| **Specialist** | NPI + specialty | Endocrinology, Nephrology visits |
| **Procedures** | CPT/HCPCS codes | Eye exams, foot exams, glucose monitoring |

### Clinical Coherence Scoring

```
Score 0.9+ : Strong treatment evidence → Likely valid diagnosis
Score 0.5-0.9: Partial evidence → Review recommended  
Score <0.5 : Minimal/no evidence → Flag for audit
```

## Architecture

MCCV currently includes two implementations:

### Phase 1: Rule-Based Prototype (CURRENT - WORKING)
Multimodal coherence scoring using clinical knowledge graphs and treatment coverage calculations:

```
┌─────────────────────────────────────────────────────────────────┐
│                Rule-Based Coherence Scoring                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Pharmacy   │    │  Laboratory  │    │  Specialist  │       │
│  │   Claims     │    │   Claims     │    │   Visits     │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         └───────────────────┼───────────────────┘                │
│                             ▼                                    │
│              ┌─────────────────────────────┐                     │
│              │  Clinical Knowledge Graph   │                     │
│              │  (ADA, ACC/AHA, KDIGO)      │                     │
│              └─────────────┬───────────────┘                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │  Coverage Calculation        │                     │
│              │  (Found / Expected)         │                     │
│              └─────────────┬───────────────┘                     │
│                            ▓                                     │
│              ┌─────────────────────────────┐                     │
│              │  Weighted Coherence Score   │                     │
│              │  [0,1] per diagnosis        │                     │
│              └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: GNN Enhancement (PROOF-OF-CONCEPT - IN DEVELOPMENT)
Heterogeneous Graph Neural Network for learning clinical coherence patterns:

```
┌─────────────────────────────────────────────────────────────────┐
│                  GNN Architecture (SimpleHeteroGNN)              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Pharmacy   │    │  Laboratory  │    │  Procedures  │       │
│  │   (Part D)   │    │   (LOINC)    │    │  (CPT/HCPCS) │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         └───────────────────┼───────────────────┘                │
│                             ▼                                    │
│              ┌─────────────────────────────┐                     │
│              │  Heterogeneous Graph        │                     │
│              │  Construction               │                     │
│              └─────────────┬───────────────┘                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │  Node Embeddings (Learned)  │                     │
│              │  • Beneficiaries (10K max)  │                     │
│              │  • Diagnoses (200 HCCs)     │                     │
│              │  • Treatments (1K codes)    │                     │
│              └─────────────┬───────────────┘                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │  GNN Message Passing        │                     │
│              │  • Layer 1: Aggregate       │                     │
│              │  • Layer 2: Refine          │                     │
│              │  • 752K parameters          │                     │
│              └─────────────┬───────────────┘                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │  MLP Output Layer           │                     │
│              │  64 → 32 → 1 + Sigmoid      │                     │
│              └─────────────┬───────────────┘                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │  Coherence Score [0,1]      │                     │
│              │  per diagnosis instance     │                     │
│              └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

**Status:** Proof-of-concept GNN implementation complete. Architecture is functional and generates predictions. Full training and comparison benchmarks in development.

## Installation

```bash
git clone https://github.com/fahadmehfooz/mccv.git
cd mccv
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Generate Synthetic Data

```python
from mccv import get_synthetic_generator

Generator = get_synthetic_generator()  # lite (pure python) by default
generator = Generator(n_beneficiaries=10000, fraud_rate=0.15)
data = generator.generate()  # dict of lists-of-dicts
```

### 2. Run Rule-Based Prototype (no torch required)

```bash
python examples/run_example_lite.py
```

### 3. Run GNN Proof-of-Concept Demo

```bash
python demo_gnn_simple.py
```

This demonstrates:
- ✅ Working GNN architecture (752K parameters)
- ✅ Heterogeneous graph processing
- ✅ Coherence score generation
- ✅ End-to-end forward pass

### 4. GNN Model Usage

```python
from mccv import get_simple_gnn, get_graph_builder

# Build graph from data
build_graph = get_graph_builder()
graph_data, labels, diagnosis_ids, _ = build_graph(data)

# Create model
SimpleHeteroGNN = get_simple_gnn()
model = SimpleHeteroGNN(hidden_dim=64, num_layers=2)

# Generate predictions
import torch
model.eval()
with torch.no_grad():
    coherence_scores = model(graph_data)

print(f"Generated {len(coherence_scores)} coherence scores")
```

## License

Apache License 2.0

## Author

Fahad Mehfooz - [kaggle.com/fahadmehfooz](https://kaggle.com/fahadmehfooz)
