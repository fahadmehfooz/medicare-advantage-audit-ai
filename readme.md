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

MCCV employs a Heterogeneous Graph Neural Network (HGT) with Cross-Modal Transformer Attention:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCCV Architecture                             │
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
│              │  GNN Encoder Stack          │                     │
│              │  • HINormer (heterogeneous) │                     │
│              │  • GraphSAGE (inductive)    │                     │
│              │  • GATv2 (attention)        │                     │
│              └─────────────┬───────────────┘                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │  Cross-Modal Transformer    │                     │
│              │  Attention Layer            │                     │
│              └─────────────┬───────────────┘                     │
│                            ▼                                     │
│              ┌─────────────────────────────┐                     │
│              │  Coherence Score [0,1]      │                     │
│              │  + SHAP Explanations        │                     │
│              └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

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
from mccv.data.synthetic_generator import MedicareSyntheticGenerator

generator = MedicareSyntheticGenerator(n_beneficiaries=10000, fraud_rate=0.15)
data = generator.generate()
```

### 2. Train MCCV Model

```python
from mccv.models.mccv_model import MCCVModel

model = MCCVModel(hidden_dim=256, num_heads=8, num_layers=3)
model.fit(train_data, val_data, epochs=100)
```

### 3. Generate Coherence Scores

```python
results = model.predict(test_data)
high_risk = results[results['coherence_score'] < 0.3]
```

## License

Apache License 2.0

## Author

Fahad Mehfooz - [kaggle.com/fahadmehfooz](https://kaggle.com/fahadmehfooz)
