"""
Simple GNN model for MCCV (Proof of Concept)

This is a simplified implementation to demonstrate that graph neural networks
can improve over rule-based coherence scoring. It uses a basic heterogeneous
graph structure and 2-layer message passing.

NOT intended for production - this is a proof-of-concept to show the approach works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class SimpleHeteroGNN(nn.Module):
    """
    Simplified Heterogeneous GNN for clinical coherence validation.
    
    Architecture:
    - Input: Node features for beneficiaries, diagnoses, treatments
    - 2-layer message passing with attention
    - Output: Coherence score [0,1] for each diagnosis
    """
    
    def __init__(
        self, 
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embeddings (simple approach - learnable embeddings)
        self.bene_embedding = nn.Embedding(10000, hidden_dim)  # Max 10k beneficiaries
        self.diag_embedding = nn.Embedding(200, hidden_dim)    # Max 200 diagnosis types
        self.treatment_embedding = nn.Embedding(1000, hidden_dim)  # Max 1k treatments
        
        # Message passing layers
        self.conv_layers = nn.ModuleList([
            HeteroConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output layer (diagnosis coherence score)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, graph_data: Dict) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            graph_data: Dictionary containing:
                - bene_ids: Tensor of beneficiary IDs
                - diag_ids: Tensor of diagnosis IDs  
                - treatment_ids: Tensor of treatment IDs
                - edges: Dictionary of edge types and indices
                - diag_instance_to_type: Mapping from instance to type
                - n_instances: Number of diagnosis instances
                
        Returns:
            Coherence scores [0,1] for each diagnosis INSTANCE
        """
        # Get embeddings
        bene_x = self.bene_embedding(graph_data['bene_ids'])
        diag_x = self.diag_embedding(graph_data['diag_ids'])
        treatment_x = self.treatment_embedding(graph_data['treatment_ids'])
        
        # Store in dict for message passing
        x_dict = {
            'beneficiary': bene_x,
            'diagnosis': diag_x,
            'treatment': treatment_x
        }
        
        # Message passing
        for conv in self.conv_layers:
            x_dict = conv(x_dict, graph_data['edges'])
            # Apply activation
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # Get diagnosis type embeddings
        diag_type_embeddings = x_dict['diagnosis']
        
        # Map to instances using the instance_to_type mapping
        instance_to_type = graph_data['diag_instance_to_type']
        diag_instance_embeddings = diag_type_embeddings[instance_to_type]
        
        # Output coherence scores for diagnosis instances
        diag_scores = self.output(diag_instance_embeddings)
        
        return diag_scores.squeeze(-1)


class HeteroConvLayer(nn.Module):
    """
    Simplified heterogeneous graph convolution layer.
    
    Implements message passing between different node types with attention.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        # Message functions for each edge type
        self.msg_bene_to_diag = nn.Linear(in_dim, out_dim)
        self.msg_diag_to_treatment = nn.Linear(in_dim, out_dim)
        self.msg_treatment_to_diag = nn.Linear(in_dim, out_dim)
        
        # Attention weights
        self.attn = nn.Linear(out_dim, 1)
        
        # Update functions
        self.update_diag = nn.Linear(out_dim, out_dim)
        
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor],
        edges: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Message passing step.
        
        Args:
            x_dict: Node features for each node type
            edges: Edge indices for each edge type
            
        Returns:
            Updated node features
        """
        out_dict = {}
        
        # Keep beneficiary and treatment embeddings unchanged (simplified)
        out_dict['beneficiary'] = x_dict['beneficiary']
        out_dict['treatment'] = x_dict['treatment']
        
        # Update diagnosis embeddings through message passing
        diag_messages = []
        
        # Messages from beneficiaries
        if 'bene_to_diag' in edges:
            edge_index = edges['bene_to_diag']
            if edge_index.size(1) > 0:
                src_feat = x_dict['beneficiary'][edge_index[0]]
                msg = self.msg_bene_to_diag(src_feat)
                # Aggregate to diagnoses
                diag_agg = torch.zeros(
                    x_dict['diagnosis'].size(0), 
                    msg.size(1),
                    device=msg.device
                )
                diag_agg.index_add_(0, edge_index[1], msg)
                diag_messages.append(diag_agg)
        
        # Messages from treatments
        if 'treatment_to_diag' in edges:
            edge_index = edges['treatment_to_diag']
            if edge_index.size(1) > 0:
                src_feat = x_dict['treatment'][edge_index[0]]
                msg = self.msg_treatment_to_diag(src_feat)
                # Aggregate to diagnoses
                diag_agg = torch.zeros(
                    x_dict['diagnosis'].size(0),
                    msg.size(1),
                    device=msg.device
                )
                diag_agg.index_add_(0, edge_index[1], msg)
                diag_messages.append(diag_agg)
        
        # Combine messages (simple averaging)
        if diag_messages:
            combined_msg = torch.stack(diag_messages).mean(dim=0)
            out_dict['diagnosis'] = self.update_diag(combined_msg)
        else:
            out_dict['diagnosis'] = x_dict['diagnosis']
        
        return out_dict


class MCCVGNNTrainer:
    """
    Trainer for the MCCV GNN model.
    """
    
    def __init__(
        self,
        model: SimpleHeteroGNN,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_epoch(
        self,
        graph_data: Dict,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(graph_data)
        
        # Apply mask if provided
        if mask is not None:
            predictions = predictions[mask]
        
        # Compute loss
        loss = self.criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        graph_data: Dict,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model.
        
        Returns:
            loss, predictions, labels
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(graph_data)
            
            # Apply mask if provided
            if mask is not None:
                predictions_masked = predictions[mask]
            else:
                predictions_masked = predictions
            
            loss = self.criterion(predictions_masked, labels)
        
        return (
            loss.item(),
            predictions_masked.cpu().numpy(),
            labels.cpu().numpy()
        )
    
    def predict(self, graph_data: Dict) -> np.ndarray:
        """Get predictions."""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(graph_data)
        
        return predictions.cpu().numpy()


def build_heterogeneous_graph_from_data(
    data: Dict[str, List[Dict]],
    max_samples: Optional[int] = None
) -> Tuple[Dict, np.ndarray, List[str], List[int]]:
    """
    Build heterogeneous graph from MCCV lite data format.
    
    Args:
        data: Dictionary from MedicareSyntheticGeneratorLite
        max_samples: Maximum number of beneficiaries to include
        
    Returns:
        graph_data: Dictionary for GNN
        labels: Ground truth coherence scores
        diagnosis_ids: List of diagnosis identifiers
        diag_instance_to_type: Mapping from instance index to diagnosis type index
    """
    beneficiaries = data['beneficiaries'][:max_samples] if max_samples else data['beneficiaries']
    labels_list = data['labels']
    
    # Create mappings
    bene_to_idx = {b['beneficiary_id']: i for i, b in enumerate(beneficiaries)}
    
    # Collect unique diagnoses and treatments
    unique_diagnoses = set()
    unique_treatments = set()
    
    for b in beneficiaries:
        for hcc in b.get('hcc_codes', []):
            unique_diagnoses.add(hcc)
    
    for claim_type in ['pharmacy_claims', 'lab_claims', 'specialist_visits', 'procedure_claims']:
        for claim in data.get(claim_type, []):
            bene_id = claim.get('beneficiary_id')
            if bene_id in bene_to_idx:
                # Create treatment ID
                if claim_type == 'pharmacy_claims':
                    treatment = f"med_{claim.get('medication_name', 'unknown')}"
                elif claim_type == 'lab_claims':
                    treatment = f"lab_{claim.get('lab_name', 'unknown')}"
                elif claim_type == 'specialist_visits':
                    treatment = f"spec_{claim.get('specialty_name', 'unknown')}"
                else:
                    treatment = f"proc_{claim.get('procedure_name', 'unknown')}"
                unique_treatments.add(treatment)
    
    diag_to_idx = {d: i for i, d in enumerate(sorted(unique_diagnoses))}
    treatment_to_idx = {t: i for i, t in enumerate(sorted(unique_treatments))}
    
    # Build edge lists
    bene_to_diag_edges = []
    treatment_to_diag_edges = []
    
    # Map diagnoses to beneficiaries at TYPE level
    # Track instances separately for final predictions
    diagnosis_records = []
    diag_instance_to_type = []  # Maps instance index to diagnosis type index
    diag_type_seen = set()  # Track which types have edges
    
    for b in beneficiaries:
        bene_id = b['beneficiary_id']
        bene_idx = bene_to_idx[bene_id]
        for hcc in b.get('hcc_codes', []):
            diag_type_idx = diag_to_idx[hcc]
            
            # Add edge from beneficiary to diagnosis TYPE (not instance)
            if (bene_idx, diag_type_idx) not in diag_type_seen:
                bene_to_diag_edges.append([bene_idx, diag_type_idx])
                diag_type_seen.add((bene_idx, diag_type_idx))
            
            # Track instance
            diagnosis_records.append((bene_id, hcc))
            diag_instance_to_type.append(diag_type_idx)
    
    # Map treatments to diagnosis TYPES (not instances)
    for claim_type in ['pharmacy_claims', 'lab_claims', 'specialist_visits', 'procedure_claims']:
        for claim in data.get(claim_type, []):
            bene_id = claim.get('beneficiary_id')
            hcc = claim.get('related_hcc')
            
            if bene_id in bene_to_idx and hcc in diag_to_idx:
                # Create treatment ID
                if claim_type == 'pharmacy_claims':
                    treatment = f"med_{claim.get('medication_name', 'unknown')}"
                elif claim_type == 'lab_claims':
                    treatment = f"lab_{claim.get('lab_name', 'unknown')}"
                elif claim_type == 'specialist_visits':
                    treatment = f"spec_{claim.get('specialty_name', 'unknown')}"
                else:
                    treatment = f"proc_{claim.get('procedure_name', 'unknown')}"
                
                if treatment in treatment_to_idx:
                    treatment_idx = treatment_to_idx[treatment]
                    diag_type_idx = diag_to_idx[hcc]
                    treatment_to_diag_edges.append([treatment_idx, diag_type_idx])
    
    # Convert to tensors
    bene_to_diag_edges = torch.tensor(bene_to_diag_edges, dtype=torch.long).t()
    treatment_to_diag_edges = torch.tensor(treatment_to_diag_edges, dtype=torch.long).t()
    
    # Get labels
    labels = []
    for bene_id, hcc in diagnosis_records:
        # Find label
        label_entry = next(
            (l for l in labels_list 
             if l['beneficiary_id'] == bene_id and l['hcc_code'] == hcc),
            None
        )
        if label_entry:
            # Use ground truth: 1 if not fraudulent, 0 if fraudulent
            labels.append(1.0 - float(label_entry['is_fraudulent']))
        else:
            labels.append(0.5)  # Unknown
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Create graph data structure
    graph_data = {
        'bene_ids': torch.arange(len(bene_to_idx), dtype=torch.long),
        'diag_ids': torch.arange(len(diag_to_idx), dtype=torch.long),
        'treatment_ids': torch.arange(len(treatment_to_idx), dtype=torch.long),
        'edges': {
            'bene_to_diag': bene_to_diag_edges,
            'treatment_to_diag': treatment_to_diag_edges
        },
        'diag_instance_to_type': torch.tensor(diag_instance_to_type, dtype=torch.long),
        'n_instances': len(diagnosis_records)
    }
    
    diagnosis_ids = [f"{bene_id}||{hcc}" for bene_id, hcc in diagnosis_records]
    
    return graph_data, labels.numpy(), diagnosis_ids, diag_instance_to_type
