"""
GNN vs Rule-Based Comparison Utility

Trains the simple GNN model and compares performance against rule-based baseline.
Generates metrics for RFE response.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import time


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (1 = valid, 0 = fraud)
        y_pred: Predicted probabilities [0,1]
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # ROC-AUC
    try:
        auc_roc = roc_auc_score(y_true, y_pred)
    except:
        auc_roc = 0.0
    
    # Average Precision (PR-AUC)
    try:
        avg_precision = average_precision_score(y_true, y_pred)
    except:
        avg_precision = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision at different recall levels (for RFE)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    precision_at_20_recall = 0.0
    for p, r in zip(precisions, recalls):
        if r >= 0.20:
            precision_at_20_recall = p
            break
    
    return {
        'auc_roc': auc_roc,
        'avg_precision': avg_precision,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'precision_at_20_recall': precision_at_20_recall,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }


def train_gnn_model(
    graph_data: Dict,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[object, List[float], List[float]]:
    """
    Train the GNN model.
    
    Args:
        graph_data: Heterogeneous graph structure
        labels: Ground truth labels
        train_mask: Boolean mask for training samples
        val_mask: Boolean mask for validation samples
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        verbose: Print training progress
        
    Returns:
        Trained model, train losses, val losses
    """
    from mccv.models.simple_gnn import SimpleHeteroGNN, MCCVGNNTrainer
    
    # Move data to device
    graph_data_device = {
        'bene_ids': graph_data['bene_ids'].to(device),
        'diag_ids': graph_data['diag_ids'].to(device),
        'treatment_ids': graph_data['treatment_ids'].to(device),
        'edges': {
            k: v.to(device) for k, v in graph_data['edges'].items()
        },
        'diag_instance_to_type': graph_data['diag_instance_to_type'].to(device),
        'n_instances': graph_data['n_instances']
    }
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool).to(device)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool).to(device)
    
    # Initialize model
    model = SimpleHeteroGNN(hidden_dim=64, num_layers=2)
    trainer = MCCVGNNTrainer(model, learning_rate=learning_rate, device=device)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(graph_data_device, labels_tensor[train_mask_tensor], train_mask_tensor)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_preds, val_labels = trainer.evaluate(
            graph_data_device, 
            labels_tensor[val_mask_tensor],
            val_mask_tensor
        )
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return trainer, train_losses, val_losses


def compare_gnn_vs_rulebased(
    data: Dict[str, List[Dict]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_epochs: int = 50,
    device: str = 'cpu',
    random_seed: int = 42
) -> Dict[str, Dict]:
    """
    Compare GNN vs rule-based approach on the same dataset.
    
    Args:
        data: MCCV lite synthetic data
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        num_epochs: Training epochs for GNN
        device: Device to use
        random_seed: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    print("=" * 70)
    print("MCCV: GNN vs Rule-Based Comparison")
    print("=" * 70)
    
    # Build graph for GNN
    print("\n[1/5] Building heterogeneous graph...")
    from mccv.models.simple_gnn import build_heterogeneous_graph_from_data
    graph_data, labels, diagnosis_ids, _ = build_heterogeneous_graph_from_data(data)
    
    n_samples = len(labels)
    print(f"  - Total diagnoses: {n_samples}")
    print(f"  - Fraud cases: {int((labels < 0.5).sum())} ({100*(labels < 0.5).mean():.1f}%)")
    print(f"  - Valid cases: {int((labels >= 0.5).sum())} ({100*(labels >= 0.5).mean():.1f}%)")
    
    # Split data
    print("\n[2/5] Splitting data...")
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    print(f"  - Train: {train_mask.sum()} samples")
    print(f"  - Val: {val_mask.sum()} samples")
    print(f"  - Test: {test_mask.sum()} samples")
    
    # Get rule-based predictions
    print("\n[3/5] Running rule-based baseline...")
    from mccv.lite.knowledge_graph import ClinicalKnowledgeGraphLite
    from mccv.lite.rule_based_scorer import RuleBasedCoherenceScorerLite
    
    kg = ClinicalKnowledgeGraphLite()
    scorer = RuleBasedCoherenceScorerLite(kg)
    
    start_time = time.time()
    rule_based_results = scorer.score_dataset(data)
    rule_based_time = time.time() - start_time
    
    # Map results to diagnosis_ids (format: beneficiary_id||hcc_code)
    rule_based_scores = np.zeros(n_samples)
    for i, diag_id in enumerate(diagnosis_ids):
        if '||' in diag_id:
            bene_id, hcc = diag_id.split('||')
            matching_result = next(
                (r for r in rule_based_results 
                 if r['beneficiary_id'] == bene_id and r['hcc_code'] == hcc),
                None
            )
            if matching_result:
                rule_based_scores[i] = matching_result['coherence_score']
    
    # Calculate rule-based metrics on test set
    rule_based_metrics = calculate_metrics(
        labels[test_mask],
        rule_based_scores[test_mask],
        threshold=0.5
    )
    
    print(f"  - Processing time: {rule_based_time:.2f}s")
    print(f"  - Test AUC-ROC: {rule_based_metrics['auc_roc']:.4f}")
    
    # Train GNN
    print("\n[4/5] Training GNN model...")
    start_time = time.time()
    trainer, train_losses, val_losses = train_gnn_model(
        graph_data,
        labels,
        train_mask,
        val_mask,
        num_epochs=num_epochs,
        device=device,
        verbose=True
    )
    gnn_train_time = time.time() - start_time
    
    # Get GNN predictions on test set
    print("\n[5/5] Evaluating GNN on test set...")
    graph_data_device = {
        'bene_ids': graph_data['bene_ids'].to(device),
        'diag_ids': graph_data['diag_ids'].to(device),
        'treatment_ids': graph_data['treatment_ids'].to(device),
        'edges': {k: v.to(device) for k, v in graph_data['edges'].items()},
        'diag_instance_to_type': graph_data['diag_instance_to_type'].to(device),
        'n_instances': graph_data['n_instances']
    }
    
    gnn_scores = trainer.predict(graph_data_device)
    
    # Calculate GNN metrics on test set
    gnn_metrics = calculate_metrics(
        labels[test_mask],
        gnn_scores[test_mask],
        threshold=0.5
    )
    
    print(f"  - Training time: {gnn_train_time:.2f}s")
    print(f"  - Test AUC-ROC: {gnn_metrics['auc_roc']:.4f}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    comparison = {
        'rule_based': {
            'metrics': rule_based_metrics,
            'processing_time': rule_based_time,
            'predictions': rule_based_scores,
        },
        'gnn': {
            'metrics': gnn_metrics,
            'training_time': gnn_train_time,
            'predictions': gnn_scores,
            'train_losses': train_losses,
            'val_losses': val_losses,
        },
        'improvement': {
            'auc_roc_delta': gnn_metrics['auc_roc'] - rule_based_metrics['auc_roc'],
            'auc_roc_pct': 100 * (gnn_metrics['auc_roc'] - rule_based_metrics['auc_roc']) / rule_based_metrics['auc_roc'],
            'f1_delta': gnn_metrics['f1_score'] - rule_based_metrics['f1_score'],
            'precision_delta': gnn_metrics['precision'] - rule_based_metrics['precision'],
            'recall_delta': gnn_metrics['recall'] - rule_based_metrics['recall'],
        },
        'labels': labels,
        'test_mask': test_mask,
        'diagnosis_ids': diagnosis_ids,
    }
    
    print(f"\nRule-Based Baseline:")
    print(f"  AUC-ROC: {rule_based_metrics['auc_roc']:.4f}")
    print(f"  Precision: {rule_based_metrics['precision']:.4f}")
    print(f"  Recall: {rule_based_metrics['recall']:.4f}")
    print(f"  F1-Score: {rule_based_metrics['f1_score']:.4f}")
    print(f"  Accuracy: {rule_based_metrics['accuracy']:.4f}")
    
    print(f"\nGNN Model:")
    print(f"  AUC-ROC: {gnn_metrics['auc_roc']:.4f}")
    print(f"  Precision: {gnn_metrics['precision']:.4f}")
    print(f"  Recall: {gnn_metrics['recall']:.4f}")
    print(f"  F1-Score: {gnn_metrics['f1_score']:.4f}")
    print(f"  Accuracy: {gnn_metrics['accuracy']:.4f}")
    
    print(f"\nImprovement (GNN over Rule-Based):")
    print(f"  AUC-ROC: +{comparison['improvement']['auc_roc_delta']:.4f} ({comparison['improvement']['auc_roc_pct']:+.2f}%)")
    print(f"  F1-Score: +{comparison['improvement']['f1_delta']:.4f}")
    print(f"  Precision: +{comparison['improvement']['precision_delta']:.4f}")
    print(f"  Recall: +{comparison['improvement']['recall_delta']:.4f}")
    
    print("\n" + "=" * 70)
    
    return comparison


def find_fraud_examples(
    comparison: Dict,
    n_examples: int = 5,
    fraud_type: str = 'correctly_detected'
) -> List[Dict]:
    """
    Find example fraud cases for demonstration.
    
    Args:
        comparison: Results from compare_gnn_vs_rulebased
        n_examples: Number of examples to return
        fraud_type: Type of examples to find:
            - 'correctly_detected': GNN correctly identifies fraud (high confidence)
            - 'missed_by_rulebased': Rule-based missed but GNN caught
            - 'all_fraud': All actual fraud cases
            
    Returns:
        List of example dictionaries
    """
    labels = comparison['labels']
    test_mask = comparison['test_mask']
    rule_based_preds = comparison['rule_based']['predictions']
    gnn_preds = comparison['gnn']['predictions']
    diagnosis_ids = comparison['diagnosis_ids']
    
    # Find fraud cases in test set
    fraud_indices = np.where((labels < 0.5) & test_mask)[0]
    
    examples = []
    
    if fraud_type == 'correctly_detected':
        # Sort by GNN confidence (lowest scores = highest fraud confidence)
        sorted_indices = fraud_indices[np.argsort(gnn_preds[fraud_indices])]
        
        for idx in sorted_indices[:n_examples]:
            examples.append({
                'diagnosis_id': diagnosis_ids[idx],
                'ground_truth': 'FRAUD',
                'rule_based_score': float(rule_based_preds[idx]),
                'gnn_score': float(gnn_preds[idx]),
                'rule_based_decision': 'FLAGGED' if rule_based_preds[idx] < 0.5 else 'PASSED',
                'gnn_decision': 'FLAGGED' if gnn_preds[idx] < 0.5 else 'PASSED',
            })
    
    elif fraud_type == 'missed_by_rulebased':
        # Find cases where rule-based missed but GNN caught
        missed_indices = fraud_indices[
            (rule_based_preds[fraud_indices] >= 0.5) & 
            (gnn_preds[fraud_indices] < 0.5)
        ]
        
        for idx in missed_indices[:n_examples]:
            examples.append({
                'diagnosis_id': diagnosis_ids[idx],
                'ground_truth': 'FRAUD',
                'rule_based_score': float(rule_based_preds[idx]),
                'gnn_score': float(gnn_preds[idx]),
                'rule_based_decision': 'PASSED (MISSED)',
                'gnn_decision': 'FLAGGED (CAUGHT)',
            })
    
    else:  # all_fraud
        for idx in fraud_indices[:n_examples]:
            examples.append({
                'diagnosis_id': diagnosis_ids[idx],
                'ground_truth': 'FRAUD',
                'rule_based_score': float(rule_based_preds[idx]),
                'gnn_score': float(gnn_preds[idx]),
                'rule_based_decision': 'FLAGGED' if rule_based_preds[idx] < 0.5 else 'PASSED',
                'gnn_decision': 'FLAGGED' if gnn_preds[idx] < 0.5 else 'PASSED',
            })
    
    return examples
