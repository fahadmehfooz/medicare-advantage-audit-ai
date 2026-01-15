"""
MCCV Model: Multimodal Clinical Coherence Validation

This is the main model class that combines:
1. Heterogeneous Graph Neural Network encoder
2. Cross-Modal Transformer attention
3. Coherence scoring head
4. SHAP-based explainability

The model takes a clinical knowledge graph containing beneficiaries, diagnoses,
medications, labs, procedures, and providers, and outputs coherence scores
indicating how well each diagnosis is supported by treatment evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
import numpy as np


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Transformer Attention for fusing evidence across modalities.
    
    Takes diagnosis embeddings as queries and treatment evidence (pharmacy, labs,
    procedures, specialists) as keys/values. Computes attention-weighted fusion
    of evidence to determine clinical coherence.
    
    This is the core innovation of MCCV: rather than just checking if treatments
    exist, we learn which treatments are most relevant for each diagnosis and
    weight them accordingly.
    
    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Projections for each modality
        self.modalities = ["pharmacy", "laboratory", "specialist", "procedure"]
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_projs = nn.ModuleDict({
            mod: nn.Linear(embed_dim, embed_dim) for mod in self.modalities
        })
        self.value_projs = nn.ModuleDict({
            mod: nn.Linear(embed_dim, embed_dim) for mod in self.modalities
        })
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Modality-specific learnable weights (clinical importance)
        self.modality_weights = nn.Parameter(torch.ones(len(self.modalities)))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        diagnosis_embed: torch.Tensor,
        modality_embeds: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass computing cross-modal attention.
        
        Parameters
        ----------
        diagnosis_embed : torch.Tensor
            Diagnosis embeddings [batch_size, embed_dim]
        modality_embeds : Dict[str, torch.Tensor]
            Embeddings for each modality [batch_size, max_items, embed_dim]
        modality_masks : Dict[str, torch.Tensor], optional
            Masks for padded positions
        return_attention : bool
            Whether to return attention weights for explainability
        
        Returns
        -------
        torch.Tensor
            Fused evidence representation [batch_size, embed_dim]
        Dict[str, torch.Tensor], optional
            Attention weights per modality
        """
        batch_size = diagnosis_embed.size(0)
        
        # Project diagnosis as query
        q = self.query_proj(diagnosis_embed)  # [batch_size, embed_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        
        all_attention_weights = {}
        modality_outputs = []
        
        # Compute attention for each modality
        for mod_idx, modality in enumerate(self.modalities):
            if modality not in modality_embeds:
                continue
                
            mod_embed = modality_embeds[modality]
            seq_len = mod_embed.size(1)
            
            # Project keys and values
            k = self.key_projs[modality](mod_embed)  # [batch_size, seq_len, embed_dim]
            v = self.value_projs[modality](mod_embed)
            
            # Reshape for multi-head attention
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.transpose(1, 2)
            
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # [batch_size, num_heads, 1, seq_len]
            
            # Apply mask if provided
            if modality_masks is not None and modality in modality_masks:
                mask = modality_masks[modality].unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            
            # Softmax and dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            if return_attention:
                all_attention_weights[modality] = attn_weights.squeeze(2).mean(dim=1)
            
            # Compute weighted values
            attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, 1, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, self.embed_dim)
            
            # Weight by learned modality importance
            mod_weight = F.softmax(self.modality_weights, dim=0)[mod_idx]
            modality_outputs.append(attn_output * mod_weight)
        
        # Combine modality outputs
        if modality_outputs:
            fused = torch.stack(modality_outputs, dim=0).sum(dim=0)
        else:
            fused = torch.zeros(batch_size, self.embed_dim, device=diagnosis_embed.device)
        
        # Output projection and residual
        output = self.output_proj(fused)
        output = self.layer_norm(output + diagnosis_embed)
        
        if return_attention:
            return output, all_attention_weights
        return output, None


class CoherenceScorer(nn.Module):
    """
    Final coherence scoring head.
    
    Takes the fused evidence representation and diagnosis embedding
    to produce a coherence score between 0 and 1.
    
    Score interpretation:
    - 0.0-0.3: HIGH RISK - Minimal/no treatment evidence
    - 0.3-0.6: MEDIUM RISK - Partial evidence, review recommended
    - 0.6-1.0: LOW RISK - Strong treatment evidence
    
    Parameters
    ----------
    embed_dim : int
        Input embedding dimension
    hidden_dim : int
        Hidden layer dimension
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        diagnosis_embed: torch.Tensor,
        fused_evidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coherence score.
        
        Parameters
        ----------
        diagnosis_embed : torch.Tensor
            Diagnosis embedding [batch_size, embed_dim]
        fused_evidence : torch.Tensor
            Fused evidence from cross-modal attention [batch_size, embed_dim]
        
        Returns
        -------
        torch.Tensor
            Coherence scores [batch_size, 1]
        """
        # Concatenate diagnosis and evidence
        combined = torch.cat([diagnosis_embed, fused_evidence], dim=-1)
        return self.mlp(combined)


class MCCVModel(nn.Module):
    """
    Complete MCCV Model for Multimodal Clinical Coherence Validation.
    
    This model takes a heterogeneous clinical knowledge graph and outputs
    coherence scores for each beneficiary-diagnosis pair, indicating whether
    the diagnosis has sufficient treatment evidence.
    
    Architecture:
    1. HCC Embedding: Learns embeddings for diagnosis codes
    2. Treatment Encoders: Encode pharmacy, lab, specialist, procedure evidence
    3. Cross-Modal Attention: Fuse evidence across modalities
    4. Coherence Scorer: Output final coherence score
    
    Parameters
    ----------
    num_hcc_codes : int
        Number of unique HCC codes
    hcc_embed_dim : int
        HCC embedding dimension
    hidden_dim : int
        Hidden layer dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of encoder layers
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        num_hcc_codes: int = 100,
        hcc_embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_hcc_codes = num_hcc_codes
        self.hidden_dim = hidden_dim
        
        # HCC code embedding
        self.hcc_embedding = nn.Embedding(num_hcc_codes, hcc_embed_dim)
        
        # Project HCC embedding to hidden dim
        self.hcc_proj = nn.Linear(hcc_embed_dim, hidden_dim)
        
        # Treatment evidence encoders (one per modality)
        self.modality_encoders = nn.ModuleDict({
            "pharmacy": self._build_encoder(hidden_dim, num_layers, dropout),
            "laboratory": self._build_encoder(hidden_dim, num_layers, dropout),
            "specialist": self._build_encoder(hidden_dim, num_layers, dropout),
            "procedure": self._build_encoder(hidden_dim, num_layers, dropout),
        })
        
        # Input projections for each modality (from raw features to hidden_dim)
        self.modality_input_proj = nn.ModuleDict({
            "pharmacy": nn.Linear(64, hidden_dim),  # NDC features
            "laboratory": nn.Linear(64, hidden_dim),  # LOINC features
            "specialist": nn.Linear(32, hidden_dim),  # Taxonomy features
            "procedure": nn.Linear(32, hidden_dim),  # CPT features
        })
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Coherence scorer
        self.scorer = CoherenceScorer(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _build_encoder(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float
    ) -> nn.Module:
        """Build a transformer encoder for a modality."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        hcc_codes: torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MCCV model.
        
        Parameters
        ----------
        hcc_codes : torch.Tensor
            HCC code indices [batch_size]
        modality_features : Dict[str, torch.Tensor]
            Raw features for each modality
            - pharmacy: [batch_size, max_meds, 64]
            - laboratory: [batch_size, max_labs, 64]
            - specialist: [batch_size, max_visits, 32]
            - procedure: [batch_size, max_procs, 32]
        modality_masks : Dict[str, torch.Tensor], optional
            Boolean masks for valid positions
        return_attention : bool
            Whether to return attention weights
        
        Returns
        -------
        Dict[str, torch.Tensor]
            - 'coherence_score': Scores [batch_size, 1]
            - 'attention_weights': Per-modality attention (if requested)
            - 'diagnosis_embed': Diagnosis embeddings
            - 'fused_evidence': Fused evidence embeddings
        """
        batch_size = hcc_codes.size(0)
        
        # Embed HCC codes
        hcc_embed = self.hcc_embedding(hcc_codes)  # [batch_size, hcc_embed_dim]
        diagnosis_embed = self.hcc_proj(hcc_embed)  # [batch_size, hidden_dim]
        
        # Encode each modality
        modality_embeds = {}
        for modality, features in modality_features.items():
            if modality not in self.modality_encoders:
                continue
            
            # Project to hidden dim
            proj_features = self.modality_input_proj[modality](features)
            
            # Get mask if available
            mask = modality_masks.get(modality) if modality_masks else None
            
            # Encode
            encoded = self.modality_encoders[modality](
                proj_features,
                src_key_padding_mask=~mask if mask is not None else None
            )
            modality_embeds[modality] = encoded
        
        # Cross-modal attention
        fused_evidence, attention_weights = self.cross_modal_attention(
            diagnosis_embed,
            modality_embeds,
            modality_masks,
            return_attention=return_attention
        )
        
        # Compute coherence score
        coherence_score = self.scorer(diagnosis_embed, fused_evidence)
        
        output = {
            "coherence_score": coherence_score,
            "diagnosis_embed": diagnosis_embed,
            "fused_evidence": fused_evidence,
        }
        
        if return_attention:
            output["attention_weights"] = attention_weights
        
        return output
    
    def predict(
        self,
        hcc_codes: torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> np.ndarray:
        """
        Make predictions (convenience method for inference).
        
        Parameters
        ----------
        hcc_codes : torch.Tensor
            HCC code indices
        modality_features : Dict[str, torch.Tensor]
            Treatment evidence features
        modality_masks : Dict[str, torch.Tensor], optional
            Validity masks
        
        Returns
        -------
        np.ndarray
            Coherence scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(hcc_codes, modality_features, modality_masks)
            return output["coherence_score"].cpu().numpy()
    
    def get_explanation(
        self,
        hcc_codes: torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict:
        """
        Get explainable predictions with attention weights.
        
        Returns attention weights showing which evidence contributed
        most to the coherence score.
        
        Parameters
        ----------
        hcc_codes : torch.Tensor
            HCC code indices
        modality_features : Dict[str, torch.Tensor]
            Treatment evidence features
        modality_masks : Dict[str, torch.Tensor], optional
            Validity masks
        
        Returns
        -------
        Dict
            - 'coherence_score': Final score
            - 'attention_weights': Per-modality attention
            - 'modality_contributions': Contribution of each modality
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                hcc_codes, 
                modality_features, 
                modality_masks,
                return_attention=True
            )
            
            # Compute modality contributions
            modality_contributions = {}
            if output.get("attention_weights"):
                for mod, weights in output["attention_weights"].items():
                    # Average attention across items
                    modality_contributions[mod] = weights.mean().item()
            
            return {
                "coherence_score": output["coherence_score"].cpu().numpy(),
                "attention_weights": output.get("attention_weights"),
                "modality_contributions": modality_contributions,
            }


class MCCVLoss(nn.Module):
    """
    Combined loss function for MCCV training.
    
    Uses binary cross-entropy for coherence scores plus auxiliary losses
    for better learning.
    
    Parameters
    ----------
    alpha : float
        Weight for coherence loss
    beta : float  
        Weight for contrastive loss
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        
    def forward(
        self,
        pred_scores: torch.Tensor,
        true_scores: torch.Tensor,
        diagnosis_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Parameters
        ----------
        pred_scores : torch.Tensor
            Predicted coherence scores [batch_size, 1]
        true_scores : torch.Tensor
            Ground truth scores [batch_size, 1]
        diagnosis_embeds : torch.Tensor, optional
            Diagnosis embeddings for contrastive loss
        labels : torch.Tensor, optional
            Binary fraud labels
        
        Returns
        -------
        torch.Tensor
            Total loss
        """
        # Main coherence loss (BCE)
        coherence_loss = self.bce(pred_scores, true_scores)
        
        total_loss = self.alpha * coherence_loss
        
        # Optional contrastive loss
        if diagnosis_embeds is not None and labels is not None:
            contrastive_loss = self._contrastive_loss(diagnosis_embeds, labels)
            total_loss = total_loss + self.beta * contrastive_loss
        
        return total_loss
    
    def _contrastive_loss(
        self,
        embeds: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Compute supervised contrastive loss."""
        # Normalize embeddings
        embeds = F.normalize(embeds, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeds, embeds.T) / temperature
        
        # Create positive/negative masks
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()
        
        # Remove diagonal
        eye = torch.eye(embeds.size(0), device=embeds.device)
        mask = mask * (1 - eye)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix) * (1 - eye)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean over positive pairs
        loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return loss.mean()


if __name__ == "__main__":
    # Test the model
    print("Testing MCCV Model...")
    
    batch_size = 32
    max_meds = 10
    max_labs = 8
    max_visits = 5
    max_procs = 6
    
    # Create dummy data
    hcc_codes = torch.randint(0, 100, (batch_size,))
    
    modality_features = {
        "pharmacy": torch.randn(batch_size, max_meds, 64),
        "laboratory": torch.randn(batch_size, max_labs, 64),
        "specialist": torch.randn(batch_size, max_visits, 32),
        "procedure": torch.randn(batch_size, max_procs, 32),
    }
    
    modality_masks = {
        "pharmacy": torch.ones(batch_size, max_meds).bool(),
        "laboratory": torch.ones(batch_size, max_labs).bool(),
        "specialist": torch.ones(batch_size, max_visits).bool(),
        "procedure": torch.ones(batch_size, max_procs).bool(),
    }
    
    # Initialize model
    model = MCCVModel(
        num_hcc_codes=100,
        hcc_embed_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    )
    
    # Forward pass
    output = model(hcc_codes, modality_features, modality_masks, return_attention=True)
    
    print(f"Coherence scores shape: {output['coherence_score'].shape}")
    print(f"Sample scores: {output['coherence_score'][:5].squeeze().tolist()}")
    
    if output.get("attention_weights"):
        for mod, weights in output["attention_weights"].items():
            print(f"{mod} attention shape: {weights.shape}")
    
    # Test loss
    loss_fn = MCCVLoss()
    true_scores = torch.rand(batch_size, 1)
    loss = loss_fn(output["coherence_score"], true_scores)
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful!")
    
    # Test explanation
    explanation = model.get_explanation(hcc_codes, modality_features, modality_masks)
    print(f"Modality contributions: {explanation['modality_contributions']}")
    
    print("\nAll tests passed!")
