"""
Ligand-aware folding integration for OpenFold++.

This module provides components to condition protein structure prediction
on the presence of ligands by injecting ligand embeddings into the model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

from openfold.model.primitives import Linear, LayerNorm
from openfold.data.ligand_parser import LigandFeatures, LigandEmbedder


class LigandConditionedInputEmbedder(nn.Module):
    """
    Input embedder that conditions on ligand presence.
    
    Extends the standard InputEmbedder to incorporate ligand information
    into both MSA and pair representations.
    """
    
    def __init__(self,
                 base_embedder: nn.Module,
                 ligand_embedding_dim: int = 256,
                 c_z: int = 128,
                 c_m: int = 256,
                 ligand_injection_mode: str = "pair_and_msa"):
        """
        Args:
            base_embedder: The original InputEmbedder or InputEmbedderMultimer
            ligand_embedding_dim: Dimension of ligand embeddings
            c_z: Pair representation dimension
            c_m: MSA representation dimension
            ligand_injection_mode: How to inject ligand info ("pair", "msa", "pair_and_msa")
        """
        super().__init__()
        
        self.base_embedder = base_embedder
        self.ligand_embedding_dim = ligand_embedding_dim
        self.c_z = c_z
        self.c_m = c_m
        self.injection_mode = ligand_injection_mode
        
        # Ligand conditioning layers
        if "pair" in ligand_injection_mode:
            # Project ligand embedding to pair representation
            self.ligand_to_pair = nn.Sequential(
                Linear(ligand_embedding_dim, c_z),
                nn.ReLU(),
                Linear(c_z, c_z)
            )
            
        if "msa" in ligand_injection_mode:
            # Project ligand embedding to MSA representation
            self.ligand_to_msa = nn.Sequential(
                Linear(ligand_embedding_dim, c_m),
                nn.ReLU(),
                Linear(c_m, c_m)
            )
        
        # Gating mechanism to control ligand influence
        self.ligand_gate = nn.Sequential(
            Linear(ligand_embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, *args, ligand_embeddings: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass with optional ligand conditioning.
        
        Args:
            *args, **kwargs: Arguments for the base embedder
            ligand_embeddings: [batch_size, num_ligands, ligand_embedding_dim] or None
        
        Returns:
            msa_emb, pair_emb: Ligand-conditioned representations
        """
        # Get base embeddings
        msa_emb, pair_emb = self.base_embedder(*args, **kwargs)
        
        # If no ligands, return base embeddings
        if ligand_embeddings is None:
            return msa_emb, pair_emb
        
        batch_size = msa_emb.shape[0] if len(msa_emb.shape) > 3 else 1
        n_res = pair_emb.shape[-2]
        
        # Pool multiple ligands if present
        if len(ligand_embeddings.shape) == 3:  # [batch, num_ligands, dim]
            # Simple mean pooling for multiple ligands
            ligand_emb = torch.mean(ligand_embeddings, dim=1)  # [batch, dim]
        else:
            ligand_emb = ligand_embeddings  # [batch, dim]
        
        # Compute gating factor
        gate = self.ligand_gate(ligand_emb)  # [batch, 1]
        
        # Inject into pair representation
        if "pair" in self.injection_mode:
            ligand_pair_contrib = self.ligand_to_pair(ligand_emb)  # [batch, c_z]
            
            # Broadcast to pair dimensions [batch, n_res, n_res, c_z]
            ligand_pair_contrib = ligand_pair_contrib.unsqueeze(1).unsqueeze(1)
            ligand_pair_contrib = ligand_pair_contrib.expand(-1, n_res, n_res, -1)
            
            # Apply gating and add to pair representation
            pair_emb = pair_emb + gate.unsqueeze(1).unsqueeze(1) * ligand_pair_contrib
        
        # Inject into MSA representation
        if "msa" in self.injection_mode:
            ligand_msa_contrib = self.ligand_to_msa(ligand_emb)  # [batch, c_m]
            
            # Broadcast to MSA dimensions [batch, n_seq, n_res, c_m]
            n_seq = msa_emb.shape[-3]
            ligand_msa_contrib = ligand_msa_contrib.unsqueeze(1).unsqueeze(1)
            ligand_msa_contrib = ligand_msa_contrib.expand(-1, n_seq, n_res, -1)
            
            # Apply gating and add to MSA representation
            msa_emb = msa_emb + gate.unsqueeze(1).unsqueeze(1) * ligand_msa_contrib
        
        return msa_emb, pair_emb


class LigandConditionedEvoformer(nn.Module):
    """
    Evoformer stack that can be conditioned on ligand presence.
    
    Periodically injects ligand information during the evolution process.
    """
    
    def __init__(self,
                 base_evoformer: nn.Module,
                 ligand_embedding_dim: int = 256,
                 c_z: int = 128,
                 c_m: int = 256,
                 injection_frequency: int = 4):
        """
        Args:
            base_evoformer: The original EvoformerStack
            ligand_embedding_dim: Dimension of ligand embeddings
            c_z: Pair representation dimension
            c_m: MSA representation dimension
            injection_frequency: Inject ligand info every N blocks
        """
        super().__init__()
        
        self.base_evoformer = base_evoformer
        self.ligand_embedding_dim = ligand_embedding_dim
        self.c_z = c_z
        self.c_m = c_m
        self.injection_frequency = injection_frequency
        
        # Ligand injection layers
        self.ligand_pair_injection = nn.Sequential(
            Linear(ligand_embedding_dim, c_z),
            nn.ReLU(),
            Linear(c_z, c_z)
        )
        
        self.ligand_msa_injection = nn.Sequential(
            Linear(ligand_embedding_dim, c_m),
            nn.ReLU(),
            Linear(c_m, c_m)
        )
        
        # Adaptive gating
        self.ligand_gate = nn.Sequential(
            Linear(ligand_embedding_dim, 2),  # Separate gates for pair and MSA
            nn.Sigmoid()
        )
    
    def forward(self, 
                m: torch.Tensor,
                z: torch.Tensor,
                ligand_embeddings: Optional[torch.Tensor] = None,
                **kwargs):
        """
        Forward pass with periodic ligand injection.
        
        Args:
            m: MSA representation
            z: Pair representation
            ligand_embeddings: Ligand embeddings
            **kwargs: Other arguments for base evoformer
        
        Returns:
            m, z, s: Updated representations
        """
        # If no ligands, use base evoformer
        if ligand_embeddings is None:
            return self.base_evoformer(m, z, **kwargs)
        
        # Pool multiple ligands
        if len(ligand_embeddings.shape) == 3:
            ligand_emb = torch.mean(ligand_embeddings, dim=1)
        else:
            ligand_emb = ligand_embeddings
        
        # Prepare ligand contributions
        ligand_pair_contrib = self.ligand_pair_injection(ligand_emb)
        ligand_msa_contrib = self.ligand_msa_injection(ligand_emb)
        gates = self.ligand_gate(ligand_emb)  # [batch, 2]
        pair_gate, msa_gate = gates[:, 0:1], gates[:, 1:2]
        
        # Get dimensions
        batch_size, n_seq, n_res = m.shape[:3]
        
        # Broadcast ligand contributions
        ligand_pair_contrib = ligand_pair_contrib.unsqueeze(1).unsqueeze(1)
        ligand_pair_contrib = ligand_pair_contrib.expand(-1, n_res, n_res, -1)
        
        ligand_msa_contrib = ligand_msa_contrib.unsqueeze(1).unsqueeze(1)
        ligand_msa_contrib = ligand_msa_contrib.expand(-1, n_seq, n_res, -1)
        
        # Apply gating
        ligand_pair_contrib = pair_gate.unsqueeze(1).unsqueeze(1) * ligand_pair_contrib
        ligand_msa_contrib = msa_gate.unsqueeze(1).unsqueeze(1) * ligand_msa_contrib
        
        # Inject ligand information periodically during evolution
        # For now, inject at the beginning - could be extended to inject periodically
        z = z + ligand_pair_contrib
        m = m + ligand_msa_contrib
        
        # Run base evoformer
        return self.base_evoformer(m, z, **kwargs)


class LigandConditionedStructureModule(nn.Module):
    """
    Structure module that can be conditioned on ligand presence.
    
    Modifies the structure prediction to be aware of ligand binding sites.
    """
    
    def __init__(self,
                 base_structure_module: nn.Module,
                 ligand_embedding_dim: int = 256,
                 c_s: int = 384):
        """
        Args:
            base_structure_module: The original StructureModule
            ligand_embedding_dim: Dimension of ligand embeddings
            c_s: Single representation dimension
        """
        super().__init__()
        
        self.base_structure_module = base_structure_module
        self.ligand_embedding_dim = ligand_embedding_dim
        self.c_s = c_s
        
        # Ligand conditioning for structure prediction
        self.ligand_to_structure = nn.Sequential(
            Linear(ligand_embedding_dim, c_s),
            nn.ReLU(),
            Linear(c_s, c_s)
        )
        
        # Binding site attention
        self.binding_site_attention = nn.MultiheadAttention(
            embed_dim=c_s,
            num_heads=8,
            batch_first=True
        )
        
        # Gating for ligand influence on structure
        self.structure_gate = nn.Sequential(
            Linear(ligand_embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                representations: Dict[str, torch.Tensor],
                aatype: torch.Tensor,
                ligand_embeddings: Optional[torch.Tensor] = None,
                **kwargs):
        """
        Forward pass with ligand-aware structure prediction.
        
        Args:
            representations: Dict containing 'single', 'pair', etc.
            aatype: Amino acid types
            ligand_embeddings: Ligand embeddings
            **kwargs: Other arguments for base structure module
        
        Returns:
            Structure module outputs
        """
        # If no ligands, use base structure module
        if ligand_embeddings is None:
            return self.base_structure_module(representations, aatype, **kwargs)
        
        # Pool multiple ligands
        if len(ligand_embeddings.shape) == 3:
            ligand_emb = torch.mean(ligand_embeddings, dim=1)
        else:
            ligand_emb = ligand_embeddings
        
        # Modify single representation with ligand information
        single_repr = representations["single"]  # [batch, n_res, c_s]
        
        # Project ligand to structure space
        ligand_structure_contrib = self.ligand_to_structure(ligand_emb)  # [batch, c_s]
        
        # Broadcast to residue dimension
        ligand_structure_contrib = ligand_structure_contrib.unsqueeze(1)  # [batch, 1, c_s]
        ligand_structure_contrib = ligand_structure_contrib.expand(-1, single_repr.shape[1], -1)
        
        # Apply binding site attention
        attended_ligand, _ = self.binding_site_attention(
            single_repr, ligand_structure_contrib, ligand_structure_contrib
        )
        
        # Gate the ligand contribution
        gate = self.structure_gate(ligand_emb).unsqueeze(1)  # [batch, 1, 1]
        
        # Update single representation
        modified_representations = representations.copy()
        modified_representations["single"] = single_repr + gate * attended_ligand
        
        # Run base structure module with modified representations
        return self.base_structure_module(modified_representations, aatype, **kwargs)


class LigandAwareAlphaFold(nn.Module):
    """
    Ligand-aware version of AlphaFold that conditions structure prediction on ligands.
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 ligand_embedder: Optional[LigandEmbedder] = None,
                 ligand_embedding_dim: int = 256,
                 injection_mode: str = "all"):
        """
        Args:
            base_model: The original AlphaFold model
            ligand_embedder: LigandEmbedder for processing ligands
            ligand_embedding_dim: Dimension of ligand embeddings
            injection_mode: Where to inject ligand info ("input", "evoformer", "structure", "all")
        """
        super().__init__()
        
        self.base_model = base_model
        self.ligand_embedder = ligand_embedder or LigandEmbedder(embedding_dim=ligand_embedding_dim)
        self.ligand_embedding_dim = ligand_embedding_dim
        self.injection_mode = injection_mode
        
        # Wrap components with ligand-aware versions
        if "input" in injection_mode or injection_mode == "all":
            self.input_embedder = LigandConditionedInputEmbedder(
                base_model.input_embedder,
                ligand_embedding_dim=ligand_embedding_dim,
                c_z=base_model.config.evoformer_stack.c_z,
                c_m=base_model.config.evoformer_stack.c_m
            )
        else:
            self.input_embedder = base_model.input_embedder
            
        if "evoformer" in injection_mode or injection_mode == "all":
            self.evoformer = LigandConditionedEvoformer(
                base_model.evoformer,
                ligand_embedding_dim=ligand_embedding_dim,
                c_z=base_model.config.evoformer_stack.c_z,
                c_m=base_model.config.evoformer_stack.c_m
            )
        else:
            self.evoformer = base_model.evoformer
            
        if "structure" in injection_mode or injection_mode == "all":
            self.structure_module = LigandConditionedStructureModule(
                base_model.structure_module,
                ligand_embedding_dim=ligand_embedding_dim,
                c_s=base_model.config.structure_module.c_s
            )
        else:
            self.structure_module = base_model.structure_module
        
        # Copy other components
        self.recycling_embedder = base_model.recycling_embedder
        self.aux_heads = base_model.aux_heads
        if hasattr(base_model, 'template_embedder'):
            self.template_embedder = base_model.template_embedder
        if hasattr(base_model, 'extra_msa_embedder'):
            self.extra_msa_embedder = base_model.extra_msa_embedder
        if hasattr(base_model, 'extra_msa_stack'):
            self.extra_msa_stack = base_model.extra_msa_stack
        
        # Copy configuration
        self.globals = base_model.globals
        self.config = base_model.config
    
    def forward(self, 
                batch: Dict[str, torch.Tensor],
                ligand_features: Optional[List[LigandFeatures]] = None,
                **kwargs):
        """
        Forward pass with optional ligand conditioning.
        
        Args:
            batch: Input batch (same as AlphaFold)
            ligand_features: List of LigandFeatures for each sample in batch
            **kwargs: Other arguments
        
        Returns:
            Model outputs conditioned on ligands
        """
        # Process ligands if provided
        ligand_embeddings = None
        if ligand_features is not None:
            ligand_embeddings = []
            for lf in ligand_features:
                if lf is not None:
                    if lf.embedding is not None:
                        ligand_embeddings.append(lf.embedding)
                    else:
                        # Generate embedding on the fly
                        with torch.no_grad():
                            emb = self.ligand_embedder(lf)
                            ligand_embeddings.append(emb)
                else:
                    # No ligand for this sample
                    ligand_embeddings.append(torch.zeros(self.ligand_embedding_dim))
            
            ligand_embeddings = torch.stack(ligand_embeddings)  # [batch, dim]
        
        # Use the base model's forward logic but with ligand-aware components
        # This is a simplified version - full implementation would mirror AlphaFold.forward()
        
        # Input embedding with ligand conditioning
        if hasattr(self.input_embedder, 'forward'):
            if self.globals.is_multimer:
                m, z = self.input_embedder(batch, ligand_embeddings=ligand_embeddings)
            else:
                m, z = self.input_embedder(
                    batch["target_feat"],
                    batch["residue_index"], 
                    batch["msa_feat"],
                    ligand_embeddings=ligand_embeddings
                )
        else:
            # Fallback to base embedder
            if self.globals.is_multimer:
                m, z = self.input_embedder(batch)
            else:
                m, z = self.input_embedder(
                    batch["target_feat"],
                    batch["residue_index"],
                    batch["msa_feat"]
                )
        
        # Evoformer with ligand conditioning
        m, z, s = self.evoformer(
            m, z,
            ligand_embeddings=ligand_embeddings,
            msa_mask=batch.get("msa_mask"),
            pair_mask=batch.get("pair_mask"),
            **kwargs
        )
        
        # Structure module with ligand conditioning
        representations = {"single": s, "pair": z, "msa": m}
        structure_outputs = self.structure_module(
            representations,
            batch["aatype"],
            ligand_embeddings=ligand_embeddings,
            **kwargs
        )
        
        return {
            "msa": m,
            "pair": z, 
            "single": s,
            "sm": structure_outputs,
            "ligand_embeddings": ligand_embeddings
        }
