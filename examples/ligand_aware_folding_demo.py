#!/usr/bin/env python3
"""
Ligand-Aware Folding Demo for OdinFold

Demonstrates the complete ligand-aware protein folding system including:
- Molecular graph encoding of ligands
- Cross-attention between protein and ligand features
- Binding pocket prediction
- Ligand-conditioned structure prediction
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.modules.ligand import (
    LigandEncoder,
    LigandProteinCrossAttention,
    LigandAwareFoldingHead,
    BindingPocketAttention,
    LigandConditionedStructureModule
)
from openfoldpp.modules.ligand.ligand_utils import (
    smiles_to_graph,
    batch_process_ligands,
    compute_ligand_protein_distances,
    get_binding_pocket_mask
)


def demo_ligand_encoding():
    """Demonstrate ligand molecular graph encoding."""
    
    print("🧬 Ligand Encoding Demo")
    print("=" * 50)
    
    # Create ligand encoder
    d_model = 128
    encoder = LigandEncoder(d_model, num_gnn_layers=3)
    
    # Example ligands (drug-like molecules)
    ligands = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Penicillin": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
    }
    
    print(f"Encoding {len(ligands)} ligands...")
    
    for name, smiles in ligands.items():
        print(f"\n🔬 Processing {name}: {smiles}")
        
        # Convert SMILES to graph
        ligand_graph = smiles_to_graph(smiles)
        
        print(f"  Atoms: {ligand_graph['num_atoms']}")
        print(f"  Bonds: {ligand_graph['edge_index'].shape[1] // 2}")
        
        # Encode ligand
        ligand_output = encoder(ligand_graph)
        
        print(f"  Atom embeddings: {ligand_output['atom_embeddings'].shape}")
        print(f"  Ligand embedding: {ligand_output['ligand_embedding'].shape}")
        
        # Show some properties
        atom_types = ligand_graph['atom_types']
        unique_atoms = torch.unique(atom_types)
        atom_names = []
        for atom_idx in unique_atoms:
            if atom_idx == 0: atom_names.append('UNK')
            elif atom_idx == 1: atom_names.append('H')
            elif atom_idx == 2: atom_names.append('C')
            elif atom_idx == 3: atom_names.append('N')
            elif atom_idx == 4: atom_names.append('O')
            elif atom_idx == 7: atom_names.append('S')
            else: atom_names.append(f'Atom{atom_idx}')
        
        print(f"  Atom types: {', '.join(atom_names)}")
    
    print("\n✅ Ligand encoding completed!")


def demo_cross_attention():
    """Demonstrate ligand-protein cross-attention."""
    
    print("\n🔗 Cross-Attention Demo")
    print("=" * 50)
    
    # Parameters
    batch_size, seq_len, num_atoms = 2, 100, 25
    protein_dim, ligand_dim, d_model = 384, 128, 256
    
    # Create cross-attention module
    cross_attn = LigandProteinCrossAttention(
        protein_dim, ligand_dim, d_model, num_heads=8, max_distance=10.0
    )
    
    print(f"Cross-attention parameters:")
    print(f"  Protein dim: {protein_dim}")
    print(f"  Ligand dim: {ligand_dim}")
    print(f"  Model dim: {d_model}")
    print(f"  Max interaction distance: 10.0 Å")
    
    # Mock protein and ligand data
    protein_features = torch.randn(batch_size, seq_len, protein_dim)
    ligand_features = torch.randn(batch_size, num_atoms, ligand_dim)
    
    # Create realistic coordinates (protein in extended conformation, ligand nearby)
    protein_coords = torch.randn(batch_size, seq_len, 3) * 2.0  # Spread protein
    ligand_coords = torch.randn(batch_size, num_atoms, 3) * 1.0  # Compact ligand
    
    # Apply cross-attention
    output = cross_attn(
        protein_features, ligand_features,
        protein_coords, ligand_coords
    )
    
    print(f"\nCross-attention results:")
    print(f"  Updated protein features: {output['protein_features'].shape}")
    print(f"  Updated ligand features: {output['ligand_features'].shape}")
    
    # Analyze interactions
    distances = output['cross_attention_weights']
    interaction_mask = output['interaction_mask']
    
    min_distances = distances.min(dim=-1)[0]  # [batch, seq_len]
    interacting_residues = (min_distances <= 10.0).sum(dim=-1)
    
    print(f"  Interacting residues per protein: {interacting_residues.tolist()}")
    print(f"  Average min distance: {min_distances.mean():.2f} Å")
    
    print("✅ Cross-attention completed!")


def demo_binding_pocket_prediction():
    """Demonstrate binding pocket prediction."""
    
    print("\n🎯 Binding Pocket Prediction Demo")
    print("=" * 50)
    
    # Parameters
    batch_size, seq_len, num_atoms = 1, 150, 20
    d_model = 256
    
    # Create binding pocket attention
    pocket_attn = BindingPocketAttention(d_model, num_heads=8, pocket_radius=8.0)
    
    # Mock protein structure (create a binding pocket)
    protein_features = torch.randn(batch_size, seq_len, d_model)
    protein_coords = torch.randn(batch_size, seq_len, 3) * 5.0
    
    # Create ligand near a specific region (residues 50-70)
    ligand_coords = torch.randn(batch_size, num_atoms, 3)
    pocket_center = protein_coords[0, 50:70].mean(dim=0)  # Center of pocket region
    ligand_coords[0] = pocket_center + torch.randn(num_atoms, 3) * 2.0  # Place ligand near pocket
    
    print(f"Protein length: {seq_len} residues")
    print(f"Ligand atoms: {num_atoms}")
    print(f"Pocket center: ({pocket_center[0]:.1f}, {pocket_center[1]:.1f}, {pocket_center[2]:.1f})")
    
    # Predict binding pocket
    output = pocket_attn(protein_features, ligand_coords, protein_coords)
    
    pocket_scores = output['pocket_scores'][0]  # Remove batch dimension
    pocket_mask = output['pocket_mask'][0]
    
    # Analyze pocket prediction
    high_score_residues = (pocket_scores > 0.7).sum()
    predicted_pocket_residues = pocket_mask.sum()
    
    print(f"\nBinding pocket analysis:")
    print(f"  High-confidence pocket residues (>0.7): {high_score_residues}")
    print(f"  Predicted pocket residues: {predicted_pocket_residues}")
    print(f"  Average pocket score: {pocket_scores.mean():.3f}")
    
    # Find top pocket residues
    top_scores, top_indices = torch.topk(pocket_scores, k=10)
    print(f"  Top 10 pocket residues: {top_indices.tolist()}")
    print(f"  Their scores: {[f'{score:.3f}' for score in top_scores.tolist()]}")
    
    print("✅ Binding pocket prediction completed!")


def demo_ligand_aware_folding():
    """Demonstrate complete ligand-aware folding."""
    
    print("\n🏗️ Ligand-Aware Folding Demo")
    print("=" * 50)
    
    # Parameters
    batch_size, seq_len, num_atoms = 1, 80, 15
    protein_dim, ligand_dim, d_model = 384, 128, 256
    
    # Create ligand-aware folding system
    folding_head = LigandAwareFoldingHead(protein_dim, ligand_dim, d_model)
    structure_module = LigandConditionedStructureModule(
        protein_dim, ligand_dim, d_model, num_layers=3
    )
    
    print(f"Folding system parameters:")
    print(f"  Protein sequence length: {seq_len}")
    print(f"  Ligand atoms: {num_atoms}")
    print(f"  Refinement layers: 3")
    
    # Mock input data
    protein_features = torch.randn(batch_size, seq_len, protein_dim)
    ligand_features = torch.randn(batch_size, num_atoms, ligand_dim)
    
    # Initial protein coordinates (random coil)
    initial_coords = torch.randn(batch_size, seq_len, 3) * 3.0
    ligand_coords = torch.randn(batch_size, num_atoms, 3) * 2.0
    
    print(f"\nInitial structure statistics:")
    initial_distances = torch.cdist(initial_coords[0], initial_coords[0])
    initial_radius_gyration = torch.sqrt(((initial_coords[0] - initial_coords[0].mean(dim=0))**2).sum(dim=-1).mean())
    print(f"  Initial radius of gyration: {initial_radius_gyration:.2f} Å")
    
    # Apply single folding step
    folding_output = folding_head(
        protein_features, ligand_features,
        initial_coords, ligand_coords
    )
    
    single_step_coords = folding_output['coordinates']
    confidence_scores = folding_output['confidence']
    pocket_scores = folding_output['pocket_scores']
    
    print(f"\nSingle folding step results:")
    print(f"  Coordinate updates: {folding_output['coord_updates'].abs().mean():.3f} Å (mean)")
    print(f"  Average confidence: {confidence_scores.mean():.3f}")
    print(f"  Pocket residues identified: {(pocket_scores[0] > 0.5).sum()}")
    
    # Apply multi-step structure module
    structure_output = structure_module(
        protein_features, ligand_features,
        initial_coords, ligand_coords
    )
    
    final_coords = structure_output['final_coordinates']
    final_confidence = structure_output['final_confidence']
    coordinate_trajectory = structure_output['coordinate_trajectory']
    
    print(f"\nMulti-step refinement results:")
    print(f"  Refinement steps: {len(coordinate_trajectory) - 1}")
    print(f"  Final confidence: {final_confidence.mean():.3f}")
    
    # Analyze structural changes
    final_radius_gyration = torch.sqrt(((final_coords[0] - final_coords[0].mean(dim=0))**2).sum(dim=-1).mean())
    total_displacement = torch.norm(final_coords - initial_coords, dim=-1).mean()
    
    print(f"  Final radius of gyration: {final_radius_gyration:.2f} Å")
    print(f"  Average atom displacement: {total_displacement:.2f} Å")
    print(f"  Structure compaction: {(initial_radius_gyration - final_radius_gyration):.2f} Å")
    
    # Analyze ligand-protein interactions
    final_ligand_protein_distances = compute_ligand_protein_distances(
        ligand_coords[0], final_coords[0]
    )
    min_interaction_distance = final_ligand_protein_distances.min()
    interacting_residues = (final_ligand_protein_distances.min(dim=-1)[0] <= 5.0).sum()
    
    print(f"\nLigand-protein interactions:")
    print(f"  Closest approach: {min_interaction_distance:.2f} Å")
    print(f"  Residues within 5Å of ligand: {interacting_residues}")
    
    print("✅ Ligand-aware folding completed!")


def demo_batch_processing():
    """Demonstrate batch processing of multiple protein-ligand complexes."""
    
    print("\n📦 Batch Processing Demo")
    print("=" * 50)
    
    # Multiple ligands
    ligand_smiles = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    ]
    
    ligand_names = ["Ethanol", "Aspirin", "Caffeine"]
    
    print(f"Processing {len(ligand_smiles)} ligands in batch:")
    for name, smiles in zip(ligand_names, ligand_smiles):
        print(f"  {name}: {smiles}")
    
    # Batch process ligands
    batched_ligands = batch_process_ligands(ligand_smiles)
    
    batch_size = len(ligand_smiles)
    max_atoms = batched_ligands['atom_types'].shape[1]
    
    print(f"\nBatch processing results:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max atoms per ligand: {max_atoms}")
    print(f"  Actual atoms per ligand: {batched_ligands['num_atoms'].tolist()}")
    
    # Create batch folding system
    seq_len = 50
    protein_dim, ligand_dim, d_model = 192, 64, 128
    
    folding_head = LigandAwareFoldingHead(protein_dim, ligand_dim, d_model)
    ligand_encoder = LigandEncoder(ligand_dim)
    
    # Mock protein data for batch
    protein_features = torch.randn(batch_size, seq_len, protein_dim)
    protein_coords = torch.randn(batch_size, seq_len, 3) * 2.0
    
    # Encode ligands
    print(f"\nEncoding ligands...")
    ligand_features_list = []
    
    for i in range(batch_size):
        # Extract single ligand data
        single_ligand = {
            'atom_types': batched_ligands['atom_types'][i][:batched_ligands['num_atoms'][i]],
            'edge_index': torch.randint(0, batched_ligands['num_atoms'][i], (2, batched_ligands['num_atoms'][i] * 2)),
            'bond_types': torch.randint(0, 5, (batched_ligands['num_atoms'][i] * 2,)),
            'ring_info': batched_ligands['ring_info'][i][:batched_ligands['num_atoms'][i]],
            'pharmacophore_features': batched_ligands['pharmacophore_features'][i][:batched_ligands['num_atoms'][i]],
            'molecular_descriptors': batched_ligands['molecular_descriptors'][i][:batched_ligands['num_atoms'][i]],
            'hybridization': batched_ligands['hybridization'][i][:batched_ligands['num_atoms'][i]],
            'formal_charges': batched_ligands['formal_charges'][i][:batched_ligands['num_atoms'][i]]
        }
        
        ligand_output = ligand_encoder(single_ligand)
        ligand_features_list.append(ligand_output['atom_embeddings'])
    
    # Pad ligand features to same size
    padded_ligand_features = torch.zeros(batch_size, max_atoms, ligand_dim)
    for i, features in enumerate(ligand_features_list):
        num_atoms = features.shape[0]
        padded_ligand_features[i, :num_atoms] = features
    
    # Apply batch folding
    print(f"Applying ligand-aware folding to batch...")
    
    batch_output = folding_head(
        protein_features, padded_ligand_features,
        protein_coords, batched_ligands['coordinates']
    )
    
    print(f"\nBatch folding results:")
    print(f"  Output coordinates: {batch_output['coordinates'].shape}")
    print(f"  Average confidence per complex: {batch_output['confidence'].mean(dim=1).tolist()}")
    print(f"  Pocket residues per complex: {(batch_output['pocket_scores'] > 0.5).sum(dim=1).tolist()}")
    
    print("✅ Batch processing completed!")


def main():
    """Run all demos."""
    
    print("🧬 OdinFold Ligand-Aware Folding System Demo")
    print("=" * 60)
    print()
    
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        demo_ligand_encoding()
        demo_cross_attention()
        demo_binding_pocket_prediction()
        demo_ligand_aware_folding()
        demo_batch_processing()
        
        print("\n🎉 All demos completed successfully!")
        print("\nThe ligand-aware folding system is ready for:")
        print("  • Molecular graph encoding of ligands")
        print("  • Cross-attention between protein and ligand features")
        print("  • Binding pocket prediction and analysis")
        print("  • Ligand-conditioned protein structure prediction")
        print("  • Multi-step structure refinement with ligand awareness")
        print("  • Batch processing of protein-ligand complexes")
        print("  • Integration with OdinFold's main folding pipeline")
        
        # Model statistics
        print(f"\n📊 System Statistics:")
        
        # Count parameters
        ligand_encoder = LigandEncoder(128)
        folding_head = LigandAwareFoldingHead(384, 128, 256)
        structure_module = LigandConditionedStructureModule(384, 128, 256, num_layers=3)
        
        total_params = (
            sum(p.numel() for p in ligand_encoder.parameters()) +
            sum(p.numel() for p in folding_head.parameters()) +
            sum(p.numel() for p in structure_module.parameters())
        )
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Ligand encoder: {sum(p.numel() for p in ligand_encoder.parameters()):,}")
        print(f"  Folding head: {sum(p.numel() for p in folding_head.parameters()):,}")
        print(f"  Structure module: {sum(p.numel() for p in structure_module.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
