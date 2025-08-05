#!/usr/bin/env python3
"""
Mutation Prediction Demo for OdinFold

Demonstrates the complete ΔΔG mutation prediction system including:
- Single mutation prediction
- High-throughput mutation scanning
- Stability analysis and thermodynamic modeling
- Mutation report generation
"""

import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.modules.mutation import (
    DDGPredictor,
    MutationScanner,
    StabilityPredictor,
    create_ddg_loss
)
from openfoldpp.modules.mutation.ddg_predictor import amino_acid_to_index, index_to_amino_acid


def demo_single_mutation_prediction():
    """Demonstrate single mutation prediction."""
    
    print("🧬 Single Mutation Prediction Demo")
    print("=" * 50)
    
    # Create predictor
    structure_dim, d_model = 384, 256
    predictor = DDGPredictor(structure_dim, d_model)
    
    # Mock structure features (in real use, these come from OdinFold)
    seq_len = 100
    structure_features = torch.randn(seq_len, structure_dim)
    
    # Predict effect of A50V mutation
    wt_aa = amino_acid_to_index('A')  # Alanine
    mut_aa = amino_acid_to_index('V')  # Valine
    position = 49  # 0-indexed (position 50 in 1-indexed)
    
    result = predictor.predict_single(structure_features, wt_aa, mut_aa, position)
    
    print(f"Mutation: A50V")
    print(f"ΔΔG prediction: {result['ddg_pred']:.3f} kJ/mol")
    print(f"Uncertainty: {result['uncertainty']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if result['ddg_pred'] > 0.5:
        effect = "Destabilizing"
    elif result['ddg_pred'] < -0.5:
        effect = "Stabilizing"
    else:
        effect = "Neutral"
    
    print(f"Predicted effect: {effect}")
    print()


def demo_mutation_scanning():
    """Demonstrate high-throughput mutation scanning."""
    
    print("🔬 Mutation Scanning Demo")
    print("=" * 50)
    
    # Create components
    structure_dim, d_model = 384, 256
    predictor = DDGPredictor(structure_dim, d_model)
    scanner = MutationScanner(predictor)
    
    # Mock protein sequence and structure
    sequence = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
    
    seq_len = len(sequence)
    structure_features = torch.randn(seq_len, structure_dim)
    
    print(f"Protein sequence length: {seq_len}")
    print(f"Sequence: {sequence[:50]}...")
    print()
    
    # Scan mutations at a specific position
    position = 25
    mutations = scanner.scan_position(structure_features, sequence, position)
    
    print(f"Scanning position {position+1} (wild-type: {sequence[position]})")
    print(f"Found {len(mutations)} possible mutations")
    print()
    
    # Show top 5 stabilizing and destabilizing mutations
    print("Top 5 stabilizing mutations:")
    for i, mut in enumerate(mutations[:5]):
        print(f"  {i+1}. {mut.wt_aa}{mut.position+1}{mut.mut_aa}: ΔΔG = {mut.ddg_pred:.3f} kJ/mol ({mut.effect_category})")
    
    print("\nTop 5 destabilizing mutations:")
    for i, mut in enumerate(mutations[-5:]):
        print(f"  {i+1}. {mut.wt_aa}{mut.position+1}{mut.mut_aa}: ΔΔG = {mut.ddg_pred:.3f} kJ/mol ({mut.effect_category})")
    print()
    
    # Find globally stabilizing mutations
    stabilizing = scanner.find_stabilizing_mutations(
        structure_features, sequence, ddg_threshold=-0.2, top_k=10
    )
    
    print(f"Found {len(stabilizing)} stabilizing mutations across the protein:")
    for i, mut in enumerate(stabilizing[:5]):
        print(f"  {i+1}. {mut.wt_aa}{mut.position+1}{mut.mut_aa}: ΔΔG = {mut.ddg_pred:.3f} kJ/mol")
    print()


def demo_stability_prediction():
    """Demonstrate protein stability prediction."""
    
    print("🌡️ Stability Prediction Demo")
    print("=" * 50)
    
    # Create components
    structure_dim, d_model = 384, 256
    ddg_predictor = DDGPredictor(structure_dim, d_model)
    stability_predictor = StabilityPredictor(ddg_predictor, structure_dim=structure_dim)
    
    # Mock protein
    sequence = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
    sequence_indices = [amino_acid_to_index(aa) for aa in sequence]
    
    seq_len = len(sequence)
    structure_features = torch.randn(seq_len, structure_dim)
    
    # Predict stability at standard conditions
    result = stability_predictor(structure_features, sequence_indices)
    
    print("Stability prediction at standard conditions (25°C, pH 7.0):")
    print(f"  Stability free energy: {result['stability_free_energy']:.3f} kJ/mol")
    print(f"  Melting temperature: {result['melting_temperature']-273.15:.1f} °C")
    print(f"  Folding probability: {result['folding_probability']:.3f}")
    print()
    
    # Predict melting curve
    curve = stability_predictor.predict_melting_curve(
        structure_features, sequence_indices, num_points=10
    )
    
    print("Melting curve prediction:")
    for i, (temp, frac) in enumerate(zip(curve['temperature_celsius'], curve['fraction_folded'])):
        print(f"  {temp:.1f}°C: {frac:.3f} folded")
    print()
    
    # Predict pH stability
    ph_curve = stability_predictor.predict_ph_stability(
        structure_features, sequence_indices, num_points=5
    )
    
    print("pH stability prediction:")
    for ph, stability in zip(ph_curve['ph'], ph_curve['stability']):
        print(f"  pH {ph:.1f}: ΔG = {stability:.3f} kJ/mol")
    print()


def demo_mutation_report():
    """Demonstrate comprehensive mutation report generation."""
    
    print("📊 Mutation Report Demo")
    print("=" * 50)
    
    # Create components
    structure_dim, d_model = 384, 256
    predictor = DDGPredictor(structure_dim, d_model)
    scanner = MutationScanner(predictor)
    
    # Smaller protein for faster demo
    sequence = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"[:100]  # First 100 residues
    
    seq_len = len(sequence)
    structure_features = torch.randn(seq_len, structure_dim)
    
    print(f"Generating mutation report for protein (length: {seq_len})")
    print("This may take a moment...")
    
    # Generate comprehensive report
    report = scanner.generate_mutation_report(
        structure_features, sequence, "Demo_Protein"
    )
    
    print(f"\n📋 Mutation Report for {report['protein_name']}")
    print(f"Sequence length: {report['sequence_length']}")
    print(f"Total mutations scanned: {report['total_mutations_scanned']}")
    print()
    
    print("📈 Statistics:")
    stats = report['statistics']
    print(f"  Mean ΔΔG: {stats['mean_ddg']:.3f} ± {stats['std_ddg']:.3f} kJ/mol")
    print(f"  Range: {stats['min_ddg']:.3f} to {stats['max_ddg']:.3f} kJ/mol")
    print(f"  Mean confidence: {stats['mean_confidence']:.3f}")
    print()
    
    print("🏷️ Mutation categories:")
    for category, count in report['mutation_categories'].items():
        percentage = (count / report['total_mutations_scanned']) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    print()
    
    print("⭐ Top 3 stabilizing mutations:")
    for i, mut in enumerate(report['top_stabilizing'][:3]):
        print(f"  {i+1}. {mut['mutation']}: ΔΔG = {mut['ddg_pred']:.3f} kJ/mol")
    print()
    
    print("⚠️ Top 3 destabilizing mutations:")
    for i, mut in enumerate(report['top_destabilizing'][:3]):
        print(f"  {i+1}. {mut['mutation']}: ΔΔG = {mut['ddg_pred']:.3f} kJ/mol")
    print()
    
    print("💡 Recommendations:")
    recs = report['recommendations']
    if recs['engineering_targets']:
        print("  Engineering targets:")
        for target in recs['engineering_targets'][:3]:
            print(f"    - {target['mutation']}: {target['ddg_improvement']:.3f} kJ/mol improvement")
    
    if recs['conservation_warnings']:
        print("  Conservation warnings:")
        for warning in recs['conservation_warnings'][:3]:
            print(f"    - Position {warning['position']} ({warning['wt_aa']}): {warning['reason']}")
    print()


def demo_training_setup():
    """Demonstrate training setup for ΔΔG prediction."""
    
    print("🎯 Training Setup Demo")
    print("=" * 50)
    
    # Create model and loss function
    structure_dim, d_model = 384, 256
    predictor = DDGPredictor(structure_dim, d_model)
    loss_fn = create_ddg_loss()
    
    # Mock training data
    batch_size = 16
    seq_len = 50
    
    structure_features = torch.randn(batch_size, seq_len, structure_dim)
    wt_aa = torch.randint(0, 20, (batch_size,))
    mut_aa = torch.randint(0, 20, (batch_size,))
    position = torch.randint(0, seq_len, (batch_size,))
    target_ddg = torch.randn(batch_size) * 2.0  # ΔΔG values
    
    print(f"Training batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Structure features shape: {structure_features.shape}")
    print()
    
    # Forward pass
    predictions = predictor(structure_features, wt_aa, mut_aa, position)
    
    # Compute loss
    losses = loss_fn(predictions, target_ddg)
    
    print("Training step results:")
    print(f"  Total loss: {losses['total_loss']:.4f}")
    print(f"  MSE loss: {losses['mse_loss']:.4f}")
    print(f"  Uncertainty loss: {losses['uncertainty_loss']:.4f}")
    print(f"  MAE: {losses['mae']:.4f}")
    print()
    
    print("Model parameters:")
    total_params = sum(p.numel() for p in predictor.parameters())
    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()


def main():
    """Run all demos."""
    
    print("🧬 OdinFold Mutation Prediction System Demo")
    print("=" * 60)
    print()
    
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        demo_single_mutation_prediction()
        demo_mutation_scanning()
        demo_stability_prediction()
        demo_mutation_report()
        demo_training_setup()
        
        print("🎉 All demos completed successfully!")
        print("\nThe ΔΔG mutation prediction system is ready for:")
        print("  • Single mutation effect prediction")
        print("  • High-throughput mutation scanning")
        print("  • Protein stability analysis")
        print("  • Thermodynamic modeling")
        print("  • Mutation report generation")
        print("  • Integration with OdinFold structure prediction")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
