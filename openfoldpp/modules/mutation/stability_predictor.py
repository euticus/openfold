"""
Stability Predictor for OdinFold

Predicts protein stability and thermodynamic properties
from structure and mutation information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .ddg_predictor import DDGPredictor

logger = logging.getLogger(__name__)


class StabilityPredictor(nn.Module):
    """
    Protein stability predictor combining ΔΔG predictions with
    thermodynamic modeling for comprehensive stability analysis.
    """
    
    def __init__(self,
                 ddg_predictor: DDGPredictor,
                 structure_dim: int = 384,
                 temperature_range: Tuple[float, float] = (273.15, 373.15),
                 ph_range: Tuple[float, float] = (5.0, 9.0)):
        super().__init__()

        self.ddg_predictor = ddg_predictor
        self.structure_dim = structure_dim
        self.temperature_range = temperature_range
        self.ph_range = ph_range

        # Thermodynamic parameters
        self.register_buffer('gas_constant', torch.tensor(8.314e-3))  # kJ/(mol·K)
        self.register_buffer('reference_temp', torch.tensor(298.15))  # 25°C

        # Sequence feature projection
        self.seq_proj = nn.Linear(structure_dim, ddg_predictor.d_model)
        
        # Stability prediction head
        self.stability_head = nn.Sequential(
            nn.Linear(ddg_predictor.d_model + 32, 256),  # +32 for condition features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # Tm, ΔG_unfold, ΔH, ΔS
        )
        
        # Condition encoding
        self.condition_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                structure_features: torch.Tensor,
                sequence: List[int],
                mutations: Optional[List[Tuple[int, int, int]]] = None,
                temperature: float = 298.15,
                ph: float = 7.0,
                ionic_strength: float = 0.15) -> Dict[str, torch.Tensor]:
        """
        Predict protein stability under given conditions.
        
        Args:
            structure_features: Structure features [seq_len, structure_dim]
            sequence: Amino acid sequence as indices
            mutations: List of (position, wt_aa, mut_aa) tuples
            temperature: Temperature in Kelvin
            ph: pH value
            ionic_strength: Ionic strength in M
            
        Returns:
            Dictionary with stability predictions
        """
        
        # Apply mutations if provided
        if mutations:
            total_ddg = self._calculate_total_ddg(
                structure_features, sequence, mutations
            )
        else:
            total_ddg = torch.tensor(0.0, device=structure_features.device,
                                   dtype=structure_features.dtype)
        
        # Encode conditions
        conditions = torch.tensor([temperature, ph, ionic_strength],
                                device=structure_features.device,
                                dtype=structure_features.dtype).unsqueeze(0)
        condition_features = self.condition_encoder(conditions)
        
        # Combine features
        seq_features = structure_features.mean(dim=0, keepdim=True)  # Global average
        seq_features = self.seq_proj(seq_features)  # Project to d_model

        combined_features = torch.cat([seq_features, condition_features], dim=-1)
        
        # Predict stability parameters
        stability_params = self.stability_head(combined_features).squeeze(0)
        
        tm_pred = stability_params[0] + 273.15  # Convert to Kelvin
        dg_unfold_ref = stability_params[1]
        dh = stability_params[2]
        ds = stability_params[3]
        
        # Apply mutation effects
        dg_unfold = dg_unfold_ref - total_ddg
        
        # Temperature-dependent stability
        dg_temp = self._calculate_temperature_dependence(
            dg_unfold, dh, ds, temperature, tm_pred
        )
        
        # pH-dependent corrections (simplified)
        dg_ph = self._calculate_ph_dependence(dg_temp, ph, sequence)
        
        # Ionic strength corrections
        dg_final = self._calculate_ionic_strength_dependence(
            dg_ph, ionic_strength, sequence
        )
        
        return {
            'stability_free_energy': dg_final,
            'melting_temperature': tm_pred,
            'unfolding_enthalpy': dh,
            'unfolding_entropy': ds,
            'total_ddg_mutation': total_ddg,
            'folding_probability': torch.sigmoid(dg_final / (self.gas_constant * temperature))
        }
    
    def _calculate_total_ddg(self,
                           structure_features: torch.Tensor,
                           sequence: List[int],
                           mutations: List[Tuple[int, int, int]]) -> torch.Tensor:
        """Calculate total ΔΔG from multiple mutations."""
        
        total_ddg = torch.tensor(0.0, device=structure_features.device)
        
        for position, wt_aa, mut_aa in mutations:
            # Predict single mutation effect
            result = self.ddg_predictor.predict_single(
                structure_features, wt_aa, mut_aa, position
            )
            total_ddg += result['ddg_pred']
        
        return total_ddg
    
    def _calculate_temperature_dependence(self,
                                        dg_ref: torch.Tensor,
                                        dh: torch.Tensor,
                                        ds: torch.Tensor,
                                        temperature: float,
                                        tm: torch.Tensor) -> torch.Tensor:
        """Calculate temperature-dependent free energy."""
        
        temp_tensor = torch.tensor(temperature, device=dg_ref.device, dtype=dg_ref.dtype)
        
        # Gibbs-Helmholtz equation
        dg_temp = dh * (1 - temp_tensor / tm) - temp_tensor * ds * torch.log(temp_tensor / tm)
        
        return dg_temp
    
    def _calculate_ph_dependence(self,
                               dg: torch.Tensor,
                               ph: float,
                               sequence: List[int]) -> torch.Tensor:
        """Calculate pH-dependent stability corrections."""
        
        # Count ionizable residues
        ionizable_count = sum(1 for aa in sequence if aa in [1, 3, 6, 8, 11])  # R, D, E, H, K
        
        # Simple pH correction (more sophisticated models would use pKa values)
        ph_factor = abs(ph - 7.0) * ionizable_count * 0.1
        ph_correction = torch.tensor(ph_factor, device=dg.device, dtype=dg.dtype)
        
        return dg - ph_correction
    
    def _calculate_ionic_strength_dependence(self,
                                           dg: torch.Tensor,
                                           ionic_strength: float,
                                           sequence: List[int]) -> torch.Tensor:
        """Calculate ionic strength-dependent stability corrections."""
        
        # Count charged residues
        charged_count = sum(1 for aa in sequence if aa in [1, 3, 6, 11])  # R, D, E, K
        
        # Debye-Hückel-like correction
        ionic_correction = charged_count * 0.5 * np.sqrt(ionic_strength)
        ionic_tensor = torch.tensor(ionic_correction, device=dg.device, dtype=dg.dtype)
        
        return dg + ionic_tensor
    
    def predict_melting_curve(self,
                            structure_features: torch.Tensor,
                            sequence: List[int],
                            mutations: Optional[List[Tuple[int, int, int]]] = None,
                            temperature_range: Optional[Tuple[float, float]] = None,
                            num_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Predict protein melting curve.
        
        Args:
            structure_features: Structure features
            sequence: Amino acid sequence
            mutations: Optional mutations
            temperature_range: Temperature range in Kelvin
            num_points: Number of temperature points
            
        Returns:
            Dictionary with temperature and fraction folded arrays
        """
        
        if temperature_range is None:
            temperature_range = self.temperature_range
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_points)
        fraction_folded = []
        
        with torch.no_grad():
            for temp in temperatures:
                result = self.forward(
                    structure_features, sequence, mutations, temperature=temp
                )
                fraction_folded.append(float(result['folding_probability']))
        
        return {
            'temperature': temperatures,
            'fraction_folded': np.array(fraction_folded),
            'temperature_celsius': temperatures - 273.15
        }
    
    def predict_ph_stability(self,
                           structure_features: torch.Tensor,
                           sequence: List[int],
                           mutations: Optional[List[Tuple[int, int, int]]] = None,
                           ph_range: Optional[Tuple[float, float]] = None,
                           num_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Predict pH-dependent stability.
        
        Args:
            structure_features: Structure features
            sequence: Amino acid sequence
            mutations: Optional mutations
            ph_range: pH range
            num_points: Number of pH points
            
        Returns:
            Dictionary with pH and stability arrays
        """
        
        if ph_range is None:
            ph_range = self.ph_range
        
        ph_values = np.linspace(ph_range[0], ph_range[1], num_points)
        stability_values = []
        
        with torch.no_grad():
            for ph in ph_values:
                result = self.forward(
                    structure_features, sequence, mutations, ph=ph
                )
                stability_values.append(float(result['stability_free_energy']))
        
        return {
            'ph': ph_values,
            'stability': np.array(stability_values)
        }
    
    def optimize_stability(self,
                         structure_features: torch.Tensor,
                         sequence: List[int],
                         target_positions: List[int],
                         max_mutations: int = 3,
                         temperature: float = 298.15) -> Dict:
        """
        Optimize protein stability through mutations.
        
        Args:
            structure_features: Structure features
            sequence: Amino acid sequence
            target_positions: Positions to consider for mutation
            max_mutations: Maximum number of mutations
            temperature: Target temperature
            
        Returns:
            Optimization results
        """
        
        from itertools import combinations
        
        best_stability = float('-inf')
        best_mutations = []
        
        # Try all combinations of mutations up to max_mutations
        for num_muts in range(1, max_mutations + 1):
            for positions in combinations(target_positions, num_muts):
                # For each position, try all amino acids
                for mut_combo in self._generate_mutation_combinations(positions, sequence):
                    with torch.no_grad():
                        result = self.forward(
                            structure_features, sequence, mut_combo, temperature
                        )
                        stability = float(result['stability_free_energy'])
                        
                        if stability > best_stability:
                            best_stability = stability
                            best_mutations = mut_combo
        
        return {
            'best_mutations': best_mutations,
            'best_stability': best_stability,
            'improvement': best_stability - float(self.forward(
                structure_features, sequence, None, temperature
            )['stability_free_energy'])
        }
    
    def _generate_mutation_combinations(self, positions: Tuple[int], 
                                      sequence: List[int]) -> List[List[Tuple[int, int, int]]]:
        """Generate mutation combinations for optimization."""
        
        # Simplified: try a few common stabilizing amino acids
        stabilizing_aa = [0, 9, 10, 15, 16, 19]  # A, I, L, S, T, V
        
        combinations = []
        
        def generate_recursive(pos_idx, current_combo):
            if pos_idx == len(positions):
                combinations.append(current_combo.copy())
                return
            
            pos = positions[pos_idx]
            wt_aa = sequence[pos]
            
            for mut_aa in stabilizing_aa:
                if mut_aa != wt_aa:
                    current_combo.append((pos, wt_aa, mut_aa))
                    generate_recursive(pos_idx + 1, current_combo)
                    current_combo.pop()
        
        generate_recursive(0, [])
        return combinations[:100]  # Limit combinations for efficiency
