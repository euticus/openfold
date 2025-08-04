"""
Enhanced MD-based structure refinement for OpenFold++.

This module provides modern molecular dynamics refinement capabilities
using TorchMD, OpenMM, and enhanced Amber relaxation for post-fold
structure optimization.
"""

import os
import logging
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    import openmm
    from openmm import app as openmm_app
    from openmm import unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    logging.warning("OpenMM not available. Some MD features will be disabled.")

try:
    import torchmd
    from torchmd.systems import System
    from torchmd.integrators import Integrator
    from torchmd.forces import Forces
    TORCHMD_AVAILABLE = True
except ImportError:
    TORCHMD_AVAILABLE = False
    logging.warning("TorchMD not available. GPU MD features will be disabled.")

from openfold.np import protein
from openfold.np.relax import amber_minimize, relax
from openfold.np.relax.relax import AmberRelaxation


class TorchMDRefinement:
    """
    GPU-accelerated molecular dynamics refinement using TorchMD.
    """
    
    def __init__(self,
                 force_field: str = "amber14",
                 temperature: float = 300.0,
                 pressure: float = 1.0,
                 timestep: float = 2.0,  # fs
                 device: str = "cuda"):
        """
        Args:
            force_field: Force field to use (amber14, charmm36, etc.)
            temperature: Temperature in Kelvin
            pressure: Pressure in bar
            timestep: Integration timestep in femtoseconds
            device: Device for computation ("cuda" or "cpu")
        """
        if not TORCHMD_AVAILABLE:
            raise ImportError("TorchMD is required for GPU MD refinement")
        
        self.force_field = force_field
        self.temperature = temperature
        self.pressure = pressure
        self.timestep = timestep
        self.device = device
    
    def refine_structure(self,
                        pdb_string: str,
                        steps: int = 1000,
                        minimize_steps: int = 500,
                        equilibration_steps: int = 1000) -> Tuple[str, Dict[str, Any]]:
        """
        Refine a protein structure using TorchMD.
        
        Args:
            pdb_string: Input PDB string
            steps: Number of MD simulation steps
            minimize_steps: Number of energy minimization steps
            equilibration_steps: Number of equilibration steps
            
        Returns:
            Tuple of (refined_pdb_string, refinement_info)
        """
        try:
            # Create TorchMD system from PDB
            system = System.from_pdb(pdb_string, force_field=self.force_field)
            system = system.to(self.device)
            
            # Set up forces
            forces = Forces(system, cutoff=10.0, switch_distance=8.0)
            
            # Energy minimization
            logging.info(f"Starting energy minimization for {minimize_steps} steps")
            integrator = Integrator(system, timestep=0.1)  # Smaller timestep for minimization
            
            initial_energy = forces.compute_energy(system.positions).item()
            
            for step in range(minimize_steps):
                forces_tensor = forces.compute_forces(system.positions)
                system.positions = integrator.minimize_step(system.positions, forces_tensor)
            
            minimized_energy = forces.compute_energy(system.positions).item()
            
            # Equilibration
            logging.info(f"Starting equilibration for {equilibration_steps} steps")
            integrator = Integrator(system, timestep=self.timestep, temperature=self.temperature)
            
            for step in range(equilibration_steps):
                forces_tensor = forces.compute_forces(system.positions)
                system.positions, system.velocities = integrator.step(
                    system.positions, system.velocities, forces_tensor
                )
            
            # Production MD
            logging.info(f"Starting production MD for {steps} steps")
            energies = []
            
            for step in range(steps):
                forces_tensor = forces.compute_forces(system.positions)
                system.positions, system.velocities = integrator.step(
                    system.positions, system.velocities, forces_tensor
                )
                
                if step % 100 == 0:
                    energy = forces.compute_energy(system.positions).item()
                    energies.append(energy)
            
            final_energy = forces.compute_energy(system.positions).item()
            
            # Convert back to PDB
            refined_pdb = system.to_pdb()
            
            refinement_info = {
                "initial_energy": initial_energy,
                "minimized_energy": minimized_energy,
                "final_energy": final_energy,
                "energy_trajectory": energies,
                "steps_completed": steps,
                "method": "TorchMD",
                "force_field": self.force_field,
                "temperature": self.temperature
            }
            
            return refined_pdb, refinement_info
            
        except Exception as e:
            logging.error(f"TorchMD refinement failed: {e}")
            # Fallback to input structure
            return pdb_string, {"error": str(e), "method": "TorchMD_failed"}


class OpenMMRefinement:
    """
    Enhanced OpenMM-based molecular dynamics refinement.
    """
    
    def __init__(self,
                 force_field: str = "amber14-all.xml",
                 water_model: str = "tip3p.xml",
                 temperature: float = 300.0,
                 pressure: float = 1.0,
                 use_gpu: bool = True):
        """
        Args:
            force_field: OpenMM force field XML file
            water_model: Water model XML file
            temperature: Temperature in Kelvin
            pressure: Pressure in bar
            use_gpu: Whether to use GPU acceleration
        """
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is required for MD refinement")
        
        self.force_field = force_field
        self.water_model = water_model
        self.temperature = temperature * unit.kelvin
        self.pressure = pressure * unit.bar
        self.use_gpu = use_gpu
    
    def refine_structure(self,
                        pdb_string: str,
                        steps: int = 10000,
                        minimize_steps: int = 1000,
                        implicit_solvent: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Refine a protein structure using OpenMM.
        
        Args:
            pdb_string: Input PDB string
            steps: Number of MD simulation steps
            minimize_steps: Number of energy minimization steps
            implicit_solvent: Whether to use implicit solvent
            
        Returns:
            Tuple of (refined_pdb_string, refinement_info)
        """
        try:
            # Load PDB
            from io import StringIO
            pdb_file = StringIO(pdb_string)
            pdb = openmm_app.PDBFile(pdb_file)
            
            # Set up force field
            forcefield = openmm_app.ForceField(self.force_field)
            if implicit_solvent:
                forcefield = openmm_app.ForceField(self.force_field, 'implicit/gbn2.xml')
            
            # Create system
            if implicit_solvent:
                system = forcefield.createSystem(
                    pdb.topology,
                    nonbondedMethod=openmm_app.NoCutoff,
                    implicitSolvent=openmm_app.GBn2
                )
            else:
                system = forcefield.createSystem(
                    pdb.topology,
                    nonbondedMethod=openmm_app.PME,
                    nonbondedCutoff=1.0*unit.nanometer
                )
            
            # Set up integrator
            integrator = openmm.LangevinMiddleIntegrator(
                self.temperature,
                1/unit.picosecond,
                2*unit.femtoseconds
            )
            
            # Set up simulation
            if self.use_gpu:
                platform = openmm.Platform.getPlatformByName('CUDA')
                properties = {'CudaPrecision': 'mixed'}
            else:
                platform = openmm.Platform.getPlatformByName('CPU')
                properties = {}
            
            simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform, properties)
            simulation.context.setPositions(pdb.positions)
            
            # Energy minimization
            logging.info(f"Starting energy minimization for {minimize_steps} steps")
            initial_state = simulation.context.getState(getEnergy=True)
            initial_energy = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            
            simulation.minimizeEnergy(maxIterations=minimize_steps)
            
            minimized_state = simulation.context.getState(getEnergy=True)
            minimized_energy = minimized_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            
            # Equilibration
            logging.info("Starting equilibration")
            simulation.context.setVelocitiesToTemperature(self.temperature)
            
            # Production MD
            logging.info(f"Starting production MD for {steps} steps")
            energies = []
            
            for step in range(steps):
                simulation.step(1)
                
                if step % 1000 == 0:
                    state = simulation.context.getState(getEnergy=True)
                    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    energies.append(energy)
            
            # Get final structure
            final_state = simulation.context.getState(getPositions=True, getEnergy=True)
            final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            
            # Write refined PDB
            with StringIO() as output:
                openmm_app.PDBFile.writeFile(
                    simulation.topology,
                    final_state.getPositions(),
                    output
                )
                refined_pdb = output.getvalue()
            
            refinement_info = {
                "initial_energy": initial_energy,
                "minimized_energy": minimized_energy,
                "final_energy": final_energy,
                "energy_trajectory": energies,
                "steps_completed": steps,
                "method": "OpenMM",
                "force_field": self.force_field,
                "temperature": self.temperature.value_in_unit(unit.kelvin),
                "implicit_solvent": implicit_solvent
            }
            
            return refined_pdb, refinement_info
            
        except Exception as e:
            logging.error(f"OpenMM refinement failed: {e}")
            return pdb_string, {"error": str(e), "method": "OpenMM_failed"}


class EnhancedAmberRefinement:
    """
    Enhanced wrapper around OpenFold's existing Amber relaxation.
    """
    
    def __init__(self,
                 max_iterations: int = 0,
                 tolerance: float = 2.39,
                 stiffness: float = 10.0,
                 max_outer_iterations: int = 20,
                 use_gpu: bool = True):
        """
        Args:
            max_iterations: Maximum L-BFGS iterations (0 = no limit)
            tolerance: Energy tolerance in kcal/mol
            stiffness: Spring constant for restraints in kcal/mol/A^2
            max_outer_iterations: Maximum outer iterations
            use_gpu: Whether to use GPU acceleration
        """
        self.relaxer = AmberRelaxation(
            max_iterations=max_iterations,
            tolerance=tolerance,
            stiffness=stiffness,
            exclude_residues=[],
            max_outer_iterations=max_outer_iterations,
            use_gpu=use_gpu
        )
    
    def refine_structure(self, prot: protein.Protein) -> Tuple[str, Dict[str, Any]]:
        """
        Refine a protein structure using enhanced Amber relaxation.
        
        Args:
            prot: OpenFold protein object
            
        Returns:
            Tuple of (refined_pdb_string, refinement_info)
        """
        try:
            refined_pdb, debug_data, violations = self.relaxer.process(prot=prot)
            
            refinement_info = {
                "method": "Enhanced_Amber",
                "initial_energy": debug_data.get("initial_energy"),
                "final_energy": debug_data.get("final_energy"),
                "attempts": debug_data.get("attempts"),
                "rmsd": debug_data.get("rmsd"),
                "violations": violations.sum() if violations is not None else 0
            }
            
            return refined_pdb, refinement_info
            
        except Exception as e:
            logging.error(f"Enhanced Amber refinement failed: {e}")
            # Fallback to original PDB
            from openfold.np import protein as protein_utils
            original_pdb = protein_utils.to_pdb(prot)
            return original_pdb, {"error": str(e), "method": "Enhanced_Amber_failed"}


class MDRefinementPipeline:
    """
    Comprehensive MD refinement pipeline with multiple methods.
    """
    
    def __init__(self,
                 methods: List[str] = ["amber", "openmm"],
                 use_gpu: bool = True,
                 fallback_on_failure: bool = True):
        """
        Args:
            methods: List of refinement methods to try ["amber", "openmm", "torchmd"]
            use_gpu: Whether to use GPU acceleration
            fallback_on_failure: Whether to fallback to next method on failure
        """
        self.methods = methods
        self.use_gpu = use_gpu
        self.fallback_on_failure = fallback_on_failure
        
        # Initialize refinement engines
        self.refiners = {}
        
        if "amber" in methods:
            self.refiners["amber"] = EnhancedAmberRefinement(use_gpu=use_gpu)
        
        if "openmm" in methods and OPENMM_AVAILABLE:
            self.refiners["openmm"] = OpenMMRefinement(use_gpu=use_gpu)
        
        if "torchmd" in methods and TORCHMD_AVAILABLE:
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            self.refiners["torchmd"] = TorchMDRefinement(device=device)
    
    def refine_structure(self,
                        structure: Union[protein.Protein, str],
                        method: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Refine a protein structure using the specified or default method.
        
        Args:
            structure: OpenFold protein object or PDB string
            method: Specific method to use (if None, tries all methods)
            
        Returns:
            Tuple of (refined_pdb_string, refinement_info)
        """
        methods_to_try = [method] if method else self.methods
        
        for method_name in methods_to_try:
            if method_name not in self.refiners:
                logging.warning(f"Method {method_name} not available, skipping")
                continue
            
            try:
                refiner = self.refiners[method_name]
                
                if method_name == "amber":
                    # Amber expects protein object
                    if isinstance(structure, str):
                        prot = protein.from_pdb_string(structure)
                    else:
                        prot = structure
                    
                    refined_pdb, info = refiner.refine_structure(prot)
                    
                else:
                    # OpenMM and TorchMD expect PDB string
                    if isinstance(structure, protein.Protein):
                        pdb_string = protein.to_pdb(structure)
                    else:
                        pdb_string = structure
                    
                    refined_pdb, info = refiner.refine_structure(pdb_string)
                
                # Check if refinement was successful
                if "error" not in info:
                    logging.info(f"Refinement successful with method: {method_name}")
                    return refined_pdb, info
                else:
                    logging.warning(f"Method {method_name} failed: {info['error']}")
                    if not self.fallback_on_failure:
                        return refined_pdb, info
                        
            except Exception as e:
                logging.error(f"Method {method_name} failed with exception: {e}")
                if not self.fallback_on_failure:
                    raise
        
        # If all methods failed, return original structure
        if isinstance(structure, protein.Protein):
            original_pdb = protein.to_pdb(structure)
        else:
            original_pdb = structure
        
        return original_pdb, {
            "error": "All refinement methods failed",
            "methods_tried": methods_to_try
        }
    
    def batch_refine(self,
                    structures: List[Union[protein.Protein, str]],
                    method: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Refine multiple structures in batch.
        
        Args:
            structures: List of protein structures to refine
            method: Refinement method to use
            
        Returns:
            List of (refined_pdb_string, refinement_info) tuples
        """
        results = []
        
        for i, structure in enumerate(structures):
            logging.info(f"Refining structure {i+1}/{len(structures)}")
            try:
                refined_pdb, info = self.refine_structure(structure, method)
                results.append((refined_pdb, info))
            except Exception as e:
                logging.error(f"Failed to refine structure {i+1}: {e}")
                # Add failed result
                if isinstance(structure, protein.Protein):
                    original_pdb = protein.to_pdb(structure)
                else:
                    original_pdb = structure
                results.append((original_pdb, {"error": str(e)}))
        
        return results


def refine_openfold_output(model_output: Dict[str, torch.Tensor],
                          batch: Dict[str, torch.Tensor],
                          refinement_method: str = "amber",
                          use_gpu: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to refine OpenFold model output.
    
    Args:
        model_output: OpenFold model output dictionary
        batch: Input batch dictionary
        refinement_method: Method to use for refinement
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (refined_pdb_string, refinement_info)
    """
    # Convert model output to protein object
    final_atom_positions = model_output["final_atom_positions"]
    final_atom_mask = model_output["final_atom_mask"]
    aatype = batch["aatype"]
    
    # Create protein object
    prot = protein.Protein(
        atom_positions=final_atom_positions.cpu().numpy(),
        atom_mask=final_atom_mask.cpu().numpy(),
        aatype=aatype.cpu().numpy(),
        residue_index=batch["residue_index"].cpu().numpy(),
        b_factors=np.zeros_like(final_atom_mask.cpu().numpy())
    )
    
    # Refine structure
    pipeline = MDRefinementPipeline(methods=[refinement_method], use_gpu=use_gpu)
    return pipeline.refine_structure(prot)
