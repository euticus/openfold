#!/usr/bin/env python3
"""
Model Quantization for WASM Build

Quantizes OdinFold model for browser deployment with aggressive optimizations:
- INT8 quantization for weights
- Dynamic quantization for activations  
- Layer pruning for size reduction
- ONNX export for WASM compatibility
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import argparse
import logging
from pathlib import Path
import json
import sys
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openfoldpp.model.model import AlphaFoldModel
from openfoldpp.config import model_config

logger = logging.getLogger(__name__)


class WASMModelOptimizer:
    """Optimizes OdinFold model for WASM deployment."""
    
    def __init__(self, max_seq_len: int = 200):
        self.max_seq_len = max_seq_len
        self.optimization_stats = {}
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load the original model."""
        
        logger.info(f"Loading model from {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model with WASM-optimized config
        config = model_config()
        config = self._optimize_config_for_wasm(config)
        
        model = AlphaFoldModel(config)
        
        # Load weights if available
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        
        original_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Original model parameters: {original_size:,}")
        
        return model
    
    def _optimize_config_for_wasm(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model config for WASM constraints."""
        
        # Reduce model dimensions
        config['model']['evoformer_stack']['c_m'] = 128  # Reduce from 256
        config['model']['evoformer_stack']['c_z'] = 64   # Reduce from 128
        config['model']['evoformer_stack']['no_blocks'] = 12  # Reduce from 48
        
        # Simplify structure module
        config['model']['structure_module']['c_s'] = 256  # Reduce from 384
        config['model']['structure_module']['no_blocks'] = 4  # Reduce from 8
        
        # Disable expensive features
        config['model']['structure_module']['no_angles'] = 4  # Reduce from 7
        
        # Set sequence length limit
        config['globals']['max_seq_len'] = self.max_seq_len
        
        logger.info("Applied WASM-specific config optimizations")
        return config
    
    def prune_model(self, model: nn.Module, pruning_ratio: float = 0.3) -> nn.Module:
        """Apply structured pruning to reduce model size."""
        
        logger.info(f"Applying {pruning_ratio:.1%} structured pruning")
        
        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply magnitude-based pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune smallest magnitude weights
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), pruning_ratio)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
        
        # Count remaining parameters
        remaining_params = sum((p != 0).sum().item() for p in model.parameters())
        pruning_achieved = 1 - (remaining_params / original_params)
        
        self.optimization_stats['pruning'] = {
            'target_ratio': pruning_ratio,
            'achieved_ratio': pruning_achieved,
            'original_params': original_params,
            'remaining_params': remaining_params
        }
        
        logger.info(f"Pruning achieved: {pruning_achieved:.1%} ({remaining_params:,} params remaining)")
        
        return model
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply INT8 quantization for WASM deployment."""
        
        logger.info("Applying INT8 quantization")
        
        # Prepare model for quantization
        model.train()
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare for quantization
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with dummy data
        self._calibrate_model(model_prepared)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)
        
        # Calculate size reduction
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(model_quantized)
        size_reduction = 1 - (quantized_size / original_size)
        
        self.optimization_stats['quantization'] = {
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'size_reduction': size_reduction
        }
        
        logger.info(f"Quantization size reduction: {size_reduction:.1%}")
        logger.info(f"Model size: {original_size/(1024*1024):.1f}MB → {quantized_size/(1024*1024):.1f}MB")
        
        return model_quantized
    
    def _calibrate_model(self, model: nn.Module):
        """Calibrate quantized model with representative data."""
        
        logger.info("Calibrating quantized model")
        
        # Generate calibration data
        batch_size = 1
        seq_lengths = [50, 100, 150, 200]
        
        model.eval()
        with torch.no_grad():
            for seq_len in seq_lengths:
                # Mock input data
                aatype = torch.randint(0, 21, (batch_size, seq_len))
                residue_index = torch.arange(seq_len).unsqueeze(0)
                
                # Create mock MSA (single sequence)
                msa = torch.randint(0, 21, (batch_size, 1, seq_len))
                
                # Mock template features
                template_aatype = torch.zeros(batch_size, 1, seq_len, dtype=torch.long)
                template_all_atom_positions = torch.zeros(batch_size, 1, seq_len, 37, 3)
                template_all_atom_mask = torch.zeros(batch_size, 1, seq_len, 37)
                
                batch = {
                    'aatype': aatype,
                    'residue_index': residue_index,
                    'msa': msa,
                    'template_aatype': template_aatype,
                    'template_all_atom_positions': template_all_atom_positions,
                    'template_all_atom_mask': template_all_atom_mask,
                }
                
                try:
                    # Run forward pass for calibration
                    _ = model(batch)
                except Exception as e:
                    logger.warning(f"Calibration failed for seq_len {seq_len}: {e}")
                    continue
        
        logger.info("Model calibration completed")
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        
        return total_size
    
    def export_onnx(self, model: nn.Module, output_path: str):
        """Export model to ONNX format for WASM compatibility."""
        
        logger.info(f"Exporting to ONNX: {output_path}")
        
        # Create dummy input
        batch_size = 1
        seq_len = 100
        
        dummy_input = {
            'aatype': torch.randint(0, 21, (batch_size, seq_len)),
            'residue_index': torch.arange(seq_len).unsqueeze(0),
            'msa': torch.randint(0, 21, (batch_size, 1, seq_len)),
            'template_aatype': torch.zeros(batch_size, 1, seq_len, dtype=torch.long),
            'template_all_atom_positions': torch.zeros(batch_size, 1, seq_len, 37, 3),
            'template_all_atom_mask': torch.zeros(batch_size, 1, seq_len, 37),
        }
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['aatype', 'residue_index', 'msa', 'template_aatype', 
                           'template_all_atom_positions', 'template_all_atom_mask'],
                output_names=['coordinates', 'confidence'],
                dynamic_axes={
                    'aatype': {1: 'seq_len'},
                    'residue_index': {1: 'seq_len'},
                    'msa': {2: 'seq_len'},
                    'template_aatype': {2: 'seq_len'},
                    'template_all_atom_positions': {2: 'seq_len'},
                    'template_all_atom_mask': {2: 'seq_len'},
                    'coordinates': {1: 'seq_len'},
                    'confidence': {1: 'seq_len'}
                }
            )
            
            logger.info("ONNX export completed successfully")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            # Save as PyTorch model instead
            torch.save(model.state_dict(), output_path.replace('.onnx', '.pt'))
            logger.info(f"Saved as PyTorch model: {output_path.replace('.onnx', '.pt')}")
    
    def create_wasm_manifest(self, output_dir: str):
        """Create manifest file for WASM deployment."""
        
        manifest = {
            'model_info': {
                'name': 'OdinFold++ WASM',
                'version': '1.0.0',
                'max_sequence_length': self.max_seq_len,
                'supported_features': [
                    'single_chain_folding',
                    'confidence_scoring',
                    'pdb_output'
                ],
                'unsupported_features': [
                    'multimer_folding',
                    'ligand_binding',
                    'msa_generation',
                    'advanced_refinement'
                ]
            },
            'optimization_stats': self.optimization_stats,
            'browser_requirements': {
                'webassembly': True,
                'shared_array_buffer': False,
                'minimum_memory_mb': 512,
                'recommended_memory_mb': 1024
            },
            'performance_targets': {
                'model_size_mb': 50,
                'inference_time_100aa_seconds': 30,
                'memory_usage_mb': 512
            }
        }
        
        manifest_path = Path(output_dir) / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created WASM manifest: {manifest_path}")
    
    def optimize_for_wasm(self, 
                         input_path: str, 
                         output_dir: str,
                         pruning_ratio: float = 0.3,
                         quantize: bool = True) -> Dict[str, Any]:
        """Complete optimization pipeline for WASM deployment."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting WASM optimization pipeline")
        
        # Load original model
        model = self.load_model(input_path)
        
        # Apply pruning
        if pruning_ratio > 0:
            model = self.prune_model(model, pruning_ratio)
        
        # Apply quantization
        if quantize:
            model = self.quantize_model(model)
        
        # Save optimized PyTorch model
        torch_output = output_dir / 'model_optimized.pt'
        torch.save(model.state_dict(), torch_output)
        logger.info(f"Saved optimized PyTorch model: {torch_output}")
        
        # Export to ONNX
        onnx_output = output_dir / 'model.onnx'
        self.export_onnx(model, str(onnx_output))
        
        # Create manifest
        self.create_wasm_manifest(str(output_dir))
        
        # Final statistics
        final_size = self._get_model_size(model)
        self.optimization_stats['final'] = {
            'model_size_mb': final_size / (1024 * 1024),
            'parameters': sum(p.numel() for p in model.parameters()),
            'max_sequence_length': self.max_seq_len
        }
        
        logger.info("WASM optimization completed!")
        logger.info(f"Final model size: {final_size/(1024*1024):.1f}MB")
        
        return self.optimization_stats


def main():
    parser = argparse.ArgumentParser(description='Optimize OdinFold model for WASM deployment')
    parser.add_argument('--input', required=True, help='Input model path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--max-seq-len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--pruning-ratio', type=float, default=0.3, help='Pruning ratio (0.0-0.9)')
    parser.add_argument('--no-quantize', action='store_true', help='Skip quantization')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run optimization
    optimizer = WASMModelOptimizer(max_seq_len=args.max_seq_len)
    
    stats = optimizer.optimize_for_wasm(
        input_path=args.input,
        output_dir=args.output,
        pruning_ratio=args.pruning_ratio,
        quantize=not args.no_quantize
    )
    
    print("\n" + "="*50)
    print("WASM Optimization Summary")
    print("="*50)
    print(f"Final model size: {stats['final']['model_size_mb']:.1f}MB")
    print(f"Parameters: {stats['final']['parameters']:,}")
    print(f"Max sequence length: {stats['final']['max_sequence_length']}")
    
    if 'pruning' in stats:
        print(f"Pruning reduction: {stats['pruning']['achieved_ratio']:.1%}")
    
    if 'quantization' in stats:
        print(f"Quantization reduction: {stats['quantization']['size_reduction']:.1%}")
    
    print(f"\nOptimized model saved to: {args.output}")


if __name__ == '__main__':
    main()
