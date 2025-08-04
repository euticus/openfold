#!/usr/bin/env python3
"""
Slim EvoFormer Configuration for OpenFold++

This module provides optimized configurations for Phase B:
- Halved layer depth (48 → 24 blocks)
- Grouped-Query Attention settings
- SwiGLU MLP configurations
- Weight sharing parameters
"""

import copy
from typing import Dict, Any


def get_slim_evoformer_config(base_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create slim EvoFormer configuration for Phase B optimization.
    
    Args:
        base_config: Base OpenFold configuration to modify
        
    Returns:
        Modified configuration with Phase B optimizations
    """
    
    # Default base configuration (simplified OpenFold config)
    if base_config is None:
        base_config = get_default_openfold_config()
    
    # Create a deep copy to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # PBX-10: Halve layer depth (48 → 24)
    config["model"]["evoformer_stack"]["no_blocks"] = 24
    
    # PBX-11: Grouped-Query Attention settings
    config["model"]["evoformer_stack"]["use_gqa"] = True
    config["model"]["evoformer_stack"]["gqa_groups"] = 4  # k=4 for KV sharing
    
    # PBX-12: SwiGLU MLP settings
    config["model"]["evoformer_stack"]["use_swiglu"] = True
    config["model"]["evoformer_stack"]["swiglu_hidden_ratio"] = 2.0  # 2x instead of 4x
    
    # PBX-13: Weight sharing settings
    config["model"]["evoformer_stack"]["weight_sharing_interval"] = 4  # Share every 4 layers
    config["model"]["evoformer_stack"]["use_weight_sharing"] = True
    
    # PBX-14: FlashAttention settings
    config["model"]["evoformer_stack"]["use_flash_attention"] = True
    config["model"]["evoformer_stack"]["flash_attention_version"] = 2
    
    # Additional optimizations
    config["model"]["evoformer_stack"]["blocks_per_ckpt"] = 4  # More frequent checkpointing
    config["model"]["evoformer_stack"]["clear_cache_between_blocks"] = True
    
    return config


def get_default_openfold_config() -> Dict[str, Any]:
    """
    Get default OpenFold configuration structure.
    
    This is a simplified version focusing on EvoFormer parameters.
    """
    
    # Model dimensions
    c_m = 256  # MSA channel dimension
    c_z = 128  # Pair channel dimension
    c_s = 384  # Single representation dimension
    
    config = {
        "model": {
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": c_s,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 48,  # Will be reduced to 24 in slim config
                "transition_n": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "no_column_attention": False,
                "opm_first": False,
                "fuse_projection_weights": False,
                "blocks_per_ckpt": 8,
                "clear_cache_between_blocks": False,
                "tune_chunk_size": False,
                "inf": 1e9,
                "eps": 1e-10,
            },
            "extra_msa": {
                "extra_msa_stack": {
                    "c_m": c_m,
                    "c_z": c_z,
                    "c_hidden_msa_att": 8,
                    "c_hidden_opm": 32,
                    "c_hidden_mul": 128,
                    "c_hidden_pair_att": 32,
                    "no_heads_msa": 8,
                    "no_heads_pair": 4,
                    "no_blocks": 4,
                    "transition_n": 4,
                    "msa_dropout": 0.15,
                    "pair_dropout": 0.25,
                    "opm_first": True,
                    "fuse_projection_weights": True,
                    "blocks_per_ckpt": 1,
                    "inf": 1e9,
                    "eps": 1e-10,
                }
            }
        }
    }
    
    return config


def get_teacher_transfer_config() -> Dict[str, Any]:
    """
    Configuration for teacher transfer from 48-layer to 24-layer model.
    
    Returns:
        Configuration specifying which layers to copy
    """
    
    # Copy even-numbered layers (0, 2, 4, ..., 46) to new 24-layer model
    layer_mapping = {}
    
    for new_idx in range(24):
        old_idx = new_idx * 2  # Map to even-numbered layers
        layer_mapping[new_idx] = old_idx
    
    config = {
        "teacher_transfer": {
            "enabled": True,
            "layer_mapping": layer_mapping,
            "copy_strategy": "even_layers",  # "even_layers", "uniform_sample", "first_last"
            "interpolate_missing": False,
            "freeze_transferred": False,  # Whether to freeze transferred weights initially
        }
    }
    
    return config


def get_phase_b_benchmark_config() -> Dict[str, Any]:
    """
    Configuration for Phase B benchmarking targets.
    
    Returns:
        Benchmark targets and thresholds
    """
    
    config = {
        "benchmark": {
            "targets": {
                "speed_improvement": 2.0,  # 2x faster than Phase A
                "max_parameters": 115_000_000,  # ≤ 115M parameters
                "max_tm_drop": 0.03,  # TM drop ≤ 0.03 vs Phase A
                "memory_reduction": 0.35,  # 35% memory reduction target
                "flash_attention_speedup": 0.40,  # 40% speedup from FlashAttention
            },
            "test_sequences": [
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
            ],
            "metrics": [
                "forward_time",
                "backward_time", 
                "memory_usage",
                "parameter_count",
                "tm_score",
                "gdt_ts",
                "rmsd"
            ]
        }
    }
    
    return config


def create_config_variants() -> Dict[str, Dict[str, Any]]:
    """
    Create different configuration variants for testing.
    
    Returns:
        Dictionary of configuration variants
    """
    
    base_config = get_default_openfold_config()
    
    variants = {
        "baseline": base_config,
        "slim": get_slim_evoformer_config(base_config),
        "slim_no_gqa": get_slim_evoformer_config(base_config),
        "slim_no_swiglu": get_slim_evoformer_config(base_config),
        "slim_no_sharing": get_slim_evoformer_config(base_config),
    }
    
    # Disable specific features for ablation studies
    variants["slim_no_gqa"]["model"]["evoformer_stack"]["use_gqa"] = False
    variants["slim_no_swiglu"]["model"]["evoformer_stack"]["use_swiglu"] = False
    variants["slim_no_sharing"]["model"]["evoformer_stack"]["use_weight_sharing"] = False
    
    return variants


# Configuration validation
def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for consistency and requirements.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    
    try:
        evo_config = config["model"]["evoformer_stack"]
        
        # Check required parameters
        required_params = [
            "c_m", "c_z", "no_blocks", "no_heads_msa", "no_heads_pair"
        ]
        
        for param in required_params:
            if param not in evo_config:
                print(f"Missing required parameter: {param}")
                return False
        
        # Check parameter ranges
        if evo_config["no_blocks"] <= 0:
            print("no_blocks must be positive")
            return False
        
        if evo_config.get("use_gqa", False):
            if "gqa_groups" not in evo_config:
                print("gqa_groups required when use_gqa=True")
                return False
            
            if evo_config["no_heads_pair"] % evo_config["gqa_groups"] != 0:
                print("no_heads_pair must be divisible by gqa_groups")
                return False
        
        return True
        
    except KeyError as e:
        print(f"Configuration validation error: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Create slim configuration
    slim_config = get_slim_evoformer_config()
    
    # Validate configuration
    if validate_config(slim_config):
        print("✅ Slim EvoFormer configuration is valid")
        
        # Print key changes
        evo_config = slim_config["model"]["evoformer_stack"]
        print(f"📊 Configuration Summary:")
        print(f"   Blocks: {evo_config['no_blocks']}")
        print(f"   GQA enabled: {evo_config.get('use_gqa', False)}")
        print(f"   SwiGLU enabled: {evo_config.get('use_swiglu', False)}")
        print(f"   Weight sharing: {evo_config.get('use_weight_sharing', False)}")
        print(f"   FlashAttention: {evo_config.get('use_flash_attention', False)}")
    else:
        print("❌ Configuration validation failed")
    
    # Create teacher transfer config
    transfer_config = get_teacher_transfer_config()
    print(f"\n🎓 Teacher Transfer: {len(transfer_config['teacher_transfer']['layer_mapping'])} layer mappings")
