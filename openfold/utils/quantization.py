"""
Advanced quantization and memory optimization utilities for OpenFold++.

This module provides modern quantization techniques including INT8, FP16, and BF16
quantization for memory-efficient inference on large protein sequences.
"""

import os
import logging
import warnings
from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
import torch.nn.functional as F

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("BitsAndBytes not available. Advanced quantization features disabled.")

try:
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer that can use INT8 or 4-bit quantization.
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 quantization_mode: str = "int8",
                 device: Optional[torch.device] = None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_mode = quantization_mode
        
        if quantization_mode == "int8" and BITSANDBYTES_AVAILABLE:
            self.linear = bnb.nn.Linear8bitLt(
                in_features, out_features, bias=bias, device=device
            )
        elif quantization_mode == "4bit" and BITSANDBYTES_AVAILABLE:
            self.linear = bnb.nn.Linear4bit(
                in_features, out_features, bias=bias, device=device
            )
        else:
            # Fallback to standard linear layer
            self.linear = nn.Linear(in_features, out_features, bias=bias, device=device)
            if quantization_mode != "none":
                logging.warning(f"Quantization mode {quantization_mode} not available, using standard linear")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention with optional quantization.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 quantize_qkv: bool = False,
                 quantization_mode: str = "int8"):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if quantize_qkv:
            self.q_proj = QuantizedLinear(embed_dim, embed_dim, bias=False, 
                                        quantization_mode=quantization_mode)
            self.k_proj = QuantizedLinear(embed_dim, embed_dim, bias=False,
                                        quantization_mode=quantization_mode)
            self.v_proj = QuantizedLinear(embed_dim, embed_dim, bias=False,
                                        quantization_mode=quantization_mode)
            self.out_proj = QuantizedLinear(embed_dim, embed_dim, bias=False,
                                          quantization_mode=quantization_mode)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use scaled dot-product attention with memory efficiency
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)


class ModelQuantizer:
    """
    Utility class for quantizing OpenFold models.
    """
    
    def __init__(self, quantization_config: Optional[Dict[str, Any]] = None):
        self.config = quantization_config or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "quantization_mode": "int8",  # "int8", "4bit", "fp16", "bf16"
            "quantize_embeddings": True,
            "quantize_attention": True,
            "quantize_feedforward": True,
            "quantize_output_layers": False,  # Keep output layers in full precision
            "preserve_accuracy_layers": ["structure_module"],  # Critical layers to keep in full precision
        }
    
    def quantize_model(self, model: nn.Module, inplace: bool = True) -> nn.Module:
        """
        Quantize an OpenFold model for memory-efficient inference.
        
        Args:
            model: The OpenFold model to quantize
            inplace: Whether to modify the model in-place
            
        Returns:
            Quantized model
        """
        if not inplace:
            model = model.deepcopy()
        
        quantization_mode = self.config["quantization_mode"]
        
        if quantization_mode in ["fp16", "bf16"]:
            return self._apply_mixed_precision(model, quantization_mode)
        elif quantization_mode in ["int8", "4bit"]:
            return self._apply_integer_quantization(model, quantization_mode)
        else:
            logging.warning(f"Unknown quantization mode: {quantization_mode}")
            return model
    
    def _apply_mixed_precision(self, model: nn.Module, precision: str) -> nn.Module:
        """Apply FP16 or BF16 mixed precision."""
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        
        # Convert model to half precision
        model = model.to(dtype=dtype)
        
        # Keep certain layers in full precision for stability
        preserve_layers = self.config.get("preserve_accuracy_layers", [])
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in preserve_layers):
                module.to(dtype=torch.float32)
        
        logging.info(f"Applied {precision} mixed precision quantization")
        return model
    
    def _apply_integer_quantization(self, model: nn.Module, quantization_mode: str) -> nn.Module:
        """Apply INT8 or 4-bit quantization."""
        if not BITSANDBYTES_AVAILABLE:
            logging.warning("BitsAndBytes not available. Falling back to PyTorch quantization.")
            return self._apply_pytorch_quantization(model)
        
        # Replace linear layers with quantized versions
        self._replace_linear_layers(model, quantization_mode)
        
        logging.info(f"Applied {quantization_mode} quantization")
        return model
    
    def _apply_pytorch_quantization(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch's built-in dynamic quantization."""
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d, nn.Conv2d},
            dtype=torch.qint8
        )
        
        logging.info("Applied PyTorch dynamic quantization")
        return quantized_model
    
    def _replace_linear_layers(self, model: nn.Module, quantization_mode: str):
        """Replace linear layers with quantized versions."""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Check if this layer should be quantized
                if self._should_quantize_layer(name, module):
                    quantized_layer = QuantizedLinear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        quantization_mode=quantization_mode,
                        device=module.weight.device
                    )
                    
                    # Copy weights if possible
                    if hasattr(quantized_layer.linear, 'weight'):
                        with torch.no_grad():
                            quantized_layer.linear.weight.copy_(module.weight)
                            if module.bias is not None:
                                quantized_layer.linear.bias.copy_(module.bias)
                    
                    setattr(model, name, quantized_layer)
            else:
                # Recursively quantize child modules
                self._replace_linear_layers(module, quantization_mode)
    
    def _should_quantize_layer(self, layer_name: str, layer: nn.Module) -> bool:
        """Determine if a layer should be quantized based on configuration."""
        preserve_layers = self.config.get("preserve_accuracy_layers", [])
        
        # Don't quantize layers that are explicitly preserved
        if any(preserve_name in layer_name for preserve_name in preserve_layers):
            return False
        
        # Check specific layer type configurations
        if "embedding" in layer_name.lower():
            return self.config.get("quantize_embeddings", True)
        elif "attention" in layer_name.lower() or "attn" in layer_name.lower():
            return self.config.get("quantize_attention", True)
        elif "feed_forward" in layer_name.lower() or "mlp" in layer_name.lower():
            return self.config.get("quantize_feedforward", True)
        elif "output" in layer_name.lower() or "head" in layer_name.lower():
            return self.config.get("quantize_output_layers", False)
        
        return True  # Default to quantizing


class AdvancedCheckpointing:
    """
    Advanced gradient checkpointing strategies for memory optimization.
    """
    
    def __init__(self, 
                 strategy: str = "adaptive",
                 memory_budget: Optional[int] = None):
        """
        Args:
            strategy: Checkpointing strategy ("adaptive", "uniform", "selective")
            memory_budget: Target memory usage in GB
        """
        self.strategy = strategy
        self.memory_budget = memory_budget
    
    def apply_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply advanced checkpointing to the model."""
        if self.strategy == "adaptive":
            return self._apply_adaptive_checkpointing(model)
        elif self.strategy == "uniform":
            return self._apply_uniform_checkpointing(model)
        elif self.strategy == "selective":
            return self._apply_selective_checkpointing(model)
        else:
            logging.warning(f"Unknown checkpointing strategy: {self.strategy}")
            return model
    
    def _apply_adaptive_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply adaptive checkpointing based on memory usage."""
        # Monitor memory usage and adjust checkpointing accordingly
        if hasattr(model, 'evoformer'):
            # More aggressive checkpointing for Evoformer (memory intensive)
            model.evoformer.blocks_per_ckpt = 1
        
        if hasattr(model, 'structure_module'):
            # Less aggressive for structure module (computationally intensive)
            if hasattr(model.structure_module, 'blocks_per_ckpt'):
                model.structure_module.blocks_per_ckpt = 2
        
        return model
    
    def _apply_uniform_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply uniform checkpointing across all modules."""
        for module in model.modules():
            if hasattr(module, 'blocks_per_ckpt'):
                module.blocks_per_ckpt = 1
        
        return model
    
    def _apply_selective_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply checkpointing only to memory-intensive modules."""
        memory_intensive_modules = ['evoformer', 'extra_msa_stack', 'template_embedder']
        
        for name, module in model.named_modules():
            if any(intensive_name in name for intensive_name in memory_intensive_modules):
                if hasattr(module, 'blocks_per_ckpt'):
                    module.blocks_per_ckpt = 1
        
        return model


def optimize_model_for_long_sequences(model: nn.Module,
                                    max_sequence_length: int = 3000,
                                    quantization_config: Optional[Dict[str, Any]] = None,
                                    checkpointing_strategy: str = "adaptive") -> nn.Module:
    """
    Optimize an OpenFold model for long sequence inference.
    
    Args:
        model: The OpenFold model to optimize
        max_sequence_length: Target maximum sequence length
        quantization_config: Configuration for quantization
        checkpointing_strategy: Strategy for gradient checkpointing
        
    Returns:
        Optimized model
    """
    logging.info(f"Optimizing model for sequences up to {max_sequence_length} residues")
    
    # Apply quantization
    quantizer = ModelQuantizer(quantization_config)
    model = quantizer.quantize_model(model, inplace=True)
    
    # Apply advanced checkpointing
    checkpointer = AdvancedCheckpointing(strategy=checkpointing_strategy)
    model = checkpointer.apply_checkpointing(model)
    
    # Enable memory-efficient settings
    if hasattr(model, 'config'):
        # Enable long sequence inference mode
        model.config.globals.offload_inference = True
        model.config.globals.use_lma = True  # Low-memory attention
        model.config.globals.use_flash = False  # LMA is better for very long sequences
        
        # Disable chunk size tuning for long sequences
        if hasattr(model.config.model, 'evoformer_stack'):
            model.config.model.evoformer_stack.tune_chunk_size = False
        
        # Enable template offloading
        if hasattr(model.config.model, 'template'):
            model.config.model.template.offload_inference = True
            model.config.model.template.offload_templates = True
    
    logging.info("Model optimization complete")
    return model


def estimate_memory_usage(model: nn.Module, 
                         sequence_length: int,
                         batch_size: int = 1,
                         precision: str = "fp32") -> Dict[str, float]:
    """
    Estimate memory usage for a given model and sequence length.
    
    Args:
        model: The model to analyze
        sequence_length: Input sequence length
        batch_size: Batch size
        precision: Model precision ("fp32", "fp16", "bf16", "int8")
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Calculate parameter memory
    param_memory = sum(p.numel() for p in model.parameters())
    
    # Precision multipliers
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "4bit": 0.5
    }
    
    bytes_per_param = precision_bytes.get(precision, 4)
    param_memory_gb = (param_memory * bytes_per_param) / (1024**3)
    
    # Estimate activation memory (rough approximation)
    # This is a simplified estimate based on sequence length scaling
    activation_memory_gb = (sequence_length**2 * batch_size * bytes_per_param) / (1024**3)
    
    # Add some overhead
    overhead_gb = 2.0  # GPU overhead, gradients, etc.
    
    total_memory_gb = param_memory_gb + activation_memory_gb + overhead_gb
    
    return {
        "parameters_gb": param_memory_gb,
        "activations_gb": activation_memory_gb,
        "overhead_gb": overhead_gb,
        "total_gb": total_memory_gb,
        "max_sequence_length_estimate": int(24 / (total_memory_gb / sequence_length))  # Rough estimate for 24GB GPU
    }
