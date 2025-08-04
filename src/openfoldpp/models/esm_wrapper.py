#!/usr/bin/env python3
"""
ESM-2 Protein Language Model Wrapper for OpenFold++

This module provides a clean interface to Facebook's ESM-2 model for extracting
protein sequence embeddings to replace MSA dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    logging.warning("ESM not available. Install with: pip install fair-esm")

try:
    import bitsandbytes as bnb
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logging.warning("bitsandbytes not available. Quantization disabled.")


@dataclass
class ESMConfig:
    """Configuration for ESM-2 model."""
    model_name: str = "esm2_t33_650M_UR50D"  # 650M parameter model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    quantize: bool = True  # Use 8-bit quantization
    cache_dir: Optional[str] = None
    max_sequence_length: int = 1024
    batch_size: int = 1  # Conservative for memory
    

class ESMWrapper(nn.Module):
    """
    Wrapper around Facebook's ESM-2 model for protein sequence embedding.
    
    Features:
    - Automatic model loading and caching
    - 8-bit quantization support for memory efficiency
    - Batch processing with memory management
    - Token-level and sequence-level embeddings
    """
    
    def __init__(self, config: ESMConfig = None):
        super().__init__()
        
        if not ESM_AVAILABLE:
            raise ImportError("ESM not available. Install with: pip install fair-esm")
        
        self.config = config or ESMConfig()
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
        # Load model
        self._load_model()
        
        # Setup quantization if requested
        if self.config.quantize and QUANTIZATION_AVAILABLE:
            self._quantize_model()
    
    def _load_model(self):
        """Load ESM-2 model and alphabet."""
        logging.info(f"Loading ESM-2 model: {self.config.model_name}")
        
        # Load model and alphabet
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            self.config.model_name
        )
        
        # Setup batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Move to device
        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        logging.info(f"ESM-2 model loaded on {self.config.device}")
    
    def _quantize_model(self):
        """Apply 8-bit quantization to reduce memory usage."""
        if not QUANTIZATION_AVAILABLE:
            logging.warning("Quantization requested but bitsandbytes not available")
            return
        
        logging.info("Applying 8-bit quantization to ESM-2 model")
        
        # Replace linear layers with 8-bit versions
        def replace_linear_with_8bit(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Replace with 8-bit linear layer
                    new_layer = bnb.nn.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )
                    
                    # Copy weights
                    new_layer.weight.data = child.weight.data
                    if child.bias is not None:
                        new_layer.bias.data = child.bias.data
                    
                    setattr(module, name, new_layer)
                else:
                    replace_linear_with_8bit(child)
        
        replace_linear_with_8bit(self.model)
        logging.info("8-bit quantization applied")
    
    def encode_sequences(
        self, 
        sequences: List[str], 
        labels: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode protein sequences into embeddings.
        
        Args:
            sequences: List of protein sequences (amino acid strings)
            labels: Optional labels for sequences
            
        Returns:
            Dictionary containing:
            - 'representations': Token-level embeddings [batch, seq_len, hidden_dim]
            - 'mean_representations': Sequence-level embeddings [batch, hidden_dim]
            - 'attention': Attention weights if available
        """
        if labels is None:
            labels = [f"seq_{i}" for i in range(len(sequences))]
        
        # Prepare batch data
        batch_data = list(zip(labels, sequences))
        
        # Convert to tokens
        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.config.device)
        
        # Truncate if necessary
        if batch_tokens.size(1) > self.config.max_sequence_length:
            batch_tokens = batch_tokens[:, :self.config.max_sequence_length]
            logging.warning(f"Sequences truncated to {self.config.max_sequence_length} tokens")
        
        # Extract embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        # Get token representations (last layer)
        token_representations = results["representations"][33]
        
        # Remove special tokens (BOS/EOS)
        token_representations = token_representations[:, 1:-1, :]
        
        # Compute sequence-level representations (mean pooling)
        sequence_lengths = [len(seq) for seq in sequences]
        mean_representations = []
        
        for i, length in enumerate(sequence_lengths):
            # Mean pool over actual sequence length (excluding padding)
            seq_repr = token_representations[i, :length, :].mean(dim=0)
            mean_representations.append(seq_repr)
        
        mean_representations = torch.stack(mean_representations)
        
        return {
            'representations': token_representations,
            'mean_representations': mean_representations,
            'sequence_lengths': sequence_lengths,
            'labels': batch_labels
        }
    
    def extract_embeddings_for_openfold(
        self, 
        sequences: List[str]
    ) -> torch.Tensor:
        """
        Extract embeddings in format suitable for OpenFold EvoFormer.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Embeddings tensor [batch, max_seq_len, 1280] ready for projection
        """
        results = self.encode_sequences(sequences)
        embeddings = results['representations']  # [batch, seq_len, 1280]
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.embed_dim  # 1280 for ESM-2 650M
    
    def save_cache(self, cache_path: str):
        """Save model cache for faster loading."""
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir) / cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, cache_path)
            
            logging.info(f"Model cache saved to {cache_path}")
    
    def load_cache(self, cache_path: str) -> bool:
        """Load model from cache."""
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir) / cache_path
            
            if cache_path.exists():
                checkpoint = torch.load(cache_path, map_location=self.config.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Model cache loaded from {cache_path}")
                return True
        
        return False


def create_esm_wrapper(
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = None,
    quantize: bool = True
) -> ESMWrapper:
    """
    Factory function to create ESM wrapper with common configurations.
    
    Args:
        model_name: ESM model name
        device: Device to load model on
        quantize: Whether to apply 8-bit quantization
        
    Returns:
        Configured ESMWrapper instance
    """
    config = ESMConfig(
        model_name=model_name,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        quantize=quantize
    )
    
    return ESMWrapper(config)


# Example usage and testing
if __name__ == "__main__":
    # Test sequences
    test_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
    ]
    
    # Create wrapper
    wrapper = create_esm_wrapper(quantize=False)  # Disable quantization for testing
    
    # Extract embeddings
    embeddings = wrapper.extract_embeddings_for_openfold(test_sequences)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {wrapper.get_embedding_dim()}")
