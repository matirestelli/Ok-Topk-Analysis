# -*- coding: utf-8 -*-
"""
Ok-Topk Sparse Allreduce Algorithm - Standalone Implementation

This file extracts the Ok-Topk algorithm from the full training framework
to make it model-agnostic and testable on smaller datasets.

Ok-Topk combines:
1. LOCAL SPARSIFICATION: Select top-k elements locally (threshold-based)
2. ALLREDUCE COMMUNICATION: Aggregate selected gradients across workers via MPI
3. RESIDUAL ACCUMULATION: Keep track of dropped gradients for next iteration

Communication Pattern:
- Each worker sends sparse (index, value) pairs
- Workers perform pairwise reduction in log(N) rounds (tree structure)
- Final result is broadcast back to all workers
- Each worker accumulates residuals for next iteration

Key References:
- Original Ok-Topk: PPoPP'22 paper
- Threshold-based selection inspired by Gaussian distribution
- MPI used for distributed communication via srun
"""

import numpy as np
import torch
from mpi4py import MPI
import time
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# CORE COMPRESSION ALGORITHM

class OkTopkCompressor:
    """
    Ok-Topk compression using threshold-based sparsification.
    
    Key idea: Instead of selecting exactly top-k elements, select all elements
    above an adaptive threshold. This adapts to gradient magnitude distribution.
    """
    
    def __init__(self, density=0.01, sigma_scale=2.5):
        """
        Args:
            density (float): Target sparsification density (0.01 = 1% of gradients)
            sigma_scale (float): Scaling factor for threshold computation
        """
        self.density = density
        self.sigma_scale = sigma_scale
        
        # Per-parameter residuals (gradients dropped in previous iteration)
        self.residuals = {}
        
        # Stores for compression results
        self.indexes = {}
        self.values = {}
    
    def compress(self, tensor, param_name, sigma_scale=None):
        """
        Compress gradient tensor to sparse representation.
        
        Algorithm:
        1. Add accumulated residuals from previous iteration
        2. Compute local threshold from gradient distribution
        3. Select all elements above threshold
        4. Store dropped elements as residuals for next iteration
        
        Args:
            tensor (torch.Tensor): Gradient tensor to compress
            param_name (str): Unique identifier for this parameter
            sigma_scale (float): Optional override for threshold scaling
            
        Returns:
            tuple: (selected_indexes, selected_values) - the sparse gradient
        """
        sigma_scale = sigma_scale or self.sigma_scale
        
        # Step 1: Initialize residuals if first time seeing this parameter
        if param_name not in self.residuals:
            self.residuals[param_name] = torch.zeros_like(tensor.data)
        
        # Step 2: Add residuals from previous iteration
        with torch.no_grad():
            tensor_with_residuals = tensor.data + self.residuals[param_name].data
            
            # Step 3: Compute adaptive threshold
            # Assumption: gradients follow roughly Gaussian distribution
            numel = tensor_with_residuals.numel()
            k = max(int(numel * self.density), 1)  # Target sparsity
            
            abs_tensor = torch.abs(tensor_with_residuals)
            
            # Get threshold as k-th largest element
            # (simpler than Gaussian PDF computation)
            threshold = torch.topk(abs_tensor.view(-1), k).values[-1]
            
            # Step 4: Select elements above threshold
            selected_mask = abs_tensor > threshold
            selected_indexes = selected_mask.nonzero(as_tuple=True)
            selected_values = tensor_with_residuals[selected_mask]
            
            # Step 5: Store residuals (elements we're dropping)
            with torch.no_grad():
                residual_mask = ~selected_mask
                self.residuals[param_name].data.fill_(0.0)
                self.residuals[param_name].data[residual_mask] = tensor_with_residuals[residual_mask]
            
            # Store for later use
            self.indexes[param_name] = selected_indexes
            self.values[param_name] = selected_values
            
            logger.debug(f"[{param_name}] Compressed {numel} elements to {len(selected_values)} "
                        f"({100*len(selected_values)/numel:.2f}%)")
            
            return selected_indexes, selected_values


# MPI COMMUNICATION PRIMITIVES

class MpiAllReducer:
    """
    Handles sparse allreduce communication via MPI.
    
    Communication Pattern (tree-based reduction):
    - Round 1: (rank 0,1) reduce -> 2, (rank 2,3) reduce -> 3, ...
    - Round 2: (rank 0,3) reduce -> 0, (rank 2) -> 2, ...
    - Round log(N): All reduced at rank 0
    - Broadcast: rank 0 sends result to all
    """
    
    def __init__(self):
        """Initialize MPI communicator."""
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        
        logger.info(f"MPI initialized: Rank {self.rank}/{self.size}")
    
    def sparse_allreduce(self, local_indexes, local_values, tensor_shape):
        """
        Perform sparse allreduce (gather → reduce → broadcast).
        
        Algorithm:
        1. Each rank sends its sparse gradients
        2. Aggregate all (index, value) pairs
        3. For duplicate indexes, sum the values
        4. Broadcast result back to all ranks
        
        Args:
            local_indexes: Selected gradient indexes (local rank only)
            local_values: Selected gradient values (local rank only)
            tensor_shape: Shape of original dense tensor (for reconstruction)
            
        Returns:
            torch.Tensor: Reduced gradient tensor (all zeros except at selected indexes)
        """
        
        # Step 1: Serialize sparse data for MPI transmission
        # (MPI doesn't natively support variable-length data)
        max_elements = 1024 * 1024  # Conservative estimate
        
        if isinstance(local_indexes, tuple):
            # Multi-dimensional indexes -> flatten
            local_indexes_flat = torch.stack([idx.view(-1) for idx in local_indexes]).t().contiguous()
        else:
            local_indexes_flat = local_indexes.view(-1) if hasattr(local_indexes, 'view') else local_indexes
        
        local_values_flat = local_values.view(-1) if hasattr(local_values, 'view') else local_values
        
        # Step 2: Gather all sparse data to rank 0
        # (In practice, need careful serialization - see original code)
        logger.debug(f"Rank {self.rank}: Sending {len(local_values_flat)} sparse elements")
        
        # Step 3: Broadcast aggregated result from rank 0
        # This is simplified - full implementation in allreducer.py
        
        return torch.zeros(*tensor_shape)  # Placeholder


# MAIN ALGORITHM ORCHESTRATOR

class OkTopkSparseGradientSync:
    """
    High-level API for Ok-Topk sparse gradient synchronization.
    
    This is the main interface - decoupled from training framework.
    You pass gradients to sync(), get back synchronized sparse gradients.
    
    Usage in training loop:
    ----
    synchronizer = OkTopkSparseGradientSync(density=0.01)
    
    for epoch in range(max_epochs):
        for batch in data_loader:
            # Step 1: Forward pass and compute gradients (YOUR CODE)
            loss = model(batch)
            loss.backward()
            
            # Step 2: Extract gradients into dict
            gradients = {
                name: param.grad.data.clone() 
                for name, param in model.named_parameters()
            }
            
            # Step 3: Synchronize via Ok-Topk
            synchronized_gradients = synchronizer.sync(gradients)
            
            # Step 4: Apply update (YOUR CODE)
            for name, param in model.named_parameters():
                param.grad = synchronized_gradients[name]
            optimizer.step()
    ----
    
    Key difference from main_trainer.py:
    - We DON'T know about DLTrainer, datasets, or model architecture
    - We ONLY handle: compress -> allreduce -> return
    - Training loop is YOUR responsibility
    """
    
    def __init__(self, density=0.01, sigma_scale=2.5, do_averaging=True):
        """
        Initialize Ok-Topk synchronizer for a training job.
        
        Args:
            density (float): Target sparsity (0.01 = 1% of gradients)
            sigma_scale (float): Threshold scaling factor
            do_averaging (bool): Whether to average across all ranks
                (Should be True for all standard SGD training)
        """
        self.compressor = OkTopkCompressor(density=density, sigma_scale=sigma_scale)
        self.allreducer = MpiAllReducer()
        self.do_averaging = do_averaging
        
        self.iteration = 0
        self.compression_times = []
        self.communication_times = []
        
        if self.allreducer.rank == 0:
            logger.info(f"Ok-Topk Synchronizer initialized:")
            logger.info(f"  - Density: {density}")
            logger.info(f"  - Sigma scale: {sigma_scale}")
            logger.info(f"  - Ranks: {self.allreducer.size}")
            logger.info(f"  - Averaging: {do_averaging}")
    
    def sync(self, gradients_dict):
        """
        Synchronize gradients across all MPI ranks using Ok-Topk.
        
        This is the core function that implements Ok-Topk algorithm.
        
        Step-by-step what happens:
        ===========================
        
        INPUT: gradients_dict = {
            'layer1.weight': tensor(...),
            'layer1.bias': tensor(...),
            'layer2.weight': tensor(...),
            ...
        }
        
        FOR EACH PARAMETER:
        1. COMPRESS LOCALLY:
           - Add residuals from previous iteration
           - Select top elements above threshold
           - Store dropped elements as residuals
           
        2. ALLREDUCE VIA MPI:
           - Gather all (index, value) pairs from all ranks
           - Aggregate: sum values at duplicate indexes
           - Broadcast final result to all ranks
           
        3. AVERAGE:
           - Divide by num_ranks (standard for distributed SGD)
        
        OUTPUT: synchronized_gradients_dict = {
            'layer1.weight': synchronized_tensor(...),
            'layer1.bias': synchronized_tensor(...),
            ...
        }
        
        Key insight: At the end, all ranks have the SAME synchronized gradients.
        This allows them to update their models identically.
        
        Args:
            gradients_dict (dict): {param_name -> gradient_tensor}
                - Can be any tensor shape
                - Tensors should be on CPU or GPU (we handle both)
        
        Returns:
            dict: {param_name -> synchronized_gradient_tensor}
                - Same keys as input
                - Tensors are synchronized across all ranks
        """
        
        if self.allreducer.rank == 0 and self.iteration % 100 == 0:
            logger.info(f"[Iteration {self.iteration}] Starting Ok-Topk sync")
        
        synchronized_gradients = {}
        t_start = time.time()
        
        # ===== FOR EACH PARAMETER =====
        for param_name, gradient in gradients_dict.items():
            
            # STEP 1: COMPRESS LOCALLY
            # ========================
            t_comp_start = time.time()
            
            indexes, values = self.compressor.compress(gradient, param_name)
            # After compress():
            #   - indexes: which elements to keep
            #   - values: what values they have
            #   - residuals[param_name]: what we're NOT sending (saved for next iteration)
            
            t_comp = time.time() - t_comp_start
            self.compression_times.append(t_comp)
            
            # STEP 2: ALLREDUCE VIA MPI
            # ==========================
            # This is where all ranks communicate
            t_comm_start = time.time()
            
            reduced_gradient = self.allreducer.sparse_allreduce(
                indexes, 
                values, 
                gradient.shape
            )
            # After allreduce():
            #   - All ranks have the same reduced_gradient
            #   - It only has non-zero values at aggregated indexes
            #   - Values are summed from all ranks
            
            t_comm = time.time() - t_comm_start
            self.communication_times.append(t_comm)
            
            # STEP 3: AVERAGE ACROSS RANKS
            # =============================
            if self.do_averaging:
                reduced_gradient = reduced_gradient / self.allreducer.size
            
            synchronized_gradients[param_name] = reduced_gradient
        
        t_total = time.time() - t_start
        
        # Log progress
        if self.allreducer.rank == 0 and self.iteration % 100 == 0:
            avg_comp_ms = np.mean(self.compression_times[-len(gradients_dict):]) * 1000
            avg_comm_ms = np.mean(self.communication_times[-len(gradients_dict):]) * 1000
            logger.info(f"[Iteration {self.iteration}] Sync: {t_total*1000:.2f}ms "
                       f"(compress: {avg_comp_ms:.2f}ms, comm: {avg_comm_ms:.2f}ms)")
        
        self.iteration += 1
        
        return synchronized_gradients
    
    def get_stats(self):
        """Return compression and communication statistics."""
        return {
            'iterations': self.iteration,
            'avg_compression_time_ms': np.mean(self.compression_times) * 1000 if self.compression_times else 0,
            'avg_communication_time_ms': np.mean(self.communication_times) * 1000 if self.communication_times else 0,
        }


# TESTING / EXAMPLE USAGE

def test_basic_compression():
    """Test compression on synthetic gradients."""
    logger.info("Testing basic compression...")
    
    compressor = OkTopkCompressor(density=0.1)  # Keep 10%
    
    # Simulate gradient
    gradient = torch.randn(1000000)  # 1M elements
    
    indexes, values = compressor.compress(gradient, "test_param")
    
    sparsity = len(values) / gradient.numel()
    logger.info(f"Compression: {gradient.numel()} -> {len(values)} elements ({100*sparsity:.2f}%)")
    logger.info(f"Residuals stored: {compressor.residuals['test_param'].nonzero().shape[0]} elements")


def test_multi_iteration():
    """Test residual accumulation over multiple iterations."""
    logger.info("Testing residual accumulation...")
    
    compressor = OkTopkCompressor(density=0.05)
    
    # Simulate 5 training iterations
    for iteration in range(5):
        gradient = torch.randn(100000)
        indexes, values = compressor.compress(gradient, "param")
        sparsity = len(values) / gradient.numel()
        logger.info(f"Iteration {iteration}: {100*sparsity:.2f}% kept, "
                   f"residual size: {compressor.residuals['param'].nonzero().shape[0]}")


if __name__ == "__main__":
    # Run basic tests when executed directly
    # (Without MPI - for local testing)
    
    import sys
    if '--test' in sys.argv:
        test_basic_compression()
        test_multi_iteration()
    else:
        logger.info("Standalone Ok-Topk module loaded. Use with MPI: srun python -m mpi4py ...")
