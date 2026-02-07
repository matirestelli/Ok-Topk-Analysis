#!/usr/bin/env python3
"""
Script to extract MPI message sizes and timing from training logs.
Usage: python3 analyze_log.py <logfile>
"""

import sys
import re
from pathlib import Path

def analyze_log(logfile):
    """Extract and analyze MPI metrics from log file"""
    
    if not Path(logfile).exists():
        print(f"Error: {logfile} not found")
        sys.exit(1)
    
    with open(logfile, 'r') as f:
        content = f.read()
    
    print("=" * 80)
    print("MPI MESSAGE ANALYSIS FROM LOG FILE")
    print("=" * 80)
    
    # Extract tensor sizes
    tensor_pattern = r'\[rank:\d+\].*?\[(\d+)\]'
    matches = re.findall(tensor_pattern, content)
    
    if matches:
        tensor_sizes = list(set([int(m) for m in matches]))
        print(f"\nTensor Sizes Found:")
        for size in sorted(tensor_sizes):
            print(f"  - {size:,} parameters = {size * 4 / 1e6:.2f} MB (as float32)")
    
    # Extract timing breakdowns
    print(f"\nTiming Breakdowns (ms):")
    print("Format: backward, merge, compression, allreduce, demerge, d2h, h2d")
    
    timing_pattern = r'\[rank:\d+\].*?\[(\d+)\]: ([\d.,\-]+)'
    for match in re.finditer(timing_pattern, content):
        tensor_size = match.group(1)
        timings_str = match.group(2)
        
        # Only show first few examples
        if "e-" in timings_str or "0.000" in timings_str:
            timings = [float(t) for t in timings_str.split(',')]
            
            if len(timings) == 7:
                backward, merge, compression, allreduce, demerge, d2h, h2d = timings
                
                print(f"\n  Tensor [{tensor_size}]:")
                print(f"    Compression:   {compression:.6f} ms")
                print(f"    Allreduce:     {allreduce:.6f} ms  ← MPI Communication")
                print(f"    Ratio (Comp/MPI): {compression/allreduce if allreduce > 0 else 'inf':.1f}x")
                break  # Just show one example
    
    # Extract sparsity information
    density_pattern = r'density: ([\d.]+)'
    density_matches = re.findall(density_pattern, content)
    if density_matches:
        density = float(density_matches[0])
        sparsity = (1 - density) * 100
        print(f"\nSparsity Configuration:")
        print(f"  Density: {density:.4f} ({sparsity:.1f}% sparse)")
        
        # Calculate message sizes
        if tensor_sizes:
            largest_tensor = max(tensor_sizes)
            sparse_params = int(largest_tensor * density)
            bytes_per_param = 8  # 4 bytes value + 4 bytes index
            message_size_mb = sparse_params * bytes_per_param / 1e6
            
            print(f"\nMessage Size Estimation (largest tensor):")
            print(f"  Total parameters: {largest_tensor:,}")
            print(f"  Dense message: {largest_tensor * 4 / 1e6:.2f} MB")
            print(f"  Sparse message (2% kept): {message_size_mb:.2f} MB")
            print(f"  Compression ratio: {(largest_tensor * 4 / 1e6) / message_size_mb:.1f}x")
    
    # Extract worker count
    workers_pattern = r'nworkers=(\d+)'
    workers_match = re.search(workers_pattern, content)
    if workers_match:
        nworkers = int(workers_match.group(1))
        print(f"\nDistributed Configuration:")
        print(f"  Number of workers: {nworkers}")
        print(f"  Ring reduction rounds needed: {int(__import__('math').log2(nworkers))}")
        
        if tensor_sizes and density_matches:
            largest_tensor = max(tensor_sizes)
            density = float(density_matches[0])
            sparse_params = int(largest_tensor * density)
            message_size_mb = sparse_params * 8 / 1e6
            
            print(f"\nTotal Data Exchanged Per Iteration:")
            print(f"  Allreduce (all-to-all):")
            print(f"    - Single allgather: {nworkers} × {message_size_mb:.2f} MB = {nworkers * message_size_mb:.2f} MB")
            print(f"  Ring reduction (pairwise):")
            print(f"    - Per worker: {int(__import__('math').log2(nworkers))} rounds × {message_size_mb:.2f} MB = {int(__import__('math').log2(nworkers)) * message_size_mb:.2f} MB")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_log.py <logfile>")
        print("Example: python3 analyze_log.py LSTM/logs/allreduce-comp-oktopk-baseline-gwarmup/.../x3207c0s13b0n0-0.log")
        sys.exit(1)
    
    analyze_log(sys.argv[1])
