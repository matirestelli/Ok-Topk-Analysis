#!/usr/bin/env python3
"""
Analyze MPI communication logs to understand message size distribution per collective type
Shows: "X% of ALLREDUCE_DENSE messages are 110.28MB, Y% of ALLGATHERV_FINAL are 1.48MB", etc.
"""

import os
import re
import glob
from collections import defaultdict
from pathlib import Path

def parse_log_file(filepath):
    """Parse MPI logging entries from a log file"""
    mpi_entries = []
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[.*?\] INFO ([\w_]+)\|(.+)'
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    timestamp, collective_type, metadata = match.groups()
                    
                    # Parse size_mb from metadata
                    size_mb = None
                    for item in metadata.split('|'):
                        if item.startswith('size_mb='):
                            try:
                                size_mb = float(item.split('=')[1])
                            except ValueError:
                                pass
                    
                    if size_mb is not None:
                        mpi_entries.append({
                            'collective': collective_type,
                            'size_mb': size_mb
                        })
    except Exception as e:
        pass
    
    return mpi_entries

def analyze_distribution():
    """Analyze message size distribution for each collective type"""
    log_dir = '/home/mrest/Ok-Topk-Analysis/LSTM/logs/allreduce-comp-oktopk-baseline-gwarmup/lstman4-n8-bs8-lr0.0100-ns1-sg2.50-ds0.02'
    
    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')))
    print(f"Found {len(log_files)} log files\n")
    
    # Collect all message sizes by collective type
    mpi_data = defaultdict(list)
    total_entries = 0
    
    for log_file in log_files:
        entries = parse_log_file(log_file)
        for entry in entries:
            collective_type = entry['collective']
            size_mb = entry['size_mb']
            mpi_data[collective_type].append(size_mb)
            total_entries += 1
    
    print(f"Total MPI logging entries: {total_entries}\n")
    print("=" * 100)
    print("MESSAGE SIZE DISTRIBUTION BY COLLECTIVE TYPE")
    print("=" * 100)
    print()
    
    for collective_type in sorted(mpi_data.keys()):
        sizes = mpi_data[collective_type]
        
        # Get unique sizes and their distribution
        unique_sizes = sorted(set(sizes))
        
        print(f"{collective_type}:")
        print(f"  Total messages: {len(sizes)}")
        
        if len(unique_sizes) == 1:
            # All messages same size
            size = unique_sizes[0]
            print(f"  100.0% of messages are {size:.2f} MB")
        else:
            # Multiple sizes - show distribution
            print(f"\n  {'Message Size (MB)':<20} {'Count':<10} {'Percentage':>12}")
            print(f"  {'-'*42}")
            
            for size in unique_sizes:
                count = sizes.count(size)
                percentage = (count / len(sizes)) * 100
                print(f"  {size:>18.2f} {count:>10} {percentage:>11.2f}%")
        
        print()
    
    # Summary comparison
    print("=" * 100)
    print("COMPARISON: MESSAGE SIZE RANGES")
    print("=" * 100)
    print()
    
    print(f"{'Collective Type':<30} {'Size Range (MB)':<25} {'Min':>10} {'Max':>10}")
    print("-" * 75)
    
    for collective_type in sorted(mpi_data.keys()):
        sizes = mpi_data[collective_type]
        min_size = min(sizes)
        max_size = max(sizes)
        
        if min_size == max_size:
            size_range = f"Fixed: {min_size:.2f}"
        else:
            size_range = f"{min_size:.2f} - {max_size:.2f}"
        
        print(f"{collective_type:<30} {size_range:<25} {min_size:>10.2f} {max_size:>10.2f}")
    
    print()
    
    # Sparse phase detection
    print("=" * 100)
    print("SPARSE PHASE DETECTION")
    print("=" * 100)
    print()
    
    dense_count = len(mpi_data.get('ALLREDUCE_DENSE', []))
    alltoall_count = len(mpi_data.get('ALLTOALL', []))
    allgatherv_count = len(mpi_data.get('ALLGATHERV_FINAL', []))
    
    print(f"DENSE_ALLREDUCE operations: {dense_count} (first 128 iterations)")
    print(f"ALLTOALL operations:       {alltoall_count} (sparse phase indicator)")
    print(f"ALLGATHERV_FINAL:         {allgatherv_count}")
    print()
    
    if alltoall_count > 0 or allgatherv_count > 0:
        print("✓ Sparse phase detected! Training reached iteration 128+")
        if allgatherv_count > 0:
            sparse_iters = allgatherv_count // 8  # 8 ranks
            print(f"  Sparse iterations: ~{sparse_iters}")

if __name__ == '__main__':
    analyze_distribution()
