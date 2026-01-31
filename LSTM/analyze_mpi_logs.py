#!/usr/bin/env python3
"""
Analyze MPI communication logs to understand message sizes and patterns
"""

import os
import re
import json
from collections import defaultdict, Counter
from pathlib import Path
import statistics

def parse_log_file(filepath):
    """Parse MPI logging entries from a log file"""
    mpi_entries = []
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[.*?\] INFO ([\w_]+)\|(.+)'
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                timestamp, collective_type, metadata = match.groups()
                # Parse metadata
                meta_dict = {}
                for item in metadata.split('|'):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        try:
                            meta_dict[key] = float(value) if '.' in value else int(value)
                        except ValueError:
                            meta_dict[key] = value
                
                mpi_entries.append({
                    'timestamp': timestamp,
                    'collective': collective_type,
                    'metadata': meta_dict
                })
    
    return mpi_entries

def analyze_logs(log_dir):
    """Analyze all log files in a directory"""
    
    # Collect all MPI entries
    all_entries = defaultdict(list)
    total_entries = 0
    
    log_files = sorted(Path(log_dir).glob('*.log'))
    print(f"Found {len(log_files)} log files\n")
    
    for log_file in log_files:
        entries = parse_log_file(str(log_file))
        total_entries += len(entries)
        for entry in entries:
            all_entries[entry['collective']].append(entry)
    
    print(f"Total MPI logging entries: {total_entries}\n")
    print("=" * 80)
    print("MPI COMMUNICATION ANALYSIS")
    print("=" * 80)
    
    # Analyze each collective type
    results = {}
    for collective, entries in sorted(all_entries.items()):
        if not entries:
            continue
            
        print(f"\n{collective} Operations:")
        print("-" * 80)
        
        # Extract sizes
        sizes = []
        for entry in entries:
            if 'size' in entry['metadata']:
                sizes.append(entry['metadata']['size'])
        
        if sizes:
            total_size = sum(sizes)
            count = len(sizes)
            avg_size = statistics.mean(sizes)
            min_size = min(sizes)
            max_size = max(sizes)
            
            print(f"  Count:       {count:,}")
            print(f"  Total Size:  {total_size:,} bytes ({total_size/1e6:.2f} MB)")
            print(f"  Avg Size:    {avg_size:,.0f} bytes ({avg_size/1e6:.2f} MB)")
            print(f"  Min Size:    {min_size:,} bytes ({min_size/1e6:.2f} MB)")
            print(f"  Max Size:    {max_size:,} bytes ({max_size/1e6:.2f} MB)")
            
            if len(sizes) > 1:
                std_dev = statistics.stdev(sizes)
                print(f"  Std Dev:     {std_dev:,.0f} bytes ({std_dev/1e6:.2f} MB)")
            
            results[collective] = {
                'count': count,
                'total_bytes': total_size,
                'total_mb': total_size / 1e6,
                'avg_bytes': avg_size,
                'min_bytes': min_size,
                'max_bytes': max_size
            }
    
    print("\n" + "=" * 80)
    print("COMMUNICATION BREAKDOWN BY COLLECTIVE TYPE")
    print("=" * 80)
    
    if results:
        total_bytes_all = sum(r['total_bytes'] for r in results.values())
        
        print(f"\n{'Collective Type':<30} {'Count':>10} {'Total MB':>15} {'Percentage':>12}")
        print("-" * 80)
        
        for collective in sorted(results.keys()):
            r = results[collective]
            percentage = (r['total_bytes'] / total_bytes_all * 100) if total_bytes_all > 0 else 0
            print(f"{collective:<30} {r['count']:>10,} {r['total_mb']:>15.2f} {percentage:>11.2f}%")
        
        print("-" * 80)
        print(f"{'TOTAL':<30} {sum(r['count'] for r in results.values()):>10,} {total_bytes_all/1e6:>15.2f} {100.0:>11.2f}%")
    
    # Check for sparse phase transition
    print("\n" + "=" * 80)
    print("SPARSE PHASE DETECTION")
    print("=" * 80)
    
    if all_entries:
        # Count iterations: dense phase ends at iteration 128, sparse begins after
        alltoall_count = len(all_entries.get('ALLTOALL', []))
        allreduce_dense_count = len(all_entries.get('ALLREDUCE_DENSE', []))
        
        print(f"\nDENSE_ALLREDUCE operations: {allreduce_dense_count} (first 128 iterations)")
        print(f"ALLTOALL operations:       {alltoall_count} (sparse phase indicator)")
        
        if alltoall_count > 0:
            print(f"\n✓ Sparse phase detected! Training reached iteration 128+")
            print(f"  Sparse iterations: {alltoall_count}")
        else:
            print(f"\n✗ No sparse phase operations detected - may not have trained long enough")
    
    return results

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = '/home/mrest/Ok-Topk-Analysis/LSTM/logs/allreduce-comp-oktopk-baseline-gwarmup/lstman4-n8-bs8-lr0.0100-ns1-sg2.50-ds0.02'
    
    if os.path.isdir(log_dir):
        analyze_logs(log_dir)
    else:
        print(f"Error: Directory not found: {log_dir}")
        sys.exit(1)
