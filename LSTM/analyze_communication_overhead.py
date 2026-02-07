#!/usr/bin/env python3
"""
Calculate communication time vs total training time from MPI logs
"""

import glob
import os
from datetime import datetime

def analyze_communication_timing():
    """Analyze how much time is spent on MPI communication"""
    log_dir = '/home/mrest/Ok-Topk-Analysis/LSTM/logs/allreduce-comp-oktopk-baseline-gwarmup/lstman4-n8-bs8-lr0.0100-ns1-sg2.50-ds0.02'
    
    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')))
    
    # Get timing from training output
    output_file = '/home/mrest/Ok-Topk-Analysis/LSTM/logs/results_FirstRun/lstm_oktopk_extended_${PBS_JOBID}.out'
    
    # Parse training start/end times
    with open(output_file, 'r') as f:
        content = f.read()
        
    # Extract timestamps
    start_line = [l for l in content.split('\n') if 'Job started at' in l][0]
    end_line = [l for l in content.split('\n') if 'Job finished at' in l][0]
    
    # Parse times
    start_time_str = start_line.split('Job started at: ')[1]
    end_time_str = end_line.split('Job finished at: ')[1]
    
    # Parse datetime (format: "Wed Jan 28 12:35:58 AM UTC 2026")
    start_time = datetime.strptime(start_time_str, '%a %b %d %I:%M:%S %p %Z %Y')
    end_time = datetime.strptime(end_time_str, '%a %b %d %I:%M:%S %p %Z %Y')
    
    total_training_time = (end_time - start_time).total_seconds()
    
    print(f"Training Job Timing")
    print(f"=" * 80)
    print(f"\nStart time: {start_time_str}")
    print(f"End time:   {end_time_str}")
    print(f"\nTotal training time: {total_training_time:.1f} seconds ({total_training_time/60:.2f} minutes)")
    
    # Analyze MPI logs
    all_timestamps = []
    collective_counts = {}
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if ' INFO ' in line and '|' in line:
                    # Extract timestamp and collective type
                    parts = line.split(' [')
                    if len(parts) >= 2:
                        try:
                            timestamp_str = parts[0].strip()
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                            all_timestamps.append(timestamp)
                            
                            # Extract collective type
                            info_part = line.split(' INFO ')[1]
                            collective_type = info_part.split('|')[0]
                            collective_counts[collective_type] = collective_counts.get(collective_type, 0) + 1
                        except:
                            pass
    
    if all_timestamps:
        first_mpi = min(all_timestamps)
        last_mpi = max(all_timestamps)
        mpi_duration = (last_mpi - first_mpi).total_seconds()
        
        print(f"\n" + "=" * 80)
        print(f"MPI Communication Logging Period")
        print(f"=" * 80)
        print(f"\nFirst MPI operation:  {first_mpi.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}")
        print(f"Last MPI operation:   {last_mpi.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}")
        print(f"MPI logging duration: {mpi_duration:.1f} seconds ({mpi_duration/60:.2f} minutes)")
        print(f"Overhead time (setup): {(first_mpi - start_time).total_seconds():.1f} seconds")
        
        print(f"\n" + "=" * 80)
        print(f"Communication Efficiency")
        print(f"=" * 80)
        print(f"\nTotal MPI operations logged: {sum(collective_counts.values())}")
        print(f"\nBreakdown by collective type:")
        for collective, count in sorted(collective_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {collective}: {count}")
        
        # Estimate communication time vs computation time
        # Assumption: Each MPI call takes ~10-50ms depending on size
        # Dense AllReduce (110MB): ~50ms at 2.2 GB/s network
        # Sparse AllGatherV (1-6MB): ~5-10ms
        # P2P (0.3MB): ~2-3ms
        
        print(f"\n" + "=" * 80)
        print(f"Communication Time Estimation")
        print(f"=" * 80)
        
        # Rough estimates based on message sizes and network bandwidth
        # Assuming 200 Gbps interconnect (25 GB/s)
        network_bw_gbps = 25  # GB/s
        
        dense_count = collective_counts.get('ALLREDUCE_DENSE', 0)
        sparse_count = collective_counts.get('ALLGATHERV_FINAL', 0)
        p2p_count = (collective_counts.get('ISEND_INDEX', 0) + 
                     collective_counts.get('ISEND_VALUE', 0) +
                     collective_counts.get('IRECV_INDEX', 0) +
                     collective_counts.get('IRECV_VALUE', 0))
        
        # Message sizes in MB
        dense_size = 110.28
        sparse_size_avg = 1.48  # from previous analysis
        p2p_size_avg = 0.45  # from previous analysis
        
        # Latency overhead (us)
        dense_latency = 50  # microseconds for AllReduce
        sparse_latency = 10  # microseconds for AllGatherV
        p2p_latency = 3    # microseconds for P2P
        
        # Calculate transmission time (bytes / bandwidth)
        dense_tx_time = (dense_count * dense_size) / network_bw_gbps
        sparse_tx_time = (sparse_count * sparse_size_avg) / network_bw_gbps
        p2p_tx_time = (p2p_count * p2p_size_avg) / network_bw_gbps
        
        # Calculate latency time
        dense_lat_time = (dense_count * dense_latency) / 1e6
        sparse_lat_time = (sparse_count * sparse_latency) / 1e6
        p2p_lat_time = (p2p_count * p2p_latency) / 1e6
        
        total_comm_time = (dense_tx_time + sparse_tx_time + p2p_tx_time + 
                          dense_lat_time + sparse_lat_time + p2p_lat_time)
        
        print(f"\nNetwork bandwidth: {network_bw_gbps} GB/s (200 Gbps)")
        print(f"\nDense phase (AllReduce):")
        print(f"  Operations: {dense_count}")
        print(f"  Message size: {dense_size:.2f} MB each")
        print(f"  Transmission time: {dense_tx_time:.2f} seconds")
        print(f"  Latency time: {dense_lat_time:.4f} seconds")
        print(f"  Subtotal: {dense_tx_time + dense_lat_time:.2f} seconds")
        
        print(f"\nSparse phase (AllGatherV):")
        print(f"  Operations: {sparse_count}")
        print(f"  Message size: {sparse_size_avg:.2f} MB average")
        print(f"  Transmission time: {sparse_tx_time:.4f} seconds")
        print(f"  Latency time: {sparse_lat_time:.4f} seconds")
        print(f"  Subtotal: {sparse_tx_time + sparse_lat_time:.4f} seconds")
        
        print(f"\nPoint-to-point (ISEND/IRECV):")
        print(f"  Operations: {p2p_count}")
        print(f"  Message size: {p2p_size_avg:.2f} MB average")
        print(f"  Transmission time: {p2p_tx_time:.4f} seconds")
        print(f"  Latency time: {p2p_lat_time:.4f} seconds")
        print(f"  Subtotal: {p2p_tx_time + p2p_lat_time:.4f} seconds")
        
        print(f"\n" + "-" * 80)
        print(f"TOTAL ESTIMATED COMMUNICATION TIME: {total_comm_time:.2f} seconds")
        print(f"TOTAL TRAINING TIME: {total_training_time:.2f} seconds")
        print(f"\nCommunication overhead: {(total_comm_time/total_training_time)*100:.1f}%")
        print(f"Computation time: {(1 - total_comm_time/total_training_time)*100:.1f}%")
        
        print(f"\n" + "=" * 80)
        print(f"Analysis Summary")
        print(f"=" * 80)
        
        print(f"\nConfiguration:")
        print(f"  Training epochs: 10")
        print(f"  Batch size: 8")
        print(f"  Number of workers: 8")
        print(f"  Compression: Ok-Top-k (2% density)")
        print(f"  Dataset: AN4 (218 MB)")
        
        print(f"\nCommunication Pattern:")
        print(f"  Dense phase: First 128 iterations (AllReduce 110.28 MB each)")
        print(f"  Sparse phase: After iteration 128 (AllGatherV 0.35-5.96 MB)")
        print(f"  Total iterations: ~140+ across 8 ranks")
        
        print(f"\nCommunication Efficiency Gains:")
        compression_ratio = dense_size / sparse_size_avg
        print(f"  Message size reduction: {dense_size:.2f} MB → {sparse_size_avg:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.1f}×")
        print(f"  Sparse phase overhead: {(sparse_tx_time + sparse_lat_time):.4f}s vs")
        print(f"  Dense phase overhead: {(dense_tx_time + dense_lat_time):.2f}s")
        
        if total_comm_time > 0:
            print(f"\nTime allocation in communication:")
            print(f"  Dense phase: {(dense_tx_time + dense_lat_time)/total_comm_time*100:.1f}%")
            print(f"  Sparse phase: {(sparse_tx_time + sparse_lat_time)/total_comm_time*100:.2f}%")
            print(f"  Point-to-point: {(p2p_tx_time + p2p_lat_time)/total_comm_time*100:.2f}%")

if __name__ == '__main__':
    analyze_communication_timing()
