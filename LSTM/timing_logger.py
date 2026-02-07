#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread-safe timing logger for communication and computation measurements.
Buffers timing records in a queue and flushes to disk periodically.
Designed to be independent from the main training logger.
"""

import os
import time
import threading
import queue
from collections import deque
from datetime import datetime

class TimingLogger:
    """
    Thread-safe timing logger that captures MPI and computation timings.
    
    Features:
    - Separate file per rank (timing-rank{i}.csv)
    - Buffered queue to avoid I/O blocking
    - Background flush thread
    - Precision timing with perf_counter()
    - Easy record insertion with dict-like interface
    """
    
    def __init__(self, rank, log_dir='./logs'):
        """
        Initialize the TimingLogger.
        
        Args:
            rank: MPI rank for file naming
            log_dir: Directory to write timing files
        """
        self.rank = rank
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Timing file path (rank-specific)
        self.timing_file = os.path.join(log_dir, f'timing-rank{rank}.csv')
        
        # Queue for buffering records (thread-safe)
        self.record_queue = queue.Queue(maxsize=10000)
        
        # CSV headers
        self.headers = [
            'timestamp',           # Wall-clock time
            'iteration',           # Training iteration
            'epoch',              # Training epoch
            'layer_name',         # Parameter group name
            'operation',          # MPI operation type (ALLREDUCE, ALLGATHER, etc)
            'phase',              # dense or sparse
            'message_size_bytes', # Size of data moved
            'elapsed_ms',         # Elapsed time in milliseconds
            'start_counter',      # perf_counter at start
            'end_counter',        # perf_counter at end
            'thread_name',        # Thread that recorded this (main, allreduce, etc)
        ]
        
        # Write CSV header
        self._write_header()
        
        # Background flush thread
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()
        
        if rank == 0:
            print(f"[TimingLogger] Initialized on rank {rank}")
            print(f"[TimingLogger] Writing to: {self.timing_file}")
    
    def _write_header(self):
        """Write CSV header to file."""
        try:
            with open(self.timing_file, 'w') as f:
                f.write(','.join(self.headers) + '\n')
        except Exception as e:
            print(f"[TimingLogger-{self.rank}] Error writing header: {e}")
    
    def _format_record(self, record_dict):
        """Format a timing record as CSV line."""
        try:
            values = []
            for header in self.headers:
                val = record_dict.get(header, '')
                # Handle None values
                if val is None:
                    val = ''
                # Ensure proper formatting for floats
                if isinstance(val, float):
                    val = f"{val:.6f}"
                values.append(str(val))
            return ','.join(values)
        except Exception as e:
            print(f"[TimingLogger-{self.rank}] Error formatting record: {e}")
            return None
    
    def _flush_worker(self):
        """Background worker thread that flushes records to disk."""
        buffer = []
        flush_interval = 1.0  # Flush every 1 second
        last_flush = time.time()
        
        while self._running:
            try:
                # Try to get records from queue (with timeout to periodically flush)
                try:
                    record = self.record_queue.get(timeout=0.1)
                    buffer.append(record)
                except queue.Empty:
                    pass
                
                # Flush if buffer is large or timeout elapsed
                current_time = time.time()
                if len(buffer) > 100 or (current_time - last_flush) > flush_interval:
                    if buffer:
                        self._flush_to_file(buffer)
                        buffer = []
                        last_flush = current_time
                        
            except Exception as e:
                print(f"[TimingLogger-{self.rank}] Flush worker error: {e}")
        
        # Final flush on shutdown
        if buffer:
            self._flush_to_file(buffer)
    
    def _flush_to_file(self, records):
        """Write buffered records to CSV file."""
        try:
            with open(self.timing_file, 'a') as f:
                for record in records:
                    if record:
                        f.write(record + '\n')
        except Exception as e:
            print(f"[TimingLogger-{self.rank}] Error flushing to file: {e}")
    
    def record_timing(self, operation, elapsed_ms, 
                     layer_name=None, message_size=0, 
                     phase='dense', iteration=0, epoch=0,
                     start_counter=None, end_counter=None):
        """
        Record a timing measurement.
        
        Args:
            operation: str - MPI operation type (e.g., 'ALLREDUCE', 'ALLGATHER', 'SEND', 'RECV')
            elapsed_ms: float - Elapsed time in milliseconds
            layer_name: str - Name of the parameter group (optional)
            message_size: int - Size of data in bytes (optional)
            phase: str - 'dense' or 'sparse'
            iteration: int - Current training iteration
            epoch: int - Current training epoch
            start_counter: float - perf_counter() value at start (optional)
            end_counter: float - perf_counter() value at end (optional)
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'epoch': epoch,
            'layer_name': layer_name or 'N/A',
            'operation': operation,
            'phase': phase,
            'message_size_bytes': message_size,
            'elapsed_ms': elapsed_ms,
            'start_counter': start_counter or 0.0,
            'end_counter': end_counter or 0.0,
            'thread_name': threading.current_thread().name,  # Add thread name
        }
        
        # Format and queue
        formatted = self._format_record(record)
        if formatted:
            try:
                self.record_queue.put_nowait(formatted)
            except queue.Full:
                print(f"[TimingLogger-{self.rank}] Queue full, dropping record")
    
    def record_mpi_operation(self, mpi_op_name, message_size, 
                            layer_name=None, phase='dense',
                            iteration=0, epoch=0):
        """
        Convenience method to time an MPI operation using context manager pattern.
        
        Usage:
            timer = timing_logger.record_mpi_operation('ALLREDUCE', message_size, 'layer1')
            start_time, start_counter = timer.__enter__()
            try:
                comm.Allreduce(...)
            finally:
                timer.__exit__()
        """
        return MpiOperationTimer(self, mpi_op_name, message_size, 
                                 layer_name, phase, iteration, epoch)
    
    def shutdown(self):
        """Gracefully shutdown the logger."""
        self._running = False
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5)


class MpiOperationTimer:
    """
    Context manager for timing MPI operations with automatic record insertion.
    
    Usage:
        with timing_logger.record_mpi_operation('ALLREDUCE', size, 'layer1') as timer:
            comm.Allreduce(...)
    """
    
    def __init__(self, logger, operation, message_size, 
                 layer_name=None, phase='dense', iteration=0, epoch=0):
        self.logger = logger
        self.operation = operation
        self.message_size = message_size
        self.layer_name = layer_name
        self.phase = phase
        self.iteration = iteration
        self.epoch = epoch
        self.start_time = None
        self.start_counter = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.start_counter = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        end_time = time.time()
        end_counter = time.perf_counter()
        
        elapsed_ms = (end_time - self.start_time) * 1000.0
        
        self.logger.record_timing(
            operation=self.operation,
            elapsed_ms=elapsed_ms,
            layer_name=self.layer_name,
            message_size=self.message_size,
            phase=self.phase,
            iteration=self.iteration,
            epoch=self.epoch,
            start_counter=self.start_counter,
            end_counter=end_counter
        )
        
        return False  # Don't suppress exceptions


# Global timing logger instance (initialized in main_trainer.py)
_global_timing_logger = None

def get_timing_logger():
    """Get the global timing logger instance."""
    return _global_timing_logger

def init_timing_logger(rank, log_dir='./logs'):
    """Initialize the global timing logger."""
    global _global_timing_logger
    _global_timing_logger = TimingLogger(rank, log_dir)
    return _global_timing_logger

def shutdown_timing_logger():
    """Shutdown the global timing logger."""
    global _global_timing_logger
    if _global_timing_logger:
        _global_timing_logger.shutdown()
