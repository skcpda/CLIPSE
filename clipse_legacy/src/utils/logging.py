"""
CSV logging system for research-grade metrics.
"""
import csv
import os
from typing import Dict, Any, Optional


class CSVLogger:
    """Research-grade CSV logger for training metrics."""
    
    def __init__(self, log_file: str, mode: str = 'w'):
        """
        Args:
            log_file: Path to CSV file
            mode: File mode ('w' for write, 'a' for append)
        """
        self.log_file = log_file
        self.mode = mode
        self.fieldnames = None
        self._ensure_dir()
    
    def _ensure_dir(self):
        """Ensure log directory exists."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def log(self, **kwargs):
        """
        Log metrics to CSV.
        
        Args:
            **kwargs: Metric name -> value pairs
        """
        # Initialize fieldnames on first write
        if self.fieldnames is None:
            self.fieldnames = list(kwargs.keys())
            with open(self.log_file, self.mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        else:
            # Update fieldnames if new fields are added
            new_fields = set(kwargs.keys()) - set(self.fieldnames)
            if new_fields:
                self.fieldnames.extend(sorted(new_fields))
        
        # Write row
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(kwargs)
    
    def log_step(self, epoch: int, step: int, loss: float, temp: float, 
                 lr: float, **stats):
        """Log training step metrics."""
        self.log(
            epoch=epoch,
            step=step,
            loss=loss,
            temp=temp,
            lr=lr,
            **stats
        )
    
    def log_epoch(self, epoch: int, val_r1: Optional[float] = None, 
                  val_r5: Optional[float] = None, val_r10: Optional[float] = None,
                  **metrics):
        """Log epoch-level metrics."""
        self.log(
            epoch=epoch,
            step='epoch',
            val_r1=val_r1,
            val_r5=val_r5,
            val_r10=val_r10,
            **metrics
        )
