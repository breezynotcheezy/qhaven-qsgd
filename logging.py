"""
Structured logging: stepwise JSONL, TensorBoard, debug/fallbacks, quantum metrics.
"""
import os
import json
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or './runs/qopt'
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.jsonl_path = os.path.join(self.log_dir, "log.jsonl")
    def log_step(self, stats):
        step = getattr(self, 'step', 0)
        stats['step'] = step
        self.writer.add_scalar('loss', stats.get('loss', 0), step)
        self.writer.add_scalar('fallback', int(stats.get('fallback', False)), step)
        if stats.get('ae_precision') is not None:
            self.writer.add_scalar('ae_precision', stats.get('ae_precision'), step)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        self.step = step + 1
    def log_qae(self, qmeta):
        # Log quantum call stats
        pass
    def log_fallback(self, qmeta):
        # Log fallback and error stats
        pass
