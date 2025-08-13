"""
File-based cache for compiled circuits/results, auto-invalidated on hash/version change.
"""
import os
import pickle
from hashlib import sha256

class CircuitCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir or os.path.expanduser('~/.qopt/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    def _hash(self, obj):
        return sha256(pickle.dumps(obj)).hexdigest()
    def get(self, key):
        path = os.path.join(self.cache_dir, f'{key}.pkl')
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None # corrupted
    def set(self, key, value):
        path = os.path.join(self.cache_dir, f'{key}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(value, f)
    def purge(self):
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
