"""
Config: env, YAML/INI support, runtime switches, QOPT_* resolution.
"""
import os

class Config:
    def __init__(self, path=None):
        self.vars = dict(os.environ)
        if path:
            try:
                import yaml
                with open(path) as f:
                    self.vars.update(yaml.safe_load(f))
            except Exception:
                pass
        self._prefix = "QOPT_"
    def get(self, key, default=None):
        return self.vars.get(self._prefix + key.upper(), default)
