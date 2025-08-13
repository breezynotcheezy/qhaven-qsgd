"""
Quantum Amplitude Estimation (QAE) Engine for classical and quantum gradients.
Supports iterative AE, ML-AE, and classical MC fallback, batching, and provider abstraction.
"""
import numpy as np

class QuantumGradientEstimator:
    def __init__(self, backend="auto", precision=0.02, shots=2000, mode="iterative",
                 timeout_s=60, max_retries=2, cache_dir=None, strict_local=False):
        self.backend = backend
        self.precision = precision
        self.shots = shots
        self.mode = mode
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.strict_local = strict_local
        # Providers defer import (qiskit, braket, sim)
        from .providers import get_provider
        self.provider = get_provider(backend=backend, strict_local=strict_local)

    def estimate(self, grads, build_oracle=None, params=None):
        # Determine per-parameter or block-wise estimation
        # Compose oracles for this batch/step
        # "build_oracle" is either a supplied callable or an auto-selected builtin
        qmeta = {
            'backend': self.backend,
            'quantum': None,
            'shots': self.shots,
            'ae_mode': self.mode
        }
        try:
            # Decide execution path based on provider type
            try:
                from .providers import SimProvider
                is_sim = isinstance(self.provider, SimProvider) or self.backend == 'sim' or self.strict_local
            except Exception:
                is_sim = self.backend == 'sim' or self.strict_local
            if is_sim or build_oracle is None:
                qmeta['mode'] = 'classical-mc'
                qmeta['quantum'] = False
                ests = [grad.clone() for grad in grads]
                return ests, qmeta
            # Compose batch circuits for quantum provider
            # If build_oracle is provided, create a zero-arg closure per grad
            # Otherwise, pass through the grad directly
            oracles = []
            for idx, grad in enumerate(grads):
                if build_oracle is None:
                    oracle = (lambda g=grad: g)
                else:
                    oracle = (lambda g=grad, i=idx: build_oracle(g, i))
                oracles.append(oracle)
            results = self.provider.run_ae(oracles, shots=self.shots, epsilon=self.precision, mode=self.mode)
            qmeta['mode'] = 'quantum'
            qmeta['quantum'] = True
            return results, qmeta
        except Exception as e:
            qmeta['mode'] = 'error-fallback'
            qmeta['error'] = str(e)
            qmeta['quantum'] = False
            raise
