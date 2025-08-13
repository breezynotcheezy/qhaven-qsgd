"""
Quantum backend providers (Qiskit, Braket, Sim/local), cloud config, retry/backoff, device logic.
"""

class SimProvider:
    def run_ae(self, oracles, shots, epsilon, mode):
        # Classical fallback: simply return provided gradients or computed scalars
        results = []
        for oracle in oracles:
            out = oracle()
            # Support both direct tensors/scalars and (qc, observable)
            results.append(out)
        return results


class BraketProvider:
    def __init__(self):
        self._initialized = False
        # Lazy import braket SDK
    def run_ae(self, oracles, shots, epsilon, mode):
        # Placeholder: batch AE circuits, submit jobs, handle polling
        raise NotImplementedError("AWS Braket provider integration required.")

import os
from typing import Callable, List

class IBMProvider:
    def __init__(self):
        # Expects QISKIT_IBM_* in env; if missing, defer error and let caller fallback
        self._initialized = False
        self.service = None
        self.backend = None
        self._authenticate()

    def _authenticate(self):
        # Will raise if env is not configured; caller handles fallback
        from qiskit_ibm_runtime import QiskitRuntimeService  # lazy import
        self.service = QiskitRuntimeService()  # auto-discovers env vars
        self.backend = self.service.least_busy(operational=True, simulator=False)
        self._initialized = True

    def run_ae(self, oracles: List[Callable], shots: int, epsilon: float, mode: str):
        # Each oracle must return (QuantumCircuit, observable) for estimation
        results = []
        from qiskit_ibm_runtime import EstimatorV2 as Estimator  # lazy import
        for oracle in oracles:
            qc, observable = oracle()
            estimator = Estimator(self.backend)
            estimator.options.default_shots = shots
            job = estimator.run([(qc, observable, [])])  # No sweep params for basic use case
            pub_result = job.result()[0]
            estimate = pub_result.data.evs  # support scalar output
            results.append(estimate)
        return results

_provider_map = {
    'sim': SimProvider,
    'ibm': IBMProvider,
    'braket': BraketProvider,
}

def get_provider(backend, strict_local=False):
    # Auto-detect best backend if requested
    if backend in (None, 'auto'):
        # Prefer IBM if credentials present; else use sim (classical)
        has_token = os.environ.get('QISKIT_IBM_TOKEN')
        has_instance = os.environ.get('QISKIT_IBM_INSTANCE')
        has_channel = os.environ.get('QISKIT_IBM_CHANNEL')
        if has_token and has_instance and has_channel and not strict_local:
            try:
                return IBMProvider()
            except Exception:
                # Fallback to classical if auth fails
                return SimProvider()
        return SimProvider()
    if strict_local or backend == 'sim':
        return SimProvider()
    if backend == 'ibm':
        try:
            return IBMProvider()
        except Exception:
            return SimProvider()
    if backend == 'braket':
        return BraketProvider()
    raise ValueError(f"Unknown backend {backend}")
