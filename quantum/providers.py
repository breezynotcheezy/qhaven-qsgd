"""
Quantum backend providers (Qiskit, Braket, Sim/local), cloud config, retry/backoff, device logic.
"""

class SimProvider:
    def run_ae(self, oracles, shots, epsilon, mode):
        # Classical MC estimation of means (simulate quantum query)
        return [oracle() for oracle in oracles]


class BraketProvider:
    def __init__(self):
        self._initialized = False
        # Lazy import braket SDK
    def run_ae(self, oracles, shots, epsilon, mode):
        # Placeholder: batch AE circuits, submit jobs, handle polling
        raise NotImplementedError("AWS Braket provider integration required.")

import os
from typing import Callable, List
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp

class IBMProvider:
    def __init__(self):
        # Expects QISKIT_IBM_TOKEN, QISKIT_IBM_INSTANCE, QISKIT_IBM_CHANNEL in environment
        self._initialized = False
        self.service = None
        self.backend = None
        self._authenticate()

    def _authenticate(self):
        self.service = QiskitRuntimeService()  # auto-discovers env vars
        self.backend = self.service.least_busy(operational=True, simulator=False)
        self._initialized = True

    def run_ae(self, oracles: List[Callable], shots: int, epsilon: float, mode: str):
        # Each oracle must return (QuantumCircuit, observable) for estimation
        results = []
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
    if strict_local or backend == 'sim':
        return SimProvider()
    if backend == 'ibm':
        return IBMProvider()
    if backend == 'braket':
        return BraketProvider()
    raise ValueError(f"Unknown backend {backend}")
