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

class IBMProvider:
    def __init__(self):
        self._initialized = False
        # Placeholder for IBM Qiskit SDK setup
    def run_ae(self, oracles, shots, epsilon, mode):
        # Placeholder: batch AE circuits, submit jobs, handle polling
        raise NotImplementedError("IBM Qiskit provider integration required.")

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
