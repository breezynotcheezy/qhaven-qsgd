import os
import sys
import time
import traceback
import math
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime import QiskitRuntimeService

# Global runtime options
SHOTS = 1024
EPSILON = 0.05

# Color output for terminal
FAIL = '\033[91m'
OK = '\033[92m'
WARN = '\033[93m'
ENDC = '\033[0m'
BOLD = '\033[1m'

# Standard exit error codes
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def info(msg):
    print(f"{BOLD}[info]{ENDC} {msg}")

def warn(msg):
    print(f"{WARN}{msg}{ENDC}")

def err(msg):
    print(f"{FAIL}{msg}{ENDC}")

def ok(msg):
    print(f"{OK}{msg}{ENDC}")


def show_backend_info(backend):
    print("\n--- Backend Information ---")
    print(f"Name: {backend.name}")
    print(f"Qubits: {backend.configuration().num_qubits}")
    print(f"Basis Gates: {backend.configuration().basis_gates}")
    print(f"Max shots: {backend.configuration().max_shots}")
    print(f"Status: {'operational' if backend.status().operational else 'not operational'}")
    print(f"Pending jobs: {backend.status().pending_jobs}")
    print("--------------------------\n")


def quantum_statevector_expectation(axis: str) -> float:
    # Simulate theoretical result for test circuit
    qc = QuantumCircuit(1)
    if axis == 'Z':
        pass
    elif axis == 'X':
        qc.h(0)
    elif axis == 'Y':
        qc.sdg(0)
        qc.h(0)
    sv = Statevector.from_instruction(qc)
    if axis == 'Z':
        op = SparsePauliOp.from_list([('Z', 1)])
    elif axis == 'X':
        op = SparsePauliOp.from_list([('X', 1)])
    elif axis == 'Y':
        op = SparsePauliOp.from_list([('Y', 1)])
    return np.real(sv.expectation_value(op))


def build_oracle(axis: str, num_qubits: int) -> Tuple[QuantumCircuit, SparsePauliOp]:
    # Prepares qubit 0 along a specific axis and matches observable dimension to hardware
    qc = QuantumCircuit(num_qubits)
    if axis == 'X':
        qc.h(0)
        label = 'X' + 'I' * (num_qubits - 1)
    elif axis == 'Y':
        # Prepare |+i> = (|0> + i|1>)/sqrt(2)
        qc.sdg(0)
        qc.h(0)
        label = 'Y' + 'I' * (num_qubits - 1)
    else:
        qc.id(0)
        label = 'Z' + 'I' * (num_qubits - 1)
    observable = SparsePauliOp.from_list([(label, 1)])
    return qc, observable


def build_parametric_oracle(num_qubits: int, angle: float) -> Tuple[QuantumCircuit, SparsePauliOp]:
    qc = QuantumCircuit(num_qubits)
    qc.ry(angle, 0)
    label = 'Z' + 'I' * (num_qubits - 1)
    observable = SparsePauliOp.from_list([(label, 1)])
    return qc, observable


def assert_close(val, ref, atol=0.25):
    if abs(val - ref) > atol:
        raise AssertionError(f"Result {val} not within tolerance {atol} of expected {ref}")

def run_single_expectation(axis: str, backend, shots=SHOTS, epsilon=EPSILON):
    """
    Run an expectation on |0>, |+>, or |+i> on qubit 0 with Z/X/Y observable, as hardware allows.
    """
    num_qubits = backend.configuration().num_qubits
    qc, observable = build_oracle(axis, num_qubits)
    qc = transpile(qc, backend=backend)
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    estimator = Estimator(backend)
    estimator.options.default_shots = shots
    start = time.time()
    job = estimator.run([(qc, observable, [])])
    pub_result = job.result()[0]
    result = pub_result.data.evs
    elapsed = time.time() - start
    return result, elapsed


def run_parametric_expectation(backend, angles: List[float]):
    """
    Run a batch of parametric circuits (RY by angle) and measure <Z> for each.
    """
    num_qubits = backend.configuration().num_qubits
    instances = []
    for angle in angles:
        qc, obs = build_parametric_oracle(num_qubits, angle)
        qc = transpile(qc, backend=backend)
        instances.append((qc, obs, []))
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    estimator = Estimator(backend)
    estimator.options.default_shots = SHOTS
    job = estimator.run(instances)
    pub_results = job.result()
    return [r.data.evs for r in pub_results]


def run_error_case_mismatch(backend):
    # This produces a circuit/observable size mismatch
    qc = QuantumCircuit(2)
    qc.h(0)
    observable = SparsePauliOp.from_list([('Z', 1)])  # Single-qubit obs on 2-qubit circuit
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    estimator = Estimator(backend)
    estimator.options.default_shots = SHOTS
    try:
        estimator.run([(qc, observable, [])])
    except Exception as e:
        ok(f"Properly raised error for circuit/observable size mismatch: {e}")
    else:
        raise RuntimeError("Expected error for size mismatch but did not get one")

def list_backends(service):
    print("\nAvailable IBM Quantum Backends:")
    for backend in service.backends():
        print(f"- {backend.name} ({backend.configuration().num_qubits} qubits)")
    print()

def main():
    print("Starting EXTENSIVE IBM Quantum AE/expectation tests...")
    try:
        service = QiskitRuntimeService()
    except Exception as e:
        err(f"Unable to load QiskitRuntimeService: {e}")
        print(traceback.format_exc())
        sys.exit(EXIT_ERROR)
    backends = service.backends(operational=True, simulator=False)
    if not backends:
        err("No available IBM Quantum devices found! Exiting.")
        sys.exit(EXIT_ERROR)
    backend = service.least_busy(operational=True, simulator=False)
    show_backend_info(backend)

    # List all backends for reference
    list_backends(service)

    # Single-qubit expectations (Z/X/Y axes)
    axis_tests = [('Z', 1.0), ('X', 0.0), ('Y', 0.0)]
    for axis, theory in axis_tests:
        try:
            info(f"Testing <{axis}> on backend {backend.name}")
            result, elapsed = run_single_expectation(axis, backend)
            ok(f"<Expected {axis}({theory})> Got: {result:.4f} in {elapsed:.2f}s")
            # Test within looser tolerance due to hardware errors
            assert_close(result, theory)
        except Exception as ex:
            err(f"FAILED for axis {axis}: {ex}")

    # Parametric expectation sweep: Expectation <Z|RY(theta)|0> = cos(theta)
    info("Performing parametric RY sweep with Estimator...")
    thetas = np.linspace(0, 2*math.pi, 7)
    theory = np.cos(thetas)
    try:
        yvals = run_parametric_expectation(backend, thetas)
        ok("RY parametric sweep results:")
        for idx, (theta, result, expected) in enumerate(zip(thetas, yvals, theory)):
            print(f"  theta={theta:.3f} | Hardware={result:.3f} | Theory={expected:.3f}")
            assert_close(result, expected, atol=0.40)  # Looser for hardware noise
    except Exception as ex:
        err(f"RY parametric sweep FAILED: {ex}")

    # Error case: size mismatch
    info("Testing size mismatch error case...")
    run_error_case_mismatch(backend)

    # Multi-qubit expectations: GHZ and Bell states
    info("Testing multi-qubit GHZ state <ZZ>")
    nq = min(backend.configuration().num_qubits, 2)
    qc = QuantumCircuit(nq)
    qc.h(0)
    qc.cx(0, 1)
    label = 'Z' * nq
    observable = SparsePauliOp.from_list([(label, 1)])
    qc = transpile(qc, backend=backend)
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    estimator = Estimator(backend)
    estimator.options.default_shots = SHOTS
    job = estimator.run([(qc, observable, [])])
    result = job.result()[0].data.evs
    try:
        ok(f"<ZZ> on GHZ | Hardware={result:.3f}")
        # Expectation is ideally 1
        assert_close(result, 1, atol=0.40)
    except Exception as ex:
        err(f"FAILED GHZ <ZZ>: {ex}")

    # Test batch runs (running several different circuits at once)
    info("Testing batch expectation interface...")
    circuits = []
    obs = []
    for i in range(3):
        qc = QuantumCircuit(backend.configuration().num_qubits)
        qc.h(0)
        for j in range(1, i + 1):
            qc.cx(0, j)
        circuits.append(transpile(qc, backend=backend))
        obs_label = 'Z' + 'I' * (backend.configuration().num_qubits - 1)
        obs.append(SparsePauliOp.from_list([(obs_label, 1)]))
    instances = [(c, o, []) for c, o in zip(circuits, obs)]
    estimator = Estimator(backend)
    estimator.options.default_shots = SHOTS
    job = estimator.run(instances)
    batch_results = [r.data.evs for r in job.result()]
    for idx, r in enumerate(batch_results):
        print(f"Batch {idx} <Z...>: {r:.3f}")

    # Timings and summary
    print('\nAll core functional tests completed.  If most tests pass, library is functional.\n')

if __name__ == "__main__":
    try:
        main()
    except AssertionError as ae:
        err(f"FAILED: {ae}")
        sys.exit(EXIT_FAIL)
    except Exception as ex:
        err(f"Test run failed with error: {ex}")
        print(traceback.format_exc())
        sys.exit(EXIT_ERROR)
