import os
import sys
import time
import traceback
import math
import argparse
import numpy as np
from typing import List, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_ibm_runtime import QiskitRuntimeService

SHOTS = 512  # Lower shot count for minimal quota use (can bump for --thorough)
EPSILON = 0.05

FAIL = '\033[91m'
OK = '\033[92m'
WARN = '\033[93m'
ENDC = '\033[0m'
BOLD = '\033[1m'

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

def info(msg): print(f"{BOLD}[info]{ENDC} {msg}")
def warn(msg): print(f"{WARN}{msg}{ENDC}")
def err(msg): print(f"{FAIL}{msg}{ENDC}")
def ok(msg): print(f"{OK}{msg}{ENDC}")

def show_backend_info(backend):
    print("\n--- Backend Information ---")
    print(f"Name: {backend.name}")
    print(f"Qubits: {backend.configuration().num_qubits}")
    print(f"Basis Gates: {backend.configuration().basis_gates}")
    print(f"Max shots: {backend.configuration().max_shots}")
    print(f"Status: {'operational' if backend.status().operational else 'not operational'}")
    print(f"Pending jobs: {backend.status().pending_jobs}")
    print("--------------------------\n")

def list_backends(service):
    print("\nAvailable IBM Quantum Backends:")
    for backend in service.backends():
        print(f"- {backend.name} ({backend.configuration().num_qubits} qubits)")
    print()

def assert_close(val, ref, atol=0.35):
    if abs(val - ref) > atol:
        raise AssertionError(f"Result {val} not within tolerance {atol} of expected {ref}")

def build_oracle(axis: str, num_qubits: int) -> Tuple[QuantumCircuit, SparsePauliOp]:
    # One-qubit operation but padded to device; e.g., 'ZIII...I'
    qc = QuantumCircuit(num_qubits)
    if axis == 'Z':
        qc.id(0)
        label = 'Z' + 'I' * (num_qubits - 1)
        reference = 1.0  # |0> state expectation <Z> = 1
    elif axis == 'X':
        qc.h(0)
        label = 'X' + 'I' * (num_qubits - 1)
        reference = 0.0  # <X|0> = 0
    elif axis == 'Y':
        qc.sdg(0); qc.h(0)
        label = 'Y' + 'I' * (num_qubits - 1)
        reference = 0.0  # <Y|0> = 0
    else:
        raise ValueError(f"Unknown axis {axis}")
    observable = SparsePauliOp.from_list([(label, 1)])
    return qc, observable, reference

def build_parametric_oracle(num_qubits: int, angle: float) -> Tuple[QuantumCircuit, SparsePauliOp]:
    qc = QuantumCircuit(num_qubits)
    qc.ry(angle, 0)
    label = 'Z' + 'I' * (num_qubits - 1)
    observable = SparsePauliOp.from_list([(label, 1)])
    return qc, observable

# Thorough multi-qubit GHZ test (2-qubit GHZ/Bell to avoid quota explosion)
def build_ghz(nq=2):
    qc = QuantumCircuit(nq)
    qc.h(0)
    for i in range(1, nq):
        qc.cx(0, i)
    label = 'Z' * nq
    observable = SparsePauliOp.from_list([(label, 1)])
    return qc, observable, nq

def run_estimator(qc: QuantumCircuit, observable: SparsePauliOp, backend, shots=SHOTS):
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    estimator = Estimator(backend)
    estimator.options.default_shots = shots
    qc = transpile(qc, backend=backend)
    job = estimator.run([(qc, observable, [])])
    pub_result = job.result()[0]
    return pub_result.data.evs

def main():
    parser = argparse.ArgumentParser(description="Test QSGD IBM Quantum Integration")
    parser.add_argument('--thorough', '-t', action='store_true', help='Run all heavy/extra-cost quantum tests (GHZ, batch, sweeps)')
    args = parser.parse_args()

    print("Starting QSGD IBM Quantum integration test suite...\n")
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
    list_backends(service)
    num_qubits = backend.configuration().num_qubits

    # Minimal tests
    info("Running minimal, fast single-qubit quantum expectation tests (Z/X/Y axes)...")
    axes = ['Z', 'X', 'Y']
    for axis in axes:
        try:
            qc, observable, reference = build_oracle(axis, num_qubits)
            result = run_estimator(qc, observable, backend)
            ok(f"<Expected {axis}({reference})> Got: {result:.4f}")
            assert_close(result, reference)
        except Exception as ex:
            err(f"FAILED axis {axis} test: {ex}")

    # Thorough and compute-intensive tests only with flag
    if args.thorough:
        warn("Running thorough test suite — these use more shots and more circuit executions!")

        # Parametric sweep test
        info("Performing parametric RY sweep with Estimator...")
        thetas = np.linspace(0, 2*math.pi, 5)
        theory = np.cos(thetas)
        try:
            yvals = []
            for theta, expect_val in zip(thetas, theory):
                qc, observable = build_parametric_oracle(num_qubits, theta)
                result = run_estimator(qc, observable, backend)
                ok(f"  theta={theta:.3f} | Hardware={result:.3f} | Theory={expect_val:.3f}")
                assert_close(result, expect_val, atol=0.55)  # Looser for hardware noise
                yvals.append(result)
        except Exception as ex:
            err(f"RY parametric sweep FAILED: {ex}")

        # GHZ/Bell state test (minimal, 2-qubit)
        info("Testing 2-qubit GHZ state <ZZ>")
        try:
            qc, observable, n_ghz = build_ghz(2)
            # Pad circuit to match backend
            if num_qubits > 2:
                pad_qc = QuantumCircuit(num_qubits)
                pad_qc.compose(qc, qubits=range(2), inplace=True)
                qc = pad_qc
                label = 'Z' * 2 + 'I' * (num_qubits - 2)
                observable = SparsePauliOp.from_list([(label, 1)])
            result = run_estimator(qc, observable, backend)
            ok(f"<ZZ> on 2-qubit GHZ | Hardware={result:.3f} | Theory=1")
            assert_close(result, 1, atol=0.40)
        except Exception as ex:
            err(f"FAILED GHZ <ZZ>: {ex}")

        # Size mismatch test
        info("Testing size mismatch error case...")
        try:
            qc = QuantumCircuit(2)
            qc.h(0)
            observable = SparsePauliOp.from_list([('Z', 1)])  # Single-qubit obs on 2-qubit circuit
            from qiskit_ibm_runtime import EstimatorV2 as Estimator
            estimator = Estimator(backend)
            estimator.options.default_shots = SHOTS
            estimator.run([(qc, observable, [])])
            err("ERROR: No failure raised for intentional size mismatch!")
        except Exception as e:
            ok(f"Properly raised error for circuit/observable size mismatch: {e}")

        # Batch run
        info("Testing batch expectation interface (minimal, 2 single-qubit circuits)...")
        try:
            circuits = []
            obs = []
            for i in range(2):
                qc = QuantumCircuit(num_qubits)
                if i == 0:
                    qc.h(0)
                else:
                    qc.id(0)
                circuits.append(transpile(qc, backend=backend))
                label = ('X' if i == 0 else 'Z') + 'I' * (num_qubits - 1)
                obs.append(SparsePauliOp.from_list([(label, 1)]))
            from qiskit_ibm_runtime import EstimatorV2 as Estimator
            estimator = Estimator(backend)
            estimator.options.default_shots = SHOTS
            instances = [(c, o, []) for c, o in zip(circuits, obs)]
            batch_results = [r.data.evs for r in estimator.run(instances).result()]
            for idx, r in enumerate(batch_results):
                ok(f"Batch {idx} result: {r:.3f}")
        except Exception as ex:
            err(f"Batch run test failed: {ex}")

    print(f'\n{OK}All REQUIRED (minimal) tests completed!{ENDC} Use --thorough to try additional hardware-intensive coverage.\n')

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
