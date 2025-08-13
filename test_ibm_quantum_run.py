import os
import sys
import traceback


def simple_oracle():
    # Creates a 1-qubit circuit and returns Pauli Z observable
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    qc = QuantumCircuit(1)
    qc.ry(1.5708, 0)  # pi/2 rotation
    observable = SparsePauliOp.from_list([('Z', 1)])
    return qc, observable


def main():
    print("Starting IBM Quantum AE test...")
    if 'IBM_QUANTUM_TOKEN' not in os.environ:
        print("ERROR: IBM_QUANTUM_TOKEN environment variable not set.")
        sys.exit(1)

    try:
        from quantum.providers import IBMProvider
        provider = IBMProvider()
        print(f"Using backend: {provider.backend.name}")
        results = provider.run_ae([simple_oracle], shots=1024, epsilon=0.05, mode='iterative')
        print("\nTest result:")
        print(f"Estimated amplitude (should be close to 0.5): {results[0]}")
        # Simple harness: check if result is within reasonable bound
        if 0.4 <= results[0] <= 0.6:
            print("Test PASSED: Quantum amplitude estimation within expected range.")
        else:
            print("Test FAILED: Result {results[0]} out of expected range [0.4, 0.6].")
    except Exception as e:
        print("\nEncountered an error during IBM Quantum run:")
        print(e)
        print(traceback.format_exc())
        sys.exit(2)


if __name__ == "__main__":
    main()
