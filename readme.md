# QSGD: Quantum Stochastic Gradient Descent

---

## Overview

QSGD provides quantum-enabled and quantum-inspired stochastic gradient descent algorithms for machine learning optimization. Using the [Qiskit IBM Runtime](https://github.com/Qiskit/qiskit-ibm-runtime), QSGD supports real workloads on IBM Quantum cloud hardware—or can run fully classical if you choose. The framework is designed for easy integration with existing Python ML workflows.

---

## Features
- Run real hybrid quantum-classical optimization on IBM Quantum hardware (no simulation by default)
- Modular oracles, accepting Qiskit QuantumCircuit & observable objects
- Command-line and library API
- Logging, configuration, and extensibility as standard Python modules

---

## Project Structure

```
qhaven-qsgd/
├── cli/                # Command-line interface
├── logging.py          # Logging utilities
├── optim/              # Optimizers
├── oracles/            # Oracle implementations
├── quantum/            # Quantum device integrations (via Qiskit Runtime)
├── runtime/            # Orchestration logic
└── utils.py            # Utility functions
```

---

## Quickstart

### Installation

```shell
git clone https://github.com/YOUR_USERNAME/qsgd.git
cd qsgd
python -m venv .qsgd-venv
.qsgd-venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install qiskit qiskit-ibm-runtime qiskit-aer
pip install -e .
```

---

## Enabling Full IBM Quantum Functionality with Qiskit

To use real IBM Quantum computers (not simulators) with QSGD, follow these steps:

### 1. Set Up Your IBM Quantum Account and Credentials

- Create an account or log in at: https://quantum-computing.ibm.com/
- Create a Quantum Service instance on IBM Cloud (this will give you access to backends and generate a **CRN**).
- In IBM Cloud, go to your Service Instance > Service Credentials and **create an API key**.

### 2. Gather Your Authentication Values
You will need:
- **QISKIT_IBM_TOKEN**  
  Your IBM Cloud API key (string of letters/numbers).
- **QISKIT_IBM_INSTANCE**  
  Your Instance CRN (Cloud Resource Name), looks like  
  `crn:v1:bluemix:public:quantum-computing:us-east:a/xxxxxx::instance:yyyyyyyyy`
- **QISKIT_IBM_CHANNEL**  
  Always set to `ibm_quantum_platform` for modern use.

### 3. Set Your Environment Variables

#### On Windows (Command Prompt):
```shell
set QISKIT_IBM_TOKEN=YOUR_API_KEY
set QISKIT_IBM_INSTANCE=YOUR_INSTANCE_CRN
set QISKIT_IBM_CHANNEL=ibm_quantum_platform
```
#### On PowerShell:
```shell
$env:QISKIT_IBM_TOKEN="YOUR_API_KEY"
$env:QISKIT_IBM_INSTANCE="YOUR_INSTANCE_CRN"
$env:QISKIT_IBM_CHANNEL="ibm_quantum_platform"
```
#### On macOS/Linux:
```shell
export QISKIT_IBM_TOKEN=YOUR_API_KEY
export QISKIT_IBM_INSTANCE=YOUR_INSTANCE_CRN
export QISKIT_IBM_CHANNEL=ibm_quantum_platform
```
> These must be set in ANY terminal you use to run quantum experiments with QSGD.

---

### 4. Activate Your Virtual Environment

```shell
.qsgd-venv\Scripts\activate      # Windows
source .qsgd-venv/bin/activate    # macOS/Linux
```

---

### 5. Verify Quantum Integration

Run:

```shell
python test_ibm_quantum_run.py
```

You should see output indicating:
- The live IBM Quantum backend in use
- Your quantum circuit expectation or amplitude value
- PASS/FAIL result (if the key and instance are valid and you have backend access)

---

### 6. Use QSGD with IBM Quantum in Your Code

When configuring QSGD’s optimizer or estimator, set `backend='ibm'` to enable live quantum hardware. Your oracles must return a tuple as:

```python
QuantumCircuit, SparsePauliOp
```

Example minimal oracle:
```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

def example_oracle():
    qc = QuantumCircuit(1)
    qc.h(0)
    observable = SparsePauliOp.from_list([('Z', 1)])
    return qc, observable
```

---
**Trouble?**
- Double-check that all environment variables are set in the active terminal.
- You must have access to at least one real backend for your IBM Cloud instance.
- For advanced errors, see [Qiskit IBM Runtime Docs](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime).

---

---

## Running Quantum Hardware Tests

To verify your credentials and quantum integration, activate your virtual environment, set your environment variables, and run:

```shell
python test_ibm_quantum_run.py
```

You should see:
- The detected IBM Quantum backend name
- The estimated result of a quantum test circuit (amplitude or expectation value)
- PASS/FAIL output for result sanity

If you get authentication or device errors, check your API key, instance CRN, and user rights on the IBM Quantum dashboard.

---

## Typical QSGD Training Workflow
1. Define/train your model as usual (e.g. PyTorch)
2. Select an optimizer (pass `backend="ibm"` for true quantum; `"sim"` for classical)
3. Oracles must return `(QuantumCircuit, SparsePauliOp)` (see oracles or test for examples)
4. QSGD submits jobs to the IBMQ cloud and returns estimated values

---

## Advanced Usage
- Tweak all hyperparameters in `config.py`
- Write custom oracles in `oracles/builtins.py`, returning `(circuit, observable)`
- For backend troubleshooting or advanced Qiskit use, see [Qiskit IBM Runtime Docs](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime)

---

## Extending QSGD
- Implement new optimizers: `optim/`
- Register new routines in `cli/main.py`
- Add advanced oracles or runtime logic as you wish

---

## FAQ
**Q:** Do I need a quantum computer?  
**A:** Yes, QSGD in quantum mode submits jobs to live IBM Quantum hardware using Qiskit Runtime (account required).

**Q:** What do my oracles return?  
**A:** Qiskit `QuantumCircuit` and `SparsePauliOp` observable as a tuple: `(circuit, observable)`.

**Q:** What if IBM Quantum devices are all busy?  
**A:** Jobs are queued. Free accounts have queue limits and wait times; paid/prioritized accounts are faster.

---

## License
MIT © 2025 Your Name

---

## Citation & Acknowledgements
Built on Qiskit IBM Runtime. Cite as appropriate (bibtex available soon).

---
