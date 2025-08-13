# QSGD: Quantum-Inspired Stochastic Gradient Descent

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

## Setting Up Your IBM Quantum API Credentials

To run real quantum jobs, create an IBM Quantum Platform/Cloud account and set up Qiskit Runtime authentication:

1. [Sign up or log in to IBM Quantum Platform](https://quantum-computing.ibm.com/)
2. Create a Quantum Service instance on IBM Cloud.
3. Copy your API key ("IBM Cloud API Key") and your instance CRN (Cloud Resource Name).
4. In your shell, set these environment variables (replace with your values):

```shell
set QISKIT_IBM_TOKEN=YOUR_IBM_CLOUD_API_KEY
set QISKIT_IBM_INSTANCE=YOUR_INSTANCE_CRN
set QISKIT_IBM_CHANNEL=ibm_quantum_platform
```
(Mac/Linux users: use `export` instead of `set`)

**Note:** The CRN looks like: 
```
crn:v1:bluemix:public:quantum-computing:us-east:a/xxxxxx::instance:yyyyyyyyy
```

You must run these commands in every new terminal session or add them to your shell profile for persistence.

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
