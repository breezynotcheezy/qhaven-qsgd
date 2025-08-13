"""
Quantum-accelerated SGD with QAE and graceful fallback, torch.optim API-compatible.
"""
import torch
from torch.optim.optimizer import Optimizer

from qopt.quantum.ae import QuantumGradientEstimator
from qopt.runtime.orchestrator import Scheduler
from qopt.logging import Logger
from qopt.oracles import builtins as oracles

class SGD_QAE(Optimizer):
    r"""
    Quantum-Accelerated SGD optimizer (Amplitude Estimation).
    Drop-in replacement for torch.optim.SGD.

    Args:
        params (iterable): model params
        lr (float): learning rate
        momentum (float): SGD momentum
        weight_decay (float): Weight decay
        nesterov (bool): Nesterov momentum
        use_quantum (bool): If True, use QAE else classical mean/MC
        backend (str): Backend string (ibm, braket, sim)
        ae_precision (float): AE error tolerance (epsilon)
        shots (int): shots per AE query
        ae_mode (str): 'iterative' or 'mlae'
        timeout_s (int): QPU timeout (s)
        max_retries (int): # network/device retries
        cache_dir (str): Cache path
        log_dir (str): Logging path
        strict_local (bool): Disallow cloud/
        build_oracle (callable): Custom oracle builder (optional)
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False,
                 use_quantum=True, backend="sim", ae_precision=0.02, shots=2000,
                 ae_mode="iterative", timeout_s=60, max_retries=2, cache_dir=None,
                 log_dir=None, strict_local=False, build_oracle=None, **kwargs):
        # Standard torch param formatting/checks
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

        self.use_quantum = use_quantum
        self.backend = backend
        self.ae_precision = ae_precision
        self.shots = shots
        self.ae_mode = ae_mode
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.log_dir = log_dir
        self.strict_local = strict_local
        self.build_oracle = build_oracle
        self.fallback = False

        # Quantum engine/provider scheduler/logger
        self.quantum_engine = QuantumGradientEstimator(
            backend=backend, precision=ae_precision, shots=shots,
            mode=ae_mode, timeout_s=timeout_s, max_retries=max_retries,
            cache_dir=cache_dir, strict_local=strict_local
        )
        self.scheduler = Scheduler()
        self.logger = Logger(log_dir=log_dir)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        # Gather gradients after regular backward pass
        grads = []
        params_with_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad.detach().clone())

        # Quantum or classical estimate (with fallback)
        try:
            if self.use_quantum and not self.fallback:
                est_grads, qmeta = self.quantum_engine.estimate(
                    grads,
                    build_oracle=self.build_oracle,
                    params=params_with_grad
                )
                self.logger.log_qae(qmeta)
            else:
                est_grads = grads
                qmeta = {'mode': 'classical'}
                self.logger.log_fallback(qmeta)
        except Exception as e:
            # Fallback: log and use classical gradients for this step
            self.fallback = True
            qmeta = {'error': str(e), 'mode': 'classical-fallback'}
            self.logger.log_fallback(qmeta)
            est_grads = grads

        # SGD/Nesterov update (same as torch)
        for group in self.param_groups:
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            lr = group['lr']

            for idx, p in enumerate(params_with_grad):
                grad = est_grads[idx]
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state.setdefault(p, {})
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = grad.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                p.data.add_(grad, alpha=-lr)

        self.logger.log_step({
            'loss': loss.item() if loss is not None else None,
            'fallback': self.fallback,
            'quantum_mode': qmeta.get('mode'),
            'ae_precision': self.ae_precision,
        })
        return loss

# Alias for API convenience
SGD_QAE = SGD_QAE
