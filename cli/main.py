"""
CLI entry: qsgd bench, qsgd doctor, qsgd cache purge, qsgd providers list
"""
import os
import time
from typing import Optional
import typer
from quantum.providers import _provider_map, get_provider
from quantum.cache import CircuitCache
from optim import SGD_QAE
cli = typer.Typer()

@cli.command()
def doctor():
    """Check environment, quantum SDKs, GPU, credentials."""
    try:
        import torch
        print('Torch:', torch.__version__)
    except Exception:
        print('Torch not found')
    try:
        import qiskit
        print('Qiskit OK')
    except Exception:
        print('Qiskit not found')
    try:
        import braket
        print('Braket SDK OK')
    except Exception:
        print('Braket not found')
    # IBM creds status
    print('IBM Quantum credentials:')
    print(' - QISKIT_IBM_TOKEN:', 'set' if os.environ.get('QISKIT_IBM_TOKEN') else 'missing')
    print(' - QISKIT_IBM_INSTANCE:', 'set' if os.environ.get('QISKIT_IBM_INSTANCE') else 'missing')
    print(' - QISKIT_IBM_CHANNEL:', 'set' if os.environ.get('QISKIT_IBM_CHANNEL') else 'missing')
    print('Doctor check done.')

@cli.command()
def providers():
    """List supported quantum providers."""
    for k in _provider_map.keys():
        print(f"Provider: {k}")
    # Show auto-detected selection
    prov = get_provider('auto')
    print('Auto-detected provider:', prov.__class__.__name__)

@cli.command()
def cache_purge():
    """Remove all cached circuit files."""
    cc = CircuitCache(None)
    cc.purge()
    print("Cache purged.")

@cli.command()
def bench(
	epochs: int = typer.Option(50, help="Training epochs for each run"),
	n: int = typer.Option(512, help="Number of samples"),
	d: int = typer.Option(32, help="Feature dimension"),
	lr: float = typer.Option(0.01, help="Learning rate"),
	backend: str = typer.Option("auto", help="Backend for QSGD (auto|ibm|sim)"),
	use_quantum: bool = typer.Option(True, help="Enable quantum path for QSGD if available"),
):
    """Compare Torch SGD vs QSGD (classical or quantum if configured).

    Note: To exercise real quantum, you must set IBM env vars and provide a
    model-specific oracle via the optimizer (not included in this generic bench).
    Otherwise QSGD will run in classical mode by design.
    """
    try:
        import torch
    except Exception as e:
        print('PyTorch is required for benchmarking:', e)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(n, d, device=device)
    true_w = torch.randn(d, 1, device=device)
    y = x @ true_w + 0.1 * torch.randn(n, 1, device=device)
    loss_fn = torch.nn.MSELoss()

    def run_torch_sgd():
        model = torch.nn.Linear(d, 1).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        start = time.perf_counter()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        dur = time.perf_counter() - start
        return float(loss.item()), dur

    def run_qsgd(q_enabled: bool):
        model = torch.nn.Linear(d, 1).to(device)
        opt = SGD_QAE(
            model.parameters(), lr=lr, backend=backend, use_quantum=q_enabled
        )
        start = time.perf_counter()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        dur = time.perf_counter() - start
        return float(loss.item()), dur, bool(getattr(opt, 'fallback', False))

    torch_loss, torch_sec = run_torch_sgd()
    qsgd_class_loss, qsgd_class_sec, qsgd_class_fb = run_qsgd(False)
    qsgd_q_loss, qsgd_q_sec, qsgd_q_fb = run_qsgd(use_quantum)

    print('\nResults:')
    print(f"- Torch SGD      : loss={torch_loss:.6f}  time={torch_sec:.3f}s")
    print(f"- QSGD (classical): loss={qsgd_class_loss:.6f}  time={qsgd_class_sec:.3f}s  fallback={qsgd_class_fb}")
    print(f"- QSGD (requested {backend}): loss={qsgd_q_loss:.6f}  time={qsgd_q_sec:.3f}s  fallback={qsgd_q_fb}")

    # Best time winner
    times = {
        'torch': torch_sec,
        'qsgd_classical': qsgd_class_sec,
        f'qsgd_{backend}': qsgd_q_sec,
    }
    fastest = min(times, key=times.get)
    print(f"\nFastest: {fastest} ({times[fastest]:.3f}s)")

if __name__ == "__main__":
    cli()
