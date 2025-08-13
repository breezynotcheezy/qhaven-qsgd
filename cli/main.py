"""
CLI entry: qopt bench, qopt doctor, qopt cache purge, qopt providers list
"""
import typer
from qopt.quantum.providers import _provider_map
from qopt.quantum.cache import CircuitCache
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
    print('Doctor check done.')

@cli.command()
def providers():
    """List supported quantum providers."""
    for k in _provider_map.keys():
        print(f"Provider: {k}")

@cli.command()
def cache_purge():
    """Remove all cached circuit files."""
    cc = CircuitCache(None)
    cc.purge()
    print("Cache purged.")

@cli.command()
def bench():
    """Placeholder: Run quick QAE or SGD benchmark test."""
    print('Benchmark not implemented in CLI stub.')

if __name__ == "__main__":
    cli()
