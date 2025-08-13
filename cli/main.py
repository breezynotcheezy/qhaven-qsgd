"""
CLI entry: qsgd bench, qsgd doctor, qsgd cache purge, qsgd providers list
"""
import os
import typer
from quantum.providers import _provider_map, get_provider
from quantum.cache import CircuitCache
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
def bench():
    """Placeholder: Run quick QAE or SGD benchmark test."""
    print('Benchmark not implemented in CLI stub.')

if __name__ == "__main__":
    cli()
