"""CLI-support modules split out of run.py.

run.py stays the public entry point and re-exports every name defined here:
external pipelines (lada-ex / jasna workers) import run.py with a dummy
sys.argv and reuse its functions and globals. Anything added to these modules
that should be reachable as run.<name> must also be re-exported in run.py.
"""
