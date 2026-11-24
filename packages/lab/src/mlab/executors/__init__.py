"""Hardware executors: where the slow flywheel's measurement runs execute.

`standin` is built into the worker (deterministic invented numbers with designed discrepancies). `remote` sends
each run to a GPU sandbox service that speaks the `mlab.executor.v1` job API in `base.py`; that is where real
sandboxes plug in, with no change to the lab.
"""
