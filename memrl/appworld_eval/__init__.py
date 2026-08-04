"""AppWorld evaluation support.

Split across a process boundary on purpose: `worker.py` runs under a separate
interpreter that has `appworld` installed (pydantic 1.x), and `client.py` runs
inside MemRL (pydantic 2.x) and never imports appworld. See worker.py's module
docstring for why.
"""
