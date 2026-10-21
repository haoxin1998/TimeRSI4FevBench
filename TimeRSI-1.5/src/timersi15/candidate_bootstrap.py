"""Candidate-only startup: preserve the shared eight-thread fitting policy."""
import os
import sys

if os.environ.get("TIMERSI15_CANDIDATE") == "1" and not sys.argv[0].endswith(("composition_probe.py", "probe_client.py")):
    from timersi_u1.compute import install
    from timersi_u1.lasso186 import install as install_lasso
    install()
    install_lasso()
