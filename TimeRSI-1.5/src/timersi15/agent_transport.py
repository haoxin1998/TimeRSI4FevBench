"""Keep the historical evaluator resident for one code-agent decision session."""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time


@contextmanager
def evaluator(workspace):
    workspace=Path(workspace)
    path=workspace/'.composition.sock'
    with (workspace/'probe-service.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','timersi15.probe',
            '--directory',str(workspace/'composition-past'),'--socket','.composition.sock'],
            cwd=workspace,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        try:
            started=time.monotonic()
            while not path.exists():
                if process.poll() is not None:raise RuntimeError('Historical evaluator failed to start')
                if time.monotonic()-started>60:raise TimeoutError('Historical evaluator startup')
                time.sleep(.02)
            yield
        finally:
            if process.poll() is None:process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            path.unlink(missing_ok=True)


def propose(session,*args,**kwargs):
    from timersi_u1.agent import propose as original
    with evaluator(Path(session)/'workspace'):
        return original(session,*args,**kwargs)
