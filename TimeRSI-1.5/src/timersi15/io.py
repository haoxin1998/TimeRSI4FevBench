"""Read an optional local JSON record."""
import json
from pathlib import Path

def read(path, default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.is_file() else default
