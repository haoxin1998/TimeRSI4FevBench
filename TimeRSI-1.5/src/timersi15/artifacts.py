"""Task-local programs and historical state, exposed only at recorded cutoffs."""
from collections import OrderedDict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
from urllib.parse import unquote, urlparse
import zipfile

import numpy as np


class TaskArtifacts:
    def __init__(self, task_name, folder=None):
        root = Path(folder) if folder is not None else Path(__file__).with_name("artifacts")
        index = json.loads((root / "index.json").read_text())
        entry = index[task_name]
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Task artifact hash mismatch")
        self.archive = zipfile.ZipFile(path)
        self.root, self.parts, self.open_parts = root, entry.get("parts", {}), {}
        self.decoded, self.decoded_bytes = OrderedDict(), 0
        self.recipe = json.loads(self.archive.read("recipe.json"))

    def decoded_arrays(self, row):
        key = row["arrays"]
        if key in self.decoded:
            self.decoded.move_to_end(key)
            return self.decoded[key]
        with np.load(io.BytesIO(self.array_payload(row)), allow_pickle=False) as saved:
            values = {name: saved[name] for name in ("point", "quantiles", "truth")}
        size = sum(value.nbytes for value in values.values())
        if size <= 256 * 1024**2:
            while self.decoded and self.decoded_bytes + size > 256 * 1024**2:
                _, old = self.decoded.popitem(last=False)
                self.decoded_bytes -= sum(value.nbytes for value in old.values())
            for value in values.values():
                value.flags.writeable = False
            self.decoded[key] = values
            self.decoded_bytes += size
        return values

    def array_payload(self, row):
        name = row.get("archive")
        if name is None:
            return self.archive.read(row["arrays"])
        if name not in self.open_parts:
            path = self.root / name
            pointer = False
            if path.exists():
                with path.open("rb") as stream:
                    pointer = stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
            if not path.exists() or pointer:
                path = self.download_state(name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.parts[name]:
                raise ValueError("Historical state hash mismatch")
            self.open_parts[name] = zipfile.ZipFile(path)
        return self.open_parts[name].read(row["arrays"])

    def download_state(self, name):
        """VCS installers may retain LFS pointers; fetch the pinned public object."""
        distribution = importlib.metadata.distribution("timersi-fevbench15")
        source = json.loads(distribution.read_text("direct_url.json") or "{}")
        parsed = urlparse(source.get("url", ""))
        if parsed.scheme == "file":
            project = Path(unquote(parsed.path))
            local = project / "src/timersi15/artifacts" / name
            if project.is_dir() and local.is_file():
                with local.open("rb") as stream:
                    pointer = stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
                if not pointer:
                    return local
        revision = source.get("vcs_info", {}).get("commit_id")
        if not revision:
            match = re.search(r"/archive/([0-9a-f]{40})\.tar\.gz$", source.get("url", ""))
            revision = match.group(1) if match else None
        if not revision:
            raise RuntimeError("Run git lfs pull before installing from a local checkout")
        root = Path(os.environ.get("TIMERSI15_STATE_CACHE", str(Path.home() / ".cache/timersi15-state")))
        root.mkdir(parents=True, exist_ok=True)
        path = root / (self.parts[name] + ".zip")
        if path.exists():
            return path
        url = ("https://media.githubusercontent.com/media/haoxin1998/TimeRSI4FevBench/"
               + revision + "/TimeRSI-1.5/src/timersi15/artifacts/" + name)
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, stream)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != self.parts[name]:
            temporary.unlink()
            raise ValueError("Downloaded historical state hash mismatch")
        temporary.replace(path)
        return path

    def visible_memory(self, plan, view, *, historical_subset=False):
        cutoffs = {str(item["id"]): np.datetime64(item["timestamps"][item["y"].shape[1] - 1])
                   for item in view.items}
        result = []
        for allowed in plan["memory"]:
            row = self.recipe["memory"][allowed["record"]]
            indexes = allowed["rows"]
            if historical_subset:
                indexes = [i for i in indexes if str(row["ids"][i]) in cutoffs
                           and np.datetime64(row["ends"][i]) <= cutoffs[str(row["ids"][i])]]
                if not indexes:
                    continue
            ids = [row["ids"][i] for i in indexes]
            ends = [row["ends"][i] for i in indexes]
            if any(str(key) not in cutoffs or np.datetime64(end) > cutoffs[str(key)]
                   for key, end in zip(ids, ends)):
                raise ValueError("Historical state is not visible at this cutoff")
            values = {name: value[indexes].copy() for name, value in self.decoded_arrays(row).items()}
            detail = {k: row[k] for k in ("key", "source", "targets")}
            detail.update(ids=ids, ends=ends, **values)
            for name in ("starts", "official_SQL_by_target"):
                if name in row:
                    detail[name] = [row[name][i] for i in indexes]
            result.append(detail)
        return result

    def close(self):
        self.archive.close()
        for archive in self.open_parts.values():
            archive.close()


def matured_windows(bounds, view):
    observed = {str(item["id"]): np.datetime64(item["timestamps"][item["y"].shape[1] - 1])
              for item in view.items}
    return [j for j, previous in enumerate(bounds)
            if all(str(key) in observed and np.datetime64(end) <= observed[str(key)]
                   for key, end in zip(previous["ids"], previous["ends"]))]
