"""Equivalent historical composition evaluation, immutable per-window inputs.

No current/future targets or model calls. Keep FEV reduction semantics exactly;
reuse invariants instead of computing the same seasonal and quantile errors again.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import socket
import numpy as np
from fev.metrics import Metric, _abs_seasonal_error_per_item, _quantile_loss


def mixed_arrays(arrays, manifest, decision):
    names = manifest['candidates']
    ids = list(map(str, arrays['item_ids'].tolist()))

    def mixture(spec, rows=None, target=None):
        weights = spec['weights']
        values = np.asarray(list(weights.values()), float)
        if not weights or not set(weights) <= set(names):
            raise ValueError('Unknown or unavailable past candidates')
        if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
            raise ValueError('Invalid weights')
        values = values / values.sum()
        def array(name, suffix):
            a = arrays[name + suffix]
            return a if rows is None else a[rows, :, target]
        return tuple(sum(w * array(name, suffix) for name, w in zip(weights, values))
                     for suffix in ('__point', '__quantiles'))

    point, quantiles = mixture(decision['default'])
    point, quantiles = point.copy(), quantiles.copy()
    groups = {g['key']: g for g in manifest['groups']}
    if not set(decision.get('groups', {})) <= set(groups):
        raise ValueError('Unknown group')
    for key, spec in decision.get('groups', {}).items():
        group = groups[key]
        target = manifest['targets'].index(group['target'])
        rows = [i for i, ident in enumerate(ids) if ident in group['members']]
        p, q = mixture(spec, rows, target)
        point[rows, :, target], quantiles[rows, :, target] = p, q
    return point, quantiles


class PreparedEpisode:
    def __init__(self, directory, manifest, episode):
        self.manifest, self.episode = manifest, episode
        with np.load(Path(directory) / episode['arrays'], allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        a = self.arrays
        self.y = a['y_true']
        self.seasonal = np.clip(_abs_seasonal_error_per_item(
            y_past=a['y_past'], y_past_lengths=a['y_past_lengths'],
            seasonality=manifest['seasonality']), 0., None)
        self.abs_true = np.maximum(np.nanmean(np.abs(self.y), axis=(0, 1)), 1.)
        ids = list(map(str, a['item_ids'].tolist()))
        self.groups = [(g, manifest['targets'].index(g['target']),
                        [i for i, ident in enumerate(ids) if ident in g['members']])
                       for g in manifest['groups']]

    def evaluate(self, decision):
        point, quantiles = mixed_arrays(self.arrays, self.manifest, decision)
        ql = _quantile_loss(y_true=self.y, q_pred=quantiles,
                            quantile_levels=self.manifest['quantile_levels'])
        scaled = ql / self.seasonal[:, None, :, None]
        cells = np.mean(Metric._safemean(scaled, axis=(1,)), axis=-1)
        sql = float(np.mean(np.mean(Metric._safemean(scaled, axis=(0, 1)), axis=-1)))
        absolute = np.abs(self.y - point)
        mase = float(np.mean(Metric._safemean(absolute / self.seasonal[:, None, :], axis=(0, 1))))
        wape = float(np.mean(np.nanmean(absolute, axis=(0, 1)) / self.abs_true))
        wql = float(np.mean(np.mean(np.nanmean(ql, axis=(0, 1)) / self.abs_true[..., None], axis=-1)))
        groups = []
        for group, target, rows in self.groups:
            values = cells[rows, target]
            groups.append(dict(key=group['key'], target=group['target'],
                SQL=float(np.nanmean(values)) if np.isfinite(values).any() else None,
                visible_cells=int(np.isfinite(values).sum())))
        return dict(context=self.episode['context'],
                    metrics=dict(SQL=sql, MASE=mase, WAPE=wape, WQL=wql), groups=groups)


class Engine:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory/'MANIFEST.json').read_text())
        self.episodes = [PreparedEpisode(directory, self.manifest, e)
                         for e in self.manifest['episodes']]
        self.cache = {}

    def evaluate(self, decision):
        # Exact decision JSON only; no rounding of weights or posterior pruning.
        key = json.dumps(decision, sort_keys=True, allow_nan=False)
        if key not in self.cache:
            self.cache[key] = dict(decision=deepcopy(decision),
                episodes=[e.evaluate(decision) for e in self.episodes],
                current_labels_used=False, new_native_calls=0)
        return deepcopy(self.cache[key])


def serve(directory, path):
    engine = Engine(directory)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        server.listen(8)
        running = True
        while running:
            connection, _ = server.accept()
            with connection, connection.makefile('rwb') as stream:
                request = json.loads(stream.readline())
                if request.get('shutdown'):
                    response, running = {'shutdown': True}, False
                else:
                    try:
                        response = {'result': engine.evaluate(request['decision'])}
                    except Exception as error:
                        response = {'error': repr(error)}
                stream.write(json.dumps(response, allow_nan=False).encode()+b'\n')
                stream.flush()


def evaluate(directory, decision):
    return Engine(directory).evaluate(decision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    args = parser.parse_args()
    serve(args.directory, args.socket)
