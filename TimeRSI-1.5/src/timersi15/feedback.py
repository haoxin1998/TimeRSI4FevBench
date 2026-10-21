"""Observation-time-filtered decision feedback."""
import numpy as np

def mature_indices(ends, view):
    observed = {str(i['id']): np.datetime64(i['timestamps'][i['y'].shape[1]-1]) for i in view.items}
    return [j for j, row in enumerate(ends)
            if all(k in observed and np.datetime64(v) <= observed[k] for k, v in row.items())]

def safe_history(rows, ends, view, limit=3):
    mature = mature_indices(ends, view)[-limit:]
    return [rows[i] for i in mature]
