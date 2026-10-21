"""Causal persistence and local revisions over the fixed167 forecasting pool."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .probe import evaluate, mixed_arrays
from .io import read
from timersi_u1.core import write
from .feedback import safe_history
from .prompts import COMMON, LOCAL_PATCH

GUIDANCE = '''
## Preserve useful decisions across forecast origins
INPUT.json previous_own_decision is YOUR earlier decision, not an external answer.
Apply those weights to current candidates before considering changes. The incumbent
itself may change at each origin, so carrying weights does not freeze predictions.
decision_review contains causal review reasons and priority groups. Inspect these
first, but you may inspect and change any group. Do not chase marginal historical
rank differences. A different calendar, trajectory, or mature failure can justify
a revision; absence of evidence can justify keeping your previous decision.
FIRST write a valid decision.json at required_output_path, using the supplied
executable starting_decision. Then test at most four meaningful revisions, updating
decision.json immediately after each adopted change. Leave a complete valid final
JSON even if analysis is interrupted. Do not overfit historical data.
'''


def signature(packet):
    return [(g['key'], g['target'], tuple(sorted(g['members']))) for g in packet['groups']]


def clean_decision(value):
    return dict(default=dict(weights=deepcopy(value['default']['weights'])),
                groups={k: dict(weights=deepcopy(v['weights'])) for k, v in value.get('groups', {}).items()})


def usable(decision, view, forecasts, packet):
    from timersi_u1.window_planner import route_from_decision
    from timersi_u1.selection import route_forecasts
    # The probe also rejects unknown groups, bad weights and unavailable candidates.
    arrays = {'item_ids': np.asarray([str(i['id']) for i in view.items])}
    arrays.update({n+'__point': f.point for n, f in forecasts.items()})
    arrays.update({n+'__quantiles': f.quantiles for n, f in forecasts.items()})
    p, q = mixed_arrays(arrays, dict(candidates=list(forecasts), targets=view.targets,
                                    groups=packet['groups']), decision)
    routes = route_from_decision(view, packet['groups'], forecasts, decision)
    result = route_forecasts(view, forecasts, routes, 'incumbent167').validate()
    np.testing.assert_allclose(p, result.point, rtol=0, atol=1e-8)
    np.testing.assert_allclose(q, result.quantiles, rtol=0, atol=1e-8)
    return result


def mean_finite(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None


def turn_cap_artifact_allowed(receipt):
    return bool(receipt and receipt.get('observed_models') and not receipt.get('timeout')
                and receipt.get('exit_code') == 1 and receipt.get('results')
                and receipt['results'][-1].get('subtype') == 'error_max_turns')


def review(index, packet, previous_packet, feedback, previous_decision, last_call,
           historical_probe, correction):
    """Only receives visible packets, completed-history probes and mature feedback."""
    reasons = []
    priorities = []
    if previous_decision is None:
        reasons.append('first_origin')
    if previous_packet is not None:
        if signature(packet) != signature(previous_packet):
            reasons.append('group_membership_changed')
        if set(packet['candidates']) != set(previous_packet['candidates']):
            reasons.append('candidate_pool_changed')
    if index-last_call >= 3:
        reasons.append('three_origin_review_interval')
    if historical_probe:
        for episode in historical_probe['episodes']:
            held = episode['metrics']['SQL']
            reference = packet['candidates']['incumbent167'].get('historical_S_SQL')
            if reference is not None and np.isfinite(held) and held > reference*1.05:
                reasons.append('carried_policy_historical_SQL_worse_than_incumbent_5pct')
    # Mature errors concern the actual predictions made then, never retrospectively
    # judging current decisions with labels from the current or future forecast.
    if feedback:
        row = feedback[-1]
        inc = row['candidate_cells'].get('incumbent167')
        if inc is not None:
            new = mean_finite(row['new_cells'])
            baseline = mean_finite(inc)
            if new is not None and baseline is not None and new > baseline*1.05:
                reasons.append('latest_mature_final_SQL_worse_than_incumbent_5pct')
            for group in packet['groups']:
                d = row['targets'].index(group['target']) if group['target'] in row['targets'] else None
                rows = [i for i, ident in enumerate(row['ids']) if ident in group['members']]
                if d is None or not rows:
                    continue
                a = mean_finite([row['new_cells'][i][d] for i in rows])
                b = mean_finite([inc[i][d] for i in rows])
                if a is not None and b is not None and a > b:
                    priorities.append(dict(group=group['key'], mature_SQL=a,
                                           incumbent_SQL=b, excess_SQL=a-b))
    old = (previous_packet or {}).get('correction_summary')
    if old is not None and len(old) == len(correction):
        old, current = np.asarray(old), np.asarray(correction)
        changed = (np.abs(current-old) > .5) & ((current*old < 0) |
                    (np.abs(current) > 2*np.maximum(np.abs(old), .25)))
        if changed.any():
            reasons.append('current_normalized_correction_changed')
    return dict(call_agent=bool(reasons), reasons=list(dict.fromkeys(reasons)),
                priority_groups=sorted(priorities, key=lambda x: -x['excess_SQL'])[:5],
                last_agent_origin=last_call, current_origin=index,
                current_labels_used=False)


def correction_summary(view, forecast, raw):
    out = []
    for i, item in enumerate(view.items):
        y = np.asarray(item['y'], float)
        scale = np.nanmean(np.abs(np.diff(y, axis=1)), axis=1) if y.shape[1] > 1 else np.ones(y.shape[0])
        scale = np.maximum(np.nan_to_num(scale, nan=1.), 1e-8)
        out.extend((np.mean(forecast.point[i]-raw.point[i], axis=0)/scale).tolist())
    return out


def decide(cfg, job, arm, index, view, forecasts, base_packet, template, prior, ends):
    from .agent_transport import propose
    from timersi_u1.window_planner import GUIDE
    folder = Path(job['output'])/arm/'official'/f'window-{index:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    packet = deepcopy(base_packet)
    feedback = safe_history(prior, ends, view)
    packet['replay_feedback'] = feedback
    for group in packet['groups']:
        d = view.targets.index(group['target'])
        for name in forecasts:
            cells = [row['candidate_cells'][name][i][d] for row in feedback
                     if name in row['candidate_cells'] for i, ident in enumerate(row['ids'])
                     if ident in group['members']]
            finite = [v for v in cells if v is not None and np.isfinite(v)]
            group['mature_candidate_errors'][name] = dict(SQL=mean_finite(finite), visible_cells=len(finite))
    previous = prior[-1]['decision'] if prior else None
    previous_folder = folder.parent/f'window-{index-1:03d}'
    previous_packet = read(previous_folder/'PUBLIC-INPUT.json') if prior else None
    previous_review = read(previous_folder/'REVIEW.json', {}) if prior else {}
    last_call = previous_review.get('last_agent_origin', -3)
    starting = clean_decision(previous) if previous else dict(default=dict(weights={'incumbent167': 1.}), groups={})
    invalid_previous = None
    try:
        carried = usable(starting, view, forecasts, packet)
    except Exception:
        invalid_previous = traceback.format_exc()
        starting = dict(default=dict(weights={'incumbent167': 1.}), groups={})
        carried = forecasts['incumbent167']
    historical = None
    try:
        historical = evaluate(Path(template)/'composition-past', starting)
    except Exception:
        write(folder/'HISTORICAL-PROBE-ERROR.json', dict(error=traceback.format_exc()))
    correction = correction_summary(view, carried, forecasts['raw'])
    info = review(index, packet, previous_packet, feedback, previous, last_call, historical, correction)
    if invalid_previous:
        info['reasons'].append('previous_decision_unavailable')
        info['call_agent'] = True
    if historical is None:
        info['reasons'].append('historical_probe_unavailable')
        info['call_agent'] = True
    if arm == 'R':
        info['call_agent'] = True
        info['reasons'].append('always_review_control')
    packet.update(previous_own_decision=previous, starting_decision=starting,
                  decision_review=info, carried_historical_probe=historical,
                  correction_summary=correction)
    write(folder/'PUBLIC-INPUT.json', packet)
    if not info['call_agent']:
        write(folder/'REVIEW.json', {**info, 'reused_decision': True})
        write(folder/'DECISION.json', dict(decision=starting, receipt=None,
              operational_fallback=None, seconds=time.monotonic()-started, reused_decision=True))
        return carried, starting, None, None
    session = folder/'agent-session'
    workspace = session/'workspace'
    shutil.copytree(template, workspace)
    receipt = None
    error = None
    recovered = False
    transport_error = None
    decision = starting
    result = carried
    try:
        project = cfg['projects'][(job['index']+(arm == 'P')) % len(cfg['projects'])]
        try:
            text, receipt = propose(session, packet, [], '', [], model=cfg['agent_model'],
                effort=cfg['effort'], timeout=cfg['decision_timeout'], max_turns=cfg['max_turns'],
                guide=GUIDE+COMMON+LOCAL_PATCH+GUIDANCE, vertex_project=project,
                vertex_region='global', output_artifact='decision.json', publish_session_history=False)
            if receipt['exit_code'] != 0 or receipt['timeout'] or not receipt['observed_models']:
                raise RuntimeError('Incomplete or unobserved model session')
        except Exception:
            transport_error = traceback.format_exc()
            receipt = read(session/'AGENT-RESULT.json', receipt)
            # Preserve only a complete, validated artifact from a real model session.
            draft = workspace/'decision.json'
            if not (turn_cap_artifact_allowed(receipt) and draft.is_file()):
                raise
            text = draft.read_text()
            recovered = True
        candidate = json.loads(text)
        result = usable(candidate, view, forecasts, packet)
        decision = candidate
    except Exception:
        error = traceback.format_exc()
        recovered = False
        result, decision = carried, starting
    write(folder/'REVIEW.json', {**info, 'last_agent_origin': index,
          'reused_decision': False, 'recovered_turn_cap_artifact': recovered,
          'transport_error': transport_error, 'invalid_previous': invalid_previous})
    write(folder/'DECISION.json', dict(decision=decision, receipt=receipt,
          operational_fallback=error, seconds=time.monotonic()-started,
          recovered_turn_cap_artifact=recovered))
    return result, decision, receipt, error
