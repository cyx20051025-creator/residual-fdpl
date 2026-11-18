"""Summarize paired effects across seeds and bootstrap the evaluation units."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
populations = defaultdict(list)
for path in sorted(root.glob('external_[0-9]*.json')):
    entry = json.loads(path.read_text())
    populations[entry['population']].append(entry)
if any(len(arms) != 3 for arms in populations.values()) or len(populations) != 5:
    raise ValueError(f'Expected 3 seeds across 5 populations, got {[(k,len(v)) for k,v in populations.items()]}')
report = {}
for population, entries in populations.items():
    assert [e['seed'] for e in entries] == [56, 57, 58]
    units = sorted({r['unit'] for r in entries[0]['rows']})
    metrics = entries[0]['metrics']
    rng = np.random.default_rng(20261115)
    draws = rng.integers(len(units), size=(10000,len(units)))
    arm_values = {}
    for arm in ('original','ridge','mlp','oracle'):
        seed_values = []
        for entry in entries:
            groups = defaultdict(list)
            for row in entry['rows']:
                groups[row['unit']].append(row['arms'][arm])
            assert sorted(groups) == units and entry['metrics'] == metrics
            seed_values.append(np.array([np.mean(groups[unit],axis=0) for unit in units]))
        arm_values[arm] = np.array(seed_values)
    arm_values['mlp_minus_ridge'] = arm_values['mlp'] - arm_values['ridge']
    arm_values['mlp_minus_original'] = arm_values['mlp'] - arm_values['original']
    report[population] = {'units':len(units), 'comparisons':{}}
    for key, values in arm_values.items():
        unit_average = values.mean(axis=0)
        effects = np.array([unit_average[indices].mean(0) for indices in draws])
        lo, hi = np.quantile(effects,[.025,.975],axis=0)
        report[population]['comparisons'][key] = {
            metric:{'mean':float(unit_average[:,n].mean()),'ci':[float(lo[n]),float(hi[n])],
                    'seed_mean':values[:,:,n].mean(axis=1).tolist()}
            for n,metric in enumerate(metrics)
        }
(root/'external_summary.json').write_text(json.dumps(report,indent=2)+'\n')
for label,p in report.items():
    print(label, p['units'], {k:round(v['psnr']['mean'],4) for k,v in p['comparisons'].items()})
