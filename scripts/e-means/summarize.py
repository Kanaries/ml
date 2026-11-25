"""Read run.mjs JSON files and aggregate the five paired seeds."""
import json
import sys
from pathlib import Path
from statistics import median

for filename in sys.argv[1:]:
    result = json.loads(Path(filename).read_text())
    runs = result['runs']
    print(result['name'], result['runtime'], 'maxRSSKiB', result.get('maxRSSKiB'))
    for method in ['Lloyd', 'MiniBatch', 'EMeans', 'NoRevival']:
        matched = []
        for seed in range(5):
            baseline = next(r for r in runs if r['seed'] == seed and r['method'] == 'Lloyd' and r['epochs'] == 25)
            candidates = [r for r in runs if r['seed'] == seed and r['method'] == method and
                          r['epochs'] != 'default' and r['mse'] <= 1.01 * baseline['mse']]
            if candidates:
                matched.append(min(candidates, key=lambda r: r['epochs'])['ms'])
        selected = [r for r in runs if r['method'] == method and r['epochs'] == ('default' if method != 'NoRevival' else 5)]
        if selected:
            print(method, 'matched', len(matched), 'median ms among matches',
                  round(median(matched), 2) if matched else None,
                  'default (or 5 for NoRevival) ms/MSE',
                  *(round(median(r[key] for r in selected), 4) for key in ['ms', 'mse']))
