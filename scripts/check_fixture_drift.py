"""Compare regenerated fixtures in test_data/ against the committed baseline.

Byte-identical reproduction is impossible across platforms: np.random.randn
goes through libm log/sqrt, and glibc (linux) vs Apple libm (macOS) differ in
the last ULP, so even the raw X matrices differ in low bits between the
machine that committed a fixture and the CI runner. This checker therefore
compares NUMERICALLY:

  - integers (labels, counts, indices) must match exactly
  - floats must match within rel_tol=1e-6 / abs_tol=1e-9
  - structure (keys, lengths, types) must match exactly
  - every tracked fixture must still be generated; newly generated fixtures
    must be committed

Real drift — a sklearn behavior change or an edited gen script without a
committed fixture — still fails loudly; cross-platform float noise does not.

Usage: python scripts/check_fixture_drift.py   (after `yarn gen-data`)
"""
import json
import math
import subprocess
import sys
from pathlib import Path

RTOL = 1e-6
ATOL = 1e-9
MAX_REPORTED = 20

repo = Path(__file__).resolve().parent.parent


def committed_fixtures():
    out = subprocess.run(
        ['git', 'ls-files', 'test_data/*.json'],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    return sorted(out)


def committed_content(path):
    out = subprocess.run(
        ['git', 'show', f'HEAD:{path}'],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def compare(a, b, path, errors):
    if len(errors) > MAX_REPORTED:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(a.keys()) != sorted(b.keys()):
            errors.append(f'{path}: keys {sorted(a.keys())} != {sorted(b.keys())}')
            return
        for k in a:
            compare(a[k], b[k], f'{path}.{k}', errors)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errors.append(f'{path}: length {len(a)} != {len(b)}')
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]', errors)
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            errors.append(f'{path}: {a} != {b}')
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                errors.append(f'{path}: {a} != {b} (integer mismatch)')
            return
        if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            errors.append(f'{path}: {a} !~ {b} (beyond rtol={RTOL}, atol={ATOL})')
        return
    if a != b:
        errors.append(f'{path}: {a!r} != {b!r}')


def main():
    failed = []
    tracked = committed_fixtures()
    for rel in tracked:
        f = repo / rel
        if not f.exists():
            failed.append((rel, [f'{rel}: committed fixture was not regenerated']))
            continue
        baseline = committed_content(rel)
        regenerated = json.loads(f.read_text())
        errors = []
        compare(baseline, regenerated, rel, errors)
        if errors:
            failed.append((rel, errors))

    # generated fixtures that are not committed at all
    tracked_set = set(tracked)
    for f in sorted((repo / 'test_data').glob('*.json')):
        rel = f'test_data/{f.name}'
        if rel not in tracked_set:
            failed.append((rel, [f'{rel}: generated but not committed (git add it)']))

    if failed:
        print(f'FIXTURE DRIFT in {len(failed)} file(s):')
        for rel, errors in failed:
            print(f'\n  {rel}:')
            for e in errors[:MAX_REPORTED]:
                print(f'    {e}')
            if len(errors) > MAX_REPORTED:
                print(f'    ... and more')
        sys.exit(1)
    print(f'OK: {len(tracked)} fixtures match the committed baseline within tolerance.')


if __name__ == '__main__':
    main()
