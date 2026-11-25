"""Write row-major Float32 benchmark inputs. See README.md in this directory."""
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.datasets import load_digits

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def write(name, train, test):
    for part, rows in [('train', train), ('test', test)]:
        np.asarray(rows, dtype='<f4').tofile(out / f'{name}-{part}.f32')


def clouds(rng, n, d, rare):
    x = rng.normal(size=(n, d)).astype('float32')
    if rare:
        x[-(n // 100):] += 20
    else:
        x[np.arange(n) % 3 == 0, 0] += 4
        x[np.arange(n) % 3 == 1, 1] += 4
    return x


for name, d, rare in [('blobs', 2, False), ('rare', 8, True)]:
    rng = np.random.default_rng(2026)
    write(name, clouds(rng, 5000, d, rare), clouds(rng, 1000, d, rare))

x = load_digits().data.astype('float32') / 16
order = np.random.default_rng(2026).permutation(len(x))
write('digits', x[order[:1400]], x[order[1400:]])

for name in ['glove-25-angular', 'sift-128-euclidean']:
    source = out / (name + '.hdf5')
    if source.exists():
        with h5py.File(source) as f:
            write(name, f['train'][:10000], f['test'][:1000])
