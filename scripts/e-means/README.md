# Reproducing the e-means CPU benchmarks

Run from the repository root. The two vector files come from [ANN-Benchmarks](https://ann-benchmarks.com/). Each run reads the first 10,000 training rows and 1,000 test rows using squared Euclidean distance on the original coordinates, including for GloVe. The script also creates a two-dimensional mixture, an eight-dimensional mixture with 1% distant observations, and the sklearn digits split. Input preparation requires NumPy, h5py and scikit-learn.

```sh
mkdir -p /tmp/emeans
curl -fL https://ann-benchmarks.com/glove-25-angular.hdf5 -o /tmp/emeans/glove-25-angular.hdf5
curl -fL https://ann-benchmarks.com/sift-128-euclidean.hdf5 -o /tmp/emeans/sift-128-euclidean.hdf5
python scripts/e-means/prepare.py /tmp/emeans
yarn build
node scripts/e-means/run.mjs /tmp/emeans glove-25-angular
node scripts/e-means/run.mjs /tmp/emeans glove-25-angular --chrome
```

Replace the dataset argument with `sift-128-euclidean`, `rare`, `blobs` or `digits`. Set `CHROME_BIN` to the local Chrome or Chromium executable on Linux. Each invocation starts a fresh process, warms each method, and prints JSON for five seeds. Each controlled fit receives the same initial centers for its seed. The benchmark injects them into MiniBatchKMeans's initializer because its public constructor only exposes initialization strategies. For the rare-group stress test, one initial center covers the distant group. The default runs use each public API's own seeding, restart and early stopping settings.

Times include initialization, fitting and exact training labels at the final centers. Lloyd receives a final assignment scan because its last iteration can move centers after assigning labels. Held-out scoring and data loading are outside the timer. The `passes` field counts training assignments in units of training rows, including MiniBatchKMeans initialization updates, before the closing label scan. The Node process high-water RSS includes the runtime, decoded data, warmup and all fits for that dataset.

[Per-seed measurements](results.csv) from an Apple M4 running Node.js 24.14.0 and Chrome 158.0.8127.88 include both controlled budgets and defaults. For each seed, the documentation uses the smallest tested budget whose held-out MSE is within 1% of that seed's 25-pass Lloyd reference. The reported time is the median among qualifying seeds. To summarize new JSON runs, use `python scripts/e-means/summarize.py FILE.json ...`.

Node peak RSS for the complete process was 132 MiB on GloVe and 161 MiB on SIFT. These figures include all three methods. Individual timings are short, especially on two-dimensional inputs, and depend on the machine and runtime.

Run `node scripts/e-means/build-example.mjs` after changing the example or its estimator implementations to rebuild the standalone HTML embedded in the docs.
