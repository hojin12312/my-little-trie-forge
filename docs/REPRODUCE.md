# Reproduce the MLTF coding matrix

The measurement harness is identified in `docs/data/measured-metrics.json`.
This export adapts filesystem paths and removes review-repository publication
hooks; native `all_ready.mm` is byte-identical to the measured adapter source.
Production runtime code is unchanged. Exported Python hashes differ from the
measurement orchestration harness and are not claimed to be the same file.

Use the pinned CPython3.13.14 runtime for exact corpus reconstruction and the
reviewed MLTF runtime dependencies. The dependency source inputs are hashed
individually in the corpus manifest. Original fixture code is Apache-2.0;
CPython sources retain the PSF notices in `benchmarks/CPYTHON_LICENSE.txt`.
Corpus generation uses distinct dependency implementation blocks at complete
AST boundaries; it does not repeat files or use random filler.

Build the normal B1–B4 native library from this source checkout first. Configure
these task-owned paths for your local installed environment; no weights are
downloaded and global Pi settings are not changed by the benchmark:

```sh
export MLTF_MODEL_PATH=/path/to/your/verified/q8c
export MLTF_INSTALL_SITE=/path/to/your/venv/lib/python3.13/site-packages
export MLTF_PYTHON=/path/to/your/venv/bin/python
export MLTF_BENCH_OUTPUT="$PWD/benchmark-output"
export MLTF_CORPUS_OUTPUT="$MLTF_BENCH_OUTPUT/corpus"
export MLTF_BENCH_ADAPTER="$PWD/benchmark-build/all-ready"
mkdir -p benchmark-build
xcrun -sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror \
  -fobjc-arc -mmacosx-version-min=26.4 -DSPLASH_Q8_PREFILL_A8=1 \
  -Iruntime benchmarks/all_ready.mm build/engine/libsplash.a \
  -framework Foundation -framework Metal -framework IOKit \
  -o "$MLTF_BENCH_ADAPTER"
"$MLTF_PYTHON" benchmarks/make_corpus.py
(cd benchmarks && "$MLTF_PYTHON" -m unittest -v test_measurement)
"$MLTF_PYTHON" benchmarks/run.py dry-run
"$MLTF_PYTHON" benchmarks/run.py freeze
"$MLTF_PYTHON" benchmarks/run.py grid
"$MLTF_PYTHON" benchmarks/summarize.py
```

The path values are installation-specific inputs. Use a fresh output directory
for a new campaign; attempts and the frozen method are immutable. Do not run
another GPU model, heavy build, or video encoder concurrently. See
[the method](MEASUREMENT_METHOD.md) for exact metrics, timeouts, budgets,
window validation, cold-cache policy, and limitations.
