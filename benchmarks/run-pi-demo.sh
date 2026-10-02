#!/bin/sh
# Isolated Pi frontend for an already-running loopback MLTF backend.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/launch_pi.py" "$@"
