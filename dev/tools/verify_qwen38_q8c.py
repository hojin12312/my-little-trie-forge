#!/usr/bin/env python3
"""Prove a splash-packed-q8c package equivalent to a splash-packed-q8 package.

Both packages come from the same checkpoint; the compact one stores bf16 (s, z)
where the older one stores fp32 (s, b' = z + 128*s). The kernels rebuild
b' = fma(128, s, z), and linear-q8-plan proves every A16 kernel gives the same
bytes from either form when s matches and fma(128, s, z) equals b'. This script
checks exactly that premise for every group of every dense Q8 tensor:

1. The compact manifest and all artifacts pass the installer's full check.
2. Every packed file has the expected header and section consumption.
3. Every non-Q8 section is byte-identical.
4. For every Q8 tensor: int8 weights are byte-identical, every compact scale
   widens to the fp32 scale bit for bit, and z + 128*s evaluated in fp32
   (128*s is exact, so one rounding, the fma result) equals the fp32 b' bit for
   bit. No scale, zero point or folded bias is subnormal (fast-math kernels
   may flush them).
5. draft/, vision/ and tokenizer/ are byte-identical.

    python dev/tools/verify_qwen38_q8c.py \
        --compact ~/Models/Qwen3.8-27B-8bit-q8c-Splash \
        --folded ~/Models/Qwen3.8-27B-8bit-Splash --json RECORD.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_qwen38_q8 import (  # noqa: E402
    ATTENTION_PACKED_WIDTH,
    ATTENTION_WIDTH,
    GDN_PACKED_WIDTH,
    GROUPS,
    HIDDEN,
    LAYERS,
    VOCAB,
    sha256,
)
from verify_qwen38_q8 import PackageOps, fail  # noqa: E402

from install import models as artifacts  # noqa: E402

FULL_ATTENTION_PERIOD = 4
INTERMEDIATE = 17408


def q8(out: int, in_size: int):
    return ("q8", out, in_size)


def layer_spec(layer: int):
    spec = [("raw", 10240)]
    if (layer + 1) % FULL_ATTENTION_PERIOD == 0:
        spec += [
            q8(ATTENTION_PACKED_WIDTH, HIDDEN),
            ("raw", 512),
            ("raw", 512),
            q8(HIDDEN, ATTENTION_WIDTH),
            ("raw", 10240),
        ]
    else:
        spec += [
            q8(GDN_PACKED_WIDTH, HIDDEN),
            ("raw", 81920),
            ("raw", 192),
            ("raw", 96),
            ("raw", 256),
            q8(HIDDEN, ATTENTION_WIDTH),
            ("raw", 10240),
        ]
    return spec + [
        q8(INTERMEDIATE, HIDDEN),
        q8(INTERMEDIATE, HIDDEN),
        q8(HIDDEN, INTERMEDIATE),
    ]


def sizes(spec, compact: bool):
    result = []
    for entry in spec:
        if entry[0] == "raw":
            result.append(entry[1])
        else:
            elements = entry[1] * entry[2]
            result.append(elements + 2 * elements // (32 if compact else 16))
    return result


def subnormal(bits: np.ndarray) -> int:
    """fp32 bit patterns with a zero exponent and a nonzero mantissa."""
    return int(np.count_nonzero(((bits & 0x7F800000) == 0) & ((bits & 0x7FFFFF) != 0)))


class Tally:
    def __init__(self):
        self.tensors = 0
        self.groups = 0
        self.raw_sections = 0
        self.subnormal = 0

    def parameters(self, what, s16, z16, s32, b32):
        """Compare one tensor's parameters; every array is flat, same order."""
        if not (len(s16) == len(z16) == len(s32) == len(b32)):
            fail(f"{what}: parameter counts differ")
        scale_bits = s16.astype(np.uint32) << 16
        if not np.array_equal(scale_bits, s32):
            count = int(np.count_nonzero(scale_bits != s32))
            fail(f"{what}: {count} scales differ from the fp32 package")
        scale = scale_bits.view(np.float32)
        zero_bits = z16.astype(np.uint32) << 16
        rebuilt = (zero_bits.view(np.float32) + np.float32(128.0) * scale).view(
            np.uint32
        )
        if not np.array_equal(rebuilt, b32):
            count = int(np.count_nonzero(rebuilt != b32))
            fail(f"{what}: {count} rebuilt biases differ from the fp32 b'")
        self.subnormal += subnormal(scale_bits) + subnormal(zero_bits) + subnormal(b32)
        self.tensors += 1
        self.groups += len(s16)


def compare_file(compact_root, folded_root, path, magic, layer, type_, spec, tally):
    compact = PackageOps(compact_root, path, magic, layer, type_)
    folded = PackageOps(folded_root, path, magic, layer, type_)
    pairs = zip(
        spec,
        compact.sections(sizes(spec, True)),
        folded.sections(sizes(spec, False)),
    )
    for index, (entry, mine, theirs) in enumerate(pairs):
        what = f"{path} section {index}"
        if entry[0] == "raw":
            if not np.array_equal(mine, theirs):
                fail(f"{what} differs")
            tally.raw_sections += 1
            continue
        elements = entry[1] * entry[2]
        if not np.array_equal(mine[:elements], theirs[:elements]):
            fail(f"{what}: int8 weights differ")
        compact_parameters = mine[elements:].view(np.uint16)
        folded_parameters = theirs[elements:].view(np.uint32)
        half = len(compact_parameters) // 2
        tally.parameters(
            what,
            compact_parameters[:half],
            compact_parameters[half:],
            folded_parameters[:half],
            folded_parameters[half:],
        )


def compare_embedding(compact_root, folded_root, tally):
    path = "target/embedding.bin"
    elements = VOCAB * HIDDEN
    groups = VOCAB * GROUPS
    compact = PackageOps(compact_root, path, b"MDFE0001", VOCAB, HIDDEN)
    folded = PackageOps(folded_root, path, b"MDFE0001", VOCAB, HIDDEN)
    cw, cs, cz = compact.sections([elements, groups * 2, groups * 2])
    fw, fs, fb = folded.sections([elements, groups * 4, groups * 4])
    if not np.array_equal(cw, fw):
        fail("embedding int8 weights differ")
    tally.parameters(
        path,
        cs.view(np.uint16),
        cz.view(np.uint16),
        fs.view(np.uint32),
        fb.view(np.uint32),
    )


def compare_copies(compact_root: Path, folded_root: Path) -> int:
    count = 0
    for directory in ("draft", "vision", "tokenizer"):
        for item in sorted((folded_root / directory).rglob("*")):
            if not item.is_file():
                continue
            other = compact_root / item.relative_to(folded_root)
            if not other.is_file() or sha256(item) != sha256(other):
                fail(f"{other.relative_to(compact_root)} differs")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--folded", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="write the tallies here")
    args = parser.parse_args()

    manifest = artifacts.validate_package_manifest(args.compact / "manifest.json")
    if manifest["format"]["name"] != "splash-packed-q8c":
        fail("--compact is not a splash-packed-q8c package")
    folded_manifest = artifacts.validate_package_manifest(args.folded / "manifest.json")
    if folded_manifest["format"]["name"] != "splash-packed-q8":
        fail("--folded is not a splash-packed-q8 package")
    artifacts.verify_artifacts(args.compact, manifest, full=True)
    print(f"PASS manifest and {len(manifest['artifacts'])} artifacts (full sha256)")

    tally = Tally()
    for layer in range(LAYERS):
        full = (layer + 1) % FULL_ATTENTION_PERIOD == 0
        compare_file(
            args.compact,
            args.folded,
            f"target/layer-{layer}.bin",
            b"MDFL0006",
            layer,
            1 if full else 0,
            layer_spec(layer),
            tally,
        )
    compare_file(
        args.compact,
        args.folded,
        "target/head.bin",
        b"MDFL0002",
        LAYERS,
        2,
        [("raw", 10240), q8(VOCAB, HIDDEN)],
        tally,
    )
    compare_embedding(args.compact, args.folded, tally)
    copies = compare_copies(args.compact, args.folded)
    if tally.subnormal:
        fail(f"{tally.subnormal} subnormal parameters")
    record = {
        "compact": str(args.compact),
        "folded": str(args.folded),
        "compact_artifact_set_sha256": manifest.get("artifact_set_sha256"),
        "folded_artifact_set_sha256": folded_manifest.get("artifact_set_sha256"),
        "q8_tensors": tally.tensors,
        "q8_groups": tally.groups,
        "q8_groups_bit_identical_b": tally.groups,
        "raw_sections_identical": tally.raw_sections,
        "copied_files_identical": copies,
        "subnormal_parameters": tally.subnormal,
    }
    if args.json:
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"PASS {tally.tensors} Q8 tensors, {tally.groups} groups: weights and "
        f"scales identical, fma(128, s, z) == b' for every group; "
        f"{tally.raw_sections} other sections and {copies} copied files identical"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
