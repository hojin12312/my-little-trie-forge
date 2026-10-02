#!/usr/bin/env python3
"""Structural and numerical verification of a splash-packed-q8 package.

Checks, in order:
1. The Python installer's manifest and full artifact verification.
2. Every packed file header and the exact section consumption the native
   reader enforces (WeightStore::section / finish).
3. Q8 section math against the source checkpoint for sampled rows: the packed
   signed codes, fp32 scales and folded biases must reproduce the affine
   checkpoint weights, and concatenated rows must appear in the documented
   order (GDN qkv|z|b|a|pad, attention q|k|v, embedding row-major).
4. GDN decay equals -exp(A_log) and the embedding tables are row-major.
5. draft/, vision/ and tokenizer/ are byte-identical to the template package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_qwen38_q8 import (  # noqa: E402
    ALIGNMENT,
    ATTENTION_PACKED_WIDTH,
    ATTENTION_WIDTH,
    GDN_CONV_DIM,
    GDN_PACKED_WIDTH,
    GDN_VALUE_HEADS,
    GROUP,
    GROUPS,
    HIDDEN,
    LAYERS,
    VOCAB,
    SafetensorsIndex,
    sha256,
)

from install import models as artifacts  # noqa: E402

FULL_ATTENTION_PERIOD = 4


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def section_offsets(sizes, total: int):
    offsets = []
    offset = 16
    for size in sizes:
        offset = -(-offset // ALIGNMENT) * ALIGNMENT
        offsets.append(offset)
        offset += size
    if -(-offset // ALIGNMENT) * ALIGNMENT != total:
        fail(f"section consumption mismatch: {offset} vs file {total}")
    return offsets


class PackageOps:
    """Reads packed files in the native reader's section order."""

    def __init__(self, root: Path, path: str, magic: bytes, layer: int, type_: int):
        self.root = root
        self.path = path
        data = (root / path).read_bytes()
        self.size = len(data)
        header_magic, header_layer, header_type = (
            data[:8],
            int.from_bytes(data[8:12], "little"),
            int.from_bytes(data[12:16], "little"),
        )
        if (header_magic, header_layer, header_type) != (magic, layer, type_):
            fail(
                f"{path} header mismatch: {header_magic!r} {header_layer} {header_type}"
            )
        if self.size % ALIGNMENT:
            fail(f"{path} size {self.size} is not 16 KiB aligned")
        self.data = memoryview(data)

    def sections(self, sizes):
        offsets = section_offsets(sizes, self.size)
        for offset, size in zip(offsets, sizes):
            yield np.frombuffer(self.data[offset : offset + size], dtype=np.uint8)


def q8_packed_bytes(out: int, in_size: int) -> int:
    elements = out * in_size
    return elements + elements // 8


def dequant_row(section: np.ndarray, out: int, in_size: int, row: int):
    elements = out * in_size
    groups = in_size // GROUP
    tiles = out // 256
    weights = section[:elements]
    scales = section[elements : elements + 4 * groups * out].view(np.float32)
    biases = section[elements + 4 * groups * out :].view(np.float32)
    block = weights.reshape(tiles, groups, 256, GROUP)[row // 256, :, row % 256, :]
    signed = np.ascontiguousarray(block).view(np.int8).astype(np.int16)
    s = scales.reshape(tiles, groups, 256)[row // 256, :, row % 256]
    b = biases.reshape(tiles, groups, 256)[row // 256, :, row % 256]
    return (s[:, None] * signed + b[:, None]).reshape(-1)


def checkpoint_row(index: SafetensorsIndex, stem: str, row: int):
    """The affine weight row with the package's folded bias semantics."""
    signed, scales, biases = index.affine(stem)
    groups = signed.shape[1] // GROUP
    codes = signed[row].reshape(groups, GROUP)
    folded = biases[row] + 128.0 * scales[row]
    return (scales[row, :, None] * codes + folded[:, None]).reshape(-1)


def close(actual, expected, what: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        worst = np.abs(actual - expected).max()
        fail(f"{what} mismatch (max |diff| {worst})")


def verify_layer(
    root: Path, index: SafetensorsIndex, layer: int, sample_rows: list[int]
) -> None:
    full = (layer + 1) % FULL_ATTENTION_PERIOD == 0
    sizes = [10240]
    if full:
        sizes += [
            q8_packed_bytes(ATTENTION_PACKED_WIDTH, HIDDEN),
            512,
            512,
            q8_packed_bytes(HIDDEN, ATTENTION_WIDTH),
            10240,
        ]
    else:
        sizes += [
            q8_packed_bytes(GDN_PACKED_WIDTH, HIDDEN),
            81920,
            192,
            96,
            256,
            q8_packed_bytes(HIDDEN, ATTENTION_WIDTH),
            10240,
        ]
    sizes += [q8_packed_bytes(17408, HIDDEN)] * 2 + [q8_packed_bytes(HIDDEN, 17408)]
    ops = PackageOps(
        root, f"target/layer-{layer}.bin", b"MDFL0006", layer, 1 if full else 0
    )
    sections = list(ops.sections(sizes))
    prefix = f"language_model.model.layers.{layer}."
    if full:
        packed = sections[1]
        attention = prefix + "self_attn."
        for row in sample_rows:
            if row < 12288:
                stem = attention + "q_proj"
                source_row = row
            elif row < 13312:
                stem = attention + "k_proj"
                source_row = row - 12288
            else:
                stem = attention + "v_proj"
                source_row = row - 13312
            close(
                dequant_row(packed, ATTENTION_PACKED_WIDTH, HIDDEN, row),
                checkpoint_row(index, stem, source_row),
                f"layer-{layer} attention-input row {row} ({stem})",
            )
    else:
        packed = sections[1]
        elements = GDN_PACKED_WIDTH * HIDDEN
        gdn = prefix + "linear_attn."
        for row in sample_rows:
            if row < GDN_CONV_DIM:
                stem, source_row = gdn + "in_proj_qkv", row
            elif row < GDN_CONV_DIM + ATTENTION_WIDTH:
                stem, source_row = gdn + "in_proj_z", row - GDN_CONV_DIM
            elif row < GDN_CONV_DIM + ATTENTION_WIDTH + GDN_VALUE_HEADS:
                stem = gdn + "in_proj_b"
                source_row = row - GDN_CONV_DIM - ATTENTION_WIDTH
            elif row < GDN_PACKED_WIDTH - 160:
                stem = gdn + "in_proj_a"
                source_row = row - GDN_CONV_DIM - ATTENTION_WIDTH - GDN_VALUE_HEADS
            else:
                continue
            close(
                dequant_row(packed, GDN_PACKED_WIDTH, HIDDEN, row),
                checkpoint_row(index, stem, source_row),
                f"layer-{layer} gdn-input row {row} ({stem})",
            )
        weights = np.frombuffer(sections[1], dtype=np.int8)[:elements]
        logical = (
            weights.reshape(65, 80, 256, 64)
            .transpose(0, 2, 1, 3)
            .reshape(GDN_PACKED_WIDTH, HIDDEN)
        )
        if logical[GDN_PACKED_WIDTH - 160 :].any():
            fail(f"layer-{layer} gdn padding weights are not zero")
        params = sections[1][elements:]
        pad_scales = np.frombuffer(
            params[: 4 * GDN_PACKED_WIDTH * 80], np.float32
        ).reshape(65, 80, 256)[:, :, GDN_PACKED_WIDTH - 160 :]
        pad_biases = np.frombuffer(
            params[4 * GDN_PACKED_WIDTH * 80 :], np.float32
        ).reshape(65, 80, 256)[:, :, GDN_PACKED_WIDTH - 160 :]
        if pad_scales.any() or pad_biases.any():
            fail(f"layer-{layer} gdn padding parameters are not zero")
        decay = np.frombuffer(sections[3], dtype=np.float32)
        expected = -np.exp(
            index.bf16(
                "language_model.model.layers.%d.linear_attn.A_log" % layer
            ).astype(np.float32)
        )
        if not np.array_equal(decay, expected):
            fail(f"layer-{layer} decay != -exp(A_log)")
    print(f"PASS target/layer-{layer}.bin sections and sampled rows")


def verify_head(root: Path, index: SafetensorsIndex) -> None:
    sizes = [10240, q8_packed_bytes(VOCAB, HIDDEN)]
    ops = PackageOps(root, "target/head.bin", b"MDFL0002", LAYERS, 2)
    sections = list(ops.sections(sizes))
    close(
        np.frombuffer(sections[0], dtype=np.uint16).astype(np.uint32) << 16,
        index.bf16("language_model.model.norm.weight").view(np.uint32),
        "head final-norm",
    )
    for row in (0, 12345, VOCAB - 1):
        close(
            dequant_row(sections[1], VOCAB, HIDDEN, row),
            checkpoint_row(index, "language_model.lm_head", row),
            f"head logits row {row}",
        )
    print("PASS target/head.bin")


def verify_embedding(root: Path, index: SafetensorsIndex) -> None:
    elements = VOCAB * HIDDEN
    sizes = [elements, VOCAB * GROUPS * 4, VOCAB * GROUPS * 4]
    ops = PackageOps(root, "target/embedding.bin", b"MDFE0001", VOCAB, HIDDEN)
    weights, scales, biases = (
        np.frombuffer(s, dtype=dtype)
        for s, dtype in zip(ops.sections(sizes), (np.int8, np.float32, np.float32))
    )
    codes, ref_scales, ref_biases = index.affine("language_model.model.embed_tokens")
    for token in (0, 12345, VOCAB - 1):
        actual = weights[token * HIDDEN : (token + 1) * HIDDEN].astype(
            np.float32
        ) * np.repeat(scales[token * GROUPS : (token + 1) * GROUPS], GROUP) + np.repeat(
            biases[token * GROUPS : (token + 1) * GROUPS], GROUP
        )
        expected = (
            ref_scales[token, :, None] * codes[token].reshape(GROUPS, GROUP)
            + (ref_biases[token] + 128.0 * ref_scales[token])[:, None]
        ).reshape(-1)
        close(actual, expected, f"embedding token {token}")
    if not np.array_equal(codes[:64].reshape(-1), weights[: 64 * HIDDEN]):
        fail("embedding weights are not the row-major signed codes")
    print("PASS target/embedding.bin")


def verify_copies(root: Path, template: Path) -> None:
    for directory in ("draft", "vision", "tokenizer"):
        for item in sorted((template / directory).rglob("*")):
            if not item.is_file():
                continue
            other = root / item.relative_to(template)
            if not other.is_file() or sha256(item) != sha256(other):
                fail(f"{other.relative_to(root)} differs from the template")
        print(f"PASS {directory}/ matches the template byte-for-byte")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()

    manifest = artifacts.validate_package_manifest(args.package / "manifest.json")
    artifacts.verify_artifacts(args.package, manifest, full=True)
    print(f"PASS manifest and {len(manifest['artifacts'])} artifacts (full sha256)")

    index = SafetensorsIndex(args.checkpoint)
    verify_layer(args.package, index, 0, [0, 10240, 16384, 16432])
    verify_layer(args.package, index, 3, [0, 6144, 12288, 13312])
    verify_layer(args.package, index, 63, [0, 14335])
    verify_head(args.package, index)
    verify_embedding(args.package, index)
    verify_copies(args.package, args.template)
    print("Q8 package verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
