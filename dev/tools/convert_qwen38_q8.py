#!/usr/bin/env python3
"""Convert mlx-community/Qwen3.8-27B-8bit into a splash-packed-q8c package.

The target tensors are 8-bit MLX affine quantized (group 64, uint8 codes packed
four per little-endian U32 word). Dense Q8 sections store signed int8 codes in
the StorageN=256 packed order with the checkpoint's own bf16 scales s and zero
points z (splash-packed-q8c, the default): the kernels rebuild the folded bias
b' = fma(128, s, z), which equals the fp32 z + 128*s bit for bit. With
--parameters fp32 the converter writes the older splash-packed-q8 form with
fp32 scales and fp32 folded biases instead. Both match
WeightStore::readQ8DenseProjection. Token embeddings are rewritten row-major in
three independently aligned sections (weights, scales, biases) so the gather
kernel can bind each table directly.

The draft and vision towers and the tokenizer are not quantized by this stage;
they are copied byte-for-byte from an existing splash-packed-q4 package, whose
layouts the runtime already validates. Provenance for the copy is recorded in
the manifest.

    python dev/tools/convert_qwen38_q8.py \
        --checkpoint ~/Models/Qwen3.8-27B-8bit \
        --template ~/Models/Qwen3.8-27B-Splash \
        --output ~/Models/Qwen3.8-27B-8bit-q8c-Splash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import models as artifacts  # noqa: E402

ALIGNMENT = artifacts.ALIGNMENT
GROUP = 64
STORAGE_N = 256
HIDDEN = 5120
VOCAB = 248320
LAYERS = 64
FULL_ATTENTION_PERIOD = 4
GDN_PACKED_WIDTH = 16640
ATTENTION_PACKED_WIDTH = 14336
ATTENTION_WIDTH = 6144
GDN_CONV_DIM = 10240
GDN_VALUE_HEADS = 48
HEAD_LAYER_TARGET = LAYERS
GROUPS = HIDDEN // GROUP

TARGET_MAGIC = b"MDFL0006"
HEAD_MAGIC = b"MDFL0002"
EMBEDDING_MAGIC = b"MDFE0001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class SafetensorsIndex:
    """Raw safetensors access that keeps bf16/U32 bytes untouched."""

    def __init__(self, root: Path):
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.root = root
        self.weight_map = index["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._data: dict[str, bytes] = {}

    def _header(self, filename: str) -> tuple[dict, int]:
        if filename not in self._headers:
            raw = (self.root / filename).read_bytes()
            length = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8 : 8 + length])
            self._headers[filename] = (header, 8 + length)
            self._data[filename] = raw
        return self._headers[filename]

    def _raw(self, name: str) -> tuple[str, tuple, bytes]:
        if name not in self.weight_map:
            raise KeyError(f"checkpoint is missing tensor {name}")
        filename = self.weight_map[name]
        header, base = self._header(filename)
        entry = header[name]
        raw = self._data[filename]
        start = base + entry["data_offsets"][0]
        end = base + entry["data_offsets"][1]
        return entry["dtype"], tuple(entry["shape"]), raw[start:end]

    def bf16(self, name: str) -> np.ndarray:
        """bf16 tensor values widened to fp32."""
        dtype, shape, raw = self._raw(name)
        if dtype != "BF16":
            raise ValueError(f"{name} is not bf16: {dtype}")
        words = np.frombuffer(raw, dtype=np.uint16)
        return words.astype(np.uint32).__lshift__(16).view(np.float32).reshape(shape)

    def bf16_bytes(self, name: str) -> bytes:
        """Raw bf16 section bytes in the checkpoint's own order."""
        dtype, _, raw = self._raw(name)
        if dtype != "BF16":
            raise ValueError(f"{name} is not bf16: {dtype}")
        return raw

    def codes(self, name: str) -> np.ndarray:
        """uint8 affine codes in logical row-major [out, in] order."""
        dtype, shape, raw = self._raw(name)
        if dtype != "U32":
            raise ValueError(f"{name} is not a U32 quantized weight: {dtype}")
        out, words = shape
        return np.frombuffer(raw, dtype=np.uint8).reshape(out, words * 4)

    def affine(self, stem: str):
        """Row-major (codes, scales, zero points) for a quantized tensor stem."""
        codes = self.codes(stem + ".weight").astype(np.int16) - 128
        scales = self.bf16(stem + ".scales").astype(np.float32)
        biases = self.bf16(stem + ".biases").astype(np.float32)
        return codes.astype(np.int8), scales, biases


def bf16_bits(values: np.ndarray) -> np.ndarray:
    """The bf16 encoding of values that were widened from bf16 (exact)."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    if (bits & 0xFFFF).any():
        raise ValueError("parameter is not a bf16 value")
    return (bits >> 16).astype("<u2")


def q8_pack(
    signed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    compact: bool = False,
):
    """Signed int8 codes and affine parameters into package section bytes.

    Layout: weights in StorageN=256 packed order, then two parameter tensors
    indexed [(n / 256) * groups + g] * 256 + n % 256: fp32 scales and folded
    biases (b' = z + 128*s), or with compact the bf16 scales and zero points.
    """
    out, in_size = signed.shape
    if in_size % GROUP or out % STORAGE_N:
        raise ValueError(f"unsupported dense Q8 shape {signed.shape}")
    groups = in_size // GROUP
    tiles = out // STORAGE_N
    if scales.shape != (out, groups) or biases.shape != (out, groups):
        raise ValueError("Q8 parameter shapes do not match the weight")
    packed = (
        signed.reshape(tiles, STORAGE_N, groups, GROUP)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    order = scales.reshape(tiles, STORAGE_N, groups).transpose(0, 2, 1).reshape(-1)
    if compact:
        zeros = biases.reshape(tiles, STORAGE_N, groups).transpose(0, 2, 1)
        return (
            packed.tobytes(),
            bf16_bits(order).tobytes(),
            bf16_bits(zeros.reshape(-1)).tobytes(),
        )
    folded = (biases + 128.0 * scales).reshape(tiles, STORAGE_N, groups)
    folded = folded.transpose(0, 2, 1).reshape(-1)
    return (
        packed.tobytes(),
        order.astype("<f4").tobytes(),
        folded.astype("<f4").tobytes(),
    )


def concat_affine(parts, out_size: int, in_size: int):
    codes = np.concatenate([part[0] for part in parts], axis=0)
    scales = np.concatenate([part[1] for part in parts], axis=0)
    biases = np.concatenate([part[2] for part in parts], axis=0)
    if codes.shape != (out_size, in_size):
        raise ValueError(f"concatenated codes have shape {codes.shape}")
    return codes, scales, biases


class PackedWriter:
    """Writes one weight file with 16 KiB aligned sections in reader order."""

    def __init__(self, path: Path, magic: bytes, layer: int, type_: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("wb")
        self.offset = 16
        self.file.write(struct.pack("<8sII", magic, layer, type_))

    def section(self, data: bytes) -> None:
        pad = (-self.offset) % ALIGNMENT
        self.file.write(b"\0" * pad)
        self.offset += pad
        self.file.write(data)
        self.offset += len(data)

    def finish(self) -> None:
        pad = (-self.offset) % ALIGNMENT
        self.file.write(b"\0" * pad)
        self.offset += pad
        self.file.close()


def copy_tree(source: Path, destination: Path) -> None:
    for item in sorted(source.rglob("*")):
        if item.is_file():
            target = destination / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def convert_target(checkpoint: SafetensorsIndex, output: Path, compact: bool) -> None:
    target = output / "target"
    prefix = "language_model.model."

    def read(stem: str):
        return checkpoint.affine(prefix + stem)

    def pack(signed, scales, biases) -> bytes:
        return b"".join(q8_pack(signed, scales, biases, compact))

    def dense_section(stem: str) -> bytes:
        return pack(*read(stem))

    for layer in range(LAYERS):
        full = (layer + 1) % FULL_ATTENTION_PERIOD == 0
        name = f"layers.{layer}."
        writer = PackedWriter(
            target / f"layer-{layer}.bin", TARGET_MAGIC, layer, 1 if full else 0
        )
        writer.section(checkpoint.bf16_bytes(prefix + name + "input_layernorm.weight"))
        if full:
            attention = f"{name}self_attn."
            codes, scales, biases = concat_affine(
                [
                    read(attention + "q_proj"),
                    read(attention + "k_proj"),
                    read(attention + "v_proj"),
                ],
                ATTENTION_PACKED_WIDTH,
                HIDDEN,
            )
            writer.section(pack(codes, scales, biases))
            writer.section(checkpoint.bf16_bytes(prefix + attention + "q_norm.weight"))
            writer.section(checkpoint.bf16_bytes(prefix + attention + "k_norm.weight"))
            writer.section(dense_section(attention + "o_proj"))
        else:
            gdn = f"{name}linear_attn."
            codes, scales, biases = concat_affine(
                [read(gdn + "in_proj_qkv"), read(gdn + "in_proj_z")],
                GDN_CONV_DIM + ATTENTION_WIDTH,
                HIDDEN,
            )
            small = concat_affine(
                [read(gdn + "in_proj_b"), read(gdn + "in_proj_a")],
                2 * GDN_VALUE_HEADS,
                HIDDEN,
            )
            padding = (
                GDN_PACKED_WIDTH - GDN_CONV_DIM - ATTENTION_WIDTH - 2 * GDN_VALUE_HEADS
            )
            codes = np.concatenate(
                [codes, small[0], np.zeros((padding, HIDDEN), np.int8)]
            )
            scales = np.concatenate(
                [scales, small[1], np.zeros((padding, GROUPS), np.float32)]
            )
            biases = np.concatenate(
                [biases, small[2], np.zeros((padding, GROUPS), np.float32)]
            )
            writer.section(pack(codes, scales, biases))
            writer.section(checkpoint.bf16_bytes(prefix + gdn + "conv1d.weight"))
            decay = -np.exp(checkpoint.bf16(prefix + gdn + "A_log").astype(np.float32))
            writer.section(decay.astype("<f4").tobytes())
            writer.section(checkpoint.bf16_bytes(prefix + gdn + "dt_bias"))
            writer.section(checkpoint.bf16_bytes(prefix + gdn + "norm.weight"))
            writer.section(dense_section(gdn + "out_proj"))
        writer.section(
            checkpoint.bf16_bytes(prefix + name + "post_attention_layernorm.weight")
        )
        writer.section(dense_section(f"{name}mlp.gate_proj"))
        writer.section(dense_section(f"{name}mlp.up_proj"))
        writer.section(dense_section(f"{name}mlp.down_proj"))
        writer.finish()

    head = PackedWriter(target / "head.bin", HEAD_MAGIC, HEAD_LAYER_TARGET, 2)
    head.section(checkpoint.bf16_bytes("language_model.model.norm.weight"))
    head.section(pack(*checkpoint.affine("language_model.lm_head")))
    head.finish()

    embedding = PackedWriter(target / "embedding.bin", EMBEDDING_MAGIC, VOCAB, HIDDEN)
    signed, scales, biases = checkpoint.affine("language_model.model.embed_tokens")
    embedding.section(signed.reshape(-1).tobytes())
    if compact:
        embedding.section(bf16_bits(scales.reshape(-1)).tobytes())
        embedding.section(bf16_bits(biases.reshape(-1)).tobytes())
    else:
        embedding.section(scales.reshape(-1).astype("<f4").tobytes())
        embedding.section((biases + 128.0 * scales).reshape(-1).astype("<f4").tobytes())
    embedding.finish()


def artifact_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="existing splash-packed-q4 package root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parameters",
        choices=("bf16", "fp32"),
        default="bf16",
        help="bf16 (s, z) for splash-packed-q8c, or the older fp32 (s, b') "
        "splash-packed-q8",
    )
    parser.add_argument("--target-revision", default="")
    parser.add_argument("--template-revision", default="")
    args = parser.parse_args()

    checkpoint = SafetensorsIndex(args.checkpoint)
    template_manifest = json.loads((args.template / "manifest.json").read_text())
    if template_manifest["format"]["name"] != "splash-packed-q4":
        raise SystemExit("template package must be splash-packed-q4")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    copy_tree(args.template / "draft", args.output / "draft")
    copy_tree(args.template / "vision", args.output / "vision")
    copy_tree(args.template / "tokenizer", args.output / "tokenizer")
    compact = args.parameters == "bf16"
    convert_target(checkpoint, args.output, compact)

    records = artifact_records(args.output)
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['path']}:{record['sha256']}\n".encode())
    upstream = dict(template_manifest.get("upstream", {}))
    upstream["target"] = "mlx-community/Qwen3.8-27B-8bit" + (
        f" @ {args.target_revision}" if args.target_revision else ""
    )
    upstream["template_package"] = "incoai/Qwen3.8-27B-Splash" + (
        f" @ {args.template_revision}" if args.template_revision else ""
    )
    manifest = {
        "schema_version": 5,
        "model": template_manifest["model"],
        "format": {
            "name": "splash-packed-q8c" if compact else "splash-packed-q8",
            "q8_bits": 8,
            "q8_group_size": GROUP,
            "q8_storage_n": STORAGE_N,
            "section_alignment_bytes": ALIGNMENT,
            "target_layer_magic": TARGET_MAGIC.decode(),
            "draft_layer_magic": "MDFD0004",
            "vision_magic": "MDFV0001",
            **({"q8_parameters": "bf16-scale-zero"} if compact else {}),
        },
        "execution_geometry": template_manifest["execution_geometry"],
        "artifacts": records,
        "artifact_set_sha256": digest.hexdigest(),
        "converter": {"script_sha256": sha256(Path(__file__).resolve())},
        "upstream": upstream,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"wrote {len(records)} artifacts to {args.output} "
        f"({sum(r['size'] for r in records)} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
