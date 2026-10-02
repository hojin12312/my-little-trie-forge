"""Prepare a local q8c package from pinned public assets without remote code."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

try:
    from . import models
except ImportError:
    import models

TARGET = ("mlx-community/Qwen3.8-27B-8bit", "815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9")
TEMPLATE = ("incoai/Qwen3.8-27B-Splash", "9d27070b71f7142c6b6025f03ac011d70a73cb48")
ROLES = {
    "draft": ("incoai/Qwen3.8-27B-DFlash2", "dedf8df68adfb1afeaf7b7480c0a0243108177b4"),
    "tokenizer": (
        "mlx-community/Qwen3.8-27B-4bit",
        "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
    ),
    "vision": (
        "mlx-community/Qwen3.8-27B-4bit",
        "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
    ),
}


def safe_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError("unsafe external artifact path")
    return p


def run(argv=None):
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--accept-model-licenses", action="store_true")
    ap.add_argument("--reuse-target", type=Path)
    ap.add_argument("--reuse-template", type=Path)
    args = ap.parse_args(argv)
    if not args.accept_model_licenses:
        ap.error(
            "review MODEL_ASSETS.md and pass --accept-model-licenses before downloading"
        )
    work = args.work.resolve()
    output = args.output.resolve()
    if output.exists():
        ap.error("output already exists; preserve it and choose a new output")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    minimum_gib = 65 if args.reuse_target is not None else 110
    if shutil.disk_usage(work).free < minimum_gib * 1024**3:
        ap.error(
            f"need at least {minimum_gib} GiB free; fresh downloads retain Hub cache and staging copies"
        )
    api = HfApi(token=False)
    log = {
        "roles": {},
        "files": [],
        "network_weight_download": False,
        "status": "RUNNING",
    }

    def fetch(repo, revision, names, dest, reuse):
        info = api.model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision or info.private or info.gated:
            raise ValueError("public revision/access mismatch: " + repo)
        log["roles"][repo] = {
            "revision": info.sha,
            "license": info.card_data.get("license") if info.card_data else None,
        }
        siblings = {x.rfilename: x for x in info.siblings}
        for name in names:
            safe_path(name)
            remote = siblings[name]
            dst = dest / name
            dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            original = reuse / name if reuse else None
            if reuse is not None and (not original.is_file() or original.is_symlink()):
                raise ValueError(
                    "requested verified reuse is incomplete; use a complete source directory or fresh download with 110 GiB: "
                    + name
                )
            reused = False
            if original and original.is_file() and not original.is_symlink():
                reused = True
                src = original
            else:
                src = Path(
                    hf_hub_download(
                        repo,
                        name,
                        revision=revision,
                        cache_dir=work / "hub",
                        token=False,
                    )
                )
                if remote.lfs:
                    log["network_weight_download"] = True
            if src.stat().st_size != remote.size:
                raise ValueError("external size mismatch: " + name)
            digest = models.sha256(src)
            if remote.lfs:
                if digest != remote.lfs.sha256:
                    raise ValueError("external SHA256 mismatch: " + name)
            else:
                data = src.read_bytes()
                blob = hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                if blob != remote.blob_id:
                    raise ValueError("external Git blob mismatch: " + name)
            if dst.exists():
                if models.sha256(dst) != digest:
                    raise ValueError("work cache mismatch: " + name)
            else:
                # Immutable verified source reused with a new install/cache root.
                shutil.copy2(src, dst)
            log["files"].append(
                {
                    "repository": repo,
                    "revision": revision,
                    "path": name,
                    "sha256": digest,
                    "size": remote.size,
                    "reused_verified_blob": reused,
                }
            )

    try:
        for repo, revision in ROLES.values():
            info = api.model_info(repo, revision=revision, token=False)
            if info.sha != revision or info.private or info.gated:
                raise ValueError("role revision/access mismatch")
        target = work / "target-source"
        template = work / "template-source"
        info = api.model_info(TARGET[0], revision=TARGET[1], files_metadata=True)
        names = [
            s.rfilename
            for s in info.siblings
            if s.rfilename.endswith(".safetensors")
            or s.rfilename in ("model.safetensors.index.json", "config.json")
        ]
        fetch(*TARGET, names, target, args.reuse_target)
        fetch(*TEMPLATE, ["manifest.json"], template, args.reuse_template)
        manifest = models.validate_package_manifest(template / "manifest.json")
        for role, (repo, revision) in ROLES.items():
            if manifest["upstream"][role] != {"repo_id": repo, "revision": revision}:
                raise ValueError("template role provenance mismatch")
        names = [
            r["path"]
            for r in manifest["artifacts"]
            if r["path"].startswith(("draft/", "vision/", "tokenizer/"))
        ]
        fetch(*TEMPLATE, names, template, args.reuse_template)
        for record in manifest["artifacts"]:
            if (
                record["path"] in names
                and models.sha256(template / record["path"]) != record["sha256"]
            ):
                raise ValueError("template artifact mismatch")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = Path(tempfile.mkdtemp(prefix=".q8c-prepare-", dir=output.parent))
        log["temporary_output"] = temporary.name
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("convert_qwen38_q8.py")),
                "--checkpoint",
                str(target),
                "--template",
                str(template),
                "--output",
                str(temporary),
                "--target-revision",
                TARGET[1],
                "--template-revision",
                TEMPLATE[1],
            ],
            check=True,
        )
        result = models.validate_package_manifest(temporary / "manifest.json")
        models.verify_artifacts(temporary, result, full=True)
        os.chmod(temporary, 0o700)
        temporary.rename(output)
        log["status"] = "PASS"
        log["output_manifest_sha256"] = models.sha256(output / "manifest.json")
        print(
            "Prepared q8c package. Serve with --model local/Qwen3.8-27B-q8c --model-path",
            output,
        )
        return 0
    finally:
        (work / "preparation.json").write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(run())
