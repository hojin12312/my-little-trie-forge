#!/usr/bin/env python3
"""Build or validate a Homebrew bottle from local release artifacts."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def brew(*args, capture=False, **kwargs):
    return subprocess.run(
        ["brew", *map(str, args)],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        env={
            **os.environ,
            "HOMEBREW_NO_AUTO_UPDATE": "1",
            "HOMEBREW_NO_INSTALL_CLEANUP": "1",
        },
        **kwargs,
    ).stdout


def seed_cache(formula, artifact, option):
    # Keep published URLs intact; Homebrew still verifies the cached checksum.
    cache = Path(brew("--cache", option, formula, capture=True).strip())
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact, cache)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    args = parser.parse_args(argv)
    dist = args.dist.resolve()
    formula_file = dist / "mltf.rb"
    if not formula_file.is_file():
        parser.error(f"missing {formula_file}; run make package first")
    cellar = Path(brew("--cellar", capture=True).strip())
    prefix = Path(brew("--prefix", capture=True).strip())
    if (cellar / "mltf").exists() or os.path.lexists(prefix / "bin/mltf"):
        parser.error(
            "MLTF is already installed; use a release machine without an existing installation"
        )

    tap = f"mltf-check/package-{uuid.uuid4().hex[:8]}"
    name = f"{tap}/mltf"
    brew("tap-new", "--no-git", tap)
    try:
        tap_formula = (
            Path(brew("--repository", tap, capture=True).strip()) / "Formula/mltf.rb"
        )
        shutil.copyfile(formula_file, tap_formula)
        info = json.loads(brew("info", "--json=v2", name, capture=True))["formulae"][0]
        if info["versions"]["stable"] != args.version:
            parser.error("formula version does not match --version")

        if args.action == "build":
            archive = dist / f"mltf-{args.version}-macos-arm64.tar.gz"
            seed_cache(name, archive, "--build-from-source")
            brew("install", "--build-bottle", name)
            brew("test", name)
            root_url = info["urls"]["stable"]["url"].rsplit("/", 1)[0]
            with tempfile.TemporaryDirectory(prefix=".bottle-", dir=dist) as temporary:
                staging = Path(temporary)
                brew(
                    "bottle",
                    "--json",
                    "--no-rebuild",
                    f"--root-url={root_url}",
                    name,
                    cwd=staging,
                )
                (metadata,) = staging.glob("*.bottle.json")
                bottle = json.loads(metadata.read_text())[name]["bottle"]
                for tag in bottle["tags"].values():
                    output = dist / unquote(tag["filename"])
                    if output.exists():
                        parser.error(f"release bottle already exists: {output}")
                brew("bottle", "--merge", "--write", "--no-commit", metadata)
                for tag in bottle["tags"].values():
                    (staging / tag["local_filename"]).replace(
                        dist / unquote(tag["filename"])
                    )
                metadata.replace(dist / metadata.name)
                shutil.copyfile(tap_formula, formula_file)
            print(
                "Bottle built; publish it alongside the runtime archive and updated formula."
            )
        else:
            bottles = info.get("bottle", {}).get("stable", {}).get("files", {})
            if not bottles:
                parser.error("formula has no bottle; run make package-bottle first")
            # Cache each supplied platform so Homebrew can choose normally,
            # including an older compatible bottle on a newer macOS release.
            for tag, details in bottles.items():
                artifact = dist / unquote(Path(urlsplit(details["url"]).path).name)
                if artifact.is_file():
                    seed_cache(name, artifact, f"--bottle-tag={tag}")
            selected = Path(
                brew("--cache", "--force-bottle", name, capture=True).strip()
            )
            if not selected.is_file():
                parser.error(
                    "no local bottle for this platform; transfer the matching bottle first"
                )
            brew("install", "--formula", name)
            installed = Path(brew("--prefix", name, capture=True).strip())
            receipt = json.loads((installed / "INSTALL_RECEIPT.json").read_text())
            if not receipt.get("poured_from_bottle"):
                raise RuntimeError("Homebrew fell back to a source installation")
            # brew test sets up a developer build environment. Validate the
            # user's entry point directly so this check needs no toolchain.
            help_text = subprocess.check_output(
                [prefix / "bin/mltf", "--help"], text=True
            )
            if "serve" not in help_text:
                raise RuntimeError("installed launcher did not list serve")
            print(
                "Package check passed: Homebrew poured the bottle and the bundled launcher works."
            )
    finally:
        # The preflight above ensures these can only be our installation.
        try:
            if (cellar / "mltf").exists():
                brew("uninstall", "--formula", name)
        finally:
            brew("untap", tap)


if __name__ == "__main__":
    main()
