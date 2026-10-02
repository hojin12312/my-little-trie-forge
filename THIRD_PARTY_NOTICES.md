# Third-party notices — MLTF code archive

MLTF derives from incoai/splash under Apache-2.0. See `LICENSE` and
`UPSTREAM.md`. The oMLX SSD-sidecar work (jundot/omlx#2569, Apache-2.0)
informed MLTF's durable state/KV cache algorithm and failure scenarios.
Analogous native paths include `runtime/engine/RestoreBundle.cpp`,
`StorageCoordinator.hpp`, `RuntimeResources.mm`, and
`runtime/model/QwenStateStore.cpp`. This is conservatively attributed as
algorithm adaptation; no oMLX Python package is included. The inherited
`runtime/metal/kernels/common/q4_sgmatrix.h` acknowledges an MLX steel
design pattern. MLX/MLX-LM evaluation use is separate from runtime
dependency and source inclusion. `server/judgments.py` retains its
TheoLeeCJ MIT copyright and licence text in source.

The bundled Python 3.13 distribution carries `python/lib/python3.13/LICENSE.txt`.
The following 41 pinned Python distributions ship with the code-only archive.
Paths are relative to the archive root. Where a row notes additional licence
files, they remain in that distribution's metadata directory.

| Distribution | Version | Wheel metadata licence expression | Licence text in archive |
| --- | --- | --- | --- |
| annotated-doc | 0.0.5 | MIT | `python/lib/python3.13/site-packages/annotated_doc-0.0.5.dist-info/licenses` (+1 additional licence files) |
| anyio | 4.14.2 | MIT | `python/lib/python3.13/site-packages/anyio-4.14.2.dist-info/licenses` (+1 additional licence files) |
| attrs | 26.1.0 | MIT | `python/lib/python3.13/site-packages/attrs-26.1.0.dist-info/licenses` (+1 additional licence files) |
| certifi | 2026.7.22 | MPL-2.0 | `python/lib/python3.13/site-packages/certifi-2026.7.22.dist-info/licenses` (+1 additional licence files) |
| cffi | 2.1.1 | MIT-0 | `python/lib/python3.13/site-packages/cffi-2.1.1.dist-info/licenses` (+1 additional licence files) |
| click | 8.4.2 | BSD-3-Clause | `python/lib/python3.13/site-packages/click-8.4.2.dist-info/licenses` (+1 additional licence files) |
| cryptography | 50.0.1 | Apache-2.0 OR BSD-3-Clause | `python/lib/python3.13/site-packages/cryptography-50.0.1.dist-info/licenses` (+3 additional licence files) |
| filelock | 3.32.3 | MIT | `python/lib/python3.13/site-packages/filelock-3.32.3.dist-info/licenses` (+1 additional licence files) |
| fsspec | 2026.7.0 | BSD-3-Clause | `python/lib/python3.13/site-packages/fsspec-2026.7.0.dist-info/licenses` (+1 additional licence files) |
| h11 | 0.16.0 | MIT | `python/lib/python3.13/site-packages/h11-0.16.0.dist-info/licenses` (+1 additional licence files) |
| hf-xet | 1.6.0 | Apache-2.0 | `python/lib/python3.13/site-packages/hf_xet-1.6.0.dist-info/licenses` (+1 additional licence files) |
| httpcore | 1.0.9 | BSD-3-Clause | `python/lib/python3.13/site-packages/httpcore-1.0.9.dist-info/licenses` (+1 additional licence files) |
| httpx | 0.28.1 | BSD-3-Clause | `python/lib/python3.13/site-packages/httpx-0.28.1.dist-info/licenses` (+1 additional licence files) |
| huggingface_hub | 1.28.0 | Apache-2.0 | `python/lib/python3.13/site-packages/huggingface_hub-1.28.0.dist-info/licenses` (+1 additional licence files) |
| idna | 3.19 | BSD-3-Clause | `python/lib/python3.13/site-packages/idna-3.19.dist-info/licenses` (+1 additional licence files) |
| Jinja2 | 3.1.6 | Not declared as an expression | `python/lib/python3.13/site-packages/jinja2-3.1.6.dist-info/licenses` (+1 additional licence files) |
| jsonschema | 4.26.0 | MIT | `python/lib/python3.13/site-packages/jsonschema-4.26.0.dist-info/licenses` (+1 additional licence files) |
| jsonschema-specifications | 2025.9.1 | MIT | `python/lib/python3.13/site-packages/jsonschema_specifications-2025.9.1.dist-info/licenses` (+1 additional licence files) |
| llguidance | 1.8.0 | MIT | `python/lib/python3.13/site-packages/llguidance-1.8.0.dist-info/licenses` (+1 additional licence files) |
| markdown-it-py | 4.2.0 | Not declared as an expression | `python/lib/python3.13/site-packages/markdown_it_py-4.2.0.dist-info/licenses` (+2 additional licence files) |
| MarkupSafe | 3.0.3 | BSD-3-Clause | `python/lib/python3.13/site-packages/markupsafe-3.0.3.dist-info/licenses` (+1 additional licence files) |
| mdurl | 0.1.2 | Not declared as an expression | `python/lib/python3.13/site-packages/mdurl-0.1.2.dist-info/LICENSE` |
| numpy | 2.5.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | `python/lib/python3.13/site-packages/numpy-2.5.2.dist-info/licenses` (+17 additional licence files) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | `python/lib/python3.13/site-packages/packaging-26.3.dist-info/licenses` (+3 additional licence files) |
| Pillow | 12.3.0 | MIT-CMU | `python/lib/python3.13/site-packages/pillow-12.3.0.dist-info/licenses` (+1 additional licence files) |
| pip | 26.2.1 | MIT | `python/lib/python3.13/site-packages/pip-26.2.1.dist-info/licenses` (+21 additional licence files) |
| Pygments | 2.21.0 | BSD-2-Clause | `python/lib/python3.13/site-packages/pygments-2.21.0.dist-info/licenses` (+1 additional licence files) |
| pycparser | 3.0 | BSD-3-Clause | `python/lib/python3.13/site-packages/pycparser-3.0.dist-info/licenses` (+1 additional licence files) |
| pypdfium2 | 5.13.0 | BSD-3-Clause, Apache-2.0, dependency licenses | `python/lib/python3.13/site-packages/pypdfium2-5.13.0.dist-info/licenses` (+2 additional licence files) |
| PyYAML | 6.0.3 | MIT | `python/lib/python3.13/site-packages/pyyaml-6.0.3.dist-info/licenses` (+1 additional licence files) |
| referencing | 0.37.0 | MIT | `python/lib/python3.13/site-packages/referencing-0.37.0.dist-info/licenses` (+1 additional licence files) |
| regex | 2026.7.19 | Apache-2.0 AND CNRI-Python | `python/lib/python3.13/site-packages/regex-2026.7.19.dist-info/licenses` (+1 additional licence files) |
| rich | 15.0.0 | MIT | `python/lib/python3.13/site-packages/rich-15.0.0.dist-info/licenses` (+1 additional licence files) |
| rpds-py | 2026.6.3 | MIT | `python/lib/python3.13/site-packages/rpds_py-2026.6.3.dist-info/licenses` (+1 additional licence files) |
| safetensors | 0.8.0 | Not declared as an expression | `python/lib/python3.13/site-packages/safetensors-0.8.0.dist-info/licenses` (+1 additional licence files) |
| shellingham | 1.5.4 | ISC License | `python/lib/python3.13/site-packages/shellingham-1.5.4.dist-info/LICENSE` |
| tokenizers | 0.22.2 | Not declared as an expression | `third_party_licenses/tokenizers-0.22.2-LICENSE` (exact v0.22.2 source tag) |
| tqdm | 4.70.0 | MPL-2.0 AND MIT | `python/lib/python3.13/site-packages/tqdm-4.70.0.dist-info/licenses` |
| transformers | 5.15.1 | Apache 2.0 License | `python/lib/python3.13/site-packages/transformers-5.15.1.dist-info/licenses` (+1 additional licence files) |
| typer | 0.27.1 | MIT | `python/lib/python3.13/site-packages/typer-0.27.1.dist-info/licenses` (+1 additional licence files) |
| typing_extensions | 4.16.0 | PSF-2.0 | `python/lib/python3.13/site-packages/typing_extensions-4.16.0.dist-info/licenses` (+1 additional licence files) |

The tokenizers 0.22.2 wheel has no recorded licence file. Its exact-tag
upstream `LICENSE` (Apache-2.0 text; SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`)
is included separately. Five wheels lack a machine-readable licence
expression; their supplied text and classifier evidence are recorded in the
private dependency audit. The archive audit must check every indexed path
against the actual archive before publication.

No model weights, tokenizer assets, vision assets, packed models or evaluation datasets are bundled; see `MODEL_ASSETS.md`. Public code, model and data permissions are separate decisions.
