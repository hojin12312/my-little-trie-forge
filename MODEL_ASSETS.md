# Model assets

The MLTF code archive is separate from every model artifact. It contains no
target or draft weights, tokenizer, vision assets, packed q8c model package,
private logits, token dumps, or evaluation datasets.

The current validated profile uses these external roles and pinned revisions:

| Role | External source revision |
| --- | --- |
| Target source | `mlx-community/Qwen3.8-27B-8bit@815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` |
| Draft source | `incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4` |
| Tokenizer and vision | `mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff` |
| Template/package reference | `incoai/Qwen3.8-27B-Splash@9d27070b71f7142c6b6025f03ac011d70a73cb48` |

Run `mltf prepare-q8c --work "$HOME/mltf-model-work" --output
"$HOME/mltf-models/q8c" --accept-model-licenses`, then serve the prepared
package with `mltf serve --model local/Qwen3.8-27B-q8c --model-path
"$HOME/mltf-models/q8c"`. The local name is an API label, not a published
model ID. The raw 8-bit target repository is not an installer-ready package.
The public preparation tool pins the revisions above and verifies remote
LFS SHA256/Git blob identities and the template artifact manifest.

The pinned public repositories declare Apache-2.0 in their model cards;
access/license conditions must be reviewed by each user. Draft/tokenizer/vision
are copied from the pinned template and checked against its declared source
roles; this does not independently rederive those packed assets. No weights
are redistributed by this release. Preparation requires at least 110 GiB free for fresh downloads (65 GiB with verified target blob reuse).
Downloads need explicit `--accept-model-licenses`; retries retain work cache
and refuse existing output. Optional `--reuse-target`/`--reuse-template` reuse
only blobs verified against the pinned public remote, with a new work root.
This is distinct from fresh weight network download validation. Each source and derived asset follows its
own licence and notices. Publication of MLTF code does not grant permission
to redistribute any weight, tokenizer, vision, or evaluation asset. No
separate MLTF model package is published with this code candidate.
