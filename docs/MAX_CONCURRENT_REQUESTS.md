# Active request limit

`--max-concurrent-requests` is an opt-in startup setting added in MLTF 0.1.2; MLTF 0.1.1 does not include it.

```sh
mltf serve --model local/Qwen3.8-27B-q8c --model-path "$HOME/mltf-models/q8c" --max-context 256K --max-memory 96G --max-concurrent-requests 4
```

Supported values are 1, 2, 3 and 4. Omitting the option keeps the existing automatic concurrency; it is not silently mapped to a fixed limit. The chosen limit applies from startup and changing it requires a server restart.

The limit counts logical generation HTTP requests — from generation preparation through response completion — across the generation endpoints. Requests beyond the limit wait FIFO inside the existing ingress, connection and request-body safety limits; saturation of those limits can still reject requests as before. Queue time counts inside the request deadline. Queued disconnects and expirations leave no ticket behind, submission failures and errors release the slot exactly once, and server shutdown releases waiters.

`/status` reports `http.generation_requests.active`, `.waiting` and `.capacity` only when the option is set. Those are logical HTTP counts, separate from native scheduler fields such as `decoding` and `decode_batches_by_width`. Tokenization and control endpoints keep their own capacity and do not wait behind generation.

This controls logical requests, not physical decode width B. A limit of 4 does not wait for four requests to arrive, does not duplicate work, and does not force a native batch width — memory and state limits can still pick a smaller batch, and speculative draft/verify rows are not counted. It does not change native/Metal kernels, scheduler, wire protocol, KV layout or B1–B4 execution geometry.
