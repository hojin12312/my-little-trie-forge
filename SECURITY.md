# Security and local deployment

Report a minimal non-sensitive bug description through the repository's
GitHub Issues at https://github.com/hojin12312/my-little-trie-forge/issues.
Do not attach secrets, user prompts, model captures or exploit details publicly;
ask the owner there to coordinate a private channel for sensitive details.
No security email address is provisioned or promised.

Bind defaults to 127.0.0.1. External --host is explicit; use SPLASH_API_KEY
(environment preferred to command-line key), --allowed-host and a trusted
TLS proxy/network boundary. This local inference server is not a hardened
multi-tenant internet service. Standard logs include counts/timings/error types,
not request bodies or credentials. Console stdout/stderr have no automatic
persistent retention; user redirection determines location/retention/deletion.
Crash traces contain diagnostic stacks; review before sharing.

Installed writable runtime/cache roots are outside package files. SSD root
must be private (0700); never share cache across untrusted users. Model
preparation pins revisions, checks safe relative paths and content hashes,
uses safe archive extraction and never enables remote Python code.
Ad-hoc signatures are not Developer ID/notarization. Do not disable system
Gatekeeper globally. Native archive quarantine behavior is not certified;
pip/uv and local Homebrew are locally tested delivery paths.
