"""Cached, non-blocking notification that a newer stable MLTF release exists.

Notification only: nothing is downloaded, installed or executed, no telemetry
is sent, and no failure here may affect serving. One unauthenticated HTTPS GET
of the public GitHub "latest release" metadata is made at most once per TTL.
"""

import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPOSITORY = "hojin12312/my-little-trie-forge"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
API_HOSTS = frozenset({"api.github.com"})
RELEASE_URL_PREFIX = f"https://github.com/{REPOSITORY}/releases/tag/"
HOMEBREW_FORMULA = "hojin12312/mltf/mltf"
DISABLE_ENVIRONMENT = "MLTF_NO_UPDATE_CHECK"

TTL_SECONDS = 24 * 60 * 60
FAILURE_BACKOFF_SECONDS = 60 * 60
CLOCK_SKEW_SECONDS = 5 * 60
# Per-socket-operation limit; DNS resolution is not covered by it, so the
# total deadline below is what bounds how long the state reads "checking".
SOCKET_TIMEOUT_SECONDS = 4.0
TOTAL_DEADLINE_SECONDS = 8.0
MAX_BODY_BYTES = 512 * 1024
CACHE_SCHEMA = 1

_STABLE_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")


class UpdateCheckError(Exception):
    """Non-fatal failure; the message is a short fixed reason code."""


def disabled_by_environment(environ=None):
    value = (os.environ if environ is None else environ).get(DISABLE_ENVIRONMENT, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_stable(text):
    """Version for a strict MAJOR.MINOR.PATCH string, otherwise None."""
    if not isinstance(text, str) or not _STABLE_VERSION.fullmatch(text):
        return None
    return Version(text)


def release_url(version):
    return f"{RELEASE_URL_PREFIX}v{version}"


def is_newer(installed, latest):
    """True only when both versions parse and latest is strictly greater."""
    remote = parse_stable(latest)
    if remote is None:
        return False
    try:
        return Version(installed) < remote
    except (InvalidVersion, TypeError):
        return False


def parse_release(payload):
    """Validated release facts from GitHub's latest-release JSON."""
    if not isinstance(payload, dict):
        raise UpdateCheckError("invalid_response")
    tag = payload.get("tag_name")
    published = payload.get("published_at")
    if (
        payload.get("draft") is not False
        or payload.get("prerelease") is not False
        or not isinstance(tag, str)
        or not tag.startswith("v")
        or not isinstance(published, str)
        or not 0 < len(published) <= 64
    ):
        raise UpdateCheckError("invalid_response")
    version = parse_stable(tag[1:])
    if version is None:
        raise UpdateCheckError("invalid_version")
    # The remote URL is only checked, never used: the displayed link is built
    # from the validated tag so no remote string is ever trusted or followed.
    if payload.get("html_url") != RELEASE_URL_PREFIX + tag:
        raise UpdateCheckError("invalid_response")
    return {
        "latest_version": str(version),
        "release_url": release_url(version),
        "published_at": published,
    }


class _SameHostHttps(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.request.urlsplit(newurl)
        if target.scheme != "https" or target.hostname not in API_HOSTS:
            raise urllib.error.URLError("redirect outside the release API")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_latest(timeout=SOCKET_TIMEOUT_SECONDS, url=API_URL, opener=None):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "MLTF-update-check",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    opener = opener or urllib.request.build_opener(_SameHostHttps)
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise UpdateCheckError(f"http_{response.status}")
            body = response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise UpdateCheckError(f"http_{error.code}") from None
    except TimeoutError:
        raise UpdateCheckError("timeout") from None
    except urllib.error.URLError as error:
        timed_out = isinstance(error.reason, TimeoutError)
        raise UpdateCheckError("timeout" if timed_out else "network") from None
    except (OSError, ValueError):
        raise UpdateCheckError("network") from None
    if len(body) > MAX_BODY_BYTES:
        raise UpdateCheckError("invalid_response")
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        raise UpdateCheckError("invalid_response") from None
    return parse_release(payload)


def _number(value):
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value) if ok and value == value and abs(value) < 1e15 else None


def read_cache(path):
    """Untrusted cache contents: {'release': {...}|None, 'attempt': (t, ok)|None}."""
    empty = {"release": None, "attempt": None}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return empty
    if not isinstance(raw, dict) or raw.get("schema") != CACHE_SCHEMA:
        return empty
    result = dict(empty)
    checked = _number(raw.get("checked_at"))
    version = parse_stable(raw.get("latest_version"))
    published = raw.get("published_at")
    if (
        checked is not None
        and version is not None
        and raw.get("source") == "github_release"
        and raw.get("release_url") == release_url(version)
        and isinstance(published, str)
        and 0 < len(published) <= 64
    ):
        result["release"] = {
            "checked_at": checked,
            "latest_version": str(version),
            "release_url": release_url(version),
            "published_at": published,
        }
    attempt = _number(raw.get("last_attempt_at"))
    if attempt is not None and isinstance(raw.get("last_attempt_ok"), bool):
        result["attempt"] = (attempt, raw["last_attempt_ok"])
    return result


def write_cache(path, release, attempt_at, attempt_ok):
    path = Path(path)
    body = {
        "schema": CACHE_SCHEMA,
        "last_attempt_at": attempt_at,
        "last_attempt_ok": attempt_ok,
    }
    if release is not None:
        body.update(
            checked_at=release["checked_at"],
            latest_version=release["latest_version"],
            release_url=release["release_url"],
            published_at=release["published_at"],
            source="github_release",
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=".update-check-", delete=False
        ) as pending:
            pending_path = Path(pending.name)
            try:
                json.dump(body, pending)
                pending.flush()
                os.fsync(pending.fileno())
                os.replace(pending_path, path)
            finally:
                pending_path.unlink(missing_ok=True)
    except OSError:
        pass


def notice_text(installed, latest, url, channel=None):
    if channel == "homebrew":
        return (
            f"MLTF {latest} is available (installed: {installed}). "
            f'Run "brew upgrade {HOMEBREW_FORMULA}" or see {url}'
        )
    return f"A new MLTF version is available: {latest} (installed: {installed}). See: {url}"


class UpdateChecker:
    def __init__(
        self,
        installed_version,
        cache_path,
        *,
        enabled=True,
        install_channel=None,
        fetch=fetch_latest,
        clock=time.time,
        announce=None,
        deadline=TOTAL_DEADLINE_SECONDS,
    ):
        self.installed_version = installed_version
        self.cache_path = cache_path
        self.install_channel = install_channel
        self._fetch = fetch
        self._clock = clock
        self._announce = announce
        self._deadline = deadline
        self._lock = threading.Lock()
        self._thread = None
        self._started = None
        self._latest = self._url = self._checked_at = self._reason = None
        if not enabled or not installed_version or cache_path is None:
            self._status, self._reason = "disabled", "opt_out" if not enabled else "unconfigured"
        elif not self._valid_installed():
            self._status, self._reason = "unknown", "invalid_installed_version"
        else:
            self._status = "unknown"

    def _valid_installed(self):
        try:
            Version(self.installed_version)
        except (InvalidVersion, TypeError):
            return False
        return True

    def snapshot(self):
        with self._lock:
            self._expire_locked()
            update = self._status == "update_available"
            result = {
                "status": self._status,
                "installed_version": self.installed_version,
                "latest_version": self._latest,
                "update_available": update,
                "checked_at": self._checked_at,
                "release_url": self._url,
            }
            if self._reason is not None:
                result["reason"] = self._reason
            return result

    def _expire_locked(self):
        if self._status == "checking" and time.monotonic() - self._started > self._deadline:
            self._status, self._reason = "unavailable", "timeout"

    def start(self):
        with self._lock:
            if self._thread is not None or self._status != "unknown" or self._reason:
                return
            self._status, self._started = "checking", time.monotonic()
            self._thread = threading.Thread(
                target=self._run, name="mltf-update-check", daemon=True
            )
        self._thread.start()

    def _run(self):
        try:
            self._check()
        except Exception:
            self._finish("unavailable", reason="internal")

    def _check(self):
        now = self._clock()
        cache = read_cache(self.cache_path)
        release = cache["release"]
        if release is not None and -CLOCK_SKEW_SECONDS <= now - release["checked_at"] < TTL_SECONDS:
            self._finish_release(release)
            return
        attempt = cache["attempt"]
        if (
            attempt is not None
            and not attempt[1]
            and -CLOCK_SKEW_SECONDS <= now - attempt[0] < FAILURE_BACKOFF_SECONDS
        ):
            self._finish("unavailable", reason="recent_failure")
            return
        try:
            fetched = self._fetch()
        except UpdateCheckError as error:
            write_cache(self.cache_path, release, now, False)
            self._finish("unavailable", reason=str(error))
            return
        except Exception:
            write_cache(self.cache_path, release, now, False)
            self._finish("unavailable", reason="network")
            return
        fetched = dict(fetched, checked_at=now)
        write_cache(self.cache_path, fetched, now, True)
        self._finish_release(fetched)

    def _finish_release(self, release):
        if is_newer(self.installed_version, release["latest_version"]):
            status = "update_available"
        else:
            status = "up_to_date"
        self._finish(
            status,
            latest=release["latest_version"],
            url=release["release_url"],
            checked_at=release["checked_at"],
        )

    def _finish(self, status, *, reason=None, latest=None, url=None, checked_at=None):
        with self._lock:
            self._expire_locked()
            if self._status != "checking":
                return
            self._status, self._reason = status, reason
            if status in {"up_to_date", "update_available"}:
                self._latest, self._url, self._checked_at = latest, url, checked_at
        if status == "update_available" and self._announce is not None:
            try:
                self._announce(
                    notice_text(self.installed_version, latest, url, self.install_channel)
                )
            except Exception:
                pass


def disabled_snapshot(reason="unconfigured"):
    return {
        "status": "disabled",
        "installed_version": None,
        "latest_version": None,
        "update_available": False,
        "checked_at": None,
        "release_url": None,
        "reason": reason,
    }
