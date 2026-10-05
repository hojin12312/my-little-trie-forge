import http.client
import http.server
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from install import launcher, paths
from server import update_check as uc
from server.server import FrontendServer
from server.update_check import UpdateChecker, UpdateCheckError

ROOT = Path(__file__).resolve().parents[1]
TAG_URL = uc.RELEASE_URL_PREFIX + "v0.1.3"


def release_payload(**overrides):
    payload = {
        "tag_name": "v0.1.3",
        "name": "MLTF 0.1.3",
        "html_url": TAG_URL,
        "published_at": "2026-10-20T00:00:00Z",
        "draft": False,
        "prerelease": False,
        "assets": [{"name": "ignored.tar.gz"}],
    }
    payload.update(overrides)
    return payload


def fake_release(version="0.1.3"):
    return lambda: {
        "latest_version": version,
        "release_url": uc.release_url(version),
        "published_at": "2026-10-20T00:00:00Z",
    }


def wait_for(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def run_checker(installed, cache, fetch, **kwargs):
    notices = []
    checker = UpdateChecker(
        installed, cache, fetch=fetch, announce=notices.append, **kwargs
    )
    checker.start()
    wait_for(lambda: checker.snapshot()["status"] != "checking")
    return checker, notices


class TempCase(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.cache = self.directory / "update" / "update-check.json"


class VersionTests(unittest.TestCase):
    def test_comparisons(self):
        for installed, latest, newer in (
            ("0.1.1", "0.1.2", True),
            ("0.1.9", "0.1.10", True),
            ("0.1.99", "0.2.0", True),
            ("0.99.99", "1.0.0", True),
            ("0.1.2", "0.1.2", False),
            ("0.1.2", "0.1.1", False),
            ("0.1.10", "0.1.9", False),
            ("1.0.0", "0.99.99", False),
            ("0.1.3rc1", "0.1.3", True),
            ("0.1.3.dev1", "0.1.2", False),
        ):
            self.assertIs(uc.is_newer(installed, latest), newer, (installed, latest))

    def test_invalid_versions_never_report_an_update(self):
        for bad in ("", "latest", "v0.1.3", "0.1", "0.1.3-beta", None, 3):
            self.assertFalse(uc.is_newer("0.1.2", bad))
            if bad in ("", None, 3, "latest"):
                self.assertFalse(uc.is_newer(bad, "0.1.3"))

    def test_remote_version_is_strict_semver(self):
        self.assertEqual(str(uc.parse_stable("0.1.10")), "0.1.10")
        for bad in ("0.1.3rc1", "01.1.1", "1.2", "1.2.3.4", "1.2.3 ", "v1.2.3", "1.2.x", 1):
            self.assertIsNone(uc.parse_stable(bad), bad)


class ReleaseParsingTests(unittest.TestCase):
    def test_stable_release(self):
        release = uc.parse_release(release_payload())
        self.assertEqual(release["latest_version"], "0.1.3")
        self.assertEqual(release["release_url"], TAG_URL)

    def test_rejections(self):
        for override in (
            {"prerelease": True},
            {"draft": True},
            {"prerelease": None},
            {"tag_name": "v0.1.3-rc1", "html_url": uc.RELEASE_URL_PREFIX + "v0.1.3-rc1"},
            {"tag_name": "0.1.3"},
            {"tag_name": 7},
            {"published_at": None},
            {"html_url": "https://evil.example/releases/tag/v0.1.3"},
            {"html_url": "https://github.com/other/repo/releases/tag/v0.1.3"},
            {"html_url": None},
        ):
            with self.assertRaises(UpdateCheckError, msg=override):
                uc.parse_release(release_payload(**override))
        for payload in ([], "text", None, {}):
            with self.assertRaises(UpdateCheckError):
                uc.parse_release(payload)


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, result):
        self.result = result

    def open(self, request, timeout=None):
        self.request, self.timeout = request, timeout
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FetchTests(unittest.TestCase):
    def fetch(self, result):
        return uc.fetch_latest(opener=FakeOpener(result))

    def reason(self, result):
        with self.assertRaises(UpdateCheckError) as caught:
            self.fetch(result)
        return str(caught.exception)

    def test_success_and_request_shape(self):
        opener = FakeOpener(FakeResponse(json.dumps(release_payload()).encode()))
        self.assertEqual(uc.fetch_latest(opener=opener)["latest_version"], "0.1.3")
        headers = {k.lower(): v for k, v in opener.request.header_items()}
        self.assertEqual(
            set(headers), {"accept", "user-agent", "x-github-api-version"}
        )
        self.assertEqual(opener.request.full_url, uc.API_URL)
        self.assertEqual(opener.request.get_method(), "GET")
        self.assertIsNone(opener.request.data)
        self.assertLessEqual(opener.timeout, 5)
        self.assertLess(opener.timeout, uc.TOTAL_DEADLINE_SECONDS)

    def test_http_status_failures(self):
        for code in (403, 404, 429, 500, 503):
            error = urllib.error.HTTPError(uc.API_URL, code, "x", {}, None)
            self.assertEqual(self.reason(error), f"http_{code}")

    def test_transport_failures(self):
        self.assertEqual(self.reason(TimeoutError()), "timeout")
        self.assertEqual(self.reason(urllib.error.URLError("dns")), "network")
        self.assertEqual(self.reason(urllib.error.URLError(TimeoutError())), "timeout")
        self.assertEqual(
            self.reason(urllib.error.URLError(ssl.SSLCertVerificationError("bad"))),
            "network",
        )
        self.assertEqual(self.reason(ssl.SSLError("tls")), "network")
        self.assertEqual(self.reason(ConnectionResetError()), "network")

    def test_bad_bodies(self):
        self.assertEqual(self.reason(FakeResponse(b"{not json")), "invalid_response")
        self.assertEqual(self.reason(FakeResponse(b"[]")), "invalid_response")
        self.assertEqual(self.reason(FakeResponse(b"{}")), "invalid_response")
        self.assertEqual(self.reason(FakeResponse(b"\xff\xfe")), "invalid_response")
        big = b"[" + b"1," * uc.MAX_BODY_BYTES + b"1]"
        self.assertEqual(self.reason(FakeResponse(big)), "invalid_response")
        self.assertEqual(self.reason(FakeResponse(b"{}", status=204)), "http_204")


class LocalServerTests(unittest.TestCase):
    """Real sockets against a local stand-in, never the real GitHub."""

    def serve(self, handler):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/latest"

    def test_headers_and_success(self):
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.update(self.headers)
                body = json.dumps(release_payload()).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with patch.dict(os.environ, {"GITHUB_TOKEN": "secret", "SPLASH_API_KEY": "k"}):
            release = uc.fetch_latest(url=self.serve(Handler))
        self.assertEqual(release["latest_version"], "0.1.3")
        self.assertEqual(seen["User-Agent"], "MLTF-update-check")
        for name in seen:
            self.assertNotIn(name.lower(), {"authorization", "cookie"})
        self.assertNotIn("secret", json.dumps(seen))

    def test_read_timeout_is_bounded(self):
        release = threading.Event()
        self.addCleanup(release.set)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                release.wait(5)

            def log_message(self, *args):
                pass

        started = time.monotonic()
        with self.assertRaises(UpdateCheckError) as caught:
            uc.fetch_latest(timeout=0.3, url=self.serve(Handler))
        self.assertEqual(str(caught.exception), "timeout")
        self.assertLess(time.monotonic() - started, 3)

    def test_redirect_away_from_the_release_api_is_refused(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        with self.assertRaises(UpdateCheckError) as caught:
            uc.fetch_latest(url=self.serve(Handler))
        self.assertEqual(str(caught.exception), "network")


class CacheTests(TempCase):
    def test_roundtrip_and_atomic_write(self):
        release = {
            "checked_at": 1000.0,
            "latest_version": "0.1.3",
            "release_url": TAG_URL,
            "published_at": "2026-10-20T00:00:00Z",
        }
        with patch("server.update_check.os.replace", wraps=os.replace) as replace:
            uc.write_cache(self.cache, release, 1000.0, True)
        self.assertEqual(replace.call_count, 1)
        source, target = replace.call_args.args
        self.assertEqual(Path(target), self.cache)
        self.assertEqual(Path(source).parent, self.cache.parent)
        self.assertEqual([p.name for p in self.cache.parent.iterdir()], [self.cache.name])
        loaded = uc.read_cache(self.cache)
        self.assertEqual(loaded["release"], release)
        self.assertEqual(loaded["attempt"], (1000.0, True))

    def test_failed_replace_keeps_old_cache_and_leaves_no_temp(self):
        uc.write_cache(self.cache, None, 1.0, False)
        before = self.cache.read_bytes()
        with patch("server.update_check.os.replace", side_effect=OSError("disk")):
            uc.write_cache(self.cache, None, 2.0, True)
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertEqual([p.name for p in self.cache.parent.iterdir()], [self.cache.name])

    def test_unwritable_location_is_silent(self):
        blocker = self.directory / "file"
        blocker.write_text("x")
        uc.write_cache(blocker / "sub" / "c.json", None, 1.0, True)

    def test_untrusted_contents_are_ignored(self):
        self.cache.parent.mkdir(parents=True)
        good = {
            "schema": 1,
            "checked_at": 5.0,
            "latest_version": "0.1.3",
            "release_url": TAG_URL,
            "published_at": "p",
            "source": "github_release",
            "last_attempt_at": 5.0,
            "last_attempt_ok": True,
        }
        self.cache.write_text(json.dumps(good))
        self.assertIsNotNone(uc.read_cache(self.cache)["release"])
        bad_variants = [
            {**good, "schema": 0},
            {**good, "schema": 2},
            {**good, "latest_version": "0.1.3rc1"},
            {**good, "release_url": "https://evil.example/x"},
            {**good, "source": "other"},
            {**good, "checked_at": "now"},
            {**good, "checked_at": True},
            {**good, "checked_at": float("nan")},
            {k: v for k, v in good.items() if k != "published_at"},
        ]
        for variant in bad_variants:
            self.cache.write_text(json.dumps(variant))
            self.assertIsNone(uc.read_cache(self.cache)["release"], variant)
        for text in ("", "{", "[]", "null", '"x"', "\x00\x01", "{" * 100000):
            self.cache.write_text(text)
            self.assertEqual(uc.read_cache(self.cache), {"release": None, "attempt": None})
        self.cache.write_bytes(b"\xff\xfe")
        self.assertIsNone(uc.read_cache(self.cache)["release"])


class CheckerTests(TempCase):
    def test_no_cache_fetches_and_reports_update_once(self):
        calls = []

        def fetch():
            calls.append(1)
            return fake_release()()

        checker, notices = run_checker("0.1.2", self.cache, fetch, clock=lambda: 1000.0)
        snapshot = checker.snapshot()
        self.assertEqual(snapshot["status"], "update_available")
        self.assertTrue(snapshot["update_available"])
        self.assertEqual(snapshot["latest_version"], "0.1.3")
        self.assertEqual(snapshot["installed_version"], "0.1.2")
        self.assertEqual(snapshot["release_url"], TAG_URL)
        self.assertEqual(snapshot["checked_at"], 1000.0)
        self.assertNotIn("reason", snapshot)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(notices), 1)
        self.assertIn("0.1.3", notices[0])
        self.assertIn("0.1.2", notices[0])
        self.assertIn(TAG_URL, notices[0])
        self.assertNotIn("\n", notices[0])
        self.assertNotIn("brew", notices[0])
        self.assertTrue(self.cache.is_file())

    def test_up_to_date_and_newer_installed_are_silent(self):
        for installed, latest, status in (
            ("0.1.2", "0.1.2", "up_to_date"),
            ("0.1.2", "0.1.1", "up_to_date"),
            ("0.1.10", "0.1.9", "up_to_date"),
        ):
            with self.subTest(installed=installed, latest=latest):
                cache = self.directory / f"{installed}-{latest}.json"
                checker, notices = run_checker(
                    installed, cache, fake_release(latest), clock=lambda: 10.0
                )
                snapshot = checker.snapshot()
                self.assertEqual(snapshot["status"], status)
                self.assertFalse(snapshot["update_available"])
                self.assertEqual(notices, [])

    def test_fresh_cache_makes_no_network_call(self):
        run_checker("0.1.2", self.cache, fake_release(), clock=lambda: 1000.0)

        def forbidden():
            raise AssertionError("network used")

        later = 1000.0 + uc.TTL_SECONDS - 1
        checker, notices = run_checker("0.1.2", self.cache, forbidden, clock=lambda: later)
        self.assertEqual(checker.snapshot()["status"], "update_available")
        self.assertEqual(checker.snapshot()["checked_at"], 1000.0)
        self.assertEqual(len(notices), 1)

    def test_stale_cache_refreshes_and_is_not_presented_as_fresh(self):
        run_checker("0.1.2", self.cache, fake_release("0.1.3"), clock=lambda: 1000.0)
        later = 1000.0 + uc.TTL_SECONDS
        checker, _ = run_checker("0.1.2", self.cache, fake_release("0.1.4"), clock=lambda: later)
        snapshot = checker.snapshot()
        self.assertEqual(snapshot["latest_version"], "0.1.4")
        self.assertEqual(snapshot["checked_at"], later)

    def test_stale_cache_with_failed_refresh_is_unavailable_not_stale_data(self):
        run_checker("0.1.2", self.cache, fake_release("0.1.3"), clock=lambda: 1000.0)

        def failing():
            raise UpdateCheckError("network")

        later = 1000.0 + uc.TTL_SECONDS + 1
        checker, notices = run_checker("0.1.2", self.cache, failing, clock=lambda: later)
        snapshot = checker.snapshot()
        self.assertEqual(snapshot["status"], "unavailable")
        self.assertIsNone(snapshot["latest_version"])
        self.assertEqual(notices, [])

    def test_malformed_and_old_schema_cache_refresh_safely(self):
        for text in ("{broken", json.dumps({"schema": 0, "checked_at": 1}), "[]"):
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_text(text)
            checker, _ = run_checker("0.1.2", self.cache, fake_release(), clock=lambda: 50.0)
            self.assertEqual(checker.snapshot()["status"], "update_available")
            self.assertEqual(uc.read_cache(self.cache)["release"]["checked_at"], 50.0)

    def test_clock_anomalies(self):
        run_checker("0.1.2", self.cache, fake_release("0.1.3"), clock=lambda: 10_000_000.0)
        # Small backwards step is tolerated.
        checker, _ = run_checker(
            "0.1.2", self.cache, lambda: 1 / 0, clock=lambda: 10_000_000.0 - 60
        )
        self.assertEqual(checker.snapshot()["status"], "update_available")
        # A cache written far in the future is not trusted forever.
        calls = []

        def fetch():
            calls.append(1)
            return fake_release("0.1.4")()

        checker, _ = run_checker("0.1.2", self.cache, fetch, clock=lambda: 1000.0)
        self.assertEqual(calls, [1])
        self.assertEqual(checker.snapshot()["latest_version"], "0.1.4")

    def test_failures_back_off_instead_of_retrying_every_launch(self):
        calls = []

        def failing():
            calls.append(1)
            raise UpdateCheckError("http_403")

        checker, notices = run_checker("0.1.2", self.cache, failing, clock=lambda: 100.0)
        self.assertEqual(checker.snapshot()["status"], "unavailable")
        self.assertEqual(checker.snapshot()["reason"], "http_403")
        again, _ = run_checker("0.1.2", self.cache, failing, clock=lambda: 200.0)
        self.assertEqual(again.snapshot()["reason"], "recent_failure")
        self.assertEqual(len(calls), 1)
        later = 100.0 + uc.FAILURE_BACKOFF_SECONDS
        run_checker("0.1.2", self.cache, failing, clock=lambda: later)
        self.assertEqual(len(calls), 2)
        self.assertEqual(notices, [])

    def test_every_failure_mode_is_unavailable_and_quiet(self):
        for reason in ("http_403", "http_404", "http_429", "http_500", "timeout", "network", "invalid_response"):
            cache = self.directory / f"{reason}.json"

            def failing(reason=reason):
                raise UpdateCheckError(reason)

            checker, notices = run_checker("0.1.2", cache, failing)
            snapshot = checker.snapshot()
            self.assertEqual((snapshot["status"], snapshot["reason"]), ("unavailable", reason))
            self.assertFalse(snapshot["update_available"])
            self.assertEqual(notices, [])

    def test_unexpected_exceptions_do_not_escape_or_leak(self):
        def boom():
            raise RuntimeError("secret /Users/someone/path")

        checker, _ = run_checker("0.1.2", self.cache, boom)
        snapshot = checker.snapshot()
        self.assertEqual(snapshot["status"], "unavailable")
        self.assertNotIn("secret", json.dumps(snapshot))
        self.assertNotIn("/Users", json.dumps(snapshot))

    def test_announce_failure_does_not_matter(self):
        def announce(text):
            raise OSError("closed stdout")

        checker = UpdateChecker("0.1.2", self.cache, fetch=fake_release(), announce=announce)
        checker.start()
        self.assertTrue(wait_for(lambda: checker.snapshot()["status"] == "update_available"))

    def test_invalid_remote_version_from_fetch_is_not_an_update(self):
        checker, notices = run_checker("0.1.2", self.cache, fake_release("junk"))
        self.assertNotEqual(checker.snapshot()["status"], "update_available")
        self.assertEqual(notices, [])

    def test_homebrew_notice_is_only_for_detected_homebrew(self):
        _, plain = run_checker("0.1.2", self.directory / "a.json", fake_release())
        _, brew = run_checker(
            "0.1.2", self.directory / "b.json", fake_release(), install_channel="homebrew"
        )
        self.assertNotIn("brew", plain[0])
        self.assertIn('"brew upgrade hojin12312/mltf/mltf"', brew[0])
        self.assertIn(TAG_URL, brew[0])
        self.assertNotIn("pip", brew[0] + plain[0])


class OptOutAndBlockingTests(TempCase):
    def test_disabled_makes_no_network_call_and_ignores_cache(self):
        run_checker("0.1.2", self.cache, fake_release(), clock=lambda: 1.0)
        calls = []
        checker = UpdateChecker(
            "0.1.2", self.cache, enabled=False, fetch=lambda: calls.append(1)
        )
        checker.start()
        time.sleep(0.05)
        snapshot = checker.snapshot()
        self.assertEqual((snapshot["status"], snapshot["reason"]), ("disabled", "opt_out"))
        self.assertFalse(snapshot["update_available"])
        self.assertIsNone(snapshot["latest_version"])
        self.assertEqual(calls, [])

    def test_unconfigured_or_invalid_installed_version_never_fetches(self):
        for installed, cache in ((None, self.cache), ("0.1.2", None), ("junk", self.cache)):
            calls = []
            checker = UpdateChecker(installed, cache, fetch=lambda: calls.append(1))
            checker.start()
            time.sleep(0.03)
            self.assertIn(checker.snapshot()["status"], {"disabled", "unknown"})
            self.assertEqual(calls, [])

    def test_environment_opt_out_values(self):
        for value in ("1", "true", "YES", " on "):
            self.assertTrue(uc.disabled_by_environment({uc.DISABLE_ENVIRONMENT: value}))
        for value in ("", "0", "false", "no", "off", "maybe"):
            self.assertFalse(uc.disabled_by_environment({uc.DISABLE_ENVIRONMENT: value}))
        self.assertFalse(uc.disabled_by_environment({}))

    def test_slow_update_server_does_not_block_start_or_status(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def slow():
            release.wait(10)
            return fake_release()()

        checker = UpdateChecker("0.1.2", self.cache, fetch=slow)
        began = time.monotonic()
        checker.start()
        self.assertLess(time.monotonic() - began, 0.5)
        self.assertEqual(checker.snapshot()["status"], "checking")
        self.assertTrue(checker._thread.daemon)

    def test_total_deadline_turns_a_hung_check_into_unavailable(self):
        release = threading.Event()
        self.addCleanup(release.set)
        notices = []
        checker = UpdateChecker(
            "0.1.2",
            self.cache,
            fetch=lambda: release.wait(10) or fake_release()(),
            deadline=0.1,
            announce=notices.append,
        )
        checker.start()
        self.assertTrue(wait_for(lambda: checker.snapshot()["status"] == "unavailable"))
        self.assertEqual(checker.snapshot()["reason"], "timeout")
        release.set()
        time.sleep(0.1)
        self.assertEqual(checker.snapshot()["status"], "unavailable")
        self.assertEqual(notices, [])

    def test_process_exit_is_not_delayed_by_a_hung_check(self):
        code = (
            "import sys, threading, tempfile\n"
            "sys.path.insert(0, %r)\n"
            "from server.update_check import UpdateChecker\n"
            "c = UpdateChecker('0.1.2', tempfile.mkdtemp() + '/c.json',"
            " fetch=lambda: threading.Event().wait(60))\n"
            "c.start()\n" % str(ROOT)
        )
        started = time.monotonic()
        subprocess.run([os.sys.executable, "-c", code], check=True, timeout=20)
        self.assertLess(time.monotonic() - started, 10)


class StatusTests(TempCase):
    def status(self, checker):
        server = FrontendServer(
            ("127.0.0.1", 0),
            SimpleNamespace(model="m", status=lambda: {"physical_batch_width": 4}),
        )
        server.update_checker = checker
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/status")
        response = connection.getresponse()
        body = json.loads(response.read())
        self.assertEqual(response.status, 200)
        connection.close()
        return body

    def test_all_states(self):
        release = threading.Event()
        self.addCleanup(release.set)
        checking = UpdateChecker("0.1.2", self.cache, fetch=lambda: release.wait(10))
        checking.start()
        states = {
            "unknown": UpdateChecker("0.1.2", self.cache),
            "checking": checking,
            "update_available": run_checker("0.1.2", self.directory / "a.json", fake_release("0.1.3"))[0],
            "up_to_date": run_checker("0.1.2", self.directory / "b.json", fake_release("0.1.2"))[0],
            "unavailable": run_checker(
                "0.1.2", self.directory / "c.json", lambda: (_ for _ in ()).throw(UpdateCheckError("http_500"))
            )[0],
            "disabled": UpdateChecker("0.1.2", self.cache, enabled=False),
        }
        for name, checker in states.items():
            body = self.status(checker)
            self.assertEqual(body["update"]["status"], name)
            self.assertEqual(body["physical_batch_width"], 4)
            self.assertNotRegex(json.dumps(body["update"]), r"/(Users|var|tmp|private)/")
            self.assertEqual(
                set(body["update"]) - {"reason"},
                {"status", "installed_version", "latest_version", "update_available", "checked_at", "release_url"},
            )
        self.assertTrue(self.status(states["update_available"])["update"]["update_available"])

    def test_unavailable_does_not_change_health_fields(self):
        broken = run_checker(
            "0.1.2", self.cache, lambda: (_ for _ in ()).throw(UpdateCheckError("network"))
        )[0]
        body = self.status(broken)
        self.assertEqual(body["physical_batch_width"], 4)
        self.assertNotIn("error", body)

    def test_checker_absent_is_disabled(self):
        self.assertEqual(self.status(None)["update"]["status"], "disabled")


class VersionAuthorityTests(unittest.TestCase):
    def test_source_checkout_version_matches_pyproject(self):
        import tomllib

        expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        if not paths.PACKAGED:
            self.assertEqual(paths.installed_version(), expected)
            self.assertEqual(launcher._version(), f"MLTF {expected} (source checkout)")

    def test_candidate_version_is_consistent(self):
        import tomllib

        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        makefile = (ROOT / "dev/Makefile").read_text()
        self.assertEqual(re.search(r"RELEASE_VERSION := (\S+)", makefile).group(1), version)
        self.assertEqual(json.loads((ROOT / "RELEASE_PROFILE.json").read_text())["version"], version)

    def test_release_json_is_the_packaged_authority(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        release = directory / "release.json"
        release.write_text(json.dumps({"version": "9.8.7"}))
        with patch.object(paths, "PACKAGED", True), patch.object(paths, "RELEASE", release):
            self.assertEqual(paths.installed_version(), "9.8.7")
            self.assertEqual(launcher._version(), "MLTF 9.8.7")
            release.write_text("{")
            self.assertIsNone(paths.installed_version())

    def test_install_channel_is_homebrew_only_with_cellar_evidence(self):
        keg = "/opt/homebrew/Cellar/mltf/0.1.2/libexec"
        self.assertEqual(paths.install_channel(keg, False, True), "homebrew")
        self.assertIsNone(paths.install_channel(keg, True, True))
        self.assertIsNone(paths.install_channel(keg, False, False))
        for other in ("/Users/x/mltf-0.1.2", "/opt/homebrew/Cellar/other/1/libexec", "/x/mltf/Cellar"):
            self.assertIsNone(paths.install_channel(other, False, True))


class LauncherWiringTests(unittest.TestCase):
    def command(self, *extra):
        model = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, model, True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        args = launcher.parse_args(
            ["serve", "--model", "local/Qwen3.8-27B-q8c", "--model-path", str(model),
             "--port", str(port), *extra]
        )
        captured = {}
        with (
            patch.object(launcher.model_artifacts, "validate_package_manifest"),
            patch.object(launcher.model_artifacts, "verify_artifacts"),
            patch.object(launcher.catalog, "spawn_refresh"),
            patch.object(launcher, "RUNTIME_DIR", Path(tempfile.mkdtemp())),
            patch.object(launcher.os, "execve", lambda *call: captured.setdefault("cmd", call[1])),
        ):
            launcher.serve(args)
        return captured["cmd"]

    def test_flag_and_forwarding(self):
        enabled = self.command()
        self.assertNotIn("--no-update-check", enabled)
        self.assertEqual(enabled[enabled.index("--installed-version") + 1], paths.installed_version())
        self.assertEqual(Path(enabled[enabled.index("--update-cache") + 1]), paths.UPDATE_CACHE)
        disabled = self.command("--no-update-check")
        self.assertIn("--no-update-check", disabled)
        self.assertNotIn("--update-cache", disabled)
        self.assertEqual(disabled[disabled.index("--installed-version") + 1], paths.installed_version())

    def test_help_documents_the_opt_out(self):
        for command in (
            [os.sys.executable, "-m", "install.launcher", "serve", "--help"],
            [os.sys.executable, "server/server.py", "--help"],
        ):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--no-update-check", result.stdout)
            self.assertNotIn("--installed-version", result.stdout)


class WebUiTests(unittest.TestCase):
    html = (ROOT / "server/chat.html").read_text()

    def test_notice_is_passive_and_uses_status_only(self):
        self.assertIn("'/status'", self.html)
        self.assertNotIn("api.github.com", self.html)
        self.assertNotIn("alert(", self.html)
        self.assertNotIn("confirm(", self.html)
        self.assertNotIn("download", self.html.lower().replace("downloads", ""))
        self.assertRegex(self.html, r'id="update-link" target="_blank" rel="noopener noreferrer"')
        self.assertRegex(self.html, r'id="update-notice"[^>]* hidden')
        self.assertEqual(self.html.count("localStorage"), 2)

    @unittest.skipUnless(shutil.which("node"), "node is required to execute the page script")
    def test_notice_renders_only_for_update_available(self):
        function = re.search(r"const releasePrefix.*?\n  function showUpdate.*?\n  }\n", self.html, re.S).group(0)
        script = (
            "const nodes = {};\n"
            "const $ = id => nodes[id] ||= {hidden: true, textContent: '', href: ''};\n"
            + function
            + "\nconst good = {status:'update_available', update_available:true, latest_version:'0.1.3',"
            " release_url: releasePrefix + 'tag/v0.1.3'};\n"
            "const cases = [good, {...good, status:'up_to_date'}, {...good, update_available:false},"
            " {status:'unavailable'}, {status:'disabled'}, {status:'checking'}, {status:'unknown'},"
            " {...good, latest_version:'<img src=x>'}, {...good, release_url:'https://evil.example/x'},"
            " {...good, release_url:'javascript:alert(1)'}, undefined, null];\n"
            "const shown = cases.map(c => { showUpdate(c); return !$('#update-notice').hidden; });\n"
            "console.log(JSON.stringify(shown));\n"
            "showUpdate(good); console.log(JSON.stringify([$('#update-text').textContent, $('#update-link').href]));\n"
        )
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(json.loads(lines[0]), [True] + [False] * 11)
        self.assertEqual(
            json.loads(lines[1]),
            ["MLTF 0.1.3 is available.", uc.RELEASE_URL_PREFIX + "v0.1.3"],
        )

    def test_embedded_page_is_served(self):
        from server.server import CHAT_HTML

        self.assertIn(b"update-notice", CHAT_HTML)


if __name__ == "__main__":
    unittest.main()
