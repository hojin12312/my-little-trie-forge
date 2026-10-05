import argparse
import contextlib
import http.client
import io
import json
import os
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from install import launcher
from server import server as server_module
from server.latency import LatencyMetrics
from server.server import (
    FrontendHandler,
    FrontendServer,
    WaitingAdmission,
    _parse_request_limit,
)

MODEL = "local/Qwen3.8-27B-q8c"


def _app(backend=None, timeout=10):
    if backend is None:
        backend = SimpleNamespace(
            can_submit=lambda: True, submit=lambda job: True, cancel=lambda job: None
        )
    return SimpleNamespace(
        model=MODEL,
        backend=backend,
        latencies=LatencyMetrics(),
        request_timeout=timeout,
        request_deadline=lambda body, start: start
        + min(body.get("timeout", timeout), timeout),
        prepare=lambda body, deadline: (
            SimpleNamespace(
                tool_policy=None,
                response_history_items=(),
                response_format=None,
                stop_sequences=(),
            ),
            False,
            False,
        ),
        status=lambda: {"physical_batch_width": 4},
    )


@contextlib.contextmanager
def _running(server):
    runner = threading.Thread(target=server.serve_forever, daemon=True)
    runner.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        runner.join(2)


def _post(port, payload, timeout=4, path="/v1/chat/completions"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request(
        "POST",
        path,
        json.dumps(payload),
        {"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body


def _wait_for(predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


class GateTests(unittest.TestCase):
    def test_cap_fifo_and_wait(self):
        gate = WaitingAdmission(1)
        gate.acquire(time.monotonic() + 3, lambda: False)
        order = []
        release = threading.Event()

        def enter(n):
            gate.acquire(time.monotonic() + 3, lambda: False)
            order.append(n)
            if n == 1:
                release.wait(1)
            gate.release()

        a = threading.Thread(target=enter, args=(1,))
        b = threading.Thread(target=enter, args=(2,))
        a.start()
        self.assertTrue(_wait_for(lambda: gate.stats()["waiting"] == 1))
        b.start()
        self.assertTrue(_wait_for(lambda: gate.stats()["waiting"] == 2))
        self.assertEqual(
            gate.stats(), {"active": 1, "waiting": 2, "capacity": 1}
        )
        gate.release()
        self.assertTrue(_wait_for(lambda: order == [1]))
        release.set()
        a.join(2)
        b.join(2)
        self.assertEqual(order, [1, 2])
        self.assertEqual(gate.stats()["active"], 0)

    def test_timeout_disconnect_shutdown_leave_no_queue_ticket(self):
        gate = WaitingAdmission(1)
        gate.acquire(time.monotonic() + 2, lambda: False)
        with self.assertRaises(TimeoutError):
            gate.acquire(time.monotonic() + 0.02, lambda: False)
        with self.assertRaises(ConnectionResetError):
            gate.acquire(time.monotonic() + 2, lambda: True)
        self.assertEqual(gate.stats()["waiting"], 0)
        gate.close()
        with self.assertRaises(Exception):
            gate.acquire(time.monotonic() + 2, lambda: False)
        self.assertEqual(gate.stats()["waiting"], 0)
        gate.release()

    def test_request_limit_bounds(self):
        for value in ("1", "2", "3", "4"):
            self.assertEqual(_parse_request_limit(value), int(value))
            self.assertEqual(WaitingAdmission(int(value)).capacity, int(value))
        for value in ("0", "-1", "5", "7", "1.5", "auto", "x", "True"):
            with self.assertRaises(argparse.ArgumentTypeError):
                _parse_request_limit(value)
        for value in (0, -1, 5, 7, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                WaitingAdmission(value)

    def test_release_without_acquire_raises_and_never_goes_negative(self):
        gate = WaitingAdmission(1)
        with self.assertRaises(RuntimeError):
            gate.release()
        gate.acquire(time.monotonic() + 1, lambda: False)
        gate.release()
        with self.assertRaises(RuntimeError):
            gate.release()
        self.assertEqual(gate.stats()["active"], 0)

    def test_cli_parsers_share_the_supported_range(self):
        def serve_args(limit):
            return ["t", "d", "--tokenizer", "x", "--model", MODEL] + (
                ["--max-concurrent-requests", limit] if limit is not None else []
            )

        def mltf_args(limit):
            return ["serve", "--model", MODEL] + (
                ["--max-concurrent-requests", limit] if limit is not None else []
            )

        with patch.dict(os.environ):
            os.environ.pop("SPLASH_API_KEY", None)
            os.environ.pop("SPLASH_DEFAULT_REASONING_EFFORT", None)
            for value in ("1", "2", "3", "4"):
                self.assertEqual(
                    server_module.parse_args(serve_args(value)).max_concurrent_requests,
                    int(value),
                )
                self.assertEqual(
                    launcher.parse_args(mltf_args(value)).max_concurrent_requests,
                    int(value),
                )
            self.assertIsNone(
                server_module.parse_args(serve_args(None)).max_concurrent_requests
            )
            self.assertIsNone(
                launcher.parse_args(mltf_args(None)).max_concurrent_requests
            )
            for value in ("0", "5", "1.5", "auto", "x"):
                for parse, args in (
                    (server_module.parse_args, serve_args),
                    (launcher.parse_args, mltf_args),
                ):
                    with (
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        parse(args(value))

    def test_constructor_rejects_out_of_range_limit(self):
        for value in (0, 5, True):
            with self.assertRaises(ValueError):
                FrontendServer(("127.0.0.1", 0), _app(), max_concurrent_requests=value)


class HTTPTests(unittest.TestCase):
    def test_two_requests_wait_and_control_status_stays_available(self):
        release = threading.Event()
        entered = []

        def complete(handler, job, thinking, has_tools):
            entered.append(job)
            release.wait(2)
            handler._json(200, {"ok": True})

        server = FrontendServer(
            ("127.0.0.1", 0), _app(), max_concurrent_requests=1
        )
        with _running(server), patch.object(
            FrontendHandler, "_complete", complete
        ), ThreadPoolExecutor(2) as pool:
            one = pool.submit(_post, server.server_port, {})
            self.assertTrue(_wait_for(lambda: len(entered) == 1))
            two = pool.submit(_post, server.server_port, {})
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["waiting"] == 1
                )
            )
            control = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=2
            )
            control.request("GET", "/status")
            response = control.getresponse()
            self.assertEqual(response.status, 200)
            status = json.loads(response.read())
            control.close()
            self.assertEqual(status["http"]["generation_requests"]["active"], 1)
            self.assertEqual(status["http"]["generation_requests"]["waiting"], 1)
            self.assertEqual(
                status["http"]["generation_requests"]["capacity"], 1
            )
            self.assertEqual(status["physical_batch_width"], 4)
            self.assertEqual(len(entered), 1)
            release.set()
            self.assertEqual(one.result()[0], 200)
            self.assertEqual(two.result()[0], 200)
            self.assertEqual(server.generation_admission.stats()["active"], 0)

    def test_streaming_request_holds_the_slot_until_done(self):
        release = threading.Event()
        entered = []

        def stream(handler, job, thinking, has_tools, stream_options):
            entered.append(job)
            release.wait(2)
            handler._json(200, {"ok": True})

        server = FrontendServer(
            ("127.0.0.1", 0), _app(), max_concurrent_requests=1
        )
        with _running(server), patch.object(
            FrontendHandler, "_stream", stream
        ), ThreadPoolExecutor(2) as pool:
            one = pool.submit(_post, server.server_port, {"stream": True})
            self.assertTrue(_wait_for(lambda: len(entered) == 1))
            two = pool.submit(_post, server.server_port, {"stream": True})
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["waiting"] == 1
                )
            )
            self.assertEqual(len(entered), 1)
            release.set()
            self.assertEqual(one.result()[0], 200)
            self.assertEqual(two.result()[0], 200)
            self.assertEqual(server.generation_admission.stats()["active"], 0)

    def test_queued_deadline_expiry_times_out_and_frees_the_waiter(self):
        release = threading.Event()

        def complete(handler, job, thinking, has_tools):
            release.wait(2)
            handler._json(200, {"ok": True})

        server = FrontendServer(
            ("127.0.0.1", 0), _app(), max_concurrent_requests=1
        )
        with _running(server), patch.object(
            FrontendHandler, "_complete", complete
        ), ThreadPoolExecutor(2) as pool:
            one = pool.submit(_post, server.server_port, {"timeout": 3})
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["active"] == 1
                )
            )
            status, _ = _post(server.server_port, {"timeout": 0.2})
            self.assertEqual(status, 408)
            self.assertEqual(server.generation_admission.stats()["waiting"], 0)
            release.set()
            self.assertEqual(one.result()[0], 200)
            self.assertEqual(server.generation_admission.stats()["active"], 0)
            status, _ = _post(server.server_port, {"timeout": 3})
            self.assertEqual(status, 200)

    def test_queued_disconnect_removes_the_waiter(self):
        release = threading.Event()

        def complete(handler, job, thinking, has_tools):
            release.wait(2)
            handler._json(200, {"ok": True})

        server = FrontendServer(
            ("127.0.0.1", 0), _app(), max_concurrent_requests=1
        )
        with _running(server), patch.object(
            FrontendHandler, "_complete", complete
        ), ThreadPoolExecutor(1) as pool:
            one = pool.submit(_post, server.server_port, {})
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["active"] == 1
                )
            )
            queued = socket.create_connection(
                ("127.0.0.1", server.server_port), timeout=2
            )
            body = json.dumps({}).encode()
            queued.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["waiting"] == 1
                )
            )
            queued.close()
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["waiting"] == 0
                )
            )
            release.set()
            self.assertEqual(one.result()[0], 200)
            self.assertEqual(server.generation_admission.stats()["active"], 0)

    def test_submission_failure_does_not_leak_active_capacity(self):
        backend = SimpleNamespace(
            can_submit=lambda: True,
            submit=lambda job: False,
            cancel=lambda job: None,
        )
        server = FrontendServer(
            ("127.0.0.1", 0), _app(backend), max_concurrent_requests=1
        )
        with _running(server):
            for _ in range(2):
                status, _ = _post(server.server_port, {})
                self.assertEqual(status, 429)
                self.assertEqual(
                    server.generation_admission.stats()["active"], 0
                )

    def test_running_failure_releases_the_slot_for_the_next_request(self):
        calls = []

        def complete(handler, job, thinking, has_tools):
            calls.append(job)
            if len(calls) == 1:
                raise RuntimeError("backend failure")
            handler._json(200, {"ok": True})

        server = FrontendServer(
            ("127.0.0.1", 0), _app(), max_concurrent_requests=1
        )
        with _running(server), patch.object(
            FrontendHandler, "_complete", complete
        ):
            status, _ = _post(server.server_port, {})
            self.assertEqual(status, 500)
            self.assertEqual(server.generation_admission.stats()["active"], 0)
            status, _ = _post(server.server_port, {})
            self.assertEqual(status, 200)
            self.assertEqual(server.generation_admission.stats()["active"], 0)

    def test_waiters_stay_bounded_by_existing_ingress_limits(self):
        release = threading.Event()

        def complete(handler, job, thinking, has_tools):
            release.wait(2)
            handler._json(200, {"ok": True})

        server = FrontendServer(
            ("127.0.0.1", 0),
            _app(),
            request_capacity=2,
            max_concurrent_requests=1,
        )
        with _running(server), patch.object(
            FrontendHandler, "_complete", complete
        ), ThreadPoolExecutor(2) as pool:
            one = pool.submit(_post, server.server_port, {})
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["active"] == 1
                )
            )
            two = pool.submit(_post, server.server_port, {})
            self.assertTrue(
                _wait_for(
                    lambda: server.generation_admission.stats()["waiting"] == 1
                )
            )
            status, _ = _post(server.server_port, {})
            self.assertEqual(status, 503)
            release.set()
            self.assertEqual(one.result()[0], 200)
            self.assertEqual(two.result()[0], 200)
            self.assertEqual(server.generation_admission.stats()["active"], 0)

    def test_server_close_releases_queued_waiters(self):
        release = threading.Event()

        def complete(handler, job, thinking, has_tools):
            release.wait(2)
            handler._json(200, {"ok": True})

        server = FrontendServer(
            ("127.0.0.1", 0), _app(), max_concurrent_requests=1
        )
        runner = threading.Thread(target=server.serve_forever, daemon=True)
        runner.start()
        try:
            with patch.object(
                FrontendHandler, "_complete", complete
            ), ThreadPoolExecutor(2) as pool:
                one = pool.submit(_post, server.server_port, {})
                self.assertTrue(
                    _wait_for(
                        lambda: server.generation_admission.stats()["active"] == 1
                    )
                )
                pool.submit(_post, server.server_port, {})
                self.assertTrue(
                    _wait_for(
                        lambda: server.generation_admission.stats()["waiting"]
                        == 1
                    )
                )
                server.shutdown()
                server.server_close()
                self.assertTrue(
                    _wait_for(
                        lambda: server.generation_admission.stats()["waiting"]
                        == 0
                    )
                )
                release.set()
                self.assertEqual(one.result()[0], 200)
                self.assertEqual(
                    server.generation_admission.stats()["active"], 0
                )
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            runner.join(2)

    def test_default_automatic_path_has_no_new_gate(self):
        server = FrontendServer(("127.0.0.1", 0), _app())
        try:
            self.assertIsNone(server.generation_admission)
            self.assertNotIn("generation_requests", server.status()["http"])
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
