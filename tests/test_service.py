import contextlib
import io
import os
import plistlib
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from install import launcher, service

PRINT_RUNNING = (
    "io.github.hojin12312.mltf.serve-8000 = {\n"
    "\tactive count = 1\n"
    "\tstate = running\n"
    "\tpid = 4242\n"
    "\tlast exit code = (never exited)\n"
    "\tevent triggers = {\n"
    "\t\tstate = active\n"
    "\t\tpid = 1\n"
    "\t}\n"
    "}\n"
)
PRINT_STOPPED = "x = {\n\tstate = not running\n\tlast exit code = 0\n}\n"


def completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def parse(*argv):
    return launcher.parse_args(list(argv))


class HomeCase(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        for patcher in (
            mock.patch.object(Path, "home", return_value=self.home),
            mock.patch.object(service.sys, "platform", "darwin"),
            mock.patch.object(service.os, "getuid", return_value=501),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("SPLASH_PORT", None)
        os.environ.pop("SPLASH_API_KEY", None)
        self.exe = self.home / "bin" / "mltf"
        self.exe.parent.mkdir()
        self.exe.write_text("#!/bin/sh\n")
        self.exe.chmod(0o755)
        self.calls = []
        self.state = None  # launchctl print output while "installed"

    def launchctl(self, *args, check=True):
        self.calls.append(args)
        if args[0] == "bootout":
            self.state = None
        elif args[0] == "kill":
            self.state = PRINT_STOPPED
        if args[0] == "print":
            if self.state is None:
                return completed(113, stderr="Could not find service")
            return completed(stdout=self.state)
        return completed()

    def run_action(self, *argv):
        with mock.patch.object(service, "_launchctl", self.launchctl):
            return service.run(parse(*argv))

    def verbs(self):
        return [call[0] for call in self.calls if call[0] != "print"]


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mock.patch.dict(os.environ).stop)
        os.environ.pop("SPLASH_PORT", None)

    def fails(self, *argv):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit):
                parse(*argv)
        return err.getvalue()

    def test_install_passes_serve_arguments_after_double_dash(self):
        args = parse("service", "install", "--", "--model", "a/b", "--port", "9000")
        self.assertEqual(args.client_args, ["--model", "a/b", "--port", "9000"])
        self.assertEqual((args.serve.model, args.serve.port), ("a/b", 9000))

    def test_install_validates_with_the_serve_parser(self):
        err = self.fails("service", "install", "--", "--model", "a/b", "--max-context", "x")
        self.assertIn("--max-context", err)
        self.assertIn("--model", self.fails("service", "install", "--", "--port", "9000"))

    def test_install_requires_serve_arguments_and_rejects_its_own_port(self):
        self.assertIn("after --", self.fails("service", "install"))
        self.assertIn(
            "serve arguments",
            self.fails("service", "install", "--port", "9", "--", "--model", "a/b"),
        )

    def test_other_actions_reject_serve_arguments_and_executable(self):
        self.assertIn("only supported", self.fails("service", "stop", "--", "--model", "a/b"))
        self.assertIn("only supported", self.fails("service", "status", "--executable", "/x"))

    def test_invalid_splash_port_is_reported(self):
        os.environ["SPLASH_PORT"] = "banana"
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
            parse("service", "status")
        self.assertIn("SPLASH_PORT", err.getvalue())

    def test_port_is_unset_unless_given_by_option_or_splash_port(self):
        self.assertIsNone(parse("service", "status").port)
        os.environ["SPLASH_PORT"] = "9001"
        self.assertEqual(parse("service", "status").port, 9001)
        self.assertEqual(parse("service", "status", "--port", "9002").port, 9002)


class PlistTests(unittest.TestCase):
    def test_build_plist(self):
        plist = service.build_plist(
            8000,
            "/opt/homebrew/bin/mltf",
            ["--model", "a/b"],
            environ={"MLTF_DATA_ROOT": "/data", "SPLASH_API_KEY": "secret", "HOME": "/h"},
        )
        self.assertEqual(plist["Label"], "io.github.hojin12312.mltf.serve-8000")
        self.assertEqual(
            plist["ProgramArguments"], ["/opt/homebrew/bin/mltf", "serve", "--model", "a/b"]
        )
        self.assertTrue(plist["RunAtLoad"])
        self.assertNotIn("KeepAlive", plist)
        env = plist["EnvironmentVariables"]
        self.assertTrue(env["PATH"].startswith("/opt/homebrew/bin:"))
        self.assertEqual(env["MLTF_DATA_ROOT"], "/data")
        self.assertNotIn("SPLASH_API_KEY", env)
        self.assertEqual(plist["StandardOutPath"], plist["StandardErrorPath"])
        self.assertTrue(plist["StandardOutPath"].endswith("Logs/mltf-8000.log"))


class ExecutableTests(HomeCase):
    def test_explicit_path_is_kept_and_checked(self):
        self.assertEqual(service.resolve_executable(str(self.exe)), str(self.exe))
        with self.assertRaises(service.ServiceError):
            service.resolve_executable(str(self.home / "missing"))

    def test_symlink_is_not_resolved(self):
        link = self.home / "bin" / "stable-mltf"
        link.symlink_to(self.exe)
        self.assertEqual(service.resolve_executable(str(link)), str(link))

    def test_source_checkout_uses_the_wrapper_next_to_the_launcher(self):
        with mock.patch.object(service.paths, "PACKAGED", False), mock.patch.object(
            service.paths, "ROOT", self.home
        ):
            (self.home / "mltf").write_text("#!/bin/sh\n")
            (self.home / "mltf").chmod(0o755)
            self.assertEqual(service.resolve_executable(), str(self.home / "mltf"))

    def test_packaged_install_uses_path_lookup(self):
        with mock.patch.object(service.paths, "PACKAGED", True):
            with mock.patch.object(service.shutil, "which", return_value=str(self.exe)):
                self.assertEqual(service.resolve_executable(), str(self.exe))
            with mock.patch.object(service.shutil, "which", return_value=None):
                with self.assertRaisesRegex(service.ServiceError, "--executable"):
                    service.resolve_executable()


class InstallTests(HomeCase):
    def install(self, *serve_args, extra=()):
        with mock.patch.object(service, "_require_free") as free:
            self.free = free
            return self.run_action(
                "service", "install", "--executable", str(self.exe), *extra, "--", *serve_args
            )

    def plist(self, port=8000):
        return plistlib.loads(service.plist_path(port).read_bytes())

    def test_fresh_install_writes_a_private_plist_and_bootstraps(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.install("--model", "a/b", "--max-context", "32K"), 0)
        self.assertEqual(self.verbs(), ["bootstrap"])
        self.assertEqual(self.calls[-1][1], "gui/501")
        self.assertEqual(
            self.plist()["ProgramArguments"],
            [str(self.exe), "serve", "--model", "a/b", "--max-context", "32K", "--port", "8000"],
        )
        mode = stat.S_IMODE(service.plist_path(8000).stat().st_mode)
        self.assertEqual(mode, 0o600)
        self.free.assert_called_once()

    def test_port_is_made_explicit_even_when_only_the_environment_sets_it(self):
        os.environ["SPLASH_PORT"] = "9001"
        with contextlib.redirect_stdout(io.StringIO()):
            self.install("--model", "a/b")
        self.assertEqual(self.plist(9001)["ProgramArguments"][-2:], ["--port", "9001"])
        self.assertFalse(service.plist_path(8000).exists())

    def test_explicit_port_is_not_repeated(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.install("--model", "a/b", "--port=9000")
        arguments = self.plist(9000)["ProgramArguments"]
        self.assertEqual(arguments.count("--port=9000") + arguments.count("--port"), 1)

    def test_reinstall_replaces_the_loaded_job_and_waits_for_the_port(self):
        self.state = PRINT_RUNNING
        with contextlib.redirect_stdout(io.StringIO()):
            self.install("--model", "a/b")
        self.assertEqual(self.verbs(), ["bootout", "bootstrap"])
        self.assertEqual(self.free.call_args.kwargs["timeout"], service.STOP_TIMEOUT)

    def test_reinstall_waits_until_launchd_has_unloaded_the_job(self):
        outputs = iter([PRINT_RUNNING, PRINT_RUNNING, PRINT_RUNNING, None, None])
        calls = []

        def launchctl(*args, check=True):
            calls.append(args[0])
            if args[0] == "print":
                state = next(outputs)
                return completed(0, state) if state else completed(113)
            return completed()

        with mock.patch.object(service, "_launchctl", launchctl), mock.patch.object(
            service, "_require_free"
        ), mock.patch.object(service.time, "sleep"), contextlib.redirect_stdout(
            io.StringIO()
        ):
            service.run(
                parse("service", "install", "--executable", str(self.exe), "--", "--model", "a/b")
            )
        self.assertLess(calls.index("bootout"), calls.index("bootstrap"))
        # Two polls still saw the job loaded; bootstrap only ran after a third
        # saw it gone.
        between = calls[calls.index("bootout") + 1 : calls.index("bootstrap")]
        self.assertEqual(between, ["print", "print", "print"])

    def test_reinstall_fails_cleanly_when_the_old_job_never_unloads(self):
        self.state = PRINT_RUNNING

        def stuck(*args, check=True):
            self.calls.append(args)
            return completed(0, PRINT_RUNNING) if args[0] == "print" else completed()

        with mock.patch.object(service, "_launchctl", stuck), mock.patch.object(
            service, "STOP_TIMEOUT", 0.3
        ):
            with self.assertRaisesRegex(service.ServiceError, "unloading"):
                service.run(
                    parse("service", "install", "--executable", str(self.exe), "--", "--model", "a/b")
                )
        self.assertNotIn("bootstrap", self.verbs())

    def test_api_key_from_the_environment_is_not_captured_and_warns(self):
        os.environ["SPLASH_API_KEY"] = "secret"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            self.install("--model", "a/b")
        self.assertIn("SPLASH_API_KEY", err.getvalue())
        self.assertNotIn(b"secret", service.plist_path(8000).read_bytes())

    def test_explicit_api_key_is_stored_without_a_warning(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            self.install("--model", "a/b", "--api-key", "k")
        self.assertEqual(err.getvalue(), "")
        self.assertIn("k", self.plist()["ProgramArguments"])

    def test_relative_model_path_is_rejected(self):
        with self.assertRaisesRegex(service.ServiceError, "absolute"):
            self.install("--model", "a/b", "--model-path", "models/q8c")
        self.assertFalse(service.plist_path(8000).exists())
        self.assertEqual(self.verbs(), [])

    def test_busy_port_aborts_before_anything_is_written(self):
        with mock.patch.object(
            service, "_require_free", side_effect=service.ServiceError("busy")
        ):
            with self.assertRaisesRegex(service.ServiceError, "busy"):
                self.run_action(
                    "service", "install", "--executable", str(self.exe), "--", "--model", "a/b"
                )
        self.assertFalse(service.plist_path(8000).exists())
        self.assertEqual(self.verbs(), [])


class ControlTests(HomeCase):
    def run_quiet(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = self.run_action(*argv)
        return code, out.getvalue()

    def test_actions_on_a_missing_service_explain_how_to_install(self):
        for action in ("start", "stop", "restart"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(service.ServiceError, "not installed"):
                    self.run_action("service", action, "--port", "8000")
        self.assertEqual(self.verbs(), [])

    def test_start_kickstarts_only_a_stopped_job(self):
        self.state = PRINT_STOPPED
        self.run_quiet("service", "start", "--port", "8000")
        self.assertEqual(self.calls[-1], ("kickstart", "gui/501/" + service.label(8000)))
        self.calls.clear()
        self.state = PRINT_RUNNING
        self.assertIn("already running", self.run_quiet("service", "start", "--port", "8000")[1])
        self.assertEqual(self.verbs(), [])

    def test_stop_sends_sigterm_and_keeps_the_job_loaded(self):
        self.state = PRINT_RUNNING
        self.run_quiet("service", "stop", "--port", "8000")
        self.assertEqual(self.verbs(), ["kill"])
        self.assertIn(("kill", "SIGTERM", "gui/501/" + service.label(8000)), self.calls)
        self.calls.clear()
        self.state = PRINT_STOPPED
        self.assertIn("not running", self.run_quiet("service", "stop", "--port", "8000")[1])
        self.assertEqual(self.verbs(), [])

    def test_stop_fails_when_the_server_does_not_exit(self):
        def stuck(*args, check=True):
            self.calls.append(args)
            return completed(0, PRINT_RUNNING) if args[0] == "print" else completed()

        with mock.patch.object(service, "_launchctl", stuck), mock.patch.object(
            service, "STOP_TIMEOUT", 0.3
        ):
            with self.assertRaisesRegex(service.ServiceError, "stopping"):
                service.run(parse("service", "stop", "--port", "8000"))

    def test_restart_uses_kickstart_k(self):
        self.state = PRINT_RUNNING
        self.run_quiet("service", "restart", "--port", "8000")
        self.assertEqual(self.calls[-1][:2], ("kickstart", "-k"))

    def test_uninstall_waits_until_launchd_has_unloaded_the_job(self):
        outputs = iter([PRINT_RUNNING, None])
        calls = []

        def launchctl(*args, check=True):
            calls.append(args[0])
            if args[0] == "print":
                state = next(outputs)
                return completed(0, state) if state else completed(113)
            return completed()

        with mock.patch.object(service, "_launchctl", launchctl), mock.patch.object(
            service.time, "sleep"
        ), contextlib.redirect_stdout(io.StringIO()):
            service.run(parse("service", "uninstall", "--port", "8000"))
        self.assertEqual(calls, ["bootout", "print", "print"])

    def test_uninstall_fails_when_the_job_never_unloads(self):
        def stuck(*args, check=True):
            return completed(0, PRINT_RUNNING) if args[0] == "print" else completed()

        path = service.plist_path(8000)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        with mock.patch.object(service, "_launchctl", stuck), mock.patch.object(
            service, "STOP_TIMEOUT", 0.3
        ):
            with self.assertRaisesRegex(service.ServiceError, "unloading"):
                service.run(parse("service", "uninstall", "--port", "8000"))
        self.assertTrue(path.exists())  # still installed from the user's view

    def test_uninstall_unloads_and_removes_the_plist(self):
        path = service.plist_path(8000)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        code, out = self.run_quiet("service", "uninstall", "--port", "8000")
        self.assertEqual(code, 0)
        self.assertFalse(path.exists())
        self.assertEqual(self.verbs(), ["bootout"])
        self.assertIn(
            "is not installed", self.run_quiet("service", "uninstall", "--port", "8000")[1]
        )
        # Nothing left to discover.
        self.assertIn("no mltf service is installed", self.run_quiet("service", "uninstall")[1])

    def test_status_reports_launchd_fields_not_the_nested_ones(self):
        self.state = PRINT_RUNNING
        code, out = self.run_quiet("service", "status", "--port", "8000")
        self.assertEqual(code, 0)
        self.assertIn("serve-8000: running", out)
        self.assertIn("pid: 4242", out)
        self.assertIn("mltf-8000.log", out)

    def test_status_of_a_missing_service_fails(self):
        code, out = self.run_quiet("service", "status", "--port", "8000")
        self.assertEqual(code, 1)
        self.assertIn("serve-8000: not installed", out)

    def test_ports_are_independent_services(self):
        self.state = PRINT_RUNNING
        self.run_quiet("service", "restart", "--port", "9000")
        self.assertTrue(self.calls[-1][-1].endswith("serve-9000"))

    def test_only_macos_is_supported(self):
        with mock.patch.object(service.sys, "platform", "linux"):
            with self.assertRaisesRegex(service.ServiceError, "macOS"):
                self.run_action("service", "status", "--port", "8000")

    def test_launchctl_failure_is_reported(self):
        with mock.patch.object(
            service.subprocess, "run", return_value=completed(5, stderr="boom")
        ):
            with self.assertRaisesRegex(service.ServiceError, "boom"):
                service._launchctl("bootstrap", "x")


class DiscoveryTests(HomeCase):
    """Without --port or SPLASH_PORT the installed service is used."""

    def add(self, *ports):
        for port in ports:
            path = service.plist_path(port)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def run_quiet(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = self.run_action(*argv)
        return code, out.getvalue()

    def test_installed_ports_ignores_other_files_and_sorts_numerically(self):
        self.add(10007, 9000)
        folder = service.plist_path(0).parent
        for name in (
            "com.other.plist",
            service.LABEL_PREFIX + "abc.plist",
            service.LABEL_PREFIX + "0.plist",
            service.LABEL_PREFIX + "70000.plist",
            service.LABEL_PREFIX + "8000.plist.bak",
        ):
            (folder / name).write_bytes(b"x")
        self.assertEqual(service.installed_ports(), [9000, 10007])

    def test_installed_ports_without_a_launchagents_folder(self):
        self.assertEqual(service.installed_ports(), [])

    def test_the_only_service_is_used_without_a_port(self):
        self.add(10007)
        target = "gui/501/" + service.label(10007)
        self.state = PRINT_RUNNING
        code, out = self.run_quiet("service", "status")
        self.assertEqual(code, 0)
        self.assertIn("serve-10007: running", out)
        self.run_quiet("service", "restart")
        self.assertEqual(self.calls[-1], ("kickstart", "-k", target))
        self.run_quiet("service", "stop")
        self.assertIn(("kill", "SIGTERM", target), self.calls)
        self.state = PRINT_STOPPED
        self.run_quiet("service", "start")
        self.assertEqual(self.calls[-1], ("kickstart", target))
        self.run_quiet("service", "uninstall")
        self.assertFalse(service.plist_path(10007).exists())

    def test_status_lists_every_installed_service(self):
        self.add(9000, 10007)
        self.state = PRINT_RUNNING
        code, out = self.run_quiet("service", "status")
        self.assertEqual(code, 0)
        self.assertLess(out.index("serve-9000"), out.index("serve-10007"))
        self.assertIn("\n\n", out)

    def test_other_actions_refuse_to_guess_between_several_services(self):
        self.add(9000, 10007)
        for action in ("start", "stop", "restart", "uninstall"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(service.ServiceError, r"9000, 10007.*--port"):
                    self.run_action("service", action)
        self.assertEqual(self.verbs(), [])
        self.assertTrue(service.plist_path(9000).exists())
        self.assertTrue(service.plist_path(10007).exists())

    def test_nothing_installed(self):
        code, out = self.run_quiet("service", "status")
        self.assertEqual(code, 1)
        self.assertIn("no mltf service is installed", out)
        for action in ("start", "stop", "restart"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(service.ServiceError, "no mltf service is installed"):
                    self.run_action("service", action)
        code, out = self.run_quiet("service", "uninstall")
        self.assertEqual(code, 0)
        self.assertEqual(self.verbs(), [])

    def test_splash_port_and_the_option_win_over_discovery(self):
        self.add(10007)
        os.environ["SPLASH_PORT"] = "8000"
        code, out = self.run_quiet("service", "status")
        self.assertEqual(code, 1)
        self.assertIn("serve-8000: not installed", out)
        code, out = self.run_quiet("service", "status", "--port", "9001")
        self.assertIn("serve-9001: not installed", out)
        self.assertEqual(self.verbs(), [])


if __name__ == "__main__":
    unittest.main()
