"""Run ``mltf serve`` as a per-user macOS LaunchAgent that starts at login."""

import os
import plistlib
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from . import paths
except ImportError:  # Executed directly by the source or packaged entry point.
    import paths

ACTIONS = ("install", "uninstall", "start", "stop", "restart", "status")
LABEL_PREFIX = "io.github.hojin12312.mltf.serve-"
# Settings the launcher reads from the environment that a login service would
# not otherwise see. API keys are deliberately not carried over.
PASSTHROUGH_ENV = ("MLTF_DATA_ROOT",)
SYSTEM_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
STOP_TIMEOUT = 30
_FIELD = re.compile(r"^\t(state|pid|last exit code) = (.+)$", re.MULTILINE)


class ServiceError(RuntimeError):
    pass


def label(port):
    return f"{LABEL_PREFIX}{port}"


def plist_path(port):
    return Path.home() / "Library/LaunchAgents" / f"{label(port)}.plist"


def log_path(port):
    return Path.home() / "Library/Logs" / f"mltf-{port}.log"


def installed_ports():
    """Ports of the services whose LaunchAgent plist is installed, ascending."""
    pattern = re.compile(re.escape(LABEL_PREFIX) + r"(\d{1,5})\.plist")
    folder = plist_path(0).parent
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    ports = {int(m.group(1)) for m in map(pattern.fullmatch, names) if m}
    return sorted(port for port in ports if 1 <= port <= 65535)


def select_ports(args):
    """The ports an action applies to, as an ordered list.

    An explicit --port or SPLASH_PORT wins. Otherwise the installed services
    are used, so a service on a custom port needs no port to be remembered.
    """
    if args.port is not None:
        return [args.port]
    found = installed_ports()
    if len(found) > 1 and args.action != "status":
        listed = ", ".join(str(port) for port in found)
        raise ServiceError(
            f"several services are installed (ports {listed}); "
            f"choose one with --port to {args.action} it"
        )
    return found


def _no_service():
    return ServiceError(
        "no mltf service is installed; "
        "run `mltf service install -- --model OWNER/REPO ...`"
    )


def _target(port):
    return f"gui/{os.getuid()}/{label(port)}"


def _launchctl(*args, check=True):
    try:
        result = subprocess.run(
            ["launchctl", *args], capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise ServiceError(f"cannot run launchctl: {error}") from None
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ServiceError(f"launchctl {args[0]} failed: {detail}")
    return result


def _state(port):
    """Return launchd's view of the job, or None when it is not loaded."""
    result = _launchctl("print", _target(port), check=False)
    if result.returncode:
        return None
    return dict(_FIELD.findall(result.stdout))


def _running(state):
    return state is not None and state.get("state") == "running"


def _wait_until(done, timeout, what):
    deadline = time.monotonic() + timeout
    while not done():
        if time.monotonic() >= deadline:
            raise ServiceError(f"{what} did not finish within {timeout:g} seconds")
        time.sleep(0.2)


def _require_free(host, port, timeout=0):
    deadline = time.monotonic() + timeout
    while True:
        with socket.socket() as probe:
            # Match the HTTP listener: TIME_WAIT must not block a restart.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
                return
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise ServiceError(
                        f"cannot bind {host}:{port}: {error}; "
                        "is `mltf serve` already running there?"
                    ) from None
        time.sleep(0.5)


def resolve_executable(explicit=None):
    """Return a stable path to ``mltf``; symlinks are kept, not resolved.

    Homebrew's ``bin/mltf`` survives upgrades, unlike the versioned Cellar path
    that resolving it would give.
    """
    if explicit:
        found = str(Path(explicit).expanduser())
    elif not paths.PACKAGED:
        found = str(paths.ROOT / "mltf")
    else:
        found = shutil.which("mltf")
        if not found:
            raise ServiceError(
                "cannot find mltf on PATH; pass --executable /path/to/mltf"
            )
    found = os.path.abspath(found)
    if not (os.path.isfile(found) and os.access(found, os.X_OK)):
        raise ServiceError(f"not an executable file: {found}")
    return found


def build_plist(port, executable, serve_args, environ=None):
    environ = os.environ if environ is None else environ
    env = {"PATH": f"{Path(executable).parent}:{SYSTEM_PATH}"}
    env.update({key: environ[key] for key in PASSTHROUGH_ENV if key in environ})
    log = str(log_path(port))
    # No KeepAlive: a server that fails under memory pressure must not respawn
    # in a loop. `mltf service start` brings it back.
    return {
        "Label": label(port),
        "ProgramArguments": [executable, "serve", *serve_args],
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }


def _write_private(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The arguments may include --api-key, so keep the file private.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o600)


def _has_option(arguments, name):
    return any(item == name or item.startswith(name + "=") for item in arguments)


def install(args):
    serve = args.serve
    port = serve.port
    if serve.model_path is not None and not serve.model_path.is_absolute():
        raise ServiceError("--model-path must be an absolute path for a login service")
    serve_args = list(args.client_args)
    # SPLASH_PORT is not visible to launchd, so the port must be explicit.
    if not _has_option(serve_args, "--port"):
        serve_args += ["--port", str(port)]
    if serve.api_key and not _has_option(serve_args, "--api-key"):
        print(
            "warning: SPLASH_API_KEY is set in this shell but a login service "
            "does not see it; pass --api-key to include it (it is then stored "
            "in the plist, readable only by you)",
            file=sys.stderr,
        )
    executable = resolve_executable(args.executable)
    path = plist_path(port)
    if _state(port) is None:
        _require_free(serve.host, port)
    else:
        # bootout returns before launchd has finished removing the job, and an
        # immediate bootstrap then fails with an I/O error; wait for it, then
        # for the old server to release its port.
        _launchctl("bootout", _target(port), check=False)
        _wait_until(lambda: _state(port) is None, STOP_TIMEOUT, "unloading the old service")
        _require_free(serve.host, port, timeout=STOP_TIMEOUT)
    _write_private(path, plistlib.dumps(build_plist(port, executable, serve_args)))
    _launchctl("bootstrap", f"gui/{os.getuid()}", str(path))
    print(f"installed {label(port)}; it starts now and at every login")
    print(f"log: {log_path(port)}")
    return 0


def uninstall(args):
    ports = select_ports(args)
    if not ports:
        print("no mltf service is installed")
        return 0
    port = ports[0]
    _launchctl("bootout", _target(port), check=False)
    # bootout is asynchronous: return only once launchd has really let go.
    _wait_until(lambda: _state(port) is None, STOP_TIMEOUT, "unloading the service")
    path = plist_path(port)
    existed = path.exists()
    path.unlink(missing_ok=True)
    print(f"uninstalled {label(port)}" if existed else f"{label(port)} is not installed")
    return 0


def _select_one(args):
    """Resolve the single port an action applies to, and its launchd state."""
    ports = select_ports(args)
    if not ports:
        raise _no_service()
    port = ports[0]
    state = _state(port)
    if state is None:
        raise ServiceError(
            f"service on port {port} is not installed; "
            "run `mltf service install -- --model OWNER/REPO ...`"
        )
    return port, state


def start(args):
    port, state = _select_one(args)
    if _running(state):
        print(f"{label(port)} is already running")
        return 0
    _launchctl("kickstart", _target(port))
    print(f"started {label(port)}")
    return 0


def stop(args):
    port, state = _select_one(args)
    if not _running(state):
        print(f"{label(port)} is not running")
        return 0
    _launchctl("kill", "SIGTERM", _target(port))
    # Wait for the server to drain and exit, so `stop && start` cannot see the
    # old process still running and do nothing.
    _wait_until(lambda: not _running(_state(port)), STOP_TIMEOUT, "stopping the service")
    print(f"stopped {label(port)}; it starts again at the next login")
    return 0


def restart(args):
    port, _ = _select_one(args)
    _launchctl("kickstart", "-k", _target(port))
    print(f"restarted {label(port)}")
    return 0


def _print_status(port):
    state = _state(port)
    if state is None:
        print(f"{label(port)}: not installed")
        return False
    print(f"{label(port)}: {state.get('state', 'unknown')}")
    if "pid" in state:
        print(f"pid: {state['pid']}")
    if "last exit code" in state:
        print(f"last exit code: {state['last exit code']}")
    print(f"plist: {plist_path(port)}")
    print(f"log: {log_path(port)}")
    return True


def status(args):
    ports = select_ports(args)
    if not ports:
        print("no mltf service is installed")
        return 1
    results = []
    for index, port in enumerate(ports):
        if index:
            print()
        results.append(_print_status(port))
    return 0 if all(results) else 1


def run(args):
    if sys.platform != "darwin":
        raise ServiceError("mltf service is only available on macOS")
    return {
        "install": install,
        "uninstall": uninstall,
        "start": start,
        "stop": stop,
        "restart": restart,
        "status": status,
    }[args.action](args)
