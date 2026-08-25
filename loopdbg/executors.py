"""Sandbox execution backends.

Two interchangeable backends implement :class:`Executor`:

* :class:`KubernetesExecutor` runs commands inside an ephemeral pod using the
  official ``kubernetes`` client (``CoreV1Api`` plus ``stream(...)`` for exec).
* :class:`LocalExecutor` runs the same commands in a temporary directory on the
  host, so the example scenario and the test suite work without a cluster.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any, Final, Protocol, runtime_checkable

from .config import SandboxConfig

log = logging.getLogger(__name__)

# Channel 3 of the SPDY/WebSocket exec stream carries the termination status.
ERROR_CHANNEL: Final[int] = 3


class SandboxError(RuntimeError):
    """Raised when the sandbox cannot be created, reached, or torn down."""


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Outcome of a single command executed inside the sandbox."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self, limit: int = 8000) -> str:
        """Human/model readable rendering with a hard size bound."""
        head = (
            f"$ {' '.join(self.command)}\n"
            f"exit_code={self.exit_code} duration={self.duration_s:.2f}s"
            f"{' TIMED_OUT' if self.timed_out else ''}\n"
        )
        body = ""
        if self.stdout:
            body += f"--- stdout ---\n{self.stdout}\n"
        if self.stderr:
            body += f"--- stderr ---\n{self.stderr}\n"
        if not body:
            body = "(no output)\n"
        if len(body) > limit:
            keep = limit // 2
            body = f"{body[:keep]}\n... [{len(body) - limit} chars elided] ...\n{body[-keep:]}"
        return head + body


@runtime_checkable
class Executor(Protocol):
    """Minimal contract the agent's tools depend on."""

    name: str

    def start(self, code_volume_map: dict[str, str]) -> str: ...

    def exec(self, cmd: list[str], timeout_s: int | None = None) -> ExecResult: ...

    def sync(self, code_volume_map: dict[str, str]) -> None: ...

    def cleanup(self) -> None: ...


def _tar_bytes(sources: dict[str, str]) -> bytes:
    """Pack ``{local_path: remote_path}`` into a tar stream rooted at ``/``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for local, remote in sources.items():
            local_path = Path(local).resolve()
            if not local_path.exists():
                raise SandboxError(f"code_volume_map source does not exist: {local}")
            arcname = remote.lstrip("/")
            tar.add(
                str(local_path),
                arcname=arcname,
                filter=lambda ti: None if "/.git/" in ti.name or ti.name.endswith("/.git") else ti,
            )
    return buf.getvalue()


class LocalExecutor:
    """Runs commands in a scratch directory on the host.

    Used by the bundled example and the test suite. It mirrors the Kubernetes
    backend's semantics: an isolated filesystem seeded from ``code_volume_map``,
    commands run with a working directory of :attr:`SandboxConfig.workdir`.
    """

    name = "local"

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._tmp: TemporaryDirectory[str] | None = None
        self._root: Path | None = None
        self.pod_name: str = ""

    @property
    def root(self) -> Path:
        if self._root is None:
            raise SandboxError("sandbox not started")
        return self._root

    def start(self, code_volume_map: dict[str, str]) -> str:
        self._tmp = TemporaryDirectory(prefix="debug-sandbox-")
        self._root = Path(self._tmp.name)
        self.pod_name = f"local-{uuid.uuid4().hex[:8]}"
        self.sync(code_volume_map)
        log.info("local sandbox %s ready at %s", self.pod_name, self._root)
        return self.pod_name

    def sync(self, code_volume_map: dict[str, str]) -> None:
        for local, remote in code_volume_map.items():
            src = Path(local).resolve()
            dst = self.root / remote.lstrip("/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)

    def exec(self, cmd: list[str], timeout_s: int | None = None) -> ExecResult:
        workdir = self.root / self._config.workdir.lstrip("/")
        workdir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s or self._config.exec_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                command=cmd,
                exit_code=124,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                duration_s=time.monotonic() - started,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return ExecResult(cmd, 127, "", str(exc), time.monotonic() - started)
        return ExecResult(
            command=cmd,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=time.monotonic() - started,
        )

    def cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
            self._root = None
            log.info("local sandbox %s cleaned up", self.pod_name)

    def __enter__(self) -> LocalExecutor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cleanup()


class KubernetesExecutor:
    """Runs commands inside an ephemeral pod in a dedicated debug namespace."""

    name = "kubernetes"

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self.pod_name: str = ""
        self._api: Any = self._build_api()

    @staticmethod
    def _build_api() -> Any:
        try:
            from kubernetes import client
            from kubernetes import config as kube_config
        except ImportError as exc:  # pragma: no cover - import guard
            raise SandboxError(
                "the 'kubernetes' package is required for the kubernetes executor; "
                "install it with `pip install kubernetes`"
            ) from exc
        try:
            kube_config.load_incluster_config()
            log.info("loaded in-cluster kubeconfig")
        except Exception:  # noqa: BLE001 - fall back to a local kubeconfig
            kube_config.load_kube_config()
            log.info("loaded local kubeconfig")
        return client.CoreV1Api()

    # ------------------------------------------------------------------ pod --
    def _pod_manifest(self) -> Any:
        from kubernetes import client

        cfg = self._config
        return client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=self.pod_name,
                namespace=cfg.namespace,
                labels={**cfg.labels, "debug-agent/pod": self.pod_name},
            ),
            spec=client.V1PodSpec(
                restart_policy="Never",
                service_account_name=cfg.service_account,
                automount_service_account_token=False,
                security_context=client.V1PodSecurityContext(
                    run_as_non_root=True,
                    run_as_user=1000,
                    fs_group=1000,
                    seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault"),
                ),
                containers=[
                    client.V1Container(
                        name="sandbox",
                        image=cfg.image,
                        # Idle so the agent can exec into a stable filesystem.
                        command=["/bin/sh", "-c", "mkdir -p $WORKDIR && sleep infinity"],
                        env=[client.V1EnvVar(name="WORKDIR", value=cfg.workdir)],
                        working_dir=cfg.workdir,
                        security_context=client.V1SecurityContext(
                            allow_privilege_escalation=False,
                            read_only_root_filesystem=False,
                            capabilities=client.V1Capabilities(drop=["ALL"]),
                        ),
                        resources=client.V1ResourceRequirements(
                            requests={"cpu": cfg.cpu_request, "memory": cfg.memory_request},
                            limits={"cpu": cfg.cpu_limit, "memory": cfg.memory_limit},
                        ),
                        volume_mounts=[client.V1VolumeMount(name="workspace", mount_path=cfg.workdir)],
                    )
                ],
                volumes=[client.V1Volume(name="workspace", empty_dir=client.V1EmptyDirVolumeSource())],
            ),
        )

    def start(self, code_volume_map: dict[str, str]) -> str:
        from kubernetes.client.rest import ApiException

        self.pod_name = f"debug-{uuid.uuid4().hex[:10]}"
        try:
            self._api.create_namespaced_pod(namespace=self._config.namespace, body=self._pod_manifest())
        except ApiException as exc:
            raise SandboxError(f"failed to create pod {self.pod_name}: {exc.reason}") from exc
        self._wait_ready()
        self.sync(code_volume_map)
        return self.pod_name

    def _wait_ready(self) -> None:
        from kubernetes.client.rest import ApiException

        deadline = time.monotonic() + self._config.pod_ready_timeout_s
        last_phase = "Unknown"
        while time.monotonic() < deadline:
            try:
                pod = self._api.read_namespaced_pod(name=self.pod_name, namespace=self._config.namespace)
            except ApiException as exc:  # pragma: no cover - transient API errors
                raise SandboxError(f"failed to read pod {self.pod_name}: {exc.reason}") from exc
            last_phase = pod.status.phase or "Unknown"
            if last_phase == "Running":
                conditions = pod.status.conditions or []
                if any(c.type == "Ready" and c.status == "True" for c in conditions):
                    return
            if last_phase in {"Failed", "Succeeded"}:
                raise SandboxError(f"pod {self.pod_name} reached terminal phase {last_phase}")
            time.sleep(1.0)
        self.cleanup()
        raise SandboxError(
            f"pod {self.pod_name} not ready after {self._config.pod_ready_timeout_s}s (phase={last_phase})"
        )

    # ----------------------------------------------------------------- exec --
    def exec(self, cmd: list[str], timeout_s: int | None = None) -> ExecResult:
        from kubernetes.stream import stream

        timeout = timeout_s or self._config.exec_timeout_s
        started = time.monotonic()
        resp = stream(
            self._api.connect_get_namespaced_pod_exec,
            self.pod_name,
            self._config.namespace,
            container="sandbox",
            command=["/bin/sh", "-c", _shell_join(cmd, self._config.workdir)],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        timed_out = False
        try:
            while resp.is_open():
                if time.monotonic() - started > timeout:
                    timed_out = True
                    break
                resp.update(timeout=1)
                if resp.peek_stdout():
                    stdout_chunks.append(resp.read_stdout())
                if resp.peek_stderr():
                    stderr_chunks.append(resp.read_stderr())
            exit_code = 124 if timed_out else _exit_code_from(resp)
        finally:
            resp.close()
        return ExecResult(
            command=cmd,
            exit_code=exit_code,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            duration_s=time.monotonic() - started,
            timed_out=timed_out,
        )

    def sync(self, code_volume_map: dict[str, str]) -> None:
        """Stream the local sources into the pod with ``tar x`` over stdin."""
        from kubernetes.stream import stream

        payload = _tar_bytes(code_volume_map)
        resp = stream(
            self._api.connect_get_namespaced_pod_exec,
            self.pod_name,
            self._config.namespace,
            container="sandbox",
            command=["tar", "xmf", "-", "-C", "/"],
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        stderr_chunks: list[str] = []
        try:
            resp.write_stdin(payload)
            # tar only finishes once stdin is closed. The v4 exec protocol has no
            # EOF frame, so close the stdin channel when the v5 subprotocol gives
            # us one and fall back to waiting for the process to exit otherwise.
            try:
                resp.close_channel(0)
            except Exception:  # noqa: BLE001 - v4 server, no stdin channel close
                log.debug("stdin channel close unsupported; relying on process exit")
            deadline = time.monotonic() + self._config.exec_timeout_s
            while resp.is_open() and time.monotonic() < deadline:
                resp.update(timeout=1)
                if resp.peek_stderr():
                    stderr_chunks.append(resp.read_stderr())
                if resp.peek_stdout():
                    resp.read_stdout()
            if resp.is_open():
                raise SandboxError(f"tar extraction into {self.pod_name} did not finish in time")
            code = _exit_code_from(resp)
        finally:
            resp.close()
        if code != 0:
            raise SandboxError(
                f"tar extraction into {self.pod_name} failed with exit code {code}: "
                f"{''.join(stderr_chunks).strip() or '(no stderr)'}"
            )

    def cleanup(self) -> None:
        from kubernetes.client.rest import ApiException

        if not self.pod_name:
            return
        try:
            self._api.delete_namespaced_pod(
                name=self.pod_name,
                namespace=self._config.namespace,
                grace_period_seconds=0,
            )
            log.info("deleted pod %s", self.pod_name)
        except ApiException as exc:
            if exc.status != 404:
                log.warning("failed to delete pod %s: %s", self.pod_name, exc.reason)
        finally:
            self.pod_name = ""

    def __enter__(self) -> KubernetesExecutor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cleanup()


def _shell_join(cmd: list[str], workdir: str) -> str:
    from shlex import quote

    return f"cd {quote(workdir)} && " + " ".join(quote(part) for part in cmd)


def _exit_code_from(resp: Any) -> int:
    """Read the exit status off the exec error channel without ever raising.

    ``WSClient.returncode`` parses the channel itself but assumes a well-formed
    ``Failure`` payload; a dropped upgrade or an OOMKill produces a payload with
    no ``details.causes`` and the property raises. Read the raw channel first so
    a torn-down exec is reported as a failure rather than as success.
    """
    raw = ""
    channels = getattr(resp, "_channels", None)
    if isinstance(channels, dict):
        raw = str(channels.get(ERROR_CHANNEL, "") or "")
    if not raw:
        try:
            raw = str(resp.read_channel(ERROR_CHANNEL) or "")
        except Exception:  # noqa: BLE001 - channel already consumed or closed
            raw = ""
    if not raw.strip():
        # No status at all: the connection died mid-command. Reporting success
        # here would let a broken exec masquerade as a passing test run.
        log.warning("exec channel closed without a termination status")
        return 1
    try:
        parsed = json.loads(raw)
    except ValueError:
        try:
            import yaml

            parsed = yaml.safe_load(raw)
        except Exception:  # noqa: BLE001 - unparseable status
            return 1
    if not isinstance(parsed, dict):
        return 1
    if parsed.get("status") == "Success":
        return 0
    causes = parsed.get("details", {}).get("causes", []) if isinstance(parsed.get("details"), dict) else []
    for cause in causes:
        if isinstance(cause, dict) and cause.get("reason") == "ExitCode":
            try:
                return int(cause.get("message", "1"))
            except (TypeError, ValueError):
                return 1
    log.warning("exec failed without an ExitCode cause: %s", parsed.get("message", raw)[:200])
    return 1


def build_executor(config: SandboxConfig) -> Executor:
    """Factory selecting the backend named by :attr:`SandboxConfig.kind`."""
    if config.kind == "local":
        return LocalExecutor(config)
    return KubernetesExecutor(config)
