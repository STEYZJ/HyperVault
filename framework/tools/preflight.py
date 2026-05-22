from __future__ import annotations

import grp
import os
import pwd
import stat
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class DockerPreflightResult(BaseModel):
    ok: bool
    socket_path: str
    socket_exists: bool
    socket_owner: str | None = None
    socket_group: str | None = None
    socket_mode: str | None = None
    current_user: str
    current_groups: list[str] = Field(default_factory=list)
    can_read_write_socket: bool = False
    compose_available: bool = False
    compose_error: str | None = None
    guidance: list[str] = Field(default_factory=list)


def docker_preflight(
    socket_path: Path = Path("/var/run/docker.sock"),
    project_dir: Path | None = None,
) -> DockerPreflightResult:
    current_user = pwd.getpwuid(os.getuid()).pw_name
    current_groups = sorted(group_names_for_current_user())
    guidance: list[str] = []
    socket_exists = socket_path.exists()
    socket_owner = None
    socket_group = None
    socket_mode = None
    can_read_write_socket = False
    compose_available = False
    compose_error = None

    if not socket_exists:
        guidance.append("Docker socket does not exist. Start Docker or use an external Qdrant URL.")
    else:
        socket_stat = socket_path.stat()
        socket_owner = pwd.getpwuid(socket_stat.st_uid).pw_name
        socket_group = grp.getgrgid(socket_stat.st_gid).gr_name
        socket_mode = stat.filemode(socket_stat.st_mode)
        can_read_write_socket = os.access(socket_path, os.R_OK | os.W_OK)
        if not can_read_write_socket:
            guidance.append(
                "Current user cannot read/write /var/run/docker.sock. Ask an administrator to "
                "fix Docker socket ownership, add the user to the effective docker group and "
                "re-login, use root/rootless Docker, or configure an external Qdrant service."
            )

    if can_read_write_socket:
        process = subprocess.run(
            ["docker", "compose", "ps"],
            cwd=project_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        compose_available = process.returncode == 0
        if not compose_available:
            compose_error = (process.stderr or process.stdout).strip()
            guidance.append("`docker compose ps` failed; check Docker Compose installation.")
    return DockerPreflightResult(
        ok=socket_exists and can_read_write_socket and compose_available,
        socket_path=str(socket_path),
        socket_exists=socket_exists,
        socket_owner=socket_owner,
        socket_group=socket_group,
        socket_mode=socket_mode,
        current_user=current_user,
        current_groups=current_groups,
        can_read_write_socket=can_read_write_socket,
        compose_available=compose_available,
        compose_error=compose_error,
        guidance=guidance,
    )


def group_names_for_current_user() -> set[str]:
    names: set[str] = set()
    for gid in os.getgroups():
        try:
            names.add(grp.getgrgid(gid).gr_name)
        except KeyError:
            names.add(str(gid))
    return names
