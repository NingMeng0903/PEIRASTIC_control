"""Helpers to start/stop netft_utils netft_node via roslaunch for NetFT calibration workflows."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

_NETFT_LAUNCH_XML = """<?xml version="1.0"?>
<launch>
  <node name="ft_data_nodes" pkg="netft_utils" type="netft_node" output="screen" args="{ip}"/>
</launch>
"""


def default_ros_setup_bash() -> str:
    env = os.environ.get("PEIRASTIC_ROS_SETUP", "").strip()
    if env and os.path.isfile(env):
        return env
    candidates = [
        Path.home() / "CampUsers/Pei/open_ws/devel/setup.bash",
        Path.home() / "open_ws/devel/setup.bash",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return ""


def reset_netft_workspace_artifacts(path_prefix: str) -> None:
    """Remove prior identification YAML and acquisition logs for a clean run."""
    root = Path(path_prefix)
    for rel in (
        "config/netft_calib_result.yaml",
        "log/write_data_identify.txt",
        "log/position_data.txt",
    ):
        p = root / rel
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass


def write_netft_launch_file(ip: str) -> Path:
    fd, name = tempfile.mkstemp(prefix="peirastic_netft_", suffix=".launch", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_NETFT_LAUNCH_XML.format(ip=ip))
    return Path(name)


def _roslaunch_bash_cmd(setup_bash: str, launch_path: Path) -> str:
    setup = f"source {setup_bash}" if setup_bash else "true"
    return (
        "set -euo pipefail; "
        "source /opt/ros/noetic/setup.bash; "
        f"{setup}; "
        f"roslaunch {launch_path}"
    )


def wait_for_netft_topic(timeout_s: float = 60.0, poll_s: float = 0.5) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["bash", "-lc", "source /opt/ros/noetic/setup.bash 2>/dev/null; rostopic list"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if r.returncode == 0 and r.stdout and "/netft_data" in r.stdout:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
        time.sleep(poll_s)
    return False


def start_netft_roslaunch(
    netft_ip: str,
    ros_setup_bash: str,
    *,
    wait_for_topic_s: float = 60.0,
) -> tuple[subprocess.Popen, Path]:
    """Start roslaunch with a temp launch file; return (process, launch_path)."""
    if not netft_ip.strip():
        raise ValueError("netft_ip is empty (set --netft-ip or PEIRASTIC_NETFT_IP).")
    setup = (ros_setup_bash or "").strip() or default_ros_setup_bash()
    if not setup or not os.path.isfile(setup):
        raise FileNotFoundError(
            "ROS overlay setup.bash not found. Pass --ros-setup /path/to/open_ws/devel/setup.bash "
            "or set PEIRASTIC_ROS_SETUP."
        )
    launch_path = write_netft_launch_file(netft_ip.strip())
    cmd = _roslaunch_bash_cmd(setup, launch_path)
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    if not wait_for_netft_topic(wait_for_topic_s):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            proc.terminate()
        proc.wait(timeout=10)
        try:
            launch_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"Timed out waiting for /netft_data after {wait_for_topic_s:.0f}s. "
            "Check NetFT IP, netft_utils in ROS_PACKAGE_PATH, and ROS master."
        )
    return proc, launch_path


def stop_netft_roslaunch(proc: Optional[subprocess.Popen], launch_path: Optional[Path]) -> None:
    if proc is None:
        if launch_path is not None:
            try:
                launch_path.unlink(missing_ok=True)
            except OSError:
                pass
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            proc.kill()
    if launch_path is not None:
        try:
            launch_path.unlink(missing_ok=True)
        except OSError:
            pass
