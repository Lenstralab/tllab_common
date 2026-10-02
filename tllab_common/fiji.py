import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# the Fiji launchers that ship with Fiji, in order of preference, for the current platform
LAUNCHERS = {
    "win32": ["fiji-windows-x64.exe", "ImageJ-win64.exe", "fiji-win64.exe", "fiji.exe"],
    "darwin": ["fiji-macos", "fiji-macos-arm64", "fiji-macos-x64", "ImageJ-macosx", "fiji-macosx"],
    "linux": ["fiji", "fiji-linux-x64", "ImageJ-linux64", "fiji-linux64"],
}

# directories where Fiji is commonly installed, for the current platform
DIRS = {
    "win32": ["~/Fiji.app", "~/Fiji", "C:/Program Files/Fiji.app", "C:/Fiji.app"],
    "darwin": ["~/Applications/Fiji.app", "/Applications/Fiji.app", "~/Fiji.app"],
    "linux": ["/opt/Fiji.app", "/usr/local/Fiji.app", "~/Fiji.app", "~/Fiji"],
}


def fiji_launcher_names() -> list[str]:
    if sys.platform != "darwin":
        return LAUNCHERS.get(sys.platform, LAUNCHERS["linux"])
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(platform.machine())
    return ([f"fiji-macos-{arch}"] if arch is not None else []) + LAUNCHERS["darwin"]


def fiji_launchers(directory: Path) -> list[Path]:
    """the Fiji launchers in a Fiji installation directory, in order of preference"""
    found = [p for p in (directory / name for name in fiji_launcher_names()) if p.is_file()]
    # on macOS the launchers live in Contents/MacOS
    return found + [p for p in (directory / "Contents/MacOS" / n for n in fiji_launcher_names()) if p.is_file()]


def fiji_executable(fiji_path: Optional[Path | str] = None) -> Path:
    """find the Fiji launcher to run scripts with: the launcher itself or the first launcher in an installation
    directory, taken from fiji_path, the FIJI_PATH or FIJI_HOME environment variable, the PATH, or one of the
    standard installation directories"""
    if fiji_path is None:
        fiji_path = next((os.environ[var] for var in ("FIJI_PATH", "FIJI_HOME") if os.environ.get(var)), None)
    if fiji_path is not None:
        fiji_path = Path(fiji_path)
        launchers = fiji_launchers(fiji_path)
        if len(launchers) > 0:
            return launchers[0]
        if fiji_path.is_file():  # a launcher that this Fiji version did not ship (a symlink, renamed, ...)
            return fiji_path
        raise FileNotFoundError(f"No Fiji launcher ({', '.join(fiji_launcher_names())}) in {fiji_path}")
    for name in fiji_launcher_names():
        which = shutil.which(name)
        if which is not None:
            return Path(which)
    for directory in DIRS.get(sys.platform, DIRS["linux"]):
        launchers = fiji_launchers(Path(directory).expanduser())
        if len(launchers) > 0:
            return launchers[0]
    raise FileNotFoundError(
        "Could not find Fiji. Pass fiji_path or set the FIJI_PATH environment variable to the Fiji installation "
        f"directory or to the Fiji launcher. Looked for {', '.join(fiji_launcher_names())} on the PATH and in "
        f"{', '.join(DIRS.get(sys.platform, DIRS['linux']))}."
    )


def fiji_param(value: str | int | float | bool) -> str:
    """a single key=value pair for the parameter list of `fiji --run script '<parameters>'`"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return str(value)


def run_fiji(
    script: str | Path,
    fiji_path: Optional[str | Path],
    args: Optional[dict[Any, Any]] = None,
    env_args: Optional[dict[Any, Any]] = None,
) -> subprocess.CompletedProcess[str]:
    executable = fiji_executable(fiji_path)
    env = dict(os.environ)
    if env_args is not None:
        for k, v in env_args.items():
            env[k] = v
    headless = "-Djava.awt.headless=true"
    java_tool_options = env.get("JAVA_TOOL_OPTIONS", "")
    env["JAVA_TOOL_OPTIONS"] = (
        f"{java_tool_options} {headless}" if headless not in java_tool_options else java_tool_options
    )
    cmd = [str(executable), "--run", str(script)]
    if args is not None:
        cmd.append(",".join(f"{k}={fiji_param(v)}" for k, v in args.items()))

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
