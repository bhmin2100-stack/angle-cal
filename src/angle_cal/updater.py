from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import __version__

try:
    from .build_info import APP_VERSION, BUILD_COMMIT, BUILD_DATE, BUILD_ID, UPDATE_CHANNEL
except Exception:  # pragma: no cover - source runs can survive missing build metadata.
    APP_VERSION = __version__
    BUILD_COMMIT = "local"
    BUILD_DATE = ""
    BUILD_ID = "local"
    UPDATE_CHANNEL = "personal"


APP_NAME = "AngleCal"
EXE_ASSET_NAME = "AngleCal.exe"
VERSION_ASSET_NAME = "version.json"
PERSONAL_RELEASE_API_URL = "https://api.github.com/repos/bhmin2100-stack/angle-cal/releases/tags/windows-latest"
PERSONAL_RELEASE_PAGE_URL = "https://github.com/bhmin2100-stack/angle-cal/releases/tag/windows-latest"
PERSONAL_DIRECT_EXE_URL = "https://github.com/bhmin2100-stack/angle-cal/releases/download/windows-latest/AngleCal.exe"
PERSONAL_DIRECT_VERSION_URL = "https://github.com/bhmin2100-stack/angle-cal/releases/download/windows-latest/version.json"
COMPANY_RELEASE_API_URL = "http://github.samsungds.net/api/v3/repos/bh2-min/AngleCal/releases/latest"
COMPANY_RELEASE_PAGE_URL = "http://github.samsungds.net/bh2-min/AngleCal/releases"
USER_AGENT = f"AngleCal/{__version__}"


class UpdateAuthenticationError(RuntimeError):
    """The Enterprise release API is not readable by the current Windows session."""


@dataclass(frozen=True)
class UpdateChannel:
    name: str
    release_api_url: str
    release_page_url: str
    direct_exe_url: str = ""
    direct_version_url: str = ""
    build_id_updates: bool = False


PERSONAL_CHANNEL = UpdateChannel(
    name="personal",
    release_api_url=PERSONAL_RELEASE_API_URL,
    release_page_url=PERSONAL_RELEASE_PAGE_URL,
    direct_exe_url=PERSONAL_DIRECT_EXE_URL,
    direct_version_url=PERSONAL_DIRECT_VERSION_URL,
    build_id_updates=True,
)
COMPANY_CHANNEL = UpdateChannel(
    name="company",
    release_api_url=COMPANY_RELEASE_API_URL,
    release_page_url=COMPANY_RELEASE_PAGE_URL,
    build_id_updates=True,
)


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    current_build_id: str
    current_commit: str
    latest_version: str
    latest_build_id: str
    latest_commit: str
    latest_build_date: str
    release_url: str
    exe_url: str
    sha256: str = ""
    size: int = 0
    notes: str = ""
    channel: str = "personal"
    build_id_updates: bool = False

    @property
    def is_available(self) -> bool:
        version_cmp = compare_versions(self.latest_version, self.current_version)
        if version_cmp != 0:
            return version_cmp > 0
        if not self.build_id_updates or self.current_build_id in ("", "local"):
            return False
        return bool(self.latest_build_id and self.latest_build_id != self.current_build_id)

    @property
    def current_label(self) -> str:
        if self.current_build_id and self.current_build_id != "local":
            return f"{self.current_version} ({self.current_build_id})"
        return self.current_version

    @property
    def latest_label(self) -> str:
        if self.latest_build_id:
            return f"{self.latest_version} ({self.latest_build_id})"
        return self.latest_version

    @property
    def notification_key(self) -> str:
        build_key = self.latest_build_id or self.sha256 or self.exe_url
        return f"{self.channel}|{self.latest_version}|{build_key}"


def selected_channel() -> UpdateChannel:
    return COMPANY_CHANNEL if str(UPDATE_CHANNEL).strip().lower() == "company" else PERSONAL_CHANNEL


def is_packaged_app() -> bool:
    return bool(getattr(sys, "frozen", False)) and Path(sys.executable).suffix.lower() == ".exe"


def current_version() -> str:
    return __version__


def current_build_id() -> str:
    return str(BUILD_ID or "local")


def build_info_dict() -> dict[str, str]:
    return {
        "package_version": __version__,
        "app_version": str(APP_VERSION),
        "build_commit": str(BUILD_COMMIT),
        "build_id": str(BUILD_ID),
        "build_date": str(BUILD_DATE),
        "update_channel": str(UPDATE_CHANNEL),
    }


def should_notify_update(info: UpdateInfo, last_notification_key: Optional[str]) -> bool:
    return info.is_available and info.notification_key != last_notification_key


def compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    max_len = max(len(left_parts), len(right_parts), 1)
    left_parts += [0] * (max_len - len(left_parts))
    right_parts += [0] * (max_len - len(right_parts))
    return (left_parts > right_parts) - (left_parts < right_parts)


def fetch_update_info(timeout: int = 10) -> UpdateInfo:
    channel = selected_channel()
    try:
        return _fetch_update_info_from_api(channel, timeout=timeout)
    except UpdateAuthenticationError:
        raise
    except Exception as api_error:
        if not channel.direct_version_url:
            raise RuntimeError(
                "업데이트 정보를 확인하지 못했습니다.\n"
                f"- 배포 채널: {channel.name}\n"
                f"- API: {api_error}\n"
                f"- 확인 URL: {channel.release_page_url}"
            ) from api_error
        try:
            return _fetch_update_info_from_direct_assets(channel, timeout=timeout)
        except Exception as direct_error:
            raise RuntimeError(
                "GitHub 업데이트 확인에 실패했습니다.\n"
                f"- API: {api_error}\n"
                f"- 직접 다운로드 정보: {direct_error}\n"
                f"- 확인 URL: {channel.release_page_url}"
            ) from direct_error


def _fetch_update_info_from_api(channel: UpdateChannel, timeout: int = 10) -> UpdateInfo:
    release = _read_json(channel.release_api_url, timeout=timeout)
    assets = release.get("assets") or []
    exe_asset = _find_asset(assets, EXE_ASSET_NAME)
    version_asset = _find_asset(assets, VERSION_ASSET_NAME)
    if channel.name == COMPANY_CHANNEL.name and not exe_asset:
        raise RuntimeError(f"{EXE_ASSET_NAME} release asset not found.")
    if channel.name == COMPANY_CHANNEL.name and not version_asset:
        raise RuntimeError(f"{VERSION_ASSET_NAME} release asset not found.")
    metadata = _read_json(str(version_asset.get("browser_download_url") or ""), timeout=timeout) if version_asset else {}
    return _update_info_from_release(channel, release, metadata, exe_asset)


def _fetch_update_info_from_direct_assets(channel: UpdateChannel, timeout: int = 10) -> UpdateInfo:
    metadata = _read_json(channel.direct_version_url, timeout=timeout)
    if not str(metadata.get("version") or ""):
        raise RuntimeError(f"{channel.direct_version_url} did not contain a version.")
    return _update_info_from_release(channel, {}, metadata, None, fallback_exe_url=channel.direct_exe_url)


def _update_info_from_release(
    channel: UpdateChannel,
    release: dict,
    metadata: dict,
    exe_asset: Optional[dict],
    *,
    fallback_exe_url: str = "",
) -> UpdateInfo:
    asset_name = str(metadata.get("asset") or metadata.get("exe_asset") or EXE_ASSET_NAME)
    if exe_asset and str(exe_asset.get("name") or "") != asset_name:
        exe_asset = None
    assets = release.get("assets") or []
    exe_asset = exe_asset or _find_asset(assets, asset_name)
    exe_url = str(metadata.get("downloadUrl") or (exe_asset or {}).get("browser_download_url") or fallback_exe_url)
    if not exe_url:
        raise RuntimeError(f"{asset_name} release asset not found.")
    latest_version = str(metadata.get("version") or release.get("tag_name") or "")
    if not latest_version:
        raise RuntimeError("Release metadata did not contain a version.")
    return UpdateInfo(
        current_version=current_version(),
        current_build_id=current_build_id(),
        current_commit=str(BUILD_COMMIT or ""),
        latest_version=latest_version,
        latest_build_id=str(metadata.get("build_id") or release.get("target_commitish") or ""),
        latest_commit=str(metadata.get("commit") or ""),
        latest_build_date=str(metadata.get("publishedAtUtc") or metadata.get("build_date") or release.get("published_at") or ""),
        release_url=str(release.get("html_url") or channel.release_page_url),
        exe_url=exe_url,
        sha256=str(metadata.get("sha256") or ""),
        size=int(metadata.get("size") or (exe_asset or {}).get("size") or 0),
        notes=str(metadata.get("notes") or release.get("body") or "").strip(),
        channel=channel.name,
        build_id_updates=channel.build_id_updates,
    )


def download_update(info: UpdateInfo, progress: Optional[Callable[[int, int], None]] = None, timeout: int = 30) -> Path:
    update_dir = update_work_dir()
    update_dir.mkdir(parents=True, exist_ok=True)
    target = update_dir / f"{_safe_name(info.latest_build_id or info.latest_version)}.new.exe"
    hasher = hashlib.sha256()
    downloaded = 0
    try:
        request = urllib.request.Request(info.exe_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or info.size or 0)
            with target.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total)
    except Exception as urllib_error:
        target.unlink(missing_ok=True)
        try:
            _download_file_with_powershell(info.exe_url, target, timeout=timeout)
        except Exception as ps_error:
            raise RuntimeError(f"업데이트 다운로드 실패: urllib={urllib_error}; PowerShell={ps_error}") from ps_error
        downloaded = target.stat().st_size
        hasher = hashlib.sha256(target.read_bytes())
        if progress:
            progress(downloaded, info.size or downloaded)
    if info.sha256 and hasher.hexdigest().lower() != info.sha256.lower():
        target.unlink(missing_ok=True)
        raise RuntimeError("다운로드한 업데이트 파일이 SHA-256 검증에 실패했습니다.")
    return target


def update_install_error(current_exe: Optional[Path] = None) -> str:
    if not is_packaged_app() and current_exe is None:
        return "소스 실행 상태에서는 EXE 자동 업데이트를 적용할 수 없습니다. 배포된 AngleCal.exe에서 실행하세요."
    executable = (current_exe or Path(sys.executable)).resolve()
    parent = executable.parent
    probe = parent / f".anglecal_update_write_test_{os.getpid()}"
    try:
        with probe.open("xb"):
            pass
        probe.unlink()
    except (OSError, PermissionError) as exc:
        return (
            f"설치 폴더에 업데이트 파일을 쓸 권한이 없습니다.\n{parent}\n\n"
            "AngleCal.exe를 사용자가 쓰기 가능한 폴더로 옮긴 뒤 다시 실행하세요. "
            f"({exc})"
        )
    return ""


def launch_self_update(downloaded_exe: Path) -> None:
    if not is_packaged_app():
        raise RuntimeError("소스 실행 상태에서는 EXE 자동 업데이트를 적용할 수 없습니다.")
    current_exe = Path(sys.executable).resolve()
    if error := update_install_error(current_exe):
        raise RuntimeError(error)
    script = _write_update_script(current_exe=current_exe, downloaded_exe=downloaded_exe.resolve(), pid=os.getpid())
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        close_fds=True,
        creationflags=creationflags,
    )


def update_work_dir() -> Path:
    return Path(tempfile.gettempdir()) / "anglecal_update"


def _read_json(url: str, timeout: int) -> dict:
    return json.loads(_read_url_bytes(url, timeout=timeout).decode("utf-8-sig"))


def _read_url_bytes(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise UpdateAuthenticationError(
                "회사 GitHub Release 접근 권한이 없습니다. 배포 관리자에게 Release 읽기 권한을 확인해 달라고 요청하세요."
            ) from error
        raise RuntimeError(f"HTTP {error.code}: {error.reason}") from error
    except Exception as urllib_error:
        if os.name != "nt":
            raise
        try:
            return _read_url_with_powershell(url, timeout=timeout)
        except Exception as ps_error:
            raise RuntimeError(f"urllib={urllib_error}; PowerShell={ps_error}") from ps_error


def _read_url_with_powershell(url: str, timeout: int) -> bytes:
    command = (
        "$ProgressPreference = 'SilentlyContinue'; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
        f"$response = Invoke-WebRequest -Uri {_ps_quote(url)} -UseBasicParsing -TimeoutSec {max(1, timeout)}; "
        "$content = $response.Content; "
        "$bytes = if ($content -is [byte[]]) { $content } else { [System.Text.Encoding]::UTF8.GetBytes([string]$content) }; "
        "[Console]::Out.Write([Convert]::ToBase64String($bytes))"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        timeout=timeout + 15,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if "401" in stderr or "403" in stderr:
            raise UpdateAuthenticationError(
                "회사 GitHub Release 접근 권한이 없습니다. 배포 관리자에게 Release 읽기 권한을 확인해 달라고 요청하세요."
            )
        raise RuntimeError(stderr or f"PowerShell exited with {completed.returncode}")
    return base64.b64decode(completed.stdout.strip())


def _download_file_with_powershell(url: str, target: Path, timeout: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
        f"Invoke-WebRequest -Uri {_ps_quote(url)} -OutFile {_ps_quote(target)} -UseBasicParsing -TimeoutSec {max(1, timeout)}"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        timeout=timeout + 60,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"PowerShell exited with {completed.returncode}")
    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError("PowerShell did not create a valid download file.")


def _find_asset(assets: list[dict], name: str) -> Optional[dict]:
    return next((asset for asset in assets if asset.get("name") == name), None)


def _version_parts(value: str) -> list[int]:
    return [int(part) for part in re.findall(r"\d+", value)]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or APP_NAME


def _ps_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _write_update_script(current_exe: Path, downloaded_exe: Path, pid: int) -> Path:
    update_dir = update_work_dir()
    update_dir.mkdir(parents=True, exist_ok=True)
    script = update_dir / f"anglecal_update_{pid}.ps1"
    log_path = update_dir / "anglecal_update.log"
    backup_exe = update_dir / f"{current_exe.name}.bak"
    script.write_text(
        f"""
$ErrorActionPreference = 'Stop'
$targetExe = {_ps_quote(current_exe)}
$newExe = {_ps_quote(downloaded_exe)}
$backupExe = {_ps_quote(backup_exe)}
$logPath = {_ps_quote(log_path)}
$pidToWait = {pid}
function Move-WithRetry($source, $destination) {{ for ($i = 0; $i -lt 40; $i++) {{ try {{ Move-Item -LiteralPath $source -Destination $destination -Force; return }} catch {{ Start-Sleep -Milliseconds 500 }} }}; Move-Item -LiteralPath $source -Destination $destination -Force }}
try {{
    Wait-Process -Id $pidToWait -Timeout 60 -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 300
    if (Test-Path -LiteralPath $backupExe) {{ Remove-Item -LiteralPath $backupExe -Force -ErrorAction SilentlyContinue }}
    if (Test-Path -LiteralPath $targetExe) {{ Move-WithRetry $targetExe $backupExe }}
    try {{
        Move-WithRetry $newExe $targetExe
    }} catch {{
        if ((Test-Path -LiteralPath $backupExe) -and -not (Test-Path -LiteralPath $targetExe)) {{
            Move-Item -LiteralPath $backupExe -Destination $targetExe -Force
        }}
        throw
    }}
    Start-Process -FilePath $targetExe -WorkingDirectory (Split-Path -Parent $targetExe)
    Start-Sleep -Seconds 2
    if (Test-Path -LiteralPath $backupExe) {{ Remove-Item -LiteralPath $backupExe -Force -ErrorAction SilentlyContinue }}
}} catch {{
    $_ | Out-File -FilePath $logPath -Encoding UTF8
    exit 1
}} finally {{
    Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue
}}
""".lstrip(),
        encoding="utf-8-sig",
    )
    return script
