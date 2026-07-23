[CmdletBinding()]
param([switch]$Clean)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSCommandPath
Set-Location $root
$srcPath = Join-Path $root "src"
$env:PYTHONPATH = $srcPath

function Write-Utf8NoBom($Path, $Content) {
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

$null = & $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller installation failed." }
}

$version = (& $python -c "import tomllib; from pathlib import Path; import angle_cal; pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8')); project_version = pyproject['project']['version']; assert angle_cal.__version__ == project_version, f'__init__={angle_cal.__version__} pyproject={project_version}'; print(angle_cal.__version__)").Trim()
if (-not $version) { throw "Could not read angle_cal.__version__." }

$buildInfo = Join-Path $root "src\angle_cal\build_info.py"
$originalBuildInfo = Get-Content -LiteralPath $buildInfo -Raw
$buildDate = (Get-Date).ToUniversalTime().ToString("o")
$commit = (git rev-parse HEAD).Trim()
$buildId = "company-" + (Get-Date -Format "yyyyMMddHHmmss")
$smokeFile = Join-Path ([System.IO.Path]::GetTempPath()) ("anglecal-build-info-" + [guid]::NewGuid().ToString("N") + ".json")

try {
    $companyBuildInfo = @"
from __future__ import annotations

APP_VERSION = "$version"
BUILD_COMMIT = "$commit"
BUILD_ID = "$buildId"
BUILD_DATE = "$buildDate"
UPDATE_CHANNEL = "company"
"@
    Write-Utf8NoBom $buildInfo $companyBuildInfo

    $embeddedVersion = (& $python -c "from angle_cal import __version__; from angle_cal.build_info import APP_VERSION, UPDATE_CHANNEL; print(f'{__version__}|{APP_VERSION}|{UPDATE_CHANNEL}')").Trim()
    if ($embeddedVersion -ne "$version|$version|company") { throw "Build metadata validation failed: $embeddedVersion" }

    if ($Clean) {
        Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
    }

    & $python -m PyInstaller --noconfirm --clean AngleCal.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

    $builtExe = Join-Path $root "dist\AngleCal.exe"
    if (-not (Test-Path $builtExe)) { throw "The built EXE was not found: $builtExe" }

    $process = Start-Process -FilePath $builtExe -ArgumentList @("--build-info-json", $smokeFile) -PassThru -WindowStyle Hidden
    if (-not $process.WaitForExit(30000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "Build smoke test timed out."
    }
    if ($process.ExitCode -ne 0) { throw "Build smoke test failed with exit code $($process.ExitCode)." }
    if (-not (Test-Path $smokeFile)) { throw "Build smoke test did not create metadata file." }

    $smoke = Get-Content -LiteralPath $smokeFile -Raw | ConvertFrom-Json
    if ($smoke.package_version -ne $version -or $smoke.app_version -ne $version -or $smoke.update_channel -ne "company") {
        throw "Company channel was not embedded in AngleCal.exe. Smoke metadata: $(Get-Content -LiteralPath $smokeFile -Raw)"
    }

    Write-Host "Company EXE: $builtExe"
    Write-Host "Version: $version"
    Write-Host "Build ID: $buildId"
    Write-Host "Update channel: company"
} finally {
    Write-Utf8NoBom $buildInfo $originalBuildInfo
    Remove-Item -LiteralPath $smokeFile -Force -ErrorAction SilentlyContinue
}
