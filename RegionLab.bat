@echo off
setlocal
cd /d "%~dp0"

if exist "RegionLab.exe" (
    if /i "%~1"=="--self-test" (
        "RegionLab.exe" %*
        exit /b
    )
    start "" "RegionLab.exe" %*
    exit /b 0
)
call "%~dp0launch_desktop.bat" RegionLab %*
exit /b %errorlevel%
