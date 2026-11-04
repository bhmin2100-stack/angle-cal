@echo off
cd /d "%~dp0"
if exist "dist\RegionLab\RegionLab.exe" (
    start "" "dist\RegionLab\RegionLab.exe" %*
) else (
    start "" ".venv\Scripts\pythonw.exe" "run_region_lab.py" %*
)
