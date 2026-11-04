@echo off
rem Immutable runtime pins. Update URL, archive hash and EXE hash together.
set "ANGLECAL_RUNTIME_ID=desktop-20261104-r3"
set "ANGLECAL_RUNTIME_REQUIRED="
if /i "%~1"=="RegionLab" goto region_lab
if /i "%~1"=="AngleCal" goto angle_cal
exit /b 2
:region_lab
set "ANGLECAL_RUNTIME_KIND=zip"
set "ANGLECAL_RUNTIME_URL=https://github.com/bhmin2100-stack/angle-cal/releases/download/region-lab-20261104-r2/RegionLab-Windows-x64-20261104-r2.zip"
set "ANGLECAL_RUNTIME_SHA=87b8c3f415cc68453767805250e26e9336158eb89159d337f9531b2f24b1ba19"
set "ANGLECAL_RUNTIME_EXE=RegionLab\RegionLab.exe"
set "ANGLECAL_RUNTIME_EXE_SHA=0c13caa2baededbc01c3d9024aaa5c1cefc0b121695dbc62c16d8b2b11ae3ef7"
set "ANGLECAL_RUNTIME_REQUIRED=RegionLab\_internal\python313.dll"
exit /b 0
:angle_cal
set "ANGLECAL_RUNTIME_KIND=exe"
set "ANGLECAL_RUNTIME_URL=https://github.com/bhmin2100-stack/angle-cal/releases/download/desktop-runtime-20261104-r3/AngleCal-desktop-20261104-r3.exe"
set "ANGLECAL_RUNTIME_SHA=50480925aea1be6babc2086d7882975205b9dfabeff07be19ea854269306800e"
set "ANGLECAL_RUNTIME_EXE=AngleCal.exe"
set "ANGLECAL_RUNTIME_EXE_SHA=50480925aea1be6babc2086d7882975205b9dfabeff07be19ea854269306800e"
exit /b 0
