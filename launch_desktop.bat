@echo off
setlocal
rem GitHub Desktop checkout launcher. No Python, pip, or execution-policy changes.
if /i not "%~1"=="AngleCal" if /i not "%~1"=="RegionLab" exit /b 2
if not defined LOCALAPPDATA (
    echo [AngleCal] LOCALAPPDATA is unavailable.
    exit /b 1
)
call "%~dp0desktop_runtime.bat" "%~1"
if errorlevel 1 goto failed
set "ANGLECAL_CACHE=%LOCALAPPDATA%\AngleCal\runtimes\%ANGLECAL_RUNTIME_ID%\%~1"
if not exist "%ANGLECAL_CACHE%" mkdir "%ANGLECAL_CACHE%"
if errorlevel 1 goto failed
set "ANGLECAL_DESKTOP_LOG=%ANGLECAL_CACHE%\launcher.log"
set "ANGLECAL_EXE=%ANGLECAL_CACHE%\%ANGLECAL_RUNTIME_EXE%"
set "ANGLECAL_DOWNLOAD=%ANGLECAL_CACHE%\runtime.download"
>"%ANGLECAL_DESKTOP_LOG%" echo Runtime: %ANGLECAL_RUNTIME_ID% / %~1

if not exist "%ANGLECAL_EXE%" goto download
if defined ANGLECAL_RUNTIME_REQUIRED if not exist "%ANGLECAL_CACHE%\%ANGLECAL_RUNTIME_REQUIRED%" goto download
set "ANGLECAL_CHECK_FILE=%ANGLECAL_EXE%"
set "ANGLECAL_CHECK_HASH=%ANGLECAL_RUNTIME_EXE_SHA%"
call :verify_hash
if not errorlevel 1 goto ready

:download
echo [%~1] Preparing the verified Windows app. Python and pip are not used.
echo [%~1] First download can take a few minutes. Please keep this window open.
where curl.exe >nul 2>nul
if errorlevel 1 goto failed
curl.exe --fail --location --silent --show-error --retry 2 --connect-timeout 20 --output "%ANGLECAL_DOWNLOAD%" "%ANGLECAL_RUNTIME_URL%" >>"%ANGLECAL_DESKTOP_LOG%" 2>&1
if errorlevel 1 goto failed
set "ANGLECAL_CHECK_FILE=%ANGLECAL_DOWNLOAD%"
set "ANGLECAL_CHECK_HASH=%ANGLECAL_RUNTIME_SHA%"
call :verify_hash
if errorlevel 1 goto failed
if /i "%ANGLECAL_RUNTIME_KIND%"=="zip" (
    tar.exe -xf "%ANGLECAL_DOWNLOAD%" -C "%ANGLECAL_CACHE%" >>"%ANGLECAL_DESKTOP_LOG%" 2>&1
) else (
    copy /y "%ANGLECAL_DOWNLOAD%" "%ANGLECAL_EXE%" >>"%ANGLECAL_DESKTOP_LOG%" 2>&1
)
if errorlevel 1 goto failed
if defined ANGLECAL_RUNTIME_REQUIRED if not exist "%ANGLECAL_CACHE%\%ANGLECAL_RUNTIME_REQUIRED%" goto failed
set "ANGLECAL_CHECK_FILE=%ANGLECAL_EXE%"
set "ANGLECAL_CHECK_HASH=%ANGLECAL_RUNTIME_EXE_SHA%"
call :verify_hash
if errorlevel 1 goto failed

:ready
if /i "%~2"=="--prepare-only" (
    echo [%~1] Verified runtime ready: %ANGLECAL_EXE%
    exit /b 0
)
if /i "%~2"=="--verify" goto verify_runtime
echo [%~1] Starting verified runtime %ANGLECAL_RUNTIME_ID%.
start "" "%ANGLECAL_EXE%" %2 %3 %4 %5 %6 %7 %8 %9
if errorlevel 1 goto failed
exit /b 0

:verify_runtime
if /i "%~1"=="RegionLab" goto verify_regions
"%ANGLECAL_EXE%" --build-info-json "%~3"
exit /b %errorlevel%
:verify_regions
"%ANGLECAL_EXE%" --self-test "%~3"
exit /b %errorlevel%

:verify_hash
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $sha=[Security.Cryptography.SHA256]::Create(); $stream=[IO.File]::OpenRead($env:ANGLECAL_CHECK_FILE); try { $hash=[BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant(); if ($hash -ne $env:ANGLECAL_CHECK_HASH) { [Console]::Error.WriteLine('Runtime checksum mismatch'); exit 1 } } finally { $stream.Dispose(); $sha.Dispose() }" >>"%ANGLECAL_DESKTOP_LOG%" 2>&1
exit /b %errorlevel%

:failed
echo.
echo [%~1] Runtime preparation failed. No pip update is required.
if defined ANGLECAL_DESKTOP_LOG (
    type "%ANGLECAL_DESKTOP_LOG%"
    echo Log: %ANGLECAL_DESKTOP_LOG%
)
echo Download page: https://github.com/bhmin2100-stack/angle-cal/releases
if not defined ANGLECAL_NO_PAUSE pause
exit /b 1
