@echo off
setlocal

cd /d "%~dp0"

call "%~dp0launch_desktop.bat" AngleCal %*
exit /b %errorlevel%
