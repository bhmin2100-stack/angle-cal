@echo off
setlocal
cd /d "%~dp0"
call "%~dp0launch_source.bat" AngleCal %*
exit /b %errorlevel%
