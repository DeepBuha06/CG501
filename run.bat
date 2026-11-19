@echo off
rem From Windows PowerShell in this folder:  .\run.bat human   (or bat, test, videos, report, all)
wsl --cd "%~dp0." bash run.sh %*
if "%~1"=="" pause
