@echo off
rem ============================================================
rem  Drone VRA webapp launcher
rem  - Finds python (anaconda3/miniconda3 env: python312 or vra)
rem  - New PC setup guide: see the .md guide file in this folder
rem ============================================================
cd /d "%~dp0"

set PY=
if exist "%USERPROFILE%\anaconda3\envs\python312\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\python312\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\python312\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\python312\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\vra\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\vra\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\vra\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\vra\python.exe"

if not defined PY (
    echo [ERROR] Python environment not found.
    echo Install Miniconda and create the "vra" env first.
    echo See the setup guide file in this folder.
    pause
    exit /b 1
)

echo Using python: %PY%
"%PY%" webapp\app.py
pause
