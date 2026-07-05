@echo off
rem ============================================================
rem  Drone VRA webapp + Cloudflare Tunnel (external access)
rem  - Starts the webapp, then opens a public https URL.
rem  - The URL is printed below as  https://xxxx.trycloudflare.com
rem  - Share that URL with team members.
rem  - NOTE: the URL changes every time this window is restarted.
rem  - Keep BOTH windows open while in use.
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

if not exist "webapp\cloudflared.exe" (
    echo [INFO] Downloading cloudflared.exe ...
    curl -L -o "webapp\cloudflared.exe" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
)

set VRA_NO_BROWSER=1
start "VRA-webapp-server" /min "%PY%" webapp\app.py
timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo  Find the public URL below:  https://xxxx.trycloudflare.com
echo  Share it with your team. Keep this window open.
echo ============================================================
echo.
webapp\cloudflared.exe tunnel --url http://127.0.0.1:8000
pause
