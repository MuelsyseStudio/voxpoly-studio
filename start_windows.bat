@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Please run install_windows.bat first.
    pause
    exit /b 1
)

echo ============================================================
echo   VoxPoly Studio V2.0 Pro - http://127.0.0.1:8765
echo   (Set VOXPOLY_DOWNLOAD_SOURCE=modelscope for fast China downloads)
echo ============================================================
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8765/"
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
pause
