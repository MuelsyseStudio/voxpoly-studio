@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   VoxPoly Studio V2.0 Pro - Windows Auto GPU Installer
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10-3.12 not found in PATH. Please install Python 3.12 first.
    pause
    exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARN] FFmpeg not found in PATH. Audio/Video processing requires FFmpeg.
    echo        Install via: winget install Gyan.FFmpeg
    echo.
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment .venv ...
    python -m venv .venv
)

".venv\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools

echo.
echo [2/3] Detecting NVIDIA GPU architecture ...
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "GPU_DESC=CPU Only"

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        set "GPU_DESC=%%G"
    )
    echo Detected GPU: !GPU_DESC!
    echo !GPU_DESC! | findstr /R /I "50[5-9]0 Blackwell B100 B200" >nul
    if not errorlevel 1 (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
        echo Selected PyTorch CUDA 12.8 ^(RTX 50 / Blackwell^)
    ) else (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        echo Selected PyTorch CUDA 12.4 ^(RTX 20/30/40 / T4 / L4 / A100^)
    )
) else (
    echo No NVIDIA GPU detected; installing CPU PyTorch ^(recommend setting Qwen3 to 0.6B in Settings^).
)

".venv\Scripts\pip.exe" install torch torchvision torchaudio --index-url !TORCH_INDEX!

echo.
echo [3/3] Installing VoxPoly Studio dependencies ...
".venv\Scripts\pip.exe" install -r requirements.txt

echo.
echo ============================================================
echo   Installation complete! Run start_windows.bat to launch.
echo   (Optional: run ".venv\Scripts\pip install demucs" for AI video BGM separation)
echo ============================================================
pause
