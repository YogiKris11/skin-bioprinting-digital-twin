@echo off
setlocal
cd /d "%~dp0"
title Skin Bioprinting Digital Twin
echo ================================================
echo   SKIN BIOPRINTING DIGITAL TWIN
echo ================================================
echo.
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
        echo During installation, select "Add Python to PATH".
        pause
        exit /b 1
    )
    set "PYTHON=python"
)
echo [1/3] Checking Python...
%PYTHON% --version
echo.
echo [2/3] Installing required packages...
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 goto install_error
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 goto install_error
echo.
echo [3/3] Starting the dashboard...
%PYTHON% -m streamlit run app.py
if errorlevel 1 goto run_error
goto end
:install_error
echo.
echo Package installation failed. Check your internet connection and Python installation.
pause
exit /b 1
:run_error
echo.
echo The dashboard could not start. Check the error shown above.
pause
exit /b 1
:end
pause
