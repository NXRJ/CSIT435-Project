@echo off
setlocal
cd /d "%~dp0"

title CSCI435 Accessibility Scene Hazard Assistant

if defined CSCI435_PYTHON (
    if exist "%CSCI435_PYTHON%" (
        set "PYTHON_EXE=%CSCI435_PYTHON%"
        goto run_app
    )
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto run_app
)

echo No project virtual environment was found.
echo Creating .venv and installing the required packages...
echo.

where py >nul 2>&1
if errorlevel 1 (
    echo Python Launcher was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
)

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    py -3 -m venv .venv
) else (
    py -3.12 -m venv .venv
)

if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_failed

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

set "PYTHON_EXE=.venv\Scripts\python.exe"

:run_app
echo.
echo Starting the CSCI435 app...
echo Open http://127.0.0.1:7860 if the browser does not open automatically.
echo Press Ctrl+C in this window to stop the server.
echo.

"%PYTHON_EXE%" app.py
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo The app stopped with error code %APP_EXIT%.
    pause
)

exit /b %APP_EXIT%

:install_failed
echo.
echo Package installation failed. Check the messages above and try again.
pause
exit /b 1
