@echo off
REM Start ADB server
echo Starting ADB server...
adb start-server

REM Check if ADB started successfully
if %errorlevel% neq 0 (
    echo Failed to start ADB server. Ensure ADB is installed and added to PATH.
    pause
    exit /b
)

REM List connected devices
echo Checking for connected devices...
adb devices

REM Run Python script
echo Running Python script...
python -u "d:\codingstufffffff\python\bsai\src\main.py"

REM Check if Python script ran successfully
if %errorlevel% neq 0 (
    echo Python script encountered an error.
    pause
    exit /b
)

echo Script completed successfully.
pause
