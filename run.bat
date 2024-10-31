@echo off
REM Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and try again.
    pause
    exit /b
)

REM Install dependencies
echo Downloading dependencies
pip install -r requirements.txt

REM Check if dependencies installed successfully
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b
)

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

REM Change to the directory where the Python script is located
cd src

REM Run the Python script
echo Running Python script...
python main.py

REM Check if Python script ran successfully
if %errorlevel% neq 0 (
    echo Python script encountered an error.
    pause
    exit /b
)

echo Script completed successfully.
pause
