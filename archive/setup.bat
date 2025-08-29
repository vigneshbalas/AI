@echo off
SETLOCAL

:: -----------------------------
:: 1. Set environment variables
:: -----------------------------
set ENV_NAME=agent_demo_env

:: -----------------------------
:: 2. Create Python virtual environment
:: -----------------------------
python -m venv %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to create virtual environment. Make sure Python is in PATH.
    exit /b 1
)

echo ✅ Virtual environment '%ENV_NAME%' created.

:: -----------------------------
:: 3. Activate virtual environment
:: -----------------------------
call %ENV_NAME%\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to activate virtual environment.
    exit /b 1
)

:: -----------------------------
:: 4. Upgrade pip
:: -----------------------------
python -m pip install --upgrade pip

:: -----------------------------
:: 5. Install dependencies
:: -----------------------------
pip install --upgrade setuptools wheel

pip install llama-index==0.10.30 qdrant-client==1.10.1 protobuf==5.28.3
pip install openai>=1.0.0 transformers>=4.42.0 torch>=2.0.0 tqdm

echo ✅ All dependencies installed.

:: -----------------------------
:: 6. Instructions
:: -----------------------------
echo.
echo -------------------------------
echo To activate the environment:
echo call %ENV_NAME%\Scripts\activate.bat
echo.
echo To run the demo:
echo python agent_demo_ui.py
echo -------------------------------

ENDLOCAL
pause
