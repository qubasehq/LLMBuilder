@echo off
REM LLM Training Pipeline - Windows Batch Version
REM Usage: run.bat [stage] [options]
REM   stage: preprocess|tokenizer|train|eval|download|all (default: all)
REM   options: /cpu-only, /help

setlocal enabledelayedexpansion

REM Configuration
SET STAGE=all
SET CPU_ONLY=0
SET VERBOSE=0
SET PYTHON=python

REM Colors (Windows 10+ with Virtual Terminal support)
SET "RESET=[0m"
SET "RED=[31m"
SET "GREEN=[32m"
SET "YELLOW=[33m"
SET "BLUE=[34m"

REM Check if running on Windows 10+
ver | find "10." > nul
if %ERRORLEVEL% EQU 0 (
    SET "ESC=^["
    echo %ESC%[0m >nul
) else (
    SET "ESC="
    SET "RESET="
    SET "RED="
    SET "GREEN="
    SET "YELLOW="
    SET "BLUE="
)

:parse_arguments
if "%~1"=="" goto :arguments_parsed

if /i "%~1"=="preprocess" (
    SET STAGE=preprocess
) else if /i "%~1"=="tokenizer" (
    SET STAGE=tokenizer
) else if /i "%~1"=="train" (
    SET STAGE=train
) else if /i "%~1"=="eval" (
    SET STAGE=eval
) else if /i "%~1"=="download" (
    SET STAGE=download
) else if /i "%~1"=="all" (
    SET STAGE=all
) else if /i "%~1"=="/cpu-only" (
    SET CPU_ONLY=1
) else if /i "%~1"=="/verbose" (
    SET VERBOSE=1
) else if /i "%~1"=="/help" (
    call :show_help
    exit /b 0
) else (
    echo %ESC%!RED!Unknown option: %~1!RESET!
    call :show_help
    exit /b 1
)

shift
goto :parse_arguments

:arguments_parsed

REM Print header
echo %ESC%!BLUE!LLM Training Pipeline - Windows%ESC%!RESET!
echo Stage: !STAGE!
if !CPU_ONLY!==1 echo CPU-only mode enabled

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %ESC%!RED![ERROR] Python is not installed or not in PATH!RESET!
    exit /b 1
)

REM Get Python version components
for /f "tokens=1-3 delims=." %%a in ('python -c "import sys; print('%d.%d.%d' %% (sys.version_info.major, sys.version_info.minor, sys.version_info.micro))" 2^>nul') do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
    set PYTHON_MICRO=%%c
)

REM Check if Python version is at least 3.8
if %PYTHON_MAJOR% LSS 3 (
    echo %ESC%!RED![ERROR] Python 3.8 or higher is required. Found Python %PYTHON_MAJOR%.%PYTHON_MINOR%.%PYTHON_MICRO%!RESET!
    exit /b 1
) else if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 8 (
        echo %ESC%!RED![ERROR] Python 3.8 or higher is required. Found Python %PYTHON_MAJOR%.%PYTHON_MINOR%.%PYTHON_MICRO%!RESET!
        exit /b 1
    )
)

echo %ESC%!BLUE![INFO] Using Python %PYTHON_MAJOR%.%PYTHON_MINOR%.%PYTHON_MICRO%!RESET!

REM Setup directories
for %%d in (
    "data\raw" 
    "data\cleaned" 
    "data\tokens" 
    "tokenizer" 
    "exports\checkpoints" 
    "exports\gguf" 
    "exports\onnx" 
    "logs"
) do (
    if not exist "%%~d" mkdir "%%~d"
)

REM Run the appropriate stage
if "!STAGE!"=="preprocess" (
    call :run_preprocessing
) else if "!STAGE!"=="tokenizer" (
    call :run_tokenizer
) else if "!STAGE!"=="train" (
    call :run_training
) else if "!STAGE!"=="eval" (
    call :run_evaluation
) else if "!STAGE!"=="download" (
    call :run_download
) else if "!STAGE!"=="all" (
    call :run_preprocessing
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
    
    call :run_tokenizer
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
    
    call :run_training
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
    
    call :run_evaluation
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
)

echo %ESC%!GREEN!Pipeline completed successfully!%ESC%!RESET!
if exist "logs\training.log" (
    echo Check the logs directory for detailed logs
)

exit /b 0

:run_preprocessing
    echo %ESC%!BLUE!=== Stage 1: Data Preprocessing ===%ESC%!RESET!
    if not exist "data\raw\*" (
        echo %ESC%!YELLOW!No data found in data\raw directory!RESET!
        echo Please add your training data (.txt, .pdf, .docx files) to data\raw\
        exit /b 1
    )
    
    %PYTHON% training\preprocess.py
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
    
    if exist "data\cleaned\combined_text.txt" (
        for /f "tokens=*" %%s in ('powershell -command "(Get-Item 'data\cleaned\combined_text.txt').length" 2^>nul') do set "filesize=%%s"
        echo Combined text file size: !filesize! bytes
    )
    exit /b 0

:run_tokenizer
    echo %ESC%!BLUE!=== Stage 2: Tokenizer Training ===%ESC%!RESET!
    if not exist "data\cleaned\combined_text.txt" (
        echo %ESC%!RED!No cleaned data found. Please run preprocessing first.!RESET!
        exit /b 1
    )
    
    %PYTHON% training\train_tokenizer.py
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
    
    if exist "tokenizer\tokenizer.model" (
        echo %ESC%!GREEN!Tokenizer created successfully!%ESC%!RESET!
    ) else (
        echo %ESC%!RED!Tokenizer training failed!%ESC%!RESET!
        exit /b 1
    )
    exit /b 0

:run_training
    echo %ESC%!BLUE!=== Stage 3: Model Training ===%ESC%!RESET!
    if not exist "data\tokens\tokens.pt" (
        echo %ESC%!RED!No tokenized data found. Please run tokenizer training first.!RESET!
        exit /b 1
    )
    
    if !CPU_ONLY!==1 (
        set "TRAIN_ARGS=--device cpu"
    )
    
    echo %ESC%!YELLOW!This may take a while depending on your hardware...%ESC%!RESET!
    %PYTHON% training\train.py !TRAIN_ARGS!
    if !ERRORLEVEL! NEQ 0 exit /b !ERRORLEVEL!
    
    if exist "exports\checkpoints\best_model.pt" (
        echo %ESC%!GREEN!Training completed successfully!%ESC%!RESET!
    else (
        echo %ESC%!YELLOW!Training completed but no best model found!%ESC%!RESET!
    )
    exit /b 0

:run_evaluation
    echo %ESC%!BLUE!=== Stage 4: Model Evaluation ===%ESC%!RESET!
    if not exist "exports\checkpoints\best_model.pt" (
        echo %ESC%!RED!No trained model found. Please run training first.!RESET!
        exit /b 1
    )
    
    %PYTHON% eval\eval.py
    exit /b !ERRORLEVEL!

:show_help
    echo LLM Training Pipeline - Windows Batch Version
    echo.
    echo Usage: run.bat [stage] [options]
    echo.
    echo Stages:
    echo   preprocess  - Run data preprocessing only
    echo   tokenizer   - Run tokenizer training only
    echo   train       - Run model training only
    echo   eval        - Run model evaluation only
    echo   download    - Download a HuggingFace model
    echo   all         - Run complete pipeline (default)
    echo.
    echo Options:
    echo   /cpu-only  - Force CPU-only training
    echo   /verbose   - Enable verbose output (not fully implemented)
    echo   /help      - Show this help message
    echo.
    echo Examples:
    echo   run.bat                    ^> Run complete pipeline
    echo   run.bat preprocess         ^> Run preprocessing only
    echo   run.bat download           ^> Download a HuggingFace model
    echo   run.bat train /cpu-only    ^> Train on CPU only
    exit /b 0

:run_download
    echo %ESC%!BLUE!=== Download HuggingFace Model ===%ESC%!RESET!
    set /p MODEL_NAME="Enter HuggingFace model name (e.g. Qwen/Qwen2.5-Coder-0.5B): "
    set /p OUTPUT_DIR="Enter output directory (default: ./models/<model_name>): "
    if "%MODEL_NAME%"=="" (
        echo %ESC%!RED!Model name is required!RESET!
        exit /b 1
    )
    if "%OUTPUT_DIR%"=="" (
        %PYTHON% tools\download_hf_model.py --model "%MODEL_NAME%"
    ) else (
        %PYTHON% tools\download_hf_model.py --model "%MODEL_NAME%" --output-dir "%OUTPUT_DIR%"
    )
    exit /b !ERRORLEVEL!
