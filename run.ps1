<#
.SYNOPSIS
    LLM Training Pipeline for Windows PowerShell
.DESCRIPTION
    This script automates the LLM training pipeline including data preprocessing,
    tokenizer training, model training, and evaluation.
    
    Usage: .\run.ps1 [-Stage <stage>] [-Config <config_file>] [-Help]
    
    Stages:
    - preprocess: Run only the data preprocessing
    - tokenizer:  Train the tokenizer (requires preprocessed data)
    - train:      Train the model (requires tokenizer)
    - eval:       Run evaluation (requires trained model)
    - finetune:   Fine-tune a pre-trained model
    - inference:  Run interactive text generation
    - download:   Download a HuggingFace model
    - all:        Run all stages (default)
    
    Example: .\run.ps1 -Stage train -Config config.json
#>

param(
    [ValidateSet("preprocess", "tokenizer", "train", "eval", "finetune", "inference", "download", "all")]
    [string]$Stage = "all",
    
    [string]$Config = "config.json",
    
    [switch]$Help
)

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Script version
$VERSION = "1.0.0"

# Configuration
$PYTHON = "python"
$SCRIPT_DIR = $PSScriptRoot
$LOG_DIR = Join-Path $SCRIPT_DIR "logs"
$EXPORT_DIR = Join-Path $SCRIPT_DIR "exports"
$DATA_DIR = Join-Path $SCRIPT_DIR "data"
$RAW_DATA_DIR = Join-Path $DATA_DIR "raw"
$CLEANED_DATA_DIR = Join-Path $DATA_DIR "cleaned"
$TOKENIZED_DATA_DIR = Join-Path $DATA_DIR "tokens"
$TOKENIZER_DIR = Join-Path $EXPORT_DIR "tokenizer"
$CHECKPOINT_DIR = Join-Path $EXPORT_DIR "checkpoints"

# Create necessary directories
$null = New-Item -ItemType Directory -Force -Path $LOG_DIR, $EXPORT_DIR, $RAW_DATA_DIR, $CLEANED_DATA_DIR, $TOKENIZED_DATA_DIR, $TOKENIZER_DIR, $CHECKPOINT_DIR

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO", "WARNING", "ERROR", "SUCCESS")]
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    switch ($Level) {
        "ERROR" { Write-Host $logMessage -ForegroundColor Red }
        "WARNING" { Write-Host $logMessage -ForegroundColor Yellow }
        "SUCCESS" { Write-Host $logMessage -ForegroundColor Green }
        default { Write-Host $logMessage }
    }
    
    Add-Content -Path (Join-Path $LOG_DIR "pipeline_$(Get-Date -Format 'yyyyMMdd').log") -Value $logMessage
}

# Function to check if a command exists
function Test-CommandExists {
    param($command)
    try {
        if (Get-Command $command -ErrorAction Stop) {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

# Function to run a command with error handling
function Invoke-SafeCommand {
    param(
        [string]$Command,
        [string]$ErrorMsg = "Command failed",
        [string]$SuccessMsg = "Command completed successfully"
    )
    
    try {
        Write-Log "Executing: $Command"
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            throw "Command exited with code $LASTEXITCODE"
        }
        Write-Log $SuccessMsg "SUCCESS"
        return $true
    } catch {
        Write-Log "$ErrorMsg`: $_" "ERROR"
        return $false
    }
}

# Show help if requested
if ($Help) {
    Get-Help $PSCommandPath -Detailed
    exit 0
}

# Check Python is installed
if (-not (Test-CommandExists "python")) {
    Write-Log "Python is not installed or not in PATH" "ERROR"
    exit 1
}

# Check Python version
$pythonVersion = & $PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$pythonMajor = [int](& $PYTHON -c "import sys; print(sys.version_info.major)")
$pythonMinor = [int](& $PYTHON -c "import sys; print(sys.version_info.minor)")
$pythonMicro = [int](& $PYTHON -c "import sys; print(sys.version_info.micro)")

# Check if Python version is at least 3.8
$isValidVersion = $false

if ($pythonMajor -gt 3) {
    $isValidVersion = $true
} elseif ($pythonMajor -eq 3) {
    if ($pythonMinor -gt 8) {
        $isValidVersion = $true
    } elseif ($pythonMinor -eq 8) {
        $isValidVersion = $true  # Python 3.8.x is acceptable
    }
}

if (-not $isValidVersion) {
    Write-Log "Python 3.8 or higher is required. Found Python $pythonVersion" "ERROR"
    exit 1
}

Write-Log "Using Python version: $pythonVersion" "INFO"

Write-Log "Starting LLM Training Pipeline (v$VERSION)"
Write-Log "Python version: $pythonVersion"
Write-Log "Stage: $Stage"
Write-Log "Config: $Config"

# Check if config file exists
if (-not (Test-Path $Config)) {
    Write-Log "Config file not found: $Config" "ERROR"
    exit 1
}

# Function to run preprocessing
function Invoke-PreprocessStage {
    Write-Log "=== Starting Data Preprocessing ==="
    
    # Check if raw data directory is empty
    if (-not (Get-ChildItem -Path $RAW_DATA_DIR -File -Recurse)) {
        Write-Log "No files found in $RAW_DATA_DIR" "WARNING"
        Write-Log "Please add your training data (PDF/DOCX/TXT) to $RAW_DATA_DIR"
        return $false
    }
    
    $command = "$PYTHON training/preprocess.py --config $Config --input-dir $RAW_DATA_DIR --output-dir $CLEANED_DATA_DIR"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Preprocessing failed" -SuccessMsg "Preprocessing completed successfully")
}

# Function to train tokenizer
function Invoke-TokenizerStage {
    Write-Log "=== Starting Tokenizer Training ==="
    
    # Check if cleaned data exists
    if (-not (Get-ChildItem -Path $CLEANED_DATA_DIR -File -Recurse)) {
        Write-Log "No cleaned data found in $CLEANED_DATA_DIR" "ERROR"
        return $false
    }
    
    # Create tokenizer directory if it doesn't exist
    $null = New-Item -ItemType Directory -Force -Path $TOKENIZER_DIR
    
    $command = "$PYTHON training/train_tokenizer.py --config $Config --input-dir $CLEANED_DATA_DIR --output-dir $TOKENIZER_DIR"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Tokenizer training failed" -SuccessMsg "Tokenizer training completed successfully")
}

# Function to train the model
function Invoke-TrainStage {
    Write-Log "=== Starting Model Training ==="
    
    # Check if tokenizer exists
    if (-not (Test-Path (Join-Path $TOKENIZER_DIR "tokenizer.model"))) {
        Write-Log "Tokenizer not found in $TOKENIZER_DIR" "ERROR"
        return $false
    }
    
    # Create checkpoints directory if it doesn't exist
    $null = New-Item -ItemType Directory -Force -Path $CHECKPOINT_DIR
    
    $command = "$PYTHON training/train.py --config $Config --tokenizer-dir $TOKENIZER_DIR --checkpoint-dir $CHECKPOINT_DIR"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Training failed" -SuccessMsg "Training completed successfully")
}

# Function to run evaluation
function Invoke-EvalStage {
    Write-Log "=== Starting Model Evaluation ==="
    
    # Check if model exists
    $latestCheckpoint = Get-ChildItem -Path $CHECKPOINT_DIR -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latestCheckpoint) {
        Write-Log "No model checkpoints found in $CHECKPOINT_DIR" "ERROR"
        return $false
    }
    
    $command = "$PYTHON eval/eval.py --config $Config --tokenizer-dir $TOKENIZER_DIR --model-path $($latestCheckpoint.FullName)"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Evaluation failed" -SuccessMsg "Evaluation completed successfully")
}

# Function to run fine-tuning
function Invoke-FinetuneStage {
    Write-Log "=== Starting Model Fine-tuning ==="
    
    # Check if base model exists
    $latestCheckpoint = Get-ChildItem -Path $CHECKPOINT_DIR -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latestCheckpoint) {
        Write-Log "No base model checkpoints found in $CHECKPOINT_DIR" "ERROR"
        Write-Log "Please train a model first or specify a pre-trained model path"
        return $false
    }
    
    # Check if fine-tuning data exists
    $finetuneDataDir = Join-Path $DATA_DIR "finetune"
    if (-not (Test-Path $finetuneDataDir) -or -not (Get-ChildItem -Path $finetuneDataDir -File -Recurse)) {
        Write-Log "No fine-tuning data found in $finetuneDataDir" "WARNING"
        Write-Log "Please add your fine-tuning data to $finetuneDataDir"
        return $false
    }
    
    $command = "$PYTHON training/finetune.py --config $Config --pretrained-model $($latestCheckpoint.FullName) --train-data $finetuneDataDir --tokenizer-dir $TOKENIZER_DIR"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Fine-tuning failed" -SuccessMsg "Fine-tuning completed successfully")
}

# Function to run inference
function Invoke-InferenceStage {
    Write-Log "=== Starting Interactive Inference ==="
    
    # Check if model exists
    $latestCheckpoint = Get-ChildItem -Path $CHECKPOINT_DIR -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latestCheckpoint) {
        Write-Log "No model checkpoints found in $CHECKPOINT_DIR" "ERROR"
        return $false
    }
    
    Write-Log "Starting interactive text generation..."
    $command = "$PYTHON inference.py --model $($latestCheckpoint.FullName) --tokenizer $TOKENIZER_DIR --config $Config --interactive"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Inference failed" -SuccessMsg "Inference session completed")
}

# Function to download HuggingFace model
function Invoke-DownloadStage {
    Write-Log "=== Downloading HuggingFace Model ==="
    $modelName = Read-Host "Enter HuggingFace model name (e.g. Qwen/Qwen2.5-Coder-0.5B)"
    $outputDir = Read-Host "Enter output directory (default: ./models/<model_name>)"
    if ([string]::IsNullOrWhiteSpace($modelName)) {
        Write-Log "Model name is required" "ERROR"
        return $false
    }
    $outputArg = if ([string]::IsNullOrWhiteSpace($outputDir)) { "" } else { "--output-dir `"$outputDir`"" }
    $command = "$PYTHON tools/download_hf_model.py --model $modelName $outputArg"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Model download failed" -SuccessMsg "Model download completed")
}

# Main execution
$success = $true
$startTime = Get-Date

try {
    switch ($Stage) {
        "preprocess" {
            $success = Invoke-PreprocessStage
        }
        "tokenizer" {
            $success = Invoke-TokenizerStage
        }
        "train" {
            $success = Invoke-TrainStage
        }
        "eval" {
            $success = Invoke-EvalStage
        }
        "finetune" {
            $success = Invoke-FinetuneStage
        }
        "inference" {
            $success = Invoke-InferenceStage
        }
        "download" {
            $success = Invoke-DownloadStage
        }
        "all" {
            $success = (Invoke-PreprocessStage) -and 
                      (Invoke-TokenizerStage) -and 
                      (Invoke-TrainStage) -and 
                      (Invoke-EvalStage)
        }
    }
} catch {
    Write-Log "Unexpected error: $_" "ERROR"
    $success = $false
}

$endTime = Get-Date
$duration = $endTime - $startTime

if ($success) {
    Write-Log "Pipeline completed successfully in $($duration.ToString('hh\:mm\:ss'))" "SUCCESS"
} else {
    Write-Log "Pipeline failed after $($duration.ToString('hh\:mm\:ss'))" "ERROR"
    exit 1
}
