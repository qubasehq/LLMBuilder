<#
.SYNOPSIS
    LLM Training Pipeline for Windows PowerShell
.DESCRIPTION
    This script automates the LLM training pipeline including data preprocessing,
    tokenizer training, model training, and evaluation.
    
    Usage: .\run.ps1 [-Stage <stage>] [-Config <config_file>] [-Help]
    
    Stages:
    - ingest:     Enhanced document ingestion (HTML, EPUB, PDF with OCR)
    - dedup:      Deduplication with hash and embedding methods
    - preprocess: Run only the data preprocessing
    - tokenizer:  Train the tokenizer (requires preprocessed data)
    - train:      Train the model (requires tokenizer)
    - eval:       Run evaluation (requires trained model)
    - finetune:   Fine-tune a pre-trained model
    - inference:  Run interactive text generation
    - gguf:       Convert model to GGUF format with quantization
    - download:   Download a HuggingFace model
    - test:       Run comprehensive test suite
    - all:        Run all stages (default)
    
    Example: .\run.ps1 -Stage train -Config config.json
#>

param(
    [ValidateSet("preprocess", "tokenizer", "train", "eval", "finetune", "inference", "download", "ingest", "dedup", "gguf", "test", "all")]
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
$DEDUPED_DATA_DIR = Join-Path $DATA_DIR "deduped"
$TOKENIZER_DIR = Join-Path $EXPORT_DIR "tokenizer"
$CHECKPOINT_DIR = Join-Path $EXPORT_DIR "checkpoints"
$GGUF_DIR = Join-Path $EXPORT_DIR "gguf"

# Create necessary directories
$null = New-Item -ItemType Directory -Force -Path $LOG_DIR, $EXPORT_DIR, $RAW_DATA_DIR, $CLEANED_DATA_DIR, $TOKENIZED_DATA_DIR, $DEDUPED_DATA_DIR, $TOKENIZER_DIR, $CHECKPOINT_DIR, $GGUF_DIR

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
    
    # Use cleaned data from deduped directory if it exists and has files, otherwise use cleaned directory
    $hasDedupedFiles = $false
    if (Test-Path -Path $DEDUPED_DATA_DIR -PathType Container) {
        $hasDedupedFiles = (Get-ChildItem -Path $DEDUPED_DATA_DIR -File -Recurse).Count -gt 0
    }
    
    $inputDir = if ($hasDedupedFiles) { $DEDUPED_DATA_DIR } else { $CLEANED_DATA_DIR }
    
    # Build the command with required parameters
    $command = "$PYTHON training/train_tokenizer.py --input-dir ""$inputDir"" --output-dir ""$TOKENIZER_DIR"" --tokenizer-type sentencepiece --model-type bpe --vocab-size 16000"
    
    Write-Log "Running: $command"
    $ok = (Invoke-SafeCommand -Command $command -ErrorMsg "Tokenizer training failed" -SuccessMsg "Tokenizer training completed successfully")
    if (-not $ok) { return $false }

    # After successful training, tokenize corpus into tensors
    $spModel = Join-Path $TOKENIZER_DIR "sentencepiece.model"
    if (Test-Path $spModel) {
        $null = New-Item -ItemType Directory -Force -Path $TOKENIZED_DATA_DIR
        $outTokens = Join-Path $TOKENIZED_DATA_DIR "tokens.pt"
        $tokCmd = "$PYTHON training/tokenize_corpus.py --input-dir ""$inputDir"" --tokenizer ""$spModel"" --output ""$outTokens"""
        Write-Log "Running: $tokCmd"
        $tokOk = (Invoke-SafeCommand -Command $tokCmd -ErrorMsg "Tokenization to tensors failed" -SuccessMsg "Tokenization completed successfully")
        if (-not $tokOk) { return $false }
        if (-not (Test-Path $outTokens)) {
            Write-Log "Tokenized data not found at $outTokens after tokenization" "ERROR"
            return $false
        }
        Write-Log "Tokenized data saved to $outTokens" "SUCCESS"
        return $true
    } else {
        # If using HF tokenizer.json, we currently do not auto-tokenize
        $hfModel = Join-Path $TOKENIZER_DIR "tokenizer.json"
        if (Test-Path $hfModel) {
            Write-Log "Found HuggingFace tokenizer.json; current pipeline auto-tokenization supports SentencePiece .model only" "WARNING"
            return $true
        }
        Write-Log "Tokenizer output files not found in $TOKENIZER_DIR" "ERROR"
        return $false
    }
}

# Function to train the model
function Invoke-TrainStage {
    Write-Log "=== Starting Model Training ==="
    
    # Check if tokenizer exists
    $spModel = Join-Path $TOKENIZER_DIR "sentencepiece.model"
    $hfModel = Join-Path $TOKENIZER_DIR "tokenizer.json"
    if (-not (Test-Path $spModel) -and -not (Test-Path $hfModel)) {
        Write-Log "Tokenizer not found in $TOKENIZER_DIR (expected sentencepiece.model or tokenizer.json)" "ERROR"
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
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Evaluation failed" -SuccessMsg "Model evaluation completed")
}

# Function to run GGUF conversion
function Invoke-GgufStage {
    Write-Log "=== Starting GGUF Conversion ==="
    
    # Check if model exists
    $latestCheckpoint = Get-ChildItem -Path $CHECKPOINT_DIR -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if (-not $latestCheckpoint) {
        Write-Log "No model checkpoints found in $CHECKPOINT_DIR" "ERROR"
        return $false
    }
    
    # Create GGUF directory if it doesn't exist
    if (-not (Test-Path $GGUF_DIR)) {
        New-Item -ItemType Directory -Path $GGUF_DIR -Force | Out-Null
    }
    
    # Run GGUF conversion with multiple quantization levels
    $command = "$PYTHON tools/conversion_pipeline.py `"$($latestCheckpoint.FullName)`" `"$GGUF_DIR`" --quantization f16 q8_0 q4_0 --tokenizer `"tokenizer/tokenizer.model`""
    return (Invoke-SafeCommand -Command $command -ErrorMsg "GGUF conversion failed" -SuccessMsg "GGUF conversion completed")
}

# Function to run data preprocessing
function Invoke-PreprocessStage {
    Write-Log "=== Starting Data Preprocessing ==="
    
    # Check if cleaned data exists
    if (-not (Get-ChildItem -Path $CLEANED_DATA_DIR -File -Recurse)) {
        Write-Log "No cleaned data found in $CLEANED_DATA_DIR" "WARNING"
        Write-Log "Running ingestion stage first..."
        if (-not (Invoke-IngestStage)) {
            Write-Log "Ingestion failed, cannot proceed with preprocessing" "ERROR"
            return $false
        }
    }
    
    $command = "$PYTHON training/preprocess.py --config $Config --raw-dir $RAW_DATA_DIR --cleaned-dir $CLEANED_DATA_DIR --deduped-dir $DEDUPED_DATA_DIR"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Preprocessing failed" -SuccessMsg "Preprocessing completed successfully")
}

# Function to run enhanced document ingestion
function Invoke-IngestStage {
    Write-Log "=== Starting Enhanced Document Ingestion ==="
    
    # Check if raw data directory exists
    if (-not (Test-Path $RAW_DATA_DIR)) {
        Write-Log "Raw data directory not found: $RAW_DATA_DIR" "ERROR"
        return $false
    }
    
    # Check for supported file types
    $supportedFiles = Get-ChildItem -Path $RAW_DATA_DIR -Recurse | Where-Object { 
        $_.Extension -in @('.txt', '.pdf', '.docx', '.html', '.epub', '.md') 
    }
    
    if (-not $supportedFiles) {
        Write-Log "No supported files found in $RAW_DATA_DIR" "WARNING"
        Write-Log "Supported formats: TXT, PDF, DOCX, HTML, EPUB, Markdown"
        return $false
    }
    
    Write-Log "Found $($supportedFiles.Count) supported files for ingestion"
    
    $command = "$PYTHON scripts/run_ingestion.py --input $RAW_DATA_DIR --output $CLEANED_DATA_DIR --recursive --verbose"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Document ingestion failed" -SuccessMsg "Document ingestion completed successfully")
}

# Function to run deduplication
function Invoke-DedupStage {
    Write-Log "=== Starting Data Deduplication ==="
    
    # Check if cleaned data exists
    if (-not (Get-ChildItem -Path $CLEANED_DATA_DIR -File -Recurse)) {
        Write-Log "No cleaned data found in $CLEANED_DATA_DIR" "ERROR"
        Write-Log "Please run ingestion or preprocessing first"
        return $false
    }
    
    $dedupOutputDir = Join-Path $DATA_DIR "deduped"
    $null = New-Item -ItemType Directory -Force -Path $dedupOutputDir
    
    $command = "$PYTHON data/dedup.py --input-dir $CLEANED_DATA_DIR --output-dir $dedupOutputDir --similarity-threshold 0.85"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Deduplication failed" -SuccessMsg "Deduplication completed successfully")
}

# Function to run GGUF conversion
function Invoke-GgufStage {
    Write-Log "=== Starting GGUF Conversion ==="
    
    # Check if model exists
    $latestCheckpoint = Get-ChildItem -Path $CHECKPOINT_DIR -Filter "*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latestCheckpoint) {
        Write-Log "No model checkpoints found in $CHECKPOINT_DIR" "ERROR"
        return $false
    }
    
    $ggufOutputDir = Join-Path $EXPORT_DIR "gguf"
    $null = New-Item -ItemType Directory -Force -Path $ggufOutputDir
    
    $command = "$PYTHON tools/conversion_pipeline.py $($latestCheckpoint.FullName) $ggufOutputDir --quantization f16 q8_0 q4_0 --tokenizer $TOKENIZER_DIR"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "GGUF conversion failed" -SuccessMsg "GGUF conversion completed successfully")
}

# Function to run comprehensive tests
function Invoke-TestStage {
    Write-Log "=== Running Comprehensive Test Suite ==="
    
    # Check if pytest is available
    try {
        & $PYTHON -m pytest --version | Out-Null
    } catch {
        Write-Log "pytest not found. Installing..." "WARNING"
        & $PYTHON -m pip install pytest
    }
    
    $command = "$PYTHON -m pytest tests/ -v --tb=short"
    return (Invoke-SafeCommand -Command $command -ErrorMsg "Tests failed" -SuccessMsg "All tests passed successfully")
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
    
    $command = "$PYTHON finetune/finetune.py --config $Config --pretrained-model $($latestCheckpoint.FullName) --train-data $finetuneDataDir --tokenizer-dir $TOKENIZER_DIR"
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
    $ggufModel = "exports/gguf/best_model_f16.gguf"
    if (-not (Test-Path $ggufModel)) {
        Write-Log "GGUF model not found: $ggufModel" "ERROR"
        return $false
    }
    $command = "$PYTHON inference.py --model $ggufModel --tokenizer tokenizer/tokenizer.model --config $Config --interactive"
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
        "ingest" {
            $success = Invoke-IngestStage
        }
        "dedup" {
            $success = Invoke-DedupStage
        }
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
        "gguf" {
            $success = Invoke-GgufStage
        }
        "download" {
            $success = Invoke-DownloadStage
        }
        "test" {
            $success = Invoke-TestStage
        }
        "all" {
            $success = (Invoke-IngestStage) -and 
                      (Invoke-DedupStage) -and 
                      (Invoke-PreprocessStage) -and 
                      (Invoke-TokenizerStage) -and 
                      (Invoke-TrainStage) -and 
                      (Invoke-EvalStage) -and
                      (Invoke-GgufStage)
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