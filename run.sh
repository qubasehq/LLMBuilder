#!/bin/bash

# LLM Training Pipeline
# Comprehensive automation script for the complete training pipeline
# Usage: ./run.sh [stage] [options]
#   stage: preprocess|tokenizer|train|eval|all (default: all)
#   options: --cpu-only, --help

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CPU_ONLY=false
STAGE="all"
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check Python version meets requirements
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Get Python version components
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    python_major=$(python -c "import sys; print(sys.version_info.major)")
    python_minor=$(python -c "import sys; print(sys.version_info.minor)")
    
    # Check if Python version is at least 3.8
    if [ "$python_major" -lt 3 ] || { [ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]; }; then
        print_error "Python 3.8 or higher is required. Found Python $python_version"
        exit 1
    fi
    
    print_status "Using Python $python_version"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Check if virtual environment is recommended
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "No virtual environment detected. Consider using one."
    fi
    
    # Install dependencies if needed
    print_status "Installing/updating dependencies..."
    python -m pip install -r requirements.txt
    
    print_success "Dependencies checked"
}

# Function to create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    
    directories=("data/raw" "data/cleaned" "data/tokens" "tokenizer" "exports/checkpoints" "exports/gguf" "exports/onnx" "logs")
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "Directories created"
}

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check RAM
    if command -v free &> /dev/null; then
        total_ram=$(free -h | awk '/^Mem:/ {print $2}')
        available_ram=$(free -h | awk '/^Mem:/ {print $7}')
        print_status "Total RAM: $total_ram, Available: $available_ram"
        
        # Warn if less than 4GB
        ram_gb=$(free -g | awk '/^Mem:/ {print $2}')
        if [ "$ram_gb" -lt 4 ]; then
            print_warning "Less than 4GB RAM detected. Consider using smaller batch sizes or shorter sequences."
        fi
    else
        print_warning "Could not check RAM usage. Ensure you have at least 4GB available."
    fi
    
    # Check storage
    if command -v df &> /dev/null; then
        current_dir=$(pwd)
        storage_info=$(df -h "$current_dir" | tail -1)
        available_storage=$(echo "$storage_info" | awk '{print $4}')
        print_status "Available storage: $available_storage"
        
        # Warn if less than 5GB
        storage_gb=$(df -BG "$current_dir" | tail -1 | awk '{print $4}' | sed 's/G//')
        if [ "$storage_gb" -lt 5 ]; then
            print_warning "Less than 5GB storage available. Consider starting with smaller data."
        fi
    else
        print_warning "Could not check storage usage. Ensure you have at least 5GB available."
    fi
    
    # Check CPU cores
    if command -v nproc &> /dev/null; then
        cpu_cores=$(nproc)
        print_status "CPU cores: $cpu_cores"
    else
        cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
        print_status "CPU cores: $cpu_cores"
    fi
}

# Function to check if data exists
check_data() {
    if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
        print_warning "No data found in data/raw directory"
        print_status "Please add your training data (.txt, .pdf, .docx files) to data/raw/"
        print_status "You can also download sample data or use your own documents"
        print_status "💡 TIP: Start with 100MB of data to test the pipeline first!"
        return 1
    fi
    
    file_count=$(find data/raw -type f \( -name "*.txt" -o -name "*.pdf" -o -name "*.docx" \) | wc -l)
    total_size=$(du -sh data/raw 2>/dev/null | cut -f1 || echo "unknown")
    print_status "Found $file_count data files in data/raw (total size: $total_size)"
    
    # Warn about large datasets
    size_mb=$(du -sm data/raw 2>/dev/null | cut -f1 || echo "0")
    if [ "$size_mb" -gt 1000 ]; then
        print_warning "Large dataset detected ($size_mb MB). Training may take days, not hours."
        print_status "Consider starting with a smaller subset for testing."
    fi
    
    return 0
}

# Function to run preprocessing
run_preprocessing() {
    print_status "=== Stage 1: Data Preprocessing ==="
    
    if ! check_data; then
        print_error "Cannot proceed without training data"
        exit 1
    fi
    
    print_status "Starting data preprocessing..."
    
    if python training/preprocess.py; then
        print_success "Data preprocessing completed"
    else
        print_error "Data preprocessing failed"
        exit 1
    fi
    
    # Check output
    if [ -f "data/cleaned/combined_text.txt" ]; then
        file_size=$(stat -f%z "data/cleaned/combined_text.txt" 2>/dev/null || stat -c%s "data/cleaned/combined_text.txt" 2>/dev/null || echo "unknown")
        print_status "Combined text file size: $file_size bytes"
    fi
}

# Function to run tokenizer training
run_tokenizer_training() {
    print_status "=== Stage 2: Tokenizer Training ==="
    
    if [ ! -f "data/cleaned/combined_text.txt" ]; then
        print_error "No cleaned data found. Please run preprocessing first."
        exit 1
    fi
    
    print_status "Starting tokenizer training..."
    
    if python training/train_tokenizer.py; then
        print_success "Tokenizer training completed"
    else
        print_error "Tokenizer training failed"
        exit 1
    fi
    
    # Check output
    if [ -f "tokenizer/tokenizer.model" ] && [ -f "data/tokens/tokens.pt" ]; then
        print_success "Tokenizer and tokenized dataset created successfully"
    else
        print_error "Tokenizer output files not found"
        exit 1
    fi
}

# Function to run model training
run_model_training() {
    print_status "=== Stage 3: Model Training ==="
    
    if [ ! -f "data/tokens/tokens.pt" ]; then
        print_error "No tokenized data found. Please run tokenizer training first."
        exit 1
    fi
    
    if [ ! -f "tokenizer/tokenizer.model" ]; then
        print_error "No tokenizer found. Please run tokenizer training first."
        exit 1
    fi
    
    print_status "Starting model training..."
    print_status "This may take a while depending on your data size and hardware..."
    
    if [ "$CPU_ONLY" = true ]; then
        print_status "Training on CPU (forced)"
    fi
    
    if python training/train.py; then
        print_success "Model training completed"
    else
        print_error "Model training failed"
        exit 1
    fi
    
    # Check output
    if [ -f "exports/checkpoints/best_model.pt" ]; then
        print_success "Best model checkpoint saved"
    else
        print_warning "Best model checkpoint not found, but training may have completed"
    fi
}

# Function to run evaluation
run_evaluation() {
    if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR/*.pt 2>/dev/null)" ]; then
        print_error "No model checkpoints found in $CHECKPOINT_DIR"
        return 1
    fi
    
    print_status "Starting model evaluation..."
    
    if python eval/eval.py; then
        print_success "Model evaluation completed"
    else
        print_error "Model evaluation failed"
        exit 1
    fi
}

# Function to run fine-tuning
run_finetuning() {
    if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR/*.pt 2>/dev/null)" ]; then
        print_error "No base model checkpoints found in $CHECKPOINT_DIR"
        print_error "Please train a model first or specify a pre-trained model path"
        return 1
    fi
    
    FINETUNE_DATA_DIR="data/finetune"
    if [ ! -d "$FINETUNE_DATA_DIR" ] || [ -z "$(find $FINETUNE_DATA_DIR -type f 2>/dev/null)" ]; then
        print_warning "No fine-tuning data found in $FINETUNE_DATA_DIR"
        print_warning "Please add your fine-tuning data to $FINETUNE_DATA_DIR"
        return 1
    fi
    
    print_status "Starting model fine-tuning..."
    
    # Get the latest checkpoint
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.pt 2>/dev/null | head -1)
    
    if python finetune/finetune.py --config config.json --pretrained-model "$LATEST_CHECKPOINT" --train-data "$FINETUNE_DATA_DIR" --tokenizer-dir "$TOKENIZER_DIR"; then
        print_success "Model fine-tuning completed"
    else
        print_error "Model fine-tuning failed"
        exit 1
    fi
}

# Function to run inference
run_inference() {
    if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR/*.pt 2>/dev/null)" ]; then
        print_error "No model checkpoints found in $CHECKPOINT_DIR"
        return 1
    fi
    
    print_status "Starting interactive inference..."
    
    # Get the latest checkpoint
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.pt 2>/dev/null | head -1)
    
    if python inference.py --model "$LATEST_CHECKPOINT" --tokenizer "$TOKENIZER_DIR" --config config.json --interactive; then
        print_success "Inference session completed"
    else
        print_error "Inference failed"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "LLM Training Pipeline"
    echo "Usage: $0 [stage] [options]"
    echo ""
    echo "Stages:"
    echo "  preprocess  - Run data preprocessing only"
    echo "  tokenizer   - Run tokenizer training only"
    echo "  train       - Run model training only"
    echo "  eval        - Run model evaluation only"
    echo "  finetune    - Fine-tune a pre-trained model"
    echo "  inference   - Run interactive text generation"
    echo "  download    - Download a HuggingFace model"
    echo "  all         - Run all main stages (default)"
    echo "  eval        - Run model evaluation only"
    echo "  all         - Run complete pipeline (default)"
    echo ""
    echo "Options:"
    echo "  --cpu-only  - Force CPU-only training"
    echo "  --verbose   - Enable verbose output"
    echo "  --help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run complete pipeline"
    echo "  $0 preprocess        # Run preprocessing only"
    echo "  $0 train --cpu-only  # Train on CPU only"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        preprocess|tokenizer|train|eval|finetune|inference|download|all)
            STAGE="$1"
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting LLM Training Pipeline"
    print_status "Stage: $STAGE"
    
    if [ "$CPU_ONLY" = true ]; then
        print_status "CPU-only mode enabled"
    fi
    
    # Setup
    check_python
    check_dependencies
    setup_directories
    
    # Execute based on stage
    case $STAGE in
        "preprocess")
            run_preprocessing
            ;;
        "tokenizer")
            run_tokenizer_training
            ;;
        "train")
            run_model_training
            ;;
        "eval")
            run_evaluation
            ;;
        "finetune")
            run_finetuning
            ;;
        "inference")
            run_inference
            ;;
        "download")
            run_download
            ;;
        "all")
            run_preprocessing
            run_tokenizer_training
            run_model_training
            run_evaluation
            ;;
        *)
            print_error "Invalid stage: $STAGE"
            show_help
            exit 1
            ;;
    esac
    
    print_success "Pipeline completed successfully!"
    print_status "Check the logs/ directory for detailed logs"
    
    if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
        print_status "Your trained model is available at: exports/checkpoints/best_model.pt"
    fi
    
    if [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
        print_status "Evaluation results are in the logs"
    fi
}

# Run main function
main
