# LLMBuilder Examples

This directory contains practical examples demonstrating how to use LLMBuilder for various machine learning tasks.

## Quick Start Examples

### 1. Basic Fine-tuning Example

```bash
#!/bin/bash
# basic_finetuning.sh - Complete fine-tuning workflow

# Create a new project
llmbuilder init my-chatbot --template fine-tuning

cd my-chatbot

# Prepare your data (assuming you have text files in ./raw_data)
llmbuilder data prepare \
  --input ./raw_data \
  --output ./data/processed \
  --formats txt,json \
  --clean \
  --deduplicate \
  --min-length 50

# Split the data
llmbuilder data split \
  --input ./data/processed \
  --ratios 0.8,0.1,0.1 \
  --stratify

# Select a base model
llmbuilder model select microsoft/DialoGPT-medium

# Configure training parameters
llmbuilder config set training.epochs 5
llmbuilder config set training.batch_size 4
llmbuilder config set training.learning_rate 2e-5
llmbuilder config set training.method lora

# Start training
llmbuilder train start \
  --data ./data/train \
  --output ./checkpoints

# Evaluate the model
llmbuilder eval run \
  --model ./checkpoints/final \
  --data ./data/test \
  --metrics perplexity,bleu

# Test interactively
llmbuilder inference \
  --model ./checkpoints/final \
  --interactive

# Deploy as API
llmbuilder deploy start \
  --model ./checkpoints/final \
  --port 8000
```

### 2. Pipeline Automation Example

```bash
#!/bin/bash
# automated_pipeline.sh - Fully automated training pipeline

# Create training pipeline
llmbuilder pipeline train production-model \
  --data-path ./datasets/customer_support \
  --model microsoft/DialoGPT-large \
  --output-dir ./experiments/$(date +%Y%m%d_%H%M%S) \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-5

# Get the workflow ID (last created)
WORKFLOW_ID=$(llmbuilder pipeline list --status pending | tail -n 1 | awk '{print $1}')

# Execute the pipeline
llmbuilder pipeline run $WORKFLOW_ID

# Monitor progress
while true; do
  STATUS=$(llmbuilder pipeline status $WORKFLOW_ID | grep "Status:" | awk '{print $2}')
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  echo "Pipeline status: $STATUS"
  sleep 30
done

echo "Pipeline completed with status: $STATUS"

# If successful, create deployment pipeline
if [ "$STATUS" = "completed" ]; then
  MODEL_PATH=$(llmbuilder pipeline status $WORKFLOW_ID | grep "Output:" | awk '{print $2}')
  
  llmbuilder pipeline deploy production-deployment \
    --model-path $MODEL_PATH/checkpoints/final \
    --type api \
    --optimize \
    --quantization-format gguf
fi
```

## Advanced Examples

### 3. Multi-GPU Training Example

```bash
#!/bin/bash
# multi_gpu_training.sh - Distributed training setup

# Configure for multi-GPU training
llmbuilder config set training.distributed true
llmbuilder config set training.num_gpus 4
llmbuilder config set training.batch_size 16  # Total batch size across GPUs
llmbuilder config set training.gradient_accumulation_steps 4

# Use larger model for multi-GPU setup
llmbuilder model select microsoft/DialoGPT-large

# Start distributed training
llmbuilder train start \
  --data ./data/large_dataset \
  --output ./checkpoints/multi_gpu \
  --method qlora \
  --mixed-precision true \
  --gradient-checkpointing true

# Monitor training across GPUs
llmbuilder monitor dashboard --port 8080
```

### 4. Custom Data Processing Example

```python
# custom_processor.py - Custom data processing pipeline

from llmbuilder.core.data import DataProcessor
from llmbuilder.utils.logging import get_logger
import re
import json

logger = get_logger(__name__)

class ConversationProcessor(DataProcessor):
    """Custom processor for conversation data."""
    
    def __init__(self, config):
        super().__init__(config)
        self.min_turns = config.get('min_turns', 2)
        self.max_turns = config.get('max_turns', 10)
    
    def process_file(self, file_path):
        """Process conversation files."""
        conversations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for conversation in data:
            if self.is_valid_conversation(conversation):
                processed = self.format_conversation(conversation)
                conversations.append(processed)
        
        return conversations
    
    def is_valid_conversation(self, conversation):
        """Validate conversation quality."""
        turns = conversation.get('turns', [])
        
        # Check turn count
        if len(turns) < self.min_turns or len(turns) > self.max_turns:
            return False
        
        # Check for empty turns
        if any(not turn.get('text', '').strip() for turn in turns):
            return False
        
        # Check for appropriate length
        total_length = sum(len(turn['text']) for turn in turns)
        if total_length < 100 or total_length > 2000:
            return False
        
        return True
    
    def format_conversation(self, conversation):
        """Format conversation for training."""
        formatted_turns = []
        
        for turn in conversation['turns']:
            # Clean and format text
            text = self.clean_text(turn['text'])
            speaker = turn.get('speaker', 'user')
            
            formatted_turns.append(f"{speaker}: {text}")
        
        return {
            'text': '\n'.join(formatted_turns),
            'metadata': {
                'turns': len(formatted_turns),
                'source': conversation.get('source', 'unknown')
            }
        }
    
    def clean_text(self, text):
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize punctuation
        text = re.sub(r'[.]{2,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text

# Register the custom processor
if __name__ == "__main__":
    import sys
    from llmbuilder.core.data import register_processor
    
    register_processor("conversation", ConversationProcessor)
    print("Custom conversation processor registered successfully!")
```

```bash
#!/bin/bash
# use_custom_processor.sh - Using custom data processor

# Register the custom processor
python custom_processor.py

# Use custom processor in data preparation
llmbuilder data prepare \
  --input ./conversation_data \
  --output ./processed_conversations \
  --processor conversation \
  --config '{"min_turns": 3, "max_turns": 8}' \
  --formats json
```

### 5. Model Comparison and A/B Testing

```bash
#!/bin/bash
# model_comparison.sh - Compare multiple models

# Train multiple models with different configurations
MODELS=("gpt2" "microsoft/DialoGPT-medium" "microsoft/DialoGPT-large")
METHODS=("lora" "qlora" "full")

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    experiment_name="${model##*/}_${method}_$(date +%Y%m%d_%H%M%S)"
    
    echo "Training $experiment_name..."
    
    llmbuilder train start \
      --model "$model" \
      --method "$method" \
      --data ./data/train \
      --output "./experiments/$experiment_name" \
      --epochs 3 \
      --batch-size 4
    
    # Evaluate each model
    llmbuilder eval run \
      --model "./experiments/$experiment_name/final" \
      --data ./data/test \
      --output "./experiments/$experiment_name/evaluation.json"
  done
done

# Compare all models
llmbuilder eval compare \
  ./experiments/*/final \
  --data ./data/test \
  --output ./model_comparison_report.html
```

### 6. Production Deployment Example

```bash
#!/bin/bash
# production_deployment.sh - Production-ready deployment

# Optimize model for production
llmbuilder optimize quantize \
  --model ./checkpoints/best_model \
  --format gguf \
  --output ./production/model.gguf \
  --calibration ./data/calibration

# Create deployment package
llmbuilder deploy package \
  --model ./production/model.gguf \
  --format docker \
  --output ./production/deployment \
  --include "requirements.txt,config.json"

# Test deployment locally
llmbuilder deploy start \
  --model ./production/model.gguf \
  --port 8000 \
  --workers 4 \
  --api-key "your-api-key"

# Health check
curl -X GET "http://localhost:8000/health"

# Test inference
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_length": 100,
    "temperature": 0.7
  }'
```

### 7. Monitoring and Debugging Example

```bash
#!/bin/bash
# monitoring_setup.sh - Comprehensive monitoring

# Start monitoring dashboard
llmbuilder monitor dashboard \
  --port 8080 \
  --data-dir ./monitoring_data &

# Set up log monitoring
llmbuilder monitor logs \
  --level INFO \
  --follow \
  --search "error|warning" > ./logs/filtered.log &

# Run system diagnostics
llmbuilder monitor debug \
  --output ./diagnostics/system_report.json

# Monitor training in real-time
llmbuilder train start \
  --data ./data/train \
  --output ./checkpoints \
  --monitor-port 8081 &

TRAIN_PID=$!

# Monitor GPU usage
while kill -0 $TRAIN_PID 2>/dev/null; do
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> ./monitoring_data/gpu_usage.csv
  sleep 10
done

echo "Training completed. Check monitoring dashboard at http://localhost:8080"
```

## Integration Examples

### 8. Jupyter Notebook Integration

```python
# notebook_integration.py - Using LLMBuilder in Jupyter notebooks

import llmbuilder
from llmbuilder.utils.config import ConfigManager
from llmbuilder.core.training import TrainingManager
from llmbuilder.core.evaluation import EvaluationManager

# Initialize in notebook
config_manager = ConfigManager()
config = config_manager.get_default_config()

# Programmatic training
training_manager = TrainingManager(config)

# Start training with progress bar
training_session = training_manager.start_training(
    data_path="./data/train",
    model_path="gpt2",
    output_path="./checkpoints",
    show_progress=True
)

# Monitor training progress
for epoch, metrics in training_session.progress():
    print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}")
    
    # Plot metrics in real-time
    import matplotlib.pyplot as plt
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

# Evaluate model
eval_manager = EvaluationManager(config)
results = eval_manager.evaluate(
    model_path="./checkpoints/final",
    test_data="./data/test"
)

print(f"Evaluation Results: {results}")
```

### 9. API Integration Example

```python
# api_integration.py - Integrating with external APIs

import requests
import json
from llmbuilder.core.inference import InferenceEngine

class APIIntegratedBot:
    def __init__(self, model_path, api_key):
        self.inference_engine = InferenceEngine(model_path)
        self.api_key = api_key
    
    def generate_with_context(self, prompt, context_api_url=None):
        """Generate response with external context."""
        
        # Get context from external API if provided
        context = ""
        if context_api_url:
            try:
                response = requests.get(
                    context_api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                context = response.json().get("context", "")
            except Exception as e:
                print(f"Failed to get context: {e}")
        
        # Combine context with prompt
        full_prompt = f"Context: {context}\n\nUser: {prompt}\nAssistant:"
        
        # Generate response
        response = self.inference_engine.generate(
            prompt=full_prompt,
            max_length=200,
            temperature=0.7
        )
        
        return response
    
    def batch_process(self, prompts, output_file):
        """Process multiple prompts and save results."""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing {i+1}/{len(prompts)}")
            response = self.generate_with_context(prompt)
            results.append({
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

# Usage
bot = APIIntegratedBot("./checkpoints/final", "your-api-key")
response = bot.generate_with_context(
    "What's the weather like?",
    "https://api.weather.com/current"
)
print(response)
```

### 10. Continuous Integration Example

```yaml
# .github/workflows/llm_training.yml - CI/CD for model training

name: LLM Training Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 0'  # Weekly training

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install LLMBuilder
      run: |
        pip install llmbuilder[gpu]
    
    - name: Download training data
      run: |
        # Download or prepare your training data
        llmbuilder data prepare \
          --input ./raw_data \
          --output ./data/processed
    
    - name: Train model
      run: |
        llmbuilder train start \
          --data ./data/processed \
          --model gpt2 \
          --output ./checkpoints \
          --epochs 2 \
          --batch-size 2
    
    - name: Evaluate model
      run: |
        llmbuilder eval run \
          --model ./checkpoints/final \
          --data ./data/test \
          --output ./evaluation_results.json
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: |
          ./checkpoints/
          ./evaluation_results.json
    
    - name: Deploy to staging
      if: github.ref == 'refs/heads/main'
      run: |
        llmbuilder deploy package \
          --model ./checkpoints/final \
          --format docker \
          --output ./deployment
        
        # Deploy to your staging environment
        # docker build -t my-llm-model ./deployment
        # docker push my-registry/my-llm-model:latest
```

## Best Practices

### Configuration Management

```bash
# Use environment-specific configs
llmbuilder config set --env development training.epochs 1
llmbuilder config set --env production training.epochs 10

# Version your configurations
cp .llmbuilder/config.json .llmbuilder/config_v1.0.json
```

### Data Management

```bash
# Always validate data before training
llmbuilder data validate --input ./data --fix

# Use consistent data splits
llmbuilder data split --seed 42 --stratify

# Monitor data quality
llmbuilder data stats --input ./data --output ./data_report.html
```

### Model Management

```bash
# Tag your models
llmbuilder model tag ./checkpoints/final production-v1.0

# Compare model versions
llmbuilder eval compare \
  ./models/v1.0 \
  ./models/v2.0 \
  --data ./data/test

# Archive old models
llmbuilder model archive ./checkpoints/old_models
```

### Monitoring and Logging

```bash
# Set up comprehensive logging
llmbuilder config set logging.level DEBUG
llmbuilder config set logging.file ./logs/training.log

# Monitor resource usage
llmbuilder monitor resources --alert-threshold 90

# Set up alerts
llmbuilder monitor alert \
  --metric gpu_memory \
  --threshold 95 \
  --action "echo 'GPU memory high' | mail admin@company.com"
```

These examples demonstrate the full range of LLMBuilder capabilities, from basic fine-tuning to advanced production deployments. Each example includes practical code that you can adapt for your specific use cases.