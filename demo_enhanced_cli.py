#!/usr/bin/env python3
"""
Demo script showing the enhanced CLI in action.

This script simulates a complete LLMBuilder workflow with colors and progress.
"""

import time
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from llmbuilder.utils.colors import (
    ColorFormatter, Color, print_header, print_success, print_error, 
    print_warning, print_info, print_table, confirm_action
)
from llmbuilder.utils.progress import (
    progress_bar, spinner, long_running_task, show_step_progress
)
from llmbuilder.utils.status import ProgressTracker


def simulate_project_creation():
    """Simulate project creation with enhanced UI."""
    print_header("LLMBuilder Project Creation")
    
    project_name = "my-awesome-llm"
    print_info(f"Creating project: {project_name}")
    print_info("Template: production")
    print_info("GPU support: enabled")
    
    def create_project_files():
        """Simulate creating project files."""
        time.sleep(1.5)
        return f"Project created at ./{project_name}/"
    
    result = long_running_task(
        create_project_files,
        message="Setting up project structure",
        success_message="Project created successfully",
        timeout_message="Creating directories and configuration files..."
    )
    
    print_info(result)
    
    # Show next steps
    print_header("Next Steps")
    steps = [
        f"cd {project_name}",
        "Add training data to data/raw/",
        "Run: llmbuilder data prepare",
        "Run: llmbuilder train start"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"  {ColorFormatter.format(f'{i}.', Color.BLUE)} {step}")


def simulate_data_preparation():
    """Simulate data preparation with progress tracking."""
    print_header("Data Preparation Pipeline")
    
    # Multi-step progress tracking
    tracker = ProgressTracker("Data Processing")
    tracker.start_display()
    
    try:
        # Step 1: File discovery
        tracker.add_operation("discovery", "File Discovery", 100)
        for i in range(100):
            time.sleep(0.01)
            tracker.update_progress("discovery", i + 1)
        tracker.complete_operation("discovery", True)
        
        # Step 2: Text extraction
        tracker.add_operation("extraction", "Text Extraction", 150)
        for i in range(150):
            time.sleep(0.008)
            tracker.update_progress("extraction", i + 1)
        tracker.complete_operation("extraction", True)
        
        # Step 3: Deduplication
        tracker.add_operation("dedup", "Deduplication", 80)
        for i in range(80):
            time.sleep(0.015)
            tracker.update_progress("dedup", i + 1)
        tracker.complete_operation("dedup", True)
        
        # Step 4: Text cleaning
        tracker.add_operation("cleaning", "Text Cleaning", 120)
        for i in range(120):
            time.sleep(0.01)
            tracker.update_progress("cleaning", i + 1)
        tracker.complete_operation("cleaning", True)
        
        time.sleep(2)  # Show final state
        
    finally:
        tracker.stop_display()
    
    # Show summary
    print_success("Data preparation completed!")
    
    headers = ["Stage", "Files Processed", "Status"]
    rows = [
        ["File Discovery", "1,247", "✓ Complete"],
        ["Text Extraction", "1,247", "✓ Complete"],
        ["Deduplication", "1,247 → 1,089", "✓ Complete"],
        ["Text Cleaning", "1,089", "✓ Complete"]
    ]
    
    print_table(headers, rows, header_color=Color.GREEN)


def simulate_training_session():
    """Simulate a training session with real-time updates."""
    print_header("Model Training Session")
    
    print_info("Model: GPT-2 Small (124M parameters)")
    print_info("Training data: 1,089 documents")
    print_info("Batch size: 8")
    print_info("Learning rate: 5e-5")
    
    # Training progress with multiple metrics
    epochs = 3
    steps_per_epoch = 50
    
    for epoch in range(1, epochs + 1):
        print_header(f"Epoch {epoch}/{epochs}")
        
        with progress_bar(
            total=steps_per_epoch,
            description=f"Training Epoch {epoch}",
            unit="steps",
            color=Color.GREEN
        ) as pbar:
            
            for step in range(steps_per_epoch):
                # Simulate training step
                time.sleep(0.05)
                
                # Update with realistic metrics
                loss = 4.5 - (epoch - 1) * 0.8 - step * 0.02
                lr = 5e-5 * (0.95 ** (epoch - 1))
                
                status = f"Loss: {loss:.3f}, LR: {lr:.2e}"
                pbar.set_status(status)
                pbar.update(1)
        
        # Epoch summary
        final_loss = 4.5 - epoch * 0.8
        print_success(f"Epoch {epoch} completed - Loss: {final_loss:.3f}")
    
    print_success("Training completed successfully!")
    
    # Training summary
    headers = ["Metric", "Initial", "Final", "Improvement"]
    rows = [
        ["Loss", "4.500", "2.100", "53.3%"],
        ["Perplexity", "90.02", "8.17", "90.9%"],
        ["Learning Rate", "5.0e-5", "4.3e-5", "14.0%"],
        ["Training Time", "0:00:00", "0:07:30", "N/A"]
    ]
    
    print_table(headers, rows, header_color=Color.BLUE)


def simulate_model_evaluation():
    """Simulate model evaluation with benchmarks."""
    print_header("Model Evaluation")
    
    benchmarks = ["Perplexity", "BLEU Score", "ROUGE-L", "Accuracy"]
    
    with progress_bar(
        total=len(benchmarks),
        description="Running benchmarks",
        unit="tests",
        color=Color.BLUE
    ) as pbar:
        
        results = {}
        for benchmark in benchmarks:
            pbar.set_status(f"Running {benchmark}")
            time.sleep(1.5)  # Simulate benchmark time
            
            # Simulate realistic scores
            if benchmark == "Perplexity":
                results[benchmark] = "8.17"
            elif benchmark == "BLEU Score":
                results[benchmark] = "0.342"
            elif benchmark == "ROUGE-L":
                results[benchmark] = "0.456"
            else:
                results[benchmark] = "87.3%"
            
            pbar.update(1)
    
    print_success("Evaluation completed!")
    
    # Results table
    headers = ["Benchmark", "Score", "Baseline", "Improvement"]
    rows = [
        ["Perplexity", "8.17", "12.45", "+34.2%"],
        ["BLEU Score", "0.342", "0.298", "+14.8%"],
        ["ROUGE-L", "0.456", "0.401", "+13.7%"],
        ["Accuracy", "87.3%", "82.1%", "+6.3%"]
    ]
    
    print_table(headers, rows, header_color=Color.GREEN)


def simulate_deployment():
    """Simulate model deployment process."""
    print_header("Model Deployment")
    
    deployment_steps = [
        "Model Optimization",
        "API Server Setup", 
        "Health Checks",
        "Load Testing",
        "Production Deploy"
    ]
    
    for i, step in enumerate(deployment_steps, 1):
        show_step_progress(deployment_steps, i, step)
        
        if step == "Model Optimization":
            with spinner("Quantizing model to INT8", color=Color.YELLOW) as spin:
                time.sleep(2)
                spin.set_status("Optimizing inference speed...")
                time.sleep(1)
        elif step == "Load Testing":
            with progress_bar(
                total=100,
                description="Load testing",
                unit="requests",
                color=Color.BLUE
            ) as pbar:
                for j in range(100):
                    time.sleep(0.02)
                    pbar.set_status(f"RPS: {j*2}, Latency: {50-j*0.2:.1f}ms")
                    pbar.update(1)
        else:
            time.sleep(1.5)
    
    print_success("Deployment completed successfully!")
    
    # Deployment info
    print_info("API Endpoint: https://api.mycompany.com/llm/v1")
    print_info("Health Check: https://api.mycompany.com/health")
    print_info("Documentation: https://api.mycompany.com/docs")


def main():
    """Run the complete demo."""
    print_header("LLMBuilder Enhanced CLI Demo")
    print_info("Demonstrating the complete ML workflow with enhanced UI")
    
    try:
        simulate_project_creation()
        time.sleep(2)
        
        simulate_data_preparation()
        time.sleep(2)
        
        simulate_training_session()
        time.sleep(2)
        
        simulate_model_evaluation()
        time.sleep(2)
        
        simulate_deployment()
        
        print_header("Demo Complete")
        print_success("LLMBuilder enhanced CLI demonstration finished!")
        print_info("The CLI now features:")
        
        features = [
            "Consistent color scheme (RED, YELLOW, BLUE, LIGHT BLUE, GREEN, WHITE)",
            "Enhanced progress bars with status messages",
            "Real-time status displays and dashboards",
            "Step-by-step progress indicators",
            "Interactive menus and confirmations",
            "Informative timeout messages for long operations",
            "Professional table formatting",
            "Comprehensive error handling with colored output"
        ]
        
        for feature in features:
            print(f"  • {ColorFormatter.format(feature, Color.BLUE_LIGHT)}")
        
    except KeyboardInterrupt:
        print_warning("\nDemo interrupted by user")
    except Exception as e:
        print_error(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())