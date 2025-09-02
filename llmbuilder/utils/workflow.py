"""
Workflow management for cross-command data sharing and pipeline execution.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    command: str
    args: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class WorkflowContext:
    """Shared context between workflow steps."""
    project_path: Path
    config: Dict[str, Any]
    shared_data: Dict[str, Any]
    current_step: int = 0
    total_steps: int = 0


class WorkflowManager:
    """Manages workflow execution and cross-command data sharing."""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path or Path.cwd()
        self.workflow_dir = self.project_path / ".llmbuilder" / "workflows"
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.config_manager = ConfigManager()
        
    def create_workflow(self, name: str, steps: List[Dict[str, Any]]) -> str:
        """Create a new workflow with the given steps."""
        workflow_id = f"{name}_{int(time.time())}"
        workflow_file = self.workflow_dir / f"{workflow_id}.json"
        
        workflow_steps = [
            WorkflowStep(command=step["command"], args=step.get("args", {}))
            for step in steps
        ]
        
        workflow_data = {
            "id": workflow_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "steps": [asdict(step) for step in workflow_steps],
            "context": {
                "project_path": str(self.project_path),
                "config": self.config_manager.get_default_config(),
                "shared_data": {},
                "current_step": 0,
                "total_steps": len(workflow_steps)
            }
        }
        
        with open(workflow_file, 'w') as f:
            json.dump(workflow_data, f, indent=2, default=str)
        
        logger.info(f"Created workflow '{name}' with ID: {workflow_id}")
        return workflow_id
    
    def load_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Load a workflow by ID."""
        workflow_file = self.workflow_dir / f"{workflow_id}.json"
        if not workflow_file.exists():
            raise FileNotFoundError(f"Workflow {workflow_id} not found")
        
        with open(workflow_file, 'r') as f:
            return json.load(f)
    
    def save_workflow(self, workflow_id: str, workflow_data: Dict[str, Any]):
        """Save workflow data."""
        workflow_file = self.workflow_dir / f"{workflow_id}.json"
        with open(workflow_file, 'w') as f:
            json.dump(workflow_data, f, indent=2, default=str)
    
    def update_step_status(self, workflow_id: str, step_index: int, 
                          status: str, output: Optional[Dict[str, Any]] = None,
                          error: Optional[str] = None):
        """Update the status of a workflow step."""
        workflow_data = self.load_workflow(workflow_id)
        step = workflow_data["steps"][step_index]
        
        step["status"] = status
        if status == "running":
            step["start_time"] = datetime.now().isoformat()
        elif status in ["completed", "failed"]:
            step["end_time"] = datetime.now().isoformat()
        
        if output:
            step["output"] = output
        if error:
            step["error"] = error
        
        self.save_workflow(workflow_id, workflow_data)
    
    def get_shared_data(self, workflow_id: str, key: Optional[str] = None) -> Any:
        """Get shared data from workflow context."""
        workflow_data = self.load_workflow(workflow_id)
        shared_data = workflow_data["context"]["shared_data"]
        
        if key:
            return shared_data.get(key)
        return shared_data
    
    def set_shared_data(self, workflow_id: str, key: str, value: Any):
        """Set shared data in workflow context."""
        workflow_data = self.load_workflow(workflow_id)
        workflow_data["context"]["shared_data"][key] = value
        self.save_workflow(workflow_id, workflow_data)
    
    def get_workflow_context(self, workflow_id: str) -> WorkflowContext:
        """Get the workflow context."""
        workflow_data = self.load_workflow(workflow_id)
        context_data = workflow_data["context"]
        
        return WorkflowContext(
            project_path=Path(context_data["project_path"]),
            config=context_data["config"],
            shared_data=context_data["shared_data"],
            current_step=context_data["current_step"],
            total_steps=context_data["total_steps"]
        )
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows in the project."""
        workflows = []
        for workflow_file in self.workflow_dir.glob("*.json"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)
                    workflows.append({
                        "id": workflow_data["id"],
                        "name": workflow_data["name"],
                        "created_at": workflow_data["created_at"],
                        "status": self._get_workflow_status(workflow_data),
                        "progress": self._get_workflow_progress(workflow_data)
                    })
            except Exception as e:
                logger.warning(f"Failed to load workflow {workflow_file}: {e}")
        
        return sorted(workflows, key=lambda x: x["created_at"], reverse=True)
    
    def _get_workflow_status(self, workflow_data: Dict[str, Any]) -> str:
        """Determine the overall status of a workflow."""
        steps = workflow_data["steps"]
        if not steps:
            return "empty"
        
        if any(step["status"] == "failed" for step in steps):
            return "failed"
        elif any(step["status"] == "running" for step in steps):
            return "running"
        elif all(step["status"] == "completed" for step in steps):
            return "completed"
        else:
            return "pending"
    
    def _get_workflow_progress(self, workflow_data: Dict[str, Any]) -> float:
        """Calculate workflow progress as a percentage."""
        steps = workflow_data["steps"]
        if not steps:
            return 0.0
        
        completed_steps = sum(1 for step in steps if step["status"] == "completed")
        return (completed_steps / len(steps)) * 100


class PipelineBuilder:
    """Builder for creating common workflow pipelines."""
    
    @staticmethod
    def full_training_pipeline(
        data_path: str,
        model_name: str,
        output_dir: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Create a full training pipeline workflow."""
        return [
            {
                "command": "data prepare",
                "args": {
                    "input": data_path,
                    "output": f"{output_dir}/data",
                    **kwargs.get("data_args", {})
                }
            },
            {
                "command": "data split",
                "args": {
                    "input": f"{output_dir}/data",
                    "ratios": kwargs.get("split_ratios", [0.8, 0.1, 0.1])
                }
            },
            {
                "command": "model select",
                "args": {
                    "model": model_name,
                    "output": f"{output_dir}/model"
                }
            },
            {
                "command": "train start",
                "args": {
                    "data": f"{output_dir}/data",
                    "model": f"{output_dir}/model",
                    "output": f"{output_dir}/checkpoints",
                    **kwargs.get("train_args", {})
                }
            },
            {
                "command": "eval run",
                "args": {
                    "model": f"{output_dir}/checkpoints",
                    "data": f"{output_dir}/data/test",
                    **kwargs.get("eval_args", {})
                }
            }
        ]
    
    @staticmethod
    def deployment_pipeline(
        model_path: str,
        deployment_type: str = "api",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Create a deployment pipeline workflow."""
        steps = []
        
        if kwargs.get("optimize", True):
            steps.append({
                "command": "optimize quantize",
                "args": {
                    "model": model_path,
                    "format": kwargs.get("quantization_format", "gguf"),
                    **kwargs.get("optimize_args", {})
                }
            })
            model_path = f"{model_path}_quantized"
        
        if deployment_type == "api":
            steps.append({
                "command": "deploy start",
                "args": {
                    "model": model_path,
                    "type": "api",
                    **kwargs.get("deploy_args", {})
                }
            })
        elif deployment_type == "mobile":
            steps.append({
                "command": "deploy export-mobile",
                "args": {
                    "model": model_path,
                    **kwargs.get("mobile_args", {})
                }
            })
        
        return steps


# Global workflow manager instance
_workflow_manager = None

def get_workflow_manager(project_path: Optional[Path] = None) -> WorkflowManager:
    """Get the global workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None or (project_path and _workflow_manager.project_path != project_path):
        _workflow_manager = WorkflowManager(project_path)
    return _workflow_manager