"""
Training CLI commands for LLMBuilder.

This module provides commands for training management, including starting,
resuming, stopping training sessions, and monitoring progress.
"""

import click
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any
from loguru import logger
import os
import signal
import subprocess
import sys

from llmbuilder.utils.config import ConfigManager
from llmbuilder.utils.colors import (
    ColorFormatter, Color, print_header, print_success, print_error, 
    print_warning, print_info, print_table, confirm_action
)
from llmbuilder.utils.progress import (
    progress_bar, spinner, long_running_task, show_step_progress
)


class TrainingSession:
    """Represents a training session with metadata and status."""
    
    def __init__(self, session_id: str, config: Dict[str, Any], model_name: str = None):
        """Initialize training session."""
        self.session_id = session_id
        self.config = config
        self.model_name = model_name or "unnamed_model"
        self.status = "initialized"
        self.start_time = None
        self.end_time = None
        self.progress = 0.0
        self.metrics = {}
        self.process = None
        self.checkpoint_dir = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "progress": self.progress,
            "metrics": self.metrics,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSession':
        """Create session from dictionary."""
        session = cls(data["session_id"], data["config"], data.get("model_name"))
        session.status = data.get("status", "initialized")
        session.progress = data.get("progress", 0.0)
        session.metrics = data.get("metrics", {})
        if data.get("checkpoint_dir"):
            session.checkpoint_dir = Path(data["checkpoint_dir"])
        return session


class TrainingManager:
    """Manages training sessions and their lifecycle."""
    
    def __init__(self, sessions_file: Optional[Path] = None):
        """Initialize training manager."""
        if sessions_file is None:
            sessions_file = Path.home() / ".llmbuilder" / "training_sessions.json"
        
        self.sessions_file = sessions_file
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing sessions
        self.sessions = self._load_sessions()
    
    def _load_sessions(self) -> Dict[str, TrainingSession]:
        """Load training sessions from file."""
        if not self.sessions_file.exists():
            return {}
        
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
            
            sessions = {}
            for session_id, session_data in data.items():
                sessions[session_id] = TrainingSession.from_dict(session_data)
            
            return sessions
        except Exception as e:
            logger.warning(f"Could not load training sessions: {e}")
            return {}
    
    def _save_sessions(self):
        """Save training sessions to file."""
        try:
            data = {}
            for session_id, session in self.sessions.items():
                data[session_id] = session.to_dict()
            
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save training sessions: {e}")
    
    def create_session(self, config: Dict[str, Any], model_name: str = None) -> TrainingSession:
        """Create a new training session."""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        session = TrainingSession(session_id, config, model_name)
        self.sessions[session_id] = session
        self._save_sessions()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Get training session by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[TrainingSession]:
        """List all training sessions."""
        return list(self.sessions.values())
    
    def update_session(self, session: TrainingSession):
        """Update session and save to file."""
        self.sessions[session.session_id] = session
        self._save_sessions()
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a training session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            return True
        return False


@click.group()
def train():
    """
    Training commands for model fine-tuning and training management.
    
    This command group provides tools for:
    - Starting new training sessions
    - Resuming interrupted training
    - Monitoring training progress
    - Managing training sessions
    
    Examples:
        llmbuilder train start --config config.json
        llmbuilder train resume session_id
        llmbuilder train status
        llmbuilder train stop session_id
    """
    pass


@train.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Training configuration file'
)
@click.option(
    '--model-name',
    '-n',
    help='Name for the training session'
)
@click.option(
    '--background',
    '-b',
    is_flag=True,
    help='Run training in background'
)
@click.option(
    '--resume-from',
    type=click.Path(exists=True, path_type=Path),
    help='Resume from checkpoint directory'
)
@click.option(
    '--data-dir',
    type=click.Path(exists=True, path_type=Path),
    help='Training data directory'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory for checkpoints and logs'
)
@click.pass_context
def start(
    ctx: click.Context,
    config: Optional[Path],
    model_name: Optional[str],
    background: bool,
    resume_from: Optional[Path],
    data_dir: Optional[Path],
    output_dir: Optional[Path]
):
    """
    Start a new training session.
    
    Starts model training with the specified configuration. Can run in
    foreground with live progress monitoring or in background as a daemon.
    
    Examples:
        llmbuilder train start --config config.json --model-name my-model
        llmbuilder train start --background --data-dir ./data
        llmbuilder train start --resume-from ./checkpoints/checkpoint_1000
    """
    try:
        # Load configuration
        config_manager = ConfigManager()
        
        if config:
            training_config = config_manager.load_config(config)
        else:
            training_config = config_manager.get_default_config()
        
        # Override config with CLI arguments
        if data_dir:
            training_config['data']['input_dir'] = str(data_dir)
        if output_dir:
            training_config['training']['output_dir'] = str(output_dir)
        if resume_from:
            training_config['training']['resume_from_checkpoint'] = str(resume_from)
        
        # Create training session
        manager = TrainingManager()
        session = manager.create_session(training_config, model_name)
        
        console.print(f"[bold green]Created training session: {session.session_id}[/bold green]")
        
        if background:
            # Start training in background
            _start_background_training(session, manager)
            console.print(f"[yellow]Training started in background[/yellow]")
            console.print(f"Use 'llmbuilder train status {session.session_id}' to monitor progress")
        else:
            # Start training in foreground with live monitoring
            _start_foreground_training(session, manager)
        
    except Exception as e:
        console.print(f"[red]Error starting training: {e}[/red]")
        logger.error(f"Training start failed: {e}")
        raise click.ClickException(f"Training start failed: {e}")


@train.command()
@click.argument('session_id', required=False)
@click.option(
    '--background',
    '-b',
    is_flag=True,
    help='Resume training in background'
)
@click.pass_context
def resume(ctx: click.Context, session_id: Optional[str], background: bool):
    """
    Resume a training session from checkpoint.
    
    Resumes training from the last saved checkpoint. If no session ID is
    provided, resumes the most recent session.
    
    SESSION_ID: ID of the training session to resume
    
    Examples:
        llmbuilder train resume abc123
        llmbuilder train resume --background
    """
    try:
        manager = TrainingManager()
        
        if session_id:
            session = manager.get_session(session_id)
            if not session:
                console.print(f"[red]Session {session_id} not found[/red]")
                return
        else:
            # Find most recent session
            sessions = manager.list_sessions()
            if not sessions:
                console.print("[red]No training sessions found[/red]")
                return
            
            # Get most recent session
            session = max(sessions, key=lambda s: s.start_time or "")
            console.print(f"[yellow]Resuming most recent session: {session.session_id}[/yellow]")
        
        if session.status == "running":
            console.print(f"[yellow]Session {session.session_id} is already running[/yellow]")
            return
        
        console.print(f"[bold green]Resuming training session: {session.session_id}[/bold green]")
        
        if background:
            _start_background_training(session, manager)
            console.print(f"[yellow]Training resumed in background[/yellow]")
        else:
            _start_foreground_training(session, manager)
        
    except Exception as e:
        console.print(f"[red]Error resuming training: {e}[/red]")
        logger.error(f"Training resume failed: {e}")
        raise click.ClickException(f"Training resume failed: {e}")


@train.command()
@click.argument('session_id', required=False)
@click.option(
    '--all',
    '-a',
    is_flag=True,
    help='Show all sessions'
)
@click.option(
    '--format',
    type=click.Choice(['table', 'json']),
    default='table',
    help='Output format'
)
@click.pass_context
def status(ctx: click.Context, session_id: Optional[str], all: bool, format: str):
    """
    Show training session status and progress.
    
    Displays current status, progress, and metrics for training sessions.
    
    SESSION_ID: ID of specific session to show (optional)
    
    Examples:
        llmbuilder train status
        llmbuilder train status abc123
        llmbuilder train status --all --format json
    """
    try:
        manager = TrainingManager()
        
        if session_id:
            session = manager.get_session(session_id)
            if not session:
                console.print(f"[red]Session {session_id} not found[/red]")
                return
            sessions = [session]
        else:
            sessions = manager.list_sessions()
            if not all:
                # Show only active sessions by default
                sessions = [s for s in sessions if s.status in ["running", "paused"]]
        
        if not sessions:
            console.print("[yellow]No training sessions found[/yellow]")
            return
        
        if format == 'json':
            import json
            data = [session.to_dict() for session in sessions]
            console.print(json.dumps(data, indent=2))
            return
        
        # Display as table
        table = Table(title=f"Training Sessions ({len(sessions)} total)")
        table.add_column("Session ID", style="cyan")
        table.add_column("Model Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Progress", style="blue")
        table.add_column("Loss", style="magenta")
        table.add_column("Started", style="dim")
        
        for session in sessions:
            progress_str = f"{session.progress:.1f}%" if session.progress > 0 else "N/A"
            loss_str = f"{session.metrics.get('loss', 'N/A'):.4f}" if isinstance(session.metrics.get('loss'), (int, float)) else "N/A"
            start_time_str = session.start_time.strftime("%Y-%m-%d %H:%M") if session.start_time else "N/A"
            
            # Color code status
            status_color = {
                "running": "[green]running[/green]",
                "completed": "[blue]completed[/blue]",
                "failed": "[red]failed[/red]",
                "paused": "[yellow]paused[/yellow]",
                "initialized": "[dim]initialized[/dim]"
            }.get(session.status, session.status)
            
            table.add_row(
                session.session_id,
                session.model_name,
                status_color,
                progress_str,
                loss_str,
                start_time_str
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error getting training status: {e}[/red]")
        logger.error(f"Training status failed: {e}")
        raise click.ClickException(f"Training status failed: {e}")


@train.command()
@click.argument('session_id')
@click.option(
    '--force',
    '-f',
    is_flag=True,
    help='Force stop without confirmation'
)
@click.pass_context
def stop(ctx: click.Context, session_id: str, force: bool):
    """
    Stop a running training session.
    
    Gracefully stops a training session, saving the current checkpoint.
    
    SESSION_ID: ID of the training session to stop
    
    Examples:
        llmbuilder train stop abc123
        llmbuilder train stop abc123 --force
    """
    try:
        manager = TrainingManager()
        session = manager.get_session(session_id)
        
        if not session:
            console.print(f"[red]Session {session_id} not found[/red]")
            return
        
        if session.status != "running":
            console.print(f"[yellow]Session {session_id} is not running (status: {session.status})[/yellow]")
            return
        
        if not force:
            if not click.confirm(f"Stop training session {session_id}?"):
                console.print("Cancelled")
                return
        
        # Stop the training process
        if session.process and session.process.poll() is None:
            console.print(f"[yellow]Stopping training session {session_id}...[/yellow]")
            session.process.terminate()
            
            # Wait for graceful shutdown
            try:
                session.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                console.print("[yellow]Force killing training process...[/yellow]")
                session.process.kill()
        
        # Update session status
        session.status = "stopped"
        session.end_time = time.time()
        manager.update_session(session)
        
        console.print(f"[green]Training session {session_id} stopped[/green]")
        
    except Exception as e:
        console.print(f"[red]Error stopping training: {e}[/red]")
        logger.error(f"Training stop failed: {e}")
        raise click.ClickException(f"Training stop failed: {e}")


@train.command()
@click.argument('session_id')
@click.option(
    '--confirm/--no-confirm',
    default=True,
    help='Ask for confirmation before removing'
)
@click.pass_context
def remove(ctx: click.Context, session_id: str, confirm: bool):
    """
    Remove a training session from the registry.
    
    Removes the session metadata. Does not delete checkpoint files.
    
    SESSION_ID: ID of the training session to remove
    
    Examples:
        llmbuilder train remove abc123
        llmbuilder train remove abc123 --no-confirm
    """
    try:
        manager = TrainingManager()
        session = manager.get_session(session_id)
        
        if not session:
            console.print(f"[red]Session {session_id} not found[/red]")
            return
        
        if confirm:
            if not click.confirm(f"Remove training session {session_id}?"):
                console.print("Cancelled")
                return
        
        if manager.remove_session(session_id):
            console.print(f"[green]Training session {session_id} removed[/green]")
            console.print("[yellow]Note: Checkpoint files were not deleted[/yellow]")
        else:
            console.print(f"[red]Failed to remove session {session_id}[/red]")
        
    except Exception as e:
        console.print(f"[red]Error removing training session: {e}[/red]")
        logger.error(f"Training remove failed: {e}")
        raise click.ClickException(f"Training remove failed: {e}")


def _start_background_training(session: TrainingSession, manager: TrainingManager):
    """Start training in background process."""
    import subprocess
    import sys
    import tempfile
    
    # Update session status
    session.status = "running"
    session.start_time = time.time()
    manager.update_session(session)
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(session.config, f, indent=2)
        config_path = f.name
    
    # Create training script command
    cmd = [
        sys.executable, "-m", "llmbuilder.core.training.train",
        "--config", config_path
    ]
    
    # Add checkpoint directory if specified
    if session.config.get('training', {}).get('output_dir'):
        cmd.extend(["--checkpoint-dir", session.config['training']['output_dir']])
    
    # Start background process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        session.process = process
        session.checkpoint_dir = Path(session.config.get('training', {}).get('output_dir', './checkpoints'))
        manager.update_session(session)
        
        # Store config file path for cleanup later
        session.config['_temp_config_file'] = config_path
        
    except Exception as e:
        session.status = "failed"
        manager.update_session(session)
        # Clean up temp file on error
        try:
            os.unlink(config_path)
        except:
            pass
        raise e


def _start_foreground_training(session: TrainingSession, manager: TrainingManager):
    """Start training in foreground with live monitoring."""
    from llmbuilder.core.training.train import Trainer
    
    # Update session status
    session.status = "running"
    session.start_time = time.time()
    manager.update_session(session)
    
    try:
        # Create temporary config file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(session.config, f, indent=2)
            config_path = f.name
        
        # Initialize trainer
        trainer = Trainer(config_path)
        session.checkpoint_dir = Path(session.config.get('training', {}).get('output_dir', './checkpoints'))
        
        # Start training with progress monitoring
        console.print(f"[bold green]Starting training session: {session.session_id}[/bold green]")
        console.print(f"Model: {session.model_name}")
        console.print(f"Config: {config_path}")
        
        # Create a simple progress display
        with console.status("[bold green]Training in progress...") as status:
            # Start training
            trainer.train()
        
        # Training completed
        session.status = "completed"
        session.end_time = time.time()
        manager.update_session(session)
        
        console.print(f"[bold green]Training completed successfully![/bold green]")
        
        # Show final metrics if available
        if hasattr(trainer, 'metrics_tracker') and trainer.metrics_tracker.metrics:
            final_metrics = trainer.metrics_tracker.get_latest_metrics()
            console.print(f"Final Loss: {final_metrics.get('loss', 'N/A')}")
            console.print(f"Final Learning Rate: {final_metrics.get('lr', 'N/A')}")
        
        # Clean up temporary config file
        os.unlink(config_path)
        
    except KeyboardInterrupt:
        session.status = "interrupted"
        session.end_time = time.time()
        manager.update_session(session)
        console.print(f"[yellow]Training interrupted by user[/yellow]")
        
        # Clean up temporary config file
        try:
            os.unlink(config_path)
        except:
            pass
        
    except Exception as e:
        session.status = "failed"
        session.end_time = time.time()
        manager.update_session(session)
        console.print(f"[red]Training failed: {e}[/red]")
        
        # Clean up temporary config file
        try:
            os.unlink(config_path)
        except:
            pass
        
        raise e


if __name__ == "__main__":
    train()