"""
Command usage analytics and intelligent suggestions system.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class UsageAnalytics:
    """Track command usage and provide intelligent suggestions."""
    
    def __init__(self):
        self.analytics_dir = Path.home() / ".llmbuilder" / "analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        self.usage_file = self.analytics_dir / "usage.json"
        self.suggestions_file = self.analytics_dir / "suggestions.json"
        
        self.usage_data = self._load_usage_data()
        self.command_patterns = self._load_command_patterns()
    
    def record_command(self, command: str, args: List[str], success: bool = True, 
                      execution_time: float = 0.0, context: Optional[Dict[str, Any]] = None):
        """Record command usage for analytics."""
        timestamp = datetime.now().isoformat()
        
        usage_entry = {
            "command": command,
            "args": args,
            "success": success,
            "execution_time": execution_time,
            "timestamp": timestamp,
            "context": context or {}
        }
        
        # Add to usage data
        if "commands" not in self.usage_data:
            self.usage_data["commands"] = []
        
        self.usage_data["commands"].append(usage_entry)
        
        # Update statistics
        self._update_statistics(command, success, execution_time)
        
        # Save data
        self._save_usage_data()
        
        # Check for suggestions
        self._check_for_suggestions(command, args, success)
    
    def get_command_suggestions(self, current_command: Optional[str] = None) -> List[Dict[str, str]]:
        """Get intelligent command suggestions based on usage patterns."""
        suggestions = []
        
        # Get recent commands
        recent_commands = self._get_recent_commands(hours=24)
        
        # Pattern-based suggestions
        if current_command:
            pattern_suggestions = self._get_pattern_suggestions(current_command)
            suggestions.extend(pattern_suggestions)
        
        # Workflow-based suggestions
        workflow_suggestions = self._get_workflow_suggestions(recent_commands)
        suggestions.extend(workflow_suggestions)
        
        # Error recovery suggestions
        error_suggestions = self._get_error_recovery_suggestions()
        suggestions.extend(error_suggestions)
        
        # Optimization suggestions
        optimization_suggestions = self._get_optimization_suggestions()
        suggestions.extend(optimization_suggestions)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def show_usage_stats(self, days: int = 7) -> None:
        """Show usage statistics for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter commands by date
        recent_commands = [
            cmd for cmd in self.usage_data.get("commands", [])
            if datetime.fromisoformat(cmd["timestamp"]) > cutoff_date
        ]
        
        if not recent_commands:
            console.print(f"[yellow]No command usage data found for the last {days} days.[/yellow]")
            return
        
        # Command frequency
        command_counts = Counter(cmd["command"] for cmd in recent_commands)
        
        # Success rate
        success_rate = sum(1 for cmd in recent_commands if cmd["success"]) / len(recent_commands) * 100
        
        # Average execution time
        exec_times = [cmd["execution_time"] for cmd in recent_commands if cmd["execution_time"] > 0]
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        # Create statistics display
        stats_text = f"""
[bold]Usage Statistics (Last {days} days)[/bold]

[cyan]Total Commands:[/cyan] {len(recent_commands)}
[cyan]Success Rate:[/cyan] {success_rate:.1f}%
[cyan]Average Execution Time:[/cyan] {avg_exec_time:.2f}s
"""
        
        console.print(Panel(stats_text.strip(), title="Usage Statistics", border_style="blue"))
        
        # Most used commands table
        if command_counts:
            table = Table(title="Most Used Commands")
            table.add_column("Command", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Percentage", style="yellow")
            
            for command, count in command_counts.most_common(10):
                percentage = (count / len(recent_commands)) * 100
                table.add_row(command, str(count), f"{percentage:.1f}%")
            
            console.print(table)
    
    def show_suggestions(self, current_command: Optional[str] = None) -> None:
        """Show intelligent suggestions to the user."""
        suggestions = self.get_command_suggestions(current_command)
        
        if not suggestions:
            return
        
        console.print("\n[bold cyan]💡 Suggestions[/bold cyan]")
        
        for i, suggestion in enumerate(suggestions, 1):
            suggestion_type = suggestion.get("type", "general")
            icon = self._get_suggestion_icon(suggestion_type)
            
            console.print(f"{icon} [bold]{suggestion['title']}[/bold]")
            console.print(f"   {suggestion['description']}")
            
            if "command" in suggestion:
                console.print(f"   [green]$ {suggestion['command']}[/green]")
            
            if i < len(suggestions):
                console.print()
    
    def _load_usage_data(self) -> Dict[str, Any]:
        """Load usage data from file."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            "commands": [],
            "statistics": {
                "total_commands": 0,
                "successful_commands": 0,
                "failed_commands": 0,
                "command_counts": {},
                "average_execution_times": {}
            }
        }
    
    def _save_usage_data(self) -> None:
        """Save usage data to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except IOError:
            pass  # Fail silently for analytics
    
    def _update_statistics(self, command: str, success: bool, execution_time: float) -> None:
        """Update usage statistics."""
        stats = self.usage_data["statistics"]
        
        stats["total_commands"] = stats.get("total_commands", 0) + 1
        
        if success:
            stats["successful_commands"] = stats.get("successful_commands", 0) + 1
        else:
            stats["failed_commands"] = stats.get("failed_commands", 0) + 1
        
        # Update command counts
        if "command_counts" not in stats:
            stats["command_counts"] = {}
        stats["command_counts"][command] = stats["command_counts"].get(command, 0) + 1
        
        # Update execution times
        if execution_time > 0:
            if "average_execution_times" not in stats:
                stats["average_execution_times"] = {}
            
            current_avg = stats["average_execution_times"].get(command, 0)
            current_count = stats["command_counts"][command]
            
            # Calculate new average
            new_avg = ((current_avg * (current_count - 1)) + execution_time) / current_count
            stats["average_execution_times"][command] = new_avg
    
    def _get_recent_commands(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get commands from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            cmd for cmd in self.usage_data.get("commands", [])
            if datetime.fromisoformat(cmd["timestamp"]) > cutoff_time
        ]
    
    def _get_pattern_suggestions(self, current_command: str) -> List[Dict[str, str]]:
        """Get suggestions based on command patterns."""
        suggestions = []
        
        # Common command sequences
        patterns = self.command_patterns.get(current_command, [])
        
        for pattern in patterns:
            suggestions.append({
                "type": "pattern",
                "title": pattern["title"],
                "description": pattern["description"],
                "command": pattern.get("command", "")
            })
        
        return suggestions
    
    def _get_workflow_suggestions(self, recent_commands: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Get suggestions based on workflow analysis."""
        suggestions = []
        
        if not recent_commands:
            return suggestions
        
        # Analyze recent command sequence
        command_sequence = [cmd["command"] for cmd in recent_commands[-5:]]
        
        # Suggest next steps in common workflows
        if "init" in command_sequence and "data" not in " ".join(command_sequence):
            suggestions.append({
                "type": "workflow",
                "title": "Prepare your data",
                "description": "After initializing a project, the next step is usually data preparation",
                "command": "llmbuilder data prepare --input data/raw"
            })
        
        elif "data prepare" in " ".join(command_sequence) and "model" not in " ".join(command_sequence):
            suggestions.append({
                "type": "workflow",
                "title": "Select a model",
                "description": "With data prepared, you can now select a base model for training",
                "command": "llmbuilder model select"
            })
        
        elif "model select" in " ".join(command_sequence) and "train" not in " ".join(command_sequence):
            suggestions.append({
                "type": "workflow",
                "title": "Start training",
                "description": "Your data and model are ready. Time to start training!",
                "command": "llmbuilder train start"
            })
        
        return suggestions
    
    def _get_error_recovery_suggestions(self) -> List[Dict[str, str]]:
        """Get suggestions for error recovery."""
        suggestions = []
        
        # Check for recent failed commands
        recent_failures = [
            cmd for cmd in self._get_recent_commands(hours=1)
            if not cmd["success"]
        ]
        
        if recent_failures:
            last_failure = recent_failures[-1]
            command = last_failure["command"]
            
            # Suggest common fixes
            if "train" in command:
                suggestions.append({
                    "type": "recovery",
                    "title": "Training issues?",
                    "description": "Try reducing batch size or checking your data",
                    "command": "llmbuilder config set training.batch_size 2"
                })
            
            elif "data" in command:
                suggestions.append({
                    "type": "recovery",
                    "title": "Data processing issues?",
                    "description": "Validate your input data format and paths",
                    "command": "llmbuilder data validate --input data/raw"
                })
        
        return suggestions
    
    def _get_optimization_suggestions(self) -> List[Dict[str, str]]:
        """Get optimization suggestions based on usage patterns."""
        suggestions = []
        
        stats = self.usage_data.get("statistics", {})
        exec_times = stats.get("average_execution_times", {})
        
        # Suggest optimizations for slow commands
        for command, avg_time in exec_times.items():
            if avg_time > 60:  # Commands taking more than 1 minute
                if "data prepare" in command:
                    suggestions.append({
                        "type": "optimization",
                        "title": "Speed up data processing",
                        "description": "Use parallel processing to speed up data preparation",
                        "command": "llmbuilder data prepare --workers 8"
                    })
                
                elif "train" in command:
                    suggestions.append({
                        "type": "optimization",
                        "title": "Optimize training speed",
                        "description": "Consider using mixed precision or smaller batch sizes",
                        "command": "llmbuilder config set training.use_mixed_precision true"
                    })
        
        return suggestions
    
    def _check_for_suggestions(self, command: str, args: List[str], success: bool) -> None:
        """Check if we should show suggestions after this command."""
        # Show suggestions after certain commands or failures
        show_suggestions = (
            not success or  # After failures
            command in ["init", "data prepare", "model select"] or  # After key workflow steps
            "--help" in args  # After help requests
        )
        
        if show_suggestions:
            suggestions = self.get_command_suggestions(command)
            if suggestions:
                # Store suggestions to show later (to avoid interrupting command output)
                self._store_pending_suggestions(suggestions)
    
    def _store_pending_suggestions(self, suggestions: List[Dict[str, str]]) -> None:
        """Store suggestions to show later."""
        try:
            with open(self.suggestions_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "suggestions": suggestions
                }, f, indent=2)
        except IOError:
            pass
    
    def show_pending_suggestions(self) -> None:
        """Show any pending suggestions."""
        if not self.suggestions_file.exists():
            return
        
        try:
            with open(self.suggestions_file, 'r') as f:
                data = json.load(f)
            
            # Check if suggestions are recent (within last 5 minutes)
            suggestion_time = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - suggestion_time > timedelta(minutes=5):
                return
            
            suggestions = data.get("suggestions", [])
            if suggestions:
                self.show_suggestions()
            
            # Remove the suggestions file
            self.suggestions_file.unlink()
            
        except (json.JSONDecodeError, IOError, KeyError):
            pass
    
    def _load_command_patterns(self) -> Dict[str, List[Dict[str, str]]]:
        """Load command patterns and suggestions."""
        return {
            "init": [
                {
                    "title": "Next: Prepare your data",
                    "description": "Add your training data to the data/raw directory and process it",
                    "command": "llmbuilder data prepare --input data/raw"
                }
            ],
            "data prepare": [
                {
                    "title": "Next: Split your data",
                    "description": "Split processed data into training and validation sets",
                    "command": "llmbuilder data split --train-ratio 0.8"
                },
                {
                    "title": "Validate data quality",
                    "description": "Check your processed data for issues",
                    "command": "llmbuilder data validate"
                }
            ],
            "model select": [
                {
                    "title": "Next: Start training",
                    "description": "Begin training with your selected model and prepared data",
                    "command": "llmbuilder train start"
                }
            ],
            "train start": [
                {
                    "title": "Monitor training progress",
                    "description": "Watch training metrics and system resources",
                    "command": "llmbuilder monitor"
                }
            ],
            "train": [
                {
                    "title": "Evaluate your model",
                    "description": "Test your trained model's performance",
                    "command": "llmbuilder eval run"
                }
            ]
        }
    
    def _get_suggestion_icon(self, suggestion_type: str) -> str:
        """Get icon for suggestion type."""
        icons = {
            "pattern": "🔄",
            "workflow": "📋",
            "recovery": "🔧",
            "optimization": "⚡",
            "general": "💡"
        }
        return icons.get(suggestion_type, "💡")


# Global analytics instance
_analytics = None


def get_analytics() -> UsageAnalytics:
    """Get the global analytics instance."""
    global _analytics
    if _analytics is None:
        _analytics = UsageAnalytics()
    return _analytics


def record_command_usage(command: str, args: List[str], success: bool = True,
                        execution_time: float = 0.0, context: Optional[Dict[str, Any]] = None):
    """Record command usage for analytics."""
    analytics = get_analytics()
    analytics.record_command(command, args, success, execution_time, context)


def show_command_suggestions(command: Optional[str] = None):
    """Show command suggestions."""
    analytics = get_analytics()
    analytics.show_suggestions(command)