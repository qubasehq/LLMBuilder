"""
Package upgrade and update system.
"""

import click
import subprocess
import sys
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from llmbuilder import __version__
from ..utils.config import ConfigManager

console = Console()


class UpgradeManager:
    """Manages package upgrades and template updates."""
    
    def __init__(self):
        self.current_version = __version__
        self.pypi_url = "https://pypi.org/pypi/llmbuilder/json"
        self.templates_repo = "https://api.github.com/repos/llmbuilder/templates"
        self.config_manager = ConfigManager()
    
    def check_for_updates(self) -> Dict[str, Any]:
        """Check for available updates."""
        updates = {
            "package": None,
            "templates": None,
            "current_version": self.current_version
        }
        
        # Check package updates
        try:
            response = requests.get(self.pypi_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                latest_version = data["info"]["version"]
                if self._is_newer_version(latest_version, self.current_version):
                    updates["package"] = {
                        "current": self.current_version,
                        "latest": latest_version,
                        "description": data["info"]["summary"],
                        "release_date": data["releases"][latest_version][0]["upload_time"]
                    }
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check for package updates: {e}[/yellow]")
        
        # Check template updates
        try:
            templates_info = self._check_template_updates()
            if templates_info:
                updates["templates"] = templates_info
        except Exception as e:
            console.print(f"[yellow]Warning: Could not check for template updates: {e}[/yellow]")
        
        return updates
    
    def upgrade_package(self, version: Optional[str] = None, pre_release: bool = False) -> bool:
        """Upgrade the LLMBuilder package."""
        console.print("[yellow]Upgrading LLMBuilder package...[/yellow]")
        
        # Prepare upgrade command
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"]
        
        if version:
            cmd.append(f"llmbuilder=={version}")
        else:
            if pre_release:
                cmd.extend(["--pre", "llmbuilder"])
            else:
                cmd.append("llmbuilder")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Installing update...", total=None)
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                progress.update(task, description="Update completed!")
            
            console.print("[green]✓ Package upgraded successfully![/green]")
            console.print(f"[dim]Restart your terminal to use the new version.[/dim]")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗ Upgrade failed: {e}[/red]")
            if e.stderr:
                console.print(f"[red]Error details: {e.stderr}[/red]")
            return False
    
    def update_templates(self, force: bool = False) -> bool:
        """Update project templates."""
        console.print("[yellow]Updating project templates...[/yellow]")
        
        templates_dir = Path(__file__).parent.parent / "templates"
        
        try:
            # Get latest templates from repository
            templates_data = self._fetch_latest_templates()
            
            if not templates_data and not force:
                console.print("[green]Templates are already up to date.[/green]")
                return True
            
            # Update templates
            updated_count = 0
            for template_name, template_content in templates_data.items():
                template_path = templates_dir / f"{template_name}.json"
                
                if template_path.exists() and not force:
                    # Check if update is needed
                    with open(template_path, 'r') as f:
                        current_content = json.load(f)
                    
                    if current_content.get("version") == template_content.get("version"):
                        continue
                
                # Update template
                template_path.parent.mkdir(parents=True, exist_ok=True)
                with open(template_path, 'w') as f:
                    json.dump(template_content, f, indent=2)
                
                updated_count += 1
                console.print(f"[green]✓ Updated template: {template_name}[/green]")
            
            if updated_count > 0:
                console.print(f"[green]✓ Updated {updated_count} templates.[/green]")
            else:
                console.print("[green]All templates are up to date.[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Template update failed: {e}[/red]")
            return False
    
    def show_update_info(self, updates: Dict[str, Any]) -> None:
        """Display available updates information."""
        if not updates["package"] and not updates["templates"]:
            console.print("[green]✓ Everything is up to date![/green]")
            return
        
        console.print("\n[bold cyan]Available Updates[/bold cyan]")
        
        if updates["package"]:
            pkg_info = updates["package"]
            
            panel_content = f"""
[bold]Current Version:[/bold] {pkg_info['current']}
[bold]Latest Version:[/bold] {pkg_info['latest']}
[bold]Release Date:[/bold] {pkg_info['release_date'][:10]}

[bold]Description:[/bold]
{pkg_info['description']}

[dim]Run 'llmbuilder upgrade package' to update[/dim]
"""
            
            console.print(Panel(
                panel_content.strip(),
                title="Package Update Available",
                border_style="green"
            ))
        
        if updates["templates"]:
            template_info = updates["templates"]
            
            table = Table(title="Template Updates Available")
            table.add_column("Template", style="cyan")
            table.add_column("Current", style="yellow")
            table.add_column("Latest", style="green")
            table.add_column("Description")
            
            for template in template_info:
                table.add_row(
                    template["name"],
                    template["current_version"],
                    template["latest_version"],
                    template["description"]
                )
            
            console.print(table)
            console.print(f"[dim]Run 'llmbuilder upgrade templates' to update[/dim]")
    
    def show_changelog(self, version: Optional[str] = None) -> None:
        """Show changelog for a specific version or latest."""
        console.print("[yellow]Fetching changelog...[/yellow]")
        
        try:
            # In a real implementation, this would fetch from GitHub releases
            # For now, we'll show a placeholder
            changelog = self._get_changelog(version)
            
            if changelog:
                console.print(Panel(
                    changelog,
                    title=f"Changelog - Version {version or 'Latest'}",
                    border_style="blue"
                ))
            else:
                console.print("[yellow]No changelog available.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error fetching changelog: {e}[/red]")
    
    def _is_newer_version(self, latest: str, current: str) -> bool:
        """Check if latest version is newer than current."""
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        try:
            return version_tuple(latest) > version_tuple(current)
        except ValueError:
            return False
    
    def _check_template_updates(self) -> Optional[List[Dict[str, str]]]:
        """Check for template updates."""
        # Placeholder implementation
        # In a real system, this would check against a templates repository
        return None
    
    def _fetch_latest_templates(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest templates from repository."""
        # Placeholder implementation
        # In a real system, this would fetch from GitHub or another repository
        return {}
    
    def _get_changelog(self, version: Optional[str] = None) -> Optional[str]:
        """Get changelog for a version."""
        # Placeholder changelog
        changelog = f"""
# Version {version or self.current_version}

## New Features
- Enhanced help system with interactive documentation
- Improved tool integration and marketplace
- Better error handling and user guidance

## Improvements
- Faster data processing pipeline
- More robust model training
- Enhanced CLI user experience

## Bug Fixes
- Fixed memory issues in large dataset processing
- Resolved configuration validation errors
- Improved cross-platform compatibility

## Breaking Changes
- None in this release

For complete details, visit: https://github.com/llmbuilder/llmbuilder/releases
"""
        return changelog.strip()


@click.group()
def upgrade():
    """Package upgrade and update system."""
    pass


@upgrade.command()
@click.option('--check-only', is_flag=True, help='Only check for updates, do not install')
@click.option('--pre-release', is_flag=True, help='Include pre-release versions')
@click.option('--version', help='Upgrade to specific version')
def package(check_only, pre_release, version):
    """Upgrade the LLMBuilder package."""
    manager = UpgradeManager()
    
    if check_only:
        console.print("[yellow]Checking for package updates...[/yellow]")
        updates = manager.check_for_updates()
        manager.show_update_info(updates)
    else:
        if not version:
            # Check for updates first
            updates = manager.check_for_updates()
            if not updates["package"]:
                console.print("[green]✓ Package is already up to date![/green]")
                return
            
            version = updates["package"]["latest"]
            console.print(f"[cyan]Upgrading to version {version}[/cyan]")
        
        success = manager.upgrade_package(version, pre_release)
        if success:
            console.print(f"\n[green]Successfully upgraded to version {version}![/green]")
        else:
            console.print(f"\n[red]Upgrade failed. Please try again or install manually.[/red]")


@upgrade.command()
@click.option('--force', is_flag=True, help='Force update even if templates are current')
def templates(force):
    """Update project templates."""
    manager = UpgradeManager()
    success = manager.update_templates(force)
    
    if success:
        console.print("\n[green]Templates updated successfully![/green]")
    else:
        console.print("\n[red]Template update failed.[/red]")


@upgrade.command()
def check():
    """Check for available updates."""
    manager = UpgradeManager()
    
    console.print("[yellow]Checking for updates...[/yellow]")
    updates = manager.check_for_updates()
    manager.show_update_info(updates)


@upgrade.command()
@click.option('--version', help='Show changelog for specific version')
def changelog(version):
    """Show changelog for latest or specific version."""
    manager = UpgradeManager()
    manager.show_changelog(version)


@upgrade.command()
def all():
    """Upgrade package and update templates."""
    manager = UpgradeManager()
    
    console.print("[bold cyan]Checking for all updates...[/bold cyan]")
    updates = manager.check_for_updates()
    
    if not updates["package"] and not updates["templates"]:
        console.print("[green]✓ Everything is already up to date![/green]")
        return
    
    # Show what will be updated
    manager.show_update_info(updates)
    
    if not click.confirm("\nProceed with updates?"):
        console.print("Update cancelled.")
        return
    
    success = True
    
    # Update package if needed
    if updates["package"]:
        console.print("\n[bold]Updating package...[/bold]")
        success &= manager.upgrade_package()
    
    # Update templates if needed
    if updates["templates"]:
        console.print("\n[bold]Updating templates...[/bold]")
        success &= manager.update_templates()
    
    if success:
        console.print("\n[green]✓ All updates completed successfully![/green]")
    else:
        console.print("\n[yellow]Some updates may have failed. Check the output above.[/yellow]")


# Add the upgrade group to the main CLI
def register_commands(cli_group):
    """Register upgrade commands with the main CLI."""
    cli_group.add_command(upgrade)