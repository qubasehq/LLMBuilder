"""
Log aggregation and search for LLMBuilder monitoring.

This module provides log file aggregation, filtering, searching,
and real-time following capabilities across multiple log sources.
"""

import re
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, TextIO
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
from collections import deque

from rich.console import Console
from rich.text import Text

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    source: str
    message: str
    raw_line: str
    file_path: str


class LogAggregator:
    """
    Log file aggregation and search system.
    
    Provides powerful log aggregation, filtering, and search capabilities
    across multiple log files with real-time following support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize log aggregator.
        
        Args:
            config: Configuration dictionary with log settings
        """
        self.config = config
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.pattern = config.get('pattern', '*.log')
        self.level_filter = config.get('level')
        self.since = config.get('since')
        self.tail = config.get('tail', 100)
        self.search_term = config.get('search')
        
        self.console = Console()
        self.following = False
        
        # Log level hierarchy for filtering
        self.log_levels = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
        
        logger.info(f"Log aggregator initialized for {self.log_dir}")
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get filtered log entries.
        
        Returns:
            List of log entry dictionaries
        """
        try:
            log_files = self._find_log_files()
            if not log_files:
                logger.warning(f"No log files found matching pattern {self.pattern} in {self.log_dir}")
                return []
            
            all_entries = []
            
            for log_file in log_files:
                entries = self._parse_log_file(log_file)
                all_entries.extend(entries)
            
            # Sort by timestamp
            all_entries.sort(key=lambda x: x.get('timestamp', ''))
            
            # Apply filters
            filtered_entries = self._apply_filters(all_entries)
            
            # Apply tail limit
            if self.tail and len(filtered_entries) > self.tail:
                filtered_entries = filtered_entries[-self.tail:]
            
            return filtered_entries
            
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return []
    
    def follow_logs(self):
        """Follow log files for real-time updates."""
        try:
            self.following = True
            log_files = self._find_log_files()
            
            if not log_files:
                self.console.print("[yellow]No log files found to follow[/yellow]")
                return
            
            # Track file positions
            file_positions = {}
            for log_file in log_files:
                if log_file.exists():
                    file_positions[log_file] = log_file.stat().st_size
            
            self.console.print(f"[green]Following {len(log_files)} log files...[/green]")
            
            while self.following:
                try:
                    # Check each file for new content
                    for log_file in log_files:
                        if not log_file.exists():
                            continue
                        
                        current_size = log_file.stat().st_size
                        last_position = file_positions.get(log_file, 0)
                        
                        if current_size > last_position:
                            # Read new content
                            new_entries = self._read_new_content(log_file, last_position)
                            
                            # Display new entries
                            for entry in new_entries:
                                if self._entry_matches_filters(entry):
                                    self._display_log_entry(entry)
                            
                            file_positions[log_file] = current_size
                    
                    time.sleep(1)  # Check every second
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error following logs: {e}")
                    time.sleep(5)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Error in follow_logs: {e}")
        finally:
            self.following = False
    
    def export_logs(self, logs_data: List[Dict[str, Any]], export_path: Path):
        """
        Export log entries to file.
        
        Args:
            logs_data: Log entries to export
            export_path: Path to save the exported logs
        """
        try:
            if export_path.suffix.lower() == '.json':
                # Export as JSON
                with open(export_path, 'w') as f:
                    json.dump(logs_data, f, indent=2)
            else:
                # Export as plain text
                with open(export_path, 'w') as f:
                    for entry in logs_data:
                        f.write(entry.get('raw_line', '') + '\n')
            
            logger.info(f"Exported {len(logs_data)} log entries to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
            raise
    
    def _find_log_files(self) -> List[Path]:
        """Find log files matching the pattern."""
        try:
            if not self.log_dir.exists():
                return []
            
            log_files = list(self.log_dir.rglob(self.pattern))
            
            # Sort by modification time (newest first)
            log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            return log_files
            
        except Exception as e:
            logger.error(f"Error finding log files: {e}")
            return []
    
    def _parse_log_file(self, log_file: Path) -> List[Dict[str, Any]]:
        """Parse a single log file."""
        entries = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.rstrip('\n\r')
                    if not line.strip():
                        continue
                    
                    entry = self._parse_log_line(line, log_file, line_num)
                    if entry:
                        entries.append(entry)
            
        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {e}")
        
        return entries
    
    def _parse_log_line(self, line: str, file_path: Path, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single log line."""
        try:
            # Common log patterns
            patterns = [
                # Standard format: 2024-01-01 12:00:00,123 - INFO - module - message
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-\s*(\w+)\s*-\s*([^-]+)\s*-\s*(.+)',
                # Simple format: 2024-01-01 12:00:00 INFO message
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)',
                # Bracket format: [2024-01-01 12:00:00] [INFO] message
                r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*\[(\w+)\]\s*(.+)',
                # Training format: INFO:root:message
                r'(\w+):([^:]+):(.+)'
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    
                    if len(groups) == 4:
                        timestamp, level, source, message = groups
                    elif len(groups) == 3:
                        if groups[0].replace('-', '').replace(':', '').replace(' ', '').isdigit():
                            # Timestamp format
                            timestamp, level, message = groups
                            source = file_path.stem
                        else:
                            # Level:source:message format
                            level, source, message = groups
                            timestamp = datetime.now().isoformat()[:19]
                    else:
                        continue
                    
                    return {
                        'timestamp': timestamp,
                        'level': level.upper(),
                        'source': source.strip(),
                        'message': message.strip(),
                        'raw_line': line,
                        'file_path': str(file_path),
                        'line_number': line_num
                    }
            
            # If no pattern matches, treat as unstructured log
            return {
                'timestamp': datetime.now().isoformat()[:19],
                'level': 'INFO',
                'source': file_path.stem,
                'message': line,
                'raw_line': line,
                'file_path': str(file_path),
                'line_number': line_num
            }
            
        except Exception as e:
            logger.debug(f"Error parsing log line: {e}")
            return None
    
    def _apply_filters(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all configured filters to log entries."""
        filtered = entries
        
        # Level filter
        if self.level_filter:
            min_level = self.log_levels.get(self.level_filter.upper(), 0)
            filtered = [
                entry for entry in filtered
                if self.log_levels.get(entry.get('level', 'INFO'), 1) >= min_level
            ]
        
        # Time filter
        if self.since:
            since_time = self._parse_time_filter(self.since)
            if since_time:
                filtered = [
                    entry for entry in filtered
                    if self._entry_after_time(entry, since_time)
                ]
        
        # Search filter
        if self.search_term:
            search_lower = self.search_term.lower()
            filtered = [
                entry for entry in filtered
                if search_lower in entry.get('message', '').lower() or
                   search_lower in entry.get('source', '').lower()
            ]
        
        return filtered
    
    def _entry_matches_filters(self, entry: Dict[str, Any]) -> bool:
        """Check if a single entry matches all filters."""
        # Level filter
        if self.level_filter:
            min_level = self.log_levels.get(self.level_filter.upper(), 0)
            entry_level = self.log_levels.get(entry.get('level', 'INFO'), 1)
            if entry_level < min_level:
                return False
        
        # Search filter
        if self.search_term:
            search_lower = self.search_term.lower()
            message_match = search_lower in entry.get('message', '').lower()
            source_match = search_lower in entry.get('source', '').lower()
            if not (message_match or source_match):
                return False
        
        return True
    
    def _parse_time_filter(self, since_str: str) -> Optional[datetime]:
        """Parse time filter string."""
        try:
            # Handle relative times like "1h", "30m", "2d"
            if since_str.endswith(('h', 'm', 'd')):
                unit = since_str[-1]
                value = int(since_str[:-1])
                
                if unit == 'm':
                    delta = timedelta(minutes=value)
                elif unit == 'h':
                    delta = timedelta(hours=value)
                elif unit == 'd':
                    delta = timedelta(days=value)
                else:
                    return None
                
                return datetime.now() - delta
            
            # Handle absolute times
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%H:%M:%S']:
                try:
                    if fmt == '%H:%M:%S':
                        # Today's date with specified time
                        time_part = datetime.strptime(since_str, fmt).time()
                        return datetime.combine(datetime.now().date(), time_part)
                    else:
                        return datetime.strptime(since_str, fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing time filter '{since_str}': {e}")
            return None
    
    def _entry_after_time(self, entry: Dict[str, Any], since_time: datetime) -> bool:
        """Check if log entry is after the specified time."""
        try:
            entry_time_str = entry.get('timestamp', '')
            
            # Try to parse entry timestamp
            for fmt in ['%Y-%m-%d %H:%M:%S,%f', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']:
                try:
                    entry_time = datetime.strptime(entry_time_str[:19], fmt[:19])
                    return entry_time >= since_time
                except ValueError:
                    continue
            
            # If parsing fails, include the entry
            return True
            
        except Exception as e:
            logger.debug(f"Error comparing entry time: {e}")
            return True
    
    def _read_new_content(self, log_file: Path, start_position: int) -> List[Dict[str, Any]]:
        """Read new content from log file starting at position."""
        entries = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(start_position)
                
                for line in f:
                    line = line.rstrip('\n\r')
                    if line.strip():
                        entry = self._parse_log_line(line, log_file, 0)
                        if entry:
                            entries.append(entry)
        
        except Exception as e:
            logger.error(f"Error reading new content from {log_file}: {e}")
        
        return entries
    
    def _display_log_entry(self, entry: Dict[str, Any]):
        """Display a single log entry with formatting."""
        timestamp = entry.get('timestamp', '')[:19]
        level = entry.get('level', 'INFO')
        source = entry.get('source', '')
        message = entry.get('message', '')
        
        # Color code by level
        level_colors = {
            'DEBUG': 'dim',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold red'
        }
        
        level_color = level_colors.get(level, 'white')
        
        # Format and display
        formatted_line = Text()
        formatted_line.append(f"{timestamp} ", style="dim")
        formatted_line.append(f"{level:8} ", style=level_color)
        formatted_line.append(f"{source:15} ", style="cyan")
        formatted_line.append(message)
        
        self.console.print(formatted_line)