"""
Conversation history management for LLMBuilder.

This module provides functionality for tracking, saving, and loading
conversation histories from interactive inference sessions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationHistory:
    """
    Manager for conversation history in interactive inference sessions.
    
    Tracks messages, metadata, and provides functionality for saving
    and loading conversation histories.
    """
    
    def __init__(self, history_dir: Optional[Path] = None):
        """
        Initialize the conversation history manager.
        
        Args:
            history_dir: Directory to store history files (default: .llmbuilder/history)
        """
        if history_dir is None:
            history_dir = Path.cwd() / '.llmbuilder' / 'history'
        
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize conversation state
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            'session_id': str(uuid.uuid4()),
            'created_at': datetime.now().isoformat(),
            'model_info': {},
            'generation_params': {}
        }
        
        logger.debug(f"Conversation history initialized with directory: {self.history_dir}")
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata for the message
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'message_id': str(uuid.uuid4())
        }
        
        if metadata:
            message['metadata'] = metadata
        
        self.messages.append(message)
        logger.debug(f"Added {role} message to conversation history")
    
    def get_messages(self, role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get messages from the conversation history.
        
        Args:
            role_filter: Optional role to filter by
            
        Returns:
            List of messages
        """
        if role_filter:
            return [msg for msg in self.messages if msg['role'] == role_filter]
        return self.messages.copy()
    
    def get_last_messages(self, count: int, role_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the last N messages from the conversation.
        
        Args:
            count: Number of messages to retrieve
            role_filter: Optional role to filter by
            
        Returns:
            List of last N messages
        """
        messages = self.get_messages(role_filter)
        return messages[-count:] if count > 0 else messages
    
    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages.clear()
        self.metadata['cleared_at'] = datetime.now().isoformat()
        logger.info("Cleared conversation history")
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the conversation.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        logger.debug(f"Set conversation metadata: {key}")
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata from the conversation.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save conversation history to a file.
        
        Args:
            file_path: Path to save the conversation
        """
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare conversation data
        conversation_data = {
            'metadata': self.metadata.copy(),
            'messages': self.messages.copy(),
            'saved_at': datetime.now().isoformat(),
            'message_count': len(self.messages),
            'version': '1.0'
        }
        
        # Update metadata
        conversation_data['metadata']['last_saved'] = conversation_data['saved_at']
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved conversation history to {file_path}")
    
    def load_from_file(self, file_path: Path) -> None:
        """
        Load conversation history from a file.
        
        Args:
            file_path: Path to load the conversation from
        """
        if not file_path.exists():
            raise FileNotFoundError(f"History file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Load messages and metadata
            self.messages = conversation_data.get('messages', [])
            self.metadata = conversation_data.get('metadata', {})
            
            # Update metadata
            self.metadata['loaded_at'] = datetime.now().isoformat()
            self.metadata['loaded_from'] = str(file_path)
            
            logger.info(f"Loaded conversation history from {file_path}")
            logger.info(f"Loaded {len(self.messages)} messages")
            
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            raise
    
    def auto_save(self, filename_prefix: str = "conversation") -> Path:
        """
        Auto-save conversation with a generated filename.
        
        Args:
            filename_prefix: Prefix for the filename
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        file_path = self.history_dir / filename
        
        self.save_to_file(file_path)
        return file_path
    
    def list_histories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List available conversation history files.
        
        Args:
            limit: Maximum number of histories to return
            
        Returns:
            List of history file information
        """
        histories = []
        
        for history_file in self.history_dir.glob("*.json"):
            try:
                # Get file stats
                stat = history_file.stat()
                
                # Try to load basic info
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                history_info = {
                    'filename': history_file.name,
                    'path': str(history_file),
                    'size': self._format_file_size(stat.st_size),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'message_count': data.get('message_count', len(data.get('messages', []))),
                    'created_at': data.get('metadata', {}).get('created_at', ''),
                    'session_id': data.get('metadata', {}).get('session_id', '')
                }
                
                # Format date for display
                try:
                    modified_dt = datetime.fromtimestamp(stat.st_mtime)
                    history_info['date'] = modified_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    history_info['date'] = 'Unknown'
                
                histories.append(history_info)
                
            except Exception as e:
                logger.warning(f"Could not read history file {history_file}: {e}")
        
        # Sort by modification time (newest first)
        histories.sort(key=lambda x: x['modified'], reverse=True)
        
        # Apply limit
        if limit:
            histories = histories[:limit]
        
        return histories
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        user_messages = [msg for msg in self.messages if msg['role'] == 'user']
        assistant_messages = [msg for msg in self.messages if msg['role'] == 'assistant']
        
        # Calculate total characters
        total_chars = sum(len(msg['content']) for msg in self.messages)
        
        # Get time span
        if self.messages:
            first_message = min(self.messages, key=lambda x: x['timestamp'])
            last_message = max(self.messages, key=lambda x: x['timestamp'])
            
            first_time = datetime.fromisoformat(first_message['timestamp'])
            last_time = datetime.fromisoformat(last_message['timestamp'])
            duration = (last_time - first_time).total_seconds()
        else:
            duration = 0
        
        return {
            'total_messages': len(self.messages),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'total_characters': total_chars,
            'duration_seconds': duration,
            'session_id': self.metadata.get('session_id'),
            'created_at': self.metadata.get('created_at'),
            'model_info': self.metadata.get('model_info', {})
        }
    
    def export_to_text(self, file_path: Path, include_metadata: bool = True) -> None:
        """
        Export conversation to a human-readable text file.
        
        Args:
            file_path: Path to save the text file
            include_metadata: Whether to include metadata in the export
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write("Conversation History\n")
                f.write("=" * 50 + "\n\n")
                
                # Write metadata
                f.write(f"Session ID: {self.metadata.get('session_id', 'Unknown')}\n")
                f.write(f"Created: {self.metadata.get('created_at', 'Unknown')}\n")
                f.write(f"Messages: {len(self.messages)}\n")
                f.write("\n" + "-" * 50 + "\n\n")
            
            # Write messages
            for i, message in enumerate(self.messages, 1):
                role = message['role'].title()
                content = message['content']
                timestamp = message.get('timestamp', '')
                
                f.write(f"[{i}] {role}")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        f.write(f" ({dt.strftime('%H:%M:%S')})")
                    except:
                        pass
                f.write(":\n")
                f.write(f"{content}\n\n")
        
        logger.info(f"Exported conversation to text file: {file_path}")
    
    def search_messages(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for messages containing a specific query.
        
        Args:
            query: Search query
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching messages
        """
        if not case_sensitive:
            query = query.lower()
        
        matching_messages = []
        
        for message in self.messages:
            content = message['content']
            if not case_sensitive:
                content = content.lower()
            
            if query in content:
                matching_messages.append(message)
        
        logger.debug(f"Found {len(matching_messages)} messages matching '{query}'")
        return matching_messages