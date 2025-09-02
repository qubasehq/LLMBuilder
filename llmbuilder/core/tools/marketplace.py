"""
Tool marketplace and discovery system.
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketplaceTool:
    """Tool information from marketplace."""
    name: str
    description: str
    category: str
    version: str
    author: str
    download_url: str
    documentation_url: Optional[str] = None
    repository_url: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = None
    rating: float = 0.0
    downloads: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ToolMarketplace:
    """Tool marketplace for discovering and installing community tools."""
    
    def __init__(self, marketplace_url: Optional[str] = None, cache_dir: Optional[Path] = None):
        """Initialize the marketplace.
        
        Args:
            marketplace_url: URL of the marketplace API
            cache_dir: Directory for caching marketplace data
        """
        self.marketplace_url = marketplace_url or "https://api.llmbuilder.tools"
        
        if cache_dir is None:
            cache_dir = Path.home() / ".llmbuilder" / "marketplace"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "tools_cache.json"
        self.cache_expiry_hours = 24
        
        self._tools_cache: Optional[List[MarketplaceTool]] = None
    
    def search_tools(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        use_cache: bool = True
    ) -> List[MarketplaceTool]:
        """Search for tools in the marketplace.
        
        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            limit: Maximum number of results
            use_cache: Whether to use cached results
            
        Returns:
            List of marketplace tools
        """
        # Try to get from cache first
        if use_cache:
            cached_tools = self._get_cached_tools()
            if cached_tools is not None:
                return self._filter_tools(cached_tools, query, category, tags, limit)
        
        # Fetch from marketplace API
        try:
            tools = self._fetch_tools_from_api(query, category, tags, limit)
            self._cache_tools(tools)
            return tools
        except Exception as e:
            logger.error(f"Failed to fetch tools from marketplace: {e}")
            
            # Fallback to cache if available
            cached_tools = self._get_cached_tools(ignore_expiry=True)
            if cached_tools is not None:
                logger.info("Using cached tools as fallback")
                return self._filter_tools(cached_tools, query, category, tags, limit)
            
            # Return built-in tools as last resort
            return self._get_builtin_tools(query, category, tags, limit)
    
    def get_tool_details(self, tool_name: str) -> Optional[MarketplaceTool]:
        """Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool details or None if not found
        """
        try:
            response = requests.get(
                f"{self.marketplace_url}/tools/{tool_name}",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_tool_data(data)
            
        except Exception as e:
            logger.error(f"Failed to get tool details for {tool_name}: {e}")
            return None
    
    def download_tool(self, tool: MarketplaceTool, install_dir: Optional[Path] = None) -> Optional[Path]:
        """Download a tool from the marketplace.
        
        Args:
            tool: Tool to download
            install_dir: Directory to install the tool
            
        Returns:
            Path to the downloaded tool file or None if failed
        """
        if install_dir is None:
            install_dir = Path.home() / ".llmbuilder" / "tools" / "downloaded"
        
        install_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(tool.download_url, timeout=30)
            response.raise_for_status()
            
            # Determine file extension
            if tool.download_url.endswith('.py'):
                file_extension = '.py'
            elif tool.download_url.endswith('.zip'):
                file_extension = '.zip'
            else:
                file_extension = '.py'  # Default to Python file
            
            tool_file = install_dir / f"{tool.name}{file_extension}"
            
            with open(tool_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded tool {tool.name} to {tool_file}")
            return tool_file
            
        except Exception as e:
            logger.error(f"Failed to download tool {tool.name}: {e}")
            return None
    
    def _fetch_tools_from_api(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[MarketplaceTool]:
        """Fetch tools from the marketplace API."""
        params = {
            'limit': limit
        }
        
        if query:
            params['q'] = query
        if category:
            params['category'] = category
        if tags:
            params['tags'] = ','.join(tags)
        
        response = requests.get(
            f"{self.marketplace_url}/tools/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        tools = []
        
        for tool_data in data.get('tools', []):
            tool = self._parse_tool_data(tool_data)
            if tool:
                tools.append(tool)
        
        return tools
    
    def _parse_tool_data(self, data: Dict[str, Any]) -> Optional[MarketplaceTool]:
        """Parse tool data from API response."""
        try:
            # Parse datetime fields
            created_at = None
            updated_at = None
            
            if 'created_at' in data:
                created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            if 'updated_at' in data:
                updated_at = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
            
            return MarketplaceTool(
                name=data['name'],
                description=data['description'],
                category=data['category'],
                version=data['version'],
                author=data['author'],
                download_url=data['download_url'],
                documentation_url=data.get('documentation_url'),
                repository_url=data.get('repository_url'),
                license=data.get('license'),
                tags=data.get('tags', []),
                rating=data.get('rating', 0.0),
                downloads=data.get('downloads', 0),
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            logger.error(f"Failed to parse tool data: {e}")
            return None
    
    def _get_cached_tools(self, ignore_expiry: bool = False) -> Optional[List[MarketplaceTool]]:
        """Get tools from cache."""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check cache expiry
            if not ignore_expiry:
                cached_at = datetime.fromisoformat(cache_data['cached_at'])
                hours_since_cache = (datetime.now() - cached_at).total_seconds() / 3600
                
                if hours_since_cache > self.cache_expiry_hours:
                    return None
            
            tools = []
            for tool_data in cache_data['tools']:
                # Parse datetime fields
                if 'created_at' in tool_data and tool_data['created_at']:
                    tool_data['created_at'] = datetime.fromisoformat(tool_data['created_at'])
                if 'updated_at' in tool_data and tool_data['updated_at']:
                    tool_data['updated_at'] = datetime.fromisoformat(tool_data['updated_at'])
                
                tools.append(MarketplaceTool(**tool_data))
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to load cached tools: {e}")
            return None
    
    def _cache_tools(self, tools: List[MarketplaceTool]) -> None:
        """Cache tools to disk."""
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'tools': []
            }
            
            for tool in tools:
                tool_dict = asdict(tool)
                # Convert datetime objects to strings
                if tool_dict['created_at']:
                    tool_dict['created_at'] = tool.created_at.isoformat()
                if tool_dict['updated_at']:
                    tool_dict['updated_at'] = tool.updated_at.isoformat()
                
                cache_data['tools'].append(tool_dict)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to cache tools: {e}")
    
    def _filter_tools(
        self,
        tools: List[MarketplaceTool],
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[MarketplaceTool]:
        """Filter tools based on criteria."""
        filtered_tools = tools
        
        # Filter by category
        if category:
            filtered_tools = [t for t in filtered_tools if t.category == category]
        
        # Filter by tags
        if tags:
            filtered_tools = [
                t for t in filtered_tools
                if any(tag in t.tags for tag in tags)
            ]
        
        # Filter by query (search in name and description)
        if query:
            query_lower = query.lower()
            filtered_tools = [
                t for t in filtered_tools
                if query_lower in t.name.lower() or query_lower in t.description.lower()
            ]
        
        # Sort by rating and downloads
        filtered_tools.sort(key=lambda t: (t.rating, t.downloads), reverse=True)
        
        return filtered_tools[:limit]
    
    def _get_builtin_tools(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[MarketplaceTool]:
        """Get built-in tools as fallback."""
        builtin_tools = [
            MarketplaceTool(
                name="example_alarm",
                description="Example alarm tool for scheduling notifications",
                category="alarm",
                version="1.0.0",
                author="LLMBuilder Team",
                download_url="builtin://example_alarm.py",
                tags=["alarm", "notification", "example"],
                rating=4.5,
                downloads=100
            ),
            MarketplaceTool(
                name="example_messenger",
                description="Example messaging tool for sending messages",
                category="messaging",
                version="1.0.0",
                author="LLMBuilder Team",
                download_url="builtin://example_messenger.py",
                tags=["messaging", "communication", "example"],
                rating=4.2,
                downloads=85
            ),
            MarketplaceTool(
                name="data_processor",
                description="Example data processing tool for CSV manipulation",
                category="data_processing",
                version="1.0.0",
                author="LLMBuilder Team",
                download_url="builtin://data_processor.py",
                tags=["data", "csv", "processing", "example"],
                rating=4.7,
                downloads=150
            )
        ]
        
        return self._filter_tools(builtin_tools, query, category, tags, limit)


# Global marketplace instance
_marketplace = None


def get_marketplace() -> ToolMarketplace:
    """Get the global marketplace instance."""
    global _marketplace
    if _marketplace is None:
        _marketplace = ToolMarketplace()
    return _marketplace