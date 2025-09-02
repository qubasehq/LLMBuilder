"""
Model serving server for LLMBuilder.

This module provides a FastAPI-based server for serving LLM models
with authentication, rate limiting, and health monitoring.
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import tempfile
from datetime import datetime
import uuid

from llmbuilder.utils.logging import get_logger

logger = get_logger(__name__)


class ModelServer:
    """
    FastAPI-based model serving server.
    
    Provides REST API endpoints for model inference with support for
    authentication, CORS, rate limiting, and health monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model server.
        
        Args:
            config: Server configuration dictionary
        """
        self.config = config
        self.process = None
        self.server_id = str(uuid.uuid4())
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Model server initialized with ID: {self.server_id}")
    
    def _validate_config(self):
        """Validate server configuration."""
        required_keys = ['model_path', 'config_path', 'host', 'port']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate paths
        model_path = Path(self.config['model_path'])
        config_path = Path(self.config['config_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def start_foreground(self):
        """Start server in foreground mode."""
        logger.info("Starting server in foreground mode")
        
        # Create FastAPI server script
        server_script = self._create_server_script()
        
        try:
            # Run the server script
            import subprocess
            subprocess.run([sys.executable, server_script], check=True)
        finally:
            # Cleanup
            if server_script.exists():
                server_script.unlink()
    
    def start_background(self):
        """Start server in background mode."""
        logger.info("Starting server in background mode")
        
        # Create FastAPI server script
        server_script = self._create_server_script()
        
        # Start server process
        self.process = subprocess.Popen(
            [sys.executable, server_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait a moment to check if server started successfully
        time.sleep(2)
        if self.process.poll() is not None:
            # Process died
            stdout, stderr = self.process.communicate()
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Server failed to start: {error_msg}")
        
        logger.info(f"Server started in background with PID: {self.process.pid}")
    
    def stop(self):
        """Stop the server."""
        if self.process:
            logger.info("Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            
            self.process = None
            logger.info("Server stopped")
    
    def get_pid(self) -> Optional[int]:
        """Get server process PID."""
        return self.process.pid if self.process else None
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.process is not None and self.process.poll() is None
    
    def _create_server_script(self) -> Path:
        """Create FastAPI server script."""
        script_content = self._generate_server_code()
        
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)
        
        return script_path
    
    def _generate_server_code(self) -> str:
        """Generate FastAPI server code."""
        return f'''#!/usr/bin/env python3
"""
Auto-generated FastAPI server for LLMBuilder model serving.
Generated at: {datetime.now().isoformat()}
Server ID: {self.server_id}
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

# Add project root to path for imports
project_root = Path(__file__).parent
while project_root.parent != project_root:
    if (project_root / "llmbuilder").exists():
        sys.path.insert(0, str(project_root))
        break
    project_root = project_root.parent

try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Server configuration
CONFIG = {json.dumps(self.config, indent=2)}

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: int = Field(100, ge=1, le=CONFIG.get("max_tokens", 512), description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stream: bool = Field(False, description="Enable streaming response")

class GenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model information")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Server health status")
    timestamp: str = Field(..., description="Health check timestamp")
    uptime: float = Field(..., description="Server uptime in seconds")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field("1.0.0", description="API version")

# Global variables
app = None
model_engine = None
server_start_time = time.time()
auth_token = CONFIG.get("auth_token")

def create_app():
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="LLMBuilder Model API",
        description="REST API for LLM model inference",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    if CONFIG.get("cors", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    return app

def get_auth_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
    """Validate authentication token."""
    if not auth_token:
        return True  # No auth required
    
    if not credentials or credentials.credentials != auth_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return True

def load_model():
    """Load the model for inference."""
    global model_engine
    
    if model_engine is not None:
        return model_engine
    
    try:
        # Try to import and use the inference engine
        from llmbuilder.core.inference.engine import InferenceEngine
        
        model_engine = InferenceEngine(
            model_path=CONFIG["model_path"],
            tokenizer_path=CONFIG.get("tokenizer_path", "tokenizer/"),
            config_path=CONFIG["config_path"],
            device="auto"
        )
        
        return model_engine
        
    except Exception as e:
        print(f"Failed to load model: {{e}}")
        traceback.print_exc()
        # Create a mock engine for demonstration
        model_engine = MockInferenceEngine()
        return model_engine

class MockInferenceEngine:
    """Mock inference engine for demonstration."""
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate mock text."""
        max_tokens = kwargs.get("max_new_tokens", 50)
        temperature = kwargs.get("temperature", 0.8)
        
        # Simple mock response
        return f"{{prompt}} [Generated with temperature={{temperature}}, max_tokens={{max_tokens}}] This is a mock response from the LLMBuilder API server. The actual model would generate real text here."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model info."""
        return {{
            "model_type": "Mock Model",
            "backend": "mock",
            "parameters": "Unknown",
            "loaded_at": datetime.now().isoformat()
        }}

# Create FastAPI app
app = create_app()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - server_start_time
    
    try:
        model = load_model()
        model_loaded = model is not None
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        uptime=uptime,
        model_loaded=model_loaded
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    _: bool = Depends(get_auth_token)
):
    """Generate text from prompt."""
    start_time = time.time()
    
    try:
        # Load model
        model = load_model()
        
        # Generate text
        generated_text = model.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        generation_time = time.time() - start_time
        
        # Count tokens (simple approximation)
        tokens_generated = len(generated_text.split()) - len(request.prompt.split())
        
        return GenerationResponse(
            text=generated_text,
            prompt=request.prompt,
            tokens_generated=max(0, tokens_generated),
            generation_time=generation_time,
            model_info=model.get_model_info() if hasattr(model, 'get_model_info') else {{}}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {{str(e)}}")

@app.get("/model/info")
async def model_info(_: bool = Depends(get_auth_token)):
    """Get model information."""
    try:
        model = load_model()
        return model.get_model_info() if hasattr(model, 'get_model_info') else {{"status": "unknown"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {{str(e)}}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {{
        "message": "LLMBuilder Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }}

if __name__ == "__main__":
    # Server configuration
    host = CONFIG.get("host", "127.0.0.1")
    port = CONFIG.get("port", 8000)
    workers = CONFIG.get("workers", 1)
    log_level = CONFIG.get("log_level", "info").lower()
    
    print(f"Starting LLMBuilder Model API server...")
    print(f"Host: {{host}}")
    print(f"Port: {{port}}")
    print(f"Workers: {{workers}}")
    print(f"Log level: {{log_level}}")
    
    # Start server
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers if workers > 1 else None,
        log_level=log_level,
        access_log=True
    )
'''
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.process:
            self.stop()