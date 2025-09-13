"""
Configuration for Columbus - API keys and models only
"""

from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class Config(BaseModel):
    # Original API keys (keep these)
    cua_api_key: Optional[str] = Field(default=None, env="CUA_API_KEY")
    hud_api_key: Optional[str] = Field(default=None, env="HUD_API_KEY")

    # New API keys for our integrations
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    mem0_api_key: Optional[str] = Field(default=None, env="MEM0_API_KEY")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")

    # Service URLs
    ollama_base_url: str = Field(
        default="http://localhost:11434", env="OLLAMA_BASE_URL"
    )
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")

    # Optional services (disable warnings when not available)
    enable_memory: bool = Field(
        default=False, env="ENABLE_MEMORY"
    )  # Set to True when Qdrant is running

    # Model configuration (with sensible defaults)
    router_model: str = Field(default="ollama_chat/llama3.2:3b", env="ROUTER_MODEL")
    computer_use_model: str = Field(
        default="ollama_chat/llama3.2:8b", env="COMPUTER_USE_MODEL"
    )
    planning_model: str = Field(default="ollama_chat/llama3.2:8b", env="PLANNING_MODEL")
    reasoning_model: str = Field(
        default="ollama_chat/llama3.2:8b", env="REASONING_MODEL"
    )

    # CUA Computer configuration
    computer_os_type: str = Field(
        default="linux", env="COMPUTER_OS_TYPE"
    )  # linux, windows, macOS
    computer_provider: str = Field(
        default="local", env="COMPUTER_PROVIDER"
    )  # local, cloud

    # VM Configuration for localhost access
    vm_expose_ports: bool = Field(
        default=True, env="VM_EXPOSE_PORTS"
    )  # Expose VM ports to localhost
    vm_vnc_port: int = Field(
        default=5900, env="VM_VNC_PORT"
    )  # VNC port for direct VM access

    # Legacy model fields (keep for backward compatibility)
    generation_model_fast: Optional[str] = Field(default=None, env="GEN_MODEL_FAST")
    generation_model_reasoning: Optional[str] = Field(
        default=None, env="GEN_MODEL_REASONING"
    )
    embedding_model: Optional[str] = Field(default=None, env="EMBEDDING_MODEL")
    moderation_model: Optional[str] = Field(default=None, env="MODERATION_MODEL")

    class Config:
        env_file = ".env"
        extra = "allow"  # Allow dynamic fields without breaking


def load_config() -> Config:
    """Load configuration from environment variables and .env file."""
    load_dotenv()
    return Config()
