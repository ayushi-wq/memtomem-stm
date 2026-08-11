"""STM (Short-Term Memory) root configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from memtomem_stm.proxy.config import ProxyConfig
from memtomem_stm.surfacing.config import SurfacingConfig


class LangfuseConfig(BaseModel):
    """Langfuse tracing configuration."""

    enabled: bool = Field(default=False, description="Whether Langfuse tracing is enabled.")
    public_key: str = Field(default="", description="Langfuse public key.")
    secret_key: str = Field(default="", description="Langfuse secret key.")
    host: str = Field(default="", description="Langfuse host URL.")
    sampling_rate: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Fraction of proxy calls to trace (0.0-1.0). Default 1.0 = all."
    )

    @model_validator(mode="after")
    def _require_keys_when_enabled(self) -> "LangfuseConfig":
        if self.enabled and not (self.public_key and self.secret_key):
            raise ValueError(
                "LangfuseConfig.enabled=true requires both public_key and secret_key "
                "to be set (non-empty)."
            )
        return self


class STMConfig(BaseSettings):
    """Root configuration for the Short-Term Memory service."""

    model_config = SettingsConfigDict(
        env_prefix="MEMTOMEM_STM_",
        env_nested_delimiter="__",
    )

    proxy: ProxyConfig = Field(
        default_factory=ProxyConfig, 
        description="Proxy service configuration."
    )
    surfacing: SurfacingConfig = Field(
        default_factory=SurfacingConfig, 
        description="Surfacing logic configuration."
    )
    langfuse: LangfuseConfig = Field(
        default_factory=LangfuseConfig, 
        description="Langfuse tracing configuration."
    )
    data_dir: Path = Field(
        default=Path("~/.memtomem"), 
        description="Base directory for memory data storage."
    )

    def model_post_init(self, __context: object) -> None:
        # Propagate consumer_model from proxy to surfacing for model-aware defaults
        if self.proxy.consumer_model and not self.surfacing.consumer_model:
            self.surfacing.consumer_model = self.proxy.consumer_model
