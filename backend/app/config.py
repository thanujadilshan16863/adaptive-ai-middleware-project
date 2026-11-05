from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    database_url: str = 'sqlite:///./middleware.db'
    llm_api_key: str = ''
    llm_base_url: str = 'https://api.openai.com/v1'
    llm_model: str = 'gpt-4o-mini'
    llm_json_schema: bool = True
    llm_required: bool = False
    chronos_model: str = 'autogluon/chronos-bolt-small'
    chronos_device: str = 'cpu'
    chronos_cache_dir: str = '.cache/huggingface'
    chronos_batch_size: int = Field(default=32, ge=1, le=256)
    forecast_mode: Literal['offline', 'chronos'] = 'offline'
    forecast_strategy: Literal['fixed', 'adaptive'] = 'adaptive'
    deployment_mode: Literal['local', 'production'] = 'local'
    job_queue_size: int = Field(default=3, ge=1, le=20)
    job_directory: str = '.cache/jobs'
    api_key: str = ''
    cors_origins: list[str] = ['http://localhost:5173', 'http://127.0.0.1:5173']
    max_upload_mb: int = 25
    max_rows: int = 500000
    max_skus: int = 2000


settings = Settings()
