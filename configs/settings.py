from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    radgenome_data_dir: str = "./datasets/radgenome_sample"
    artifacts_dir: str = "./artifacts"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
