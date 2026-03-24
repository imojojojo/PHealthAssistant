from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM provider
    gemini_api_key: str
    llm_chat_model: str = "gemini-2.5-flash"
    llm_embedding_model: str = "gemini-embedding-001"

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "patient-records"

    # Application
    data_dir: str = "data/patients"
