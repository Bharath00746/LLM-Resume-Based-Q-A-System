from pathlib import Path
from pydantic_settings import BaseSettings


# ...existing code...

class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    USE_OPENAI_EMBEDDINGS: bool = False

    INDEX_DIR: str = "storage"
    RESUME_DIR: str = "data"
    RESUME_FILENAME: str = "resume.pdf"

    TOP_K: int = 5
    SIMILARITY_CUTOFF: float = 0.5

    FALLBACK_ANSWER: str = "I cannot find that in my resume."

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# ensure dirs exist
Path(settings.INDEX_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.RESUME_DIR).mkdir(parents=True, exist_ok=True)
# ...existing code...