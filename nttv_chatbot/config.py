# nttv_chatbot/config.py
import os

class Settings:
    ENV: str
    OPENROUTER_API_KEY: str | None
    OPENROUTER_MODEL: str
    OPENROUTER_BASE_URL: str

    def __init__(self):
        # "local" by default; Render will set ENV=production
        self.ENV = os.getenv("NTTV_ENV", "local")

        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        # Pick a sane default; can change later in Render env
        self.OPENROUTER_MODEL = os.getenv(
            "OPENROUTER_MODEL",
            "meta-llama/llama-3.1-8b-instruct"  # example; adjust to taste/cost
        )
        self.OPENROUTER_BASE_URL = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1/chat/completions"
        )

settings = Settings()
