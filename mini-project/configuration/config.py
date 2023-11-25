from pydantic import BaseSettings, Field


class ServerConfig(BaseSettings):
    """
    Class for server configurations
    """
    HOST: str = Field(default="0.0.0.0", description="Host of the application", env="SERVER_HOST")
    PORT: int = Field(default=8000, description="Port of the application", env="SERVER_PORT")


class PytesseractConfig(BaseSettings):
    text_lang: str = Field(default='ara', description="Tesseract language code string")


server_config = ServerConfig()
pytesseract_config = PytesseractConfig()
