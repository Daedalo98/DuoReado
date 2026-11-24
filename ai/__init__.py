"""ai/__init__.py"""
from .ollama_client import OllamaClient, OllamaError

__all__ = ["OllamaClient", "OllamaError"]