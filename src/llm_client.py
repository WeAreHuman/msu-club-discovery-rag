"""
LLM Client Module
Provides unified interface for LLM providers
Defaults to Groq for free usage on Streamlit Cloud
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import config_streamlit first (for Streamlit deployment), fall back to config
try:
    import config_streamlit as config
except ImportError:
    import config


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate text response from LLM

        Args:
            prompt: User prompt/question
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        pass


class GroqClient(BaseLLMClient):
    """
    Groq LLM client (FREE tier available)
    Supports Llama 3.1 and 3.3 models
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq client

        Args:
            api_key: Groq API key (defaults to config)
            model: Model name (defaults to config)
        """
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Please install groq: pip install groq")

        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.LLM_MODEL

        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env file")

        self.client = Groq(api_key=self.api_key)
        print(f"✓ Initialized Groq client with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate response using Groq API

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        max_tokens = max_tokens or config.LLM_MAX_TOKENS

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️  Error calling Groq API: {e}")
            return f"Error: {str(e)}"


def get_llm_client(provider: str = None, **kwargs) -> BaseLLMClient:
    """
    Factory function to get LLM client (Groq only for deployment)

    Args:
        provider: LLM provider name (defaults to config, recommend "groq")
        **kwargs: Additional arguments for client initialization

    Returns:
        Initialized LLM client (Groq)
    """
    provider = provider or config.LLM_PROVIDER
    
    if provider and provider.lower() not in ["groq", "groq-free"]:
        print(f"⚠️  Warning: Provider '{provider}' not supported on Streamlit. Using Groq.")

    # Use Groq for both deployment and fallback
    return GroqClient(**kwargs)


# ============================================================================
# TESTING / STANDALONE EXECUTION
# ============================================================================
if __name__ == "__main__":
    """Test LLM client"""

    print("\n=== Testing LLM Client ===")

    # Get client based on config
    try:
        llm = get_llm_client()

        # Test generation
        system_prompt = "You are a helpful assistant for MSU students looking for club information."
        user_prompt = "What should I ask when looking for a student club?"

        print(f"\nPrompt: {user_prompt}")
        print("\nGenerating response...\n")

        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=200
        )

        print(f"Response:\n{response}")

    except Exception as e:
        print(f"⚠️  Error testing LLM client: {e}")
        print(f"   Make sure you have set up API keys in .env file")
