import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_azure_client():
    try:
        endpoint = os.getenv("AZURE_OPENAI_MINI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_MINI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_MINI_API_VERSION")

        if not endpoint or not api_key or not api_version:
            raise ValueError("Azure OpenAI credentials not found in .env")

        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Azure OpenAI credentials not found in .env") from exc


def get_deployment_name():
    deployment = os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT")
    if not deployment:
        raise ValueError("Azure OpenAI credentials not found in .env")
    return deployment


def chat_complete(messages: list[dict], temperature=0.7, max_completion_tokens=1000) -> str:
    client = get_azure_client()
    response = client.chat.completions.create(
        model=get_deployment_name(),
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    content = response.choices[0].message.content
    return content if isinstance(content, str) else str(content or "")
