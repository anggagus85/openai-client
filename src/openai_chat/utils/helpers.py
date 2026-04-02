# src/openai_chat/utils/helpers.py

from dotenv import dotenv_values

_env = dotenv_values(verbose=True)


def get_api_key(key: str) -> str:
    if api_key := _env.get(key):
        return api_key
    return ""
