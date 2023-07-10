from environs import Env
from .Config import Config


def get_config(env_file_path: str = ".env") -> Config:
    env = Env()
    env.read_env(env_file_path)

    config = Config(
        open_ai_api_key=env.str("OPEN_AI_API_KEY"),
        web_domain=env.str("WEB_DOMAIN"),
        port=env.int("PORT"),
    )
    return config
