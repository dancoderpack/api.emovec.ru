from dataclasses import dataclass


@dataclass
class Config:
    open_ai_api_key: str
    web_domain: str
