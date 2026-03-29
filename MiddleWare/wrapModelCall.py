from dotenv import load_dotenv

load_dotenv()

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt



basic_model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.5)
advanced_model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.5)

@dataclass
class Context:
    """Custom runtime context schema."""
    user_role: str

