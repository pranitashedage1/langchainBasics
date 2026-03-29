from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    """Custom runtime context schema."""
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest[Context]) -> str:
    user_role = request.runtime.context.user_role 
    base_prompt = "Yoa are a helpful assistant"
    match user_role:
        case "expert":
            return f'{base_prompt} Provide detailed and technical explanations.'
        case "novice":
            return f'{base_prompt} Provide simple and easy-to-understand explanations.'
        case "child":
            return f'{base_prompt} Provide explanations suitable for a child.'
        case _:
            return base_prompt
        
agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    middleware=[user_role_prompt],
    context_schema=Context
)

response = agent.invoke(
    {
        'messages' : [
            {
                "role": "user",
                "content": "Explain PCA"
            }
        ]
    },
    context=Context(user_role="expert")
)


print(response)
