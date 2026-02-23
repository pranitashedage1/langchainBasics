from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
import random

SYSTEM_PROMPT = """You are a lame joke master. 
Generate a completely random, different, creative lame/cheesy joke every time.
Never repeat jokes. Be unpredictable."""

model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=1.0
)

# Add a random seed in the message so Claude gives different output
random_seed = random.randint(1, 999999)

response = model.invoke([
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"Tell me a joke. (attempt #{random_seed})")
])

print(response.content)