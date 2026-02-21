from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import ToolRuntime, tool
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy



SYSTEM_PROMT = """
You are a helpful assistant that can answer the temperature of a city. 
You give the temperature in Celsius. 
You can only answer the temperature of a city, 
and you should not provide any other information. 
If you don't know the temperature of a city, you should say 
'That you don't know the temperatue'
If user just says, hi and hello or generenral statemenet which seems like 
initiaing a converstation, you should respond with little introduction about yourself 
and ask to enter the city name to get the temperature.
if user still does not give you a city name, 
you should say 'Please enter the city name to get the temperature' and 
wait for the user to enter the city name.

"""

@tool
def get_city() -> str:
    """Get the name of the city from user input."""
    city = input("Please enter the city name to get the temperature: ")
    return city

@tool
def get_weather_for_location(city: str) -> str:
    ### Call the weather API to get the weather of the given city.
    return f"The temperature in {city} is 30°C."

@dataclass
class Context:
    """Custom runtime context schema."""
    usr_id: str

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent. This will return the temperature in Celsius and Fahrenheit."""
    response: str
    # Any interesting information about the weather if available
    temperatureInCelcious: str | None = None
    temperatureInFahrenheit: str | None = None



checkpointer = InMemorySaver()


agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMT,
    tools=[get_city],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user"}]},
    config=config,
    context=Context(usr_id="1")
)

print(response['structured_response'])


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(usr_id="1")
)

print(response['structured_response'])

