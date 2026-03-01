# for demonstration purposes only, not for production use
# This agent can answer questions about the current weather 
# in a given city. It uses two tools
# 1. get_current_time - to get the current time for a city
# 2. get_city_from_user - to get the user's city based on their user ID
# if user id is 1 - it returns Atlanta, 
# if user id is 2 - it returns Paris, 
# otherwise it returns unknown location.

# if you want to switch between the locations, 
# you can change the user_id in the context when invoking the agent.
# for example, if you want to get the weather for Paris, 
# you can set user_id to 2 in the context.
# initalize the checkpointer once again to reset the memory 
# for the new user.
# the purpose of checkpointer is to store the
#  conversation history and the tool calls,
# so that the agent can use this information in future interactions.

import requests
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from dataclasses import dataclass


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


@dataclass
class ResponseFormat:
    timezone: str
    date: str
    time: str
    dayOfWeek: str
    summary: str | None = None

@tool
def get_current_time(city: str) -> ResponseFormat:
    """Get current time for a given city. City should be a timezone like America/New_York or Europe/London."""
    try:
        response = requests.get(
            f"https://timeapi.io/api/time/current/zone?timeZone={city}",
            timeout=5
        )
        data = response.json()
        # return {
        #     "timezone": data["timeZone"],
        #     "date": data["date"],
        #     "time": data["time"],
        #     "dayOfWeek": data["dayOfWeek"]
        # }
        return data
    except Exception as e:
        return {"error": str(e)}


@tool
def get_city_from_user(runtime: ToolRuntime[Context]) -> str:
    """Get the user's city based on their user ID."""
    userId = runtime.context.user_id
    match userId:
        case "1":
            return "Atlanta"
        case "2":
            return "Paris"
        case _:
            return f"Unknown location for user_id: {userId}"


model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.5)

checkerPointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools= [get_current_time, get_city_from_user],
    system_prompt="You are a helpful timezone assistant. " \
    "You can answer questions about the current time in a city in the structured format " \
    "Response summary should be funny and informative.", 
    context_schema = Context,
    response_format=ResponseFormat,
    checkpointer=checkerPointer,
)

config = {
        'configurable': {
            "thread_id": "1"
            },
        }

respone = agent.invoke({
    "messages":[{
        "role": "user",
        "content": "what is the current time in my city?"
    }]},
    config=config,
    context=Context(user_id="1")
)


print(respone['structured_response'])
print("***************************************")

respone = agent.invoke({
    "messages":[{
        "role": "user",
        "content": "what is a current day in atlanta and can you reply in a pun for this day?"
    }]},
    config=config,
    context=Context(user_id="1")
)
print(respone['structured_response'])

print("***************************************")

checkerPointer = InMemorySaver()

respone = agent.invoke({
    "messages":[{
        "role": "user",
        "content": "how is the night life in the city"
    }]},
    # config={"configurable": {"thread_id": "user-2"}},
    config=config,
    context=Context(user_id="2")
)
print(respone['structured_response'])

print("***************************************")
