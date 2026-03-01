#@Tool is a decorator that marks a function as a tool that can be called by the agent.
# System prompt is the initial instruction given to the agent to set the context for the conversation.
# The agent will use this system prompt to understand its role and how to respond to user queries


import requests

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

   
@tool
def get_current_time(city: str) -> str:
   
   """Get current time for a given city. City should be a timezone like America/New_York or Europe/London."""
   try:
        response = requests.get(
            f"https://timeapi.io/api/time/current/zone?timeZone={city}",
            timeout=5
        )
        data = response.json()
        return {
            "timezone": data["timeZone"],
            "date": data["date"],
            "time": data["time"],
            "dayOfWeek": data["dayOfWeek"]
        }
   except Exception as e:
        return {"error": str(e)}


agent = create_agent(
    model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.5),
    tools = [get_current_time],
    system_prompt = """You are a helpful timezone assistant. 
    You can answer questions about the current time in a city. Response should be funny and informative. But is should also include
    the timezone, date, time and day of week."""
)

response = agent.invoke(
    {"messages" :[
        {"role": "user",
         "content" : "what is the current time in London?"}
    ]}
)

print(response['messages'][-1].content)

# get token details - 
# print(response['messages'][-1].usage_metadata)

print("***************************************")
for message in response['messages']:
    print(f"Type: {type(message).__name__}")
    
    # ToolMessage = tool was called and returned a result
    if hasattr(message, 'name'):
        print(f"  Tool Used: {message.name}")
        print(f"  Tool Output: {message.content}")
    
    # AIMessage with tool_calls = model decided to call a tool
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tc in message.tool_calls:
            print(f"  Tool Called: {tc['name']}")
            print(f"  Tool Input:  {tc['args']}")
    
    print("---")


