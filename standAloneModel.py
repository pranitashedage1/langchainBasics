# how to interact with a model in a standalone way without using langchain agent framework.
#  This is useful when you want to use the model for a specific task and you don't want to use the agent framework.


import requests
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage


model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0.1)

conversation = [
    SystemMessage(content="You are a helpful assistant that provides the current time in a given city."),
    HumanMessage(content="What is the current time in London?"),
    HumanMessage(content="What about Atlanta and Chicago?")

]

# response = model.invoke("What is the current time in London?")
response = model.invoke(conversation)

print(response.content)
