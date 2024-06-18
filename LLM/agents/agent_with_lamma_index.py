import os
import json
import openai
import requests
from dotenv import load_dotenv
from tabulate import tabulate
from typing import Sequence, List
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.llms import ChatMessage, MessageRole


load_dotenv()

PAT = os.getenv("PAT")
openai.api_type = "azure"
openai.api_base = os.getenv("azure_endpoint")
openai.api_version = os.getenv("api_version")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
llm = AzureOpenAI(
    model=os.getenv("model"),
    deployment_name=os.getenv("deployment_name"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("azure_endpoint"),
    api_version=os.getenv("api_version"),
)
Settings.llm = llm


def add_to_basket(id:int, amount):
    """make an API request to the  API in order to add items to the basket"""

    id = str(id)
    url = "example.com/api/basketitems"
    payload = json.dumps({
    "data": [
        {
        "type": "basketitems",
        "attributes": {
            "count": amount,
            "description": "description for item"
        },
     
        }
    ]
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer PAT:{PAT}'
    }
    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def show_basket():
    """make an API request to the API in order to show all the items that are in the basket"""

    url = "example.com/api/basketitems?page[limit]=700"

    payload = {}
    headers = {'Authorization': f'Bearer PAT:{PAT}'}

    response = requests.request("GET", url, headers=headers, data=payload)
    
    items = response.json()["data"]
    table = []
    for item in items:
        item_id = item["id"]
        item_count = item["attributes"]["count"]
        item_description = item["attributes"]["description"]
        table.append([item_id, item_count, item_description])

    return tabulate(table, headers=["ID", "Count", "Description"], tablefmt="grid")


add_to_basket_tool = FunctionTool.from_defaults(fn=add_to_basket)
show_basket_tool = FunctionTool.from_defaults(fn=show_basket)

system_prompt = """You are here to assist you with various tasks related to the example API. 
You must always help adding items to the basket and showing the basket.
You are capable of interacting directly with other software and APIs to fulfill your requests. 
Do not perform any tasks that are not related to the example API.
Only perform tasks that are within the scope of the example API.
Only perform requests to the example API.
Do not perform any tasks that are not related to the example API.
Do not describe which tasks you must perform. Just perform them.
Always answer in the same language as the user's input."""


agent = ReActAgent.from_tools(
    system_prompt= system_prompt,
    tools = [add_to_basket_tool, show_basket_tool],
    llm=llm,
    handle_parsing_errors=True,
    verbose=True,
)


task = "add 2 pieces of the item 666 to the basket"
response = agent.chat(task)


chat_history = [
    ChatMessage(role=MessageRole.USER, content=task),
    ChatMessage(role=MessageRole.ASSISTANT, content=response.response),
]

response = agent.chat(task, chat_history=chat_history)
chat_history.append(ChatMessage(role=MessageRole.USER, content=task))
chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=response.response))
