# https://platform.openai.com/docs/guides/function-calling
import os
import json
import openai
import requests

client = openai.OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")


def make_request_to_home():
    response = requests.get("http://localhost:8000/")

    return response.json()


def get_user_id(user_id: str):
    response = requests.get(f"http://localhost:8000/users/{user_id}")

    return response.json()


def get_password(username: str):
    response = requests.get(f"http://localhost:8000/users/password/{username}/")

    return response.json()


def run_conversation(command: str):
    messages = [{"role": "user", "content": f"{command}"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "make_request_to_home",
                "description": "make an API request to the home endpoint",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_id",
                "description": "get the user id of the user with a given id",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The id number of the user",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_password",
                "description": "get the password of the user with a given username",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "The name of the user whose password we want to get",
                        }
                    },
                },
            },
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        available_functions = {
            "make_request_to_home": make_request_to_home,
            "get_user_id": get_user_id,
            "get_password": get_password,
        }
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append({"role": "assistant", "content": function_response})
        return function_response
    else:
        return response_message.choices[0].message.content


user_id = 5
print(run_conversation(f"Go to the homepage"))
print(run_conversation(f"Get the user id of the user with user id {user_id}"))
print(run_conversation(f"Get the password of the user with username Alex"))
