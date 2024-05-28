from fastapi import FastAPI
import requests

app = FastAPI()


@app.get("/")
def read_root():
    response = {"message": "Hello AI"}
    return response


@app.get("/users/{user_id}")
def read_user(user_id: str):
    response = {"user_id": user_id}
    return response


@app.get("/users/password/{username}/")
def get_password(username: str):
    credentials = {"Alex": "1", "John": "2"}
    user_id = credentials.get(username)

    response = {"username": username, "password": credentials[username]}

    return response


def make_request_to_user(user_id: str):
    import requests

    response = requests.get(f"http://localhost:8000/users/{str(5)}")
    print(response.json())
    return response.json()


def make_request_to_password(username: str):
    import requests

    response = requests.get(f"http://localhost:8000/users/password/Alex/")
    print(response.json())
    return response.json()
