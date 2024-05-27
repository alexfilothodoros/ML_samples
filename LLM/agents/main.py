from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    response = {"message": "Hello AI"} 
    return response


@app.get("/users/{user_id}")
def read_user(user_id):
    response = {"user_id": str(user_id)}
    return response


@app.get("/users/{username}/{user_id}")
def read_user(username, user_id):
    credentials = {'Alex': '1', 'John': '2'}
    if username in credentials:
        user_id = credentials[username]
    else:
        user_id = None

    response = {"username": username, "user_id": user_id}

    return response

