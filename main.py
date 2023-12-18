import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.model_response import model_response

app = FastAPI()

class UserInput(BaseModel):
    user_input: str

@app.get("/")
def index():
    return "Halo Kenari APP"

@app.post("/chatbot/response")
async def response(user_input_data: UserInput):
    user_input = user_input_data.user_input

    tag, response = model_response(user_input)

    data = {
        'tag': tag,
        'user_input': user_input,
        'model_response': response,
    }

    return JSONResponse(content=data)


port = os.environ.get("PORT", 8000)
print(f"Listening to http://127.0.0.1:{port}")
uvicorn.run(app, host='127.0.0.1', port=port)