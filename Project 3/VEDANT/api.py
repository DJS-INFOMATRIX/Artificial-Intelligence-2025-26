from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chatbot import IndianCarChatbot

app = FastAPI(title="Indian Car RAG Chatbot API")
chatbot = IndianCarChatbot()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = chatbot.chat(request.message)
        return ChatResponse(response=result["response"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_memory():
    chatbot.clear_memory()
    return {"message": "Memory cleared"}

@app.get("/health")
async def health():
    return {"status": "healthy"}