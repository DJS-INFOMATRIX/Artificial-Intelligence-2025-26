from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from sentence_transformers import SentenceTransformer
import requests

class SimpleChatbot:
    def __init__(self):
        self.client = weaviate.Client("http://localhost:8080")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.conversation_history = []
    
    def search(self, query: str, top_k: int = 4):
        try:
            query_vector = self.embedder.encode(query).tolist()
            result = self.client.query.get(
                "IndianCar",
                ["manufacturer", "model", "variant", "year", "price_min", "price_max", 
                 "fuel_type", "body_type", "mileage", "features", "pros", "cons", "text_content"]
            ).with_near_vector({
                "vector": query_vector
            }).with_limit(top_k).do()
            
            if "data" in result and "Get" in result["data"]:
                return result["data"]["Get"].get("IndianCar", [])
            return []
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def chat(self, user_input: str):
        results = self.search(user_input)
        
        if not results:
            response = "I couldn't find specific information about that. Could you rephrase your question about Indian cars?"
        else:
            cars = [f"{r.get('manufacturer','')} {r.get('model','')}" for r in results]
            q = user_input.lower()
            
            if "compare" in q or "difference" in q:
                response = f"Comparing {', '.join(cars[:2])}:\n\n"
                for r in results[:2]:
                    response += f"- {r.get('manufacturer','')} {r.get('model','')}: ₹{r.get('price_min','')}-{r.get('price_max','')} lakhs, {r.get('fuel_type','')}, {r.get('mileage','')}\n"
                response += "\nRecommendation: Choose based on your priorities: budget, fuel type, and features."
            
            elif "recommend" in q or "suggest" in q or "best" in q:
                response = f"Recommendations based on your request:\n"
                for i, r in enumerate(results[:3], 1):
                    response += f"{i}. {r.get('manufacturer','')} {r.get('model','')} - ₹{r.get('price_min','')}-{r.get('price_max','')} lakhs\n"
                response += f"\nTop pick: {results[0].get('manufacturer','')} {results[0].get('model','')} - {results[0].get('text_content','')[:200]}..."
            
            else:
                response = f"Here's what I found about {cars[0]}:\n\n{results[0].get('text_content','')}"
        
        # Simple conversation history
        self.conversation_history.append({"user": user_input, "bot": response})
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
        
        return {"response": response, "sources": results}
    
    def clear_memory(self):
        self.conversation_history = []

# Create chatbot instance
chatbot = SimpleChatbot()

app = FastAPI(title="Indian Car RAG Chatbot API")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)