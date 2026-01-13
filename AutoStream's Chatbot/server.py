from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- IMPORT THIS
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from src.graph import build_graph
import uvicorn
import uuid

# 1. Initialize API
app = FastAPI(title="AutoStream Agent API")

# --- ADD THIS BLOCK ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------

# 2. Initialize Graph
memory = MemorySaver()
graph = build_graph()
agent = graph.compile(checkpointer=memory)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        input_message = HumanMessage(content=request.message)
        response = agent.invoke({"messages": [input_message]}, config)
        last_message = response['messages'][-1]
        
        return {
            "response": last_message.content,
            "thread_id": thread_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)