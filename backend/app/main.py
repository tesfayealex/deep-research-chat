from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
import os
from dotenv import load_dotenv
import asyncio
import json
from typing import List, Optional

# Import the agent stream generator function
from .agent import stream_agent_research
from .vectorstore.weaviate_client import get_weaviate_client # Import Weaviate client getter
import weaviate.classes as wvc # Import Weaviate classes
from .config import settings # Import settings for history class name

load_dotenv() # Load environment variables from .env file

app = FastAPI(title="Deep Research Agent API")

# --- CORS Configuration --- #
# Allow frontend origin (adjust in production)
origins = [
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
    "http://localhost:3000", # Default if FRONTEND_URL not set
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Pydantic Models --- #
class ResearchQuery(BaseModel):
    query: str
    session_id: str | None = None # Optional

class ResearchResponse(BaseModel):
    report: str
    session_id: str

class ResearchRequest(BaseModel):
    query: str
    max_steps: Optional[int] = None
    # conversation_id: Optional[str] = None # Add later for continuation

# Define a model for the structure of events stored in Weaviate
class HistoryEvent(BaseModel):
    conversation_id: str
    timestamp: str # ISO format string
    event_type: str
    sender: Optional[str] = None
    data_json: str # The JSON string data payload

# --- API Endpoints --- #
@app.get("/", tags=["Status"])
def read_root():
    return {"message": "Deep Research Agent Backend is running"}

@app.post("/api/research", tags=["Research"])
async def research_stream_endpoint(query: ResearchQuery):
    """
    Receives a research query, runs the agent stream, and returns a stream of status updates.
    The stream sends JSON objects line-by-line.
    """
    print(f"Received query for streaming: {query.query}")
    try:
        # Return a StreamingResponse that consumes the async generator
        return StreamingResponse(
            stream_agent_research(query.query),
            media_type="application/x-ndjson" # Use newline-delimited JSON for streaming
        )
    except Exception as e:
        # This initial exception handling might catch setup errors before streaming starts
        print(f"Error starting agent stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent stream: {str(e)}")

# Remove or comment out the old non-streaming endpoint if it exists
# @app.post("/api/research", response_model=ResearchResponse, tags=["Research"])
# async def research_endpoint(query: ResearchQuery):
#     ... (old implementation)

# --- History API Endpoints --- #

@app.get("/api/history")
async def get_conversation_list():
    """Fetches a list of unique conversation IDs and their latest timestamp (Python processing)."""
    client = get_weaviate_client()
    if not client:
        raise HTTPException(status_code=503, detail="History persistence is unavailable.")
        
    history_collection = client.collections.get(settings.WEAVIATE_HISTORY_CLASS)
    
    try:
        # Fetch objects and extract unique IDs/timestamps in Python (reverting to previous method)
        response = history_collection.query.fetch_objects(
            limit=2000,  # Fetch a larger number initially, adjust if needed
            include_vector=False, 
            # Fetch necessary properties - using event_type now
            return_properties=["conversation_id", "timestamp", "event_type"] 
        )
        
        # Use dict to track latest timestamp per conversation
        all_conversations = {}  
        
        for obj in response.objects:
            conv_id = obj.properties.get("conversation_id")
            timestamp = obj.properties.get("timestamp", "")
            # Use event_type from Weaviate properties
            event_type = obj.properties.get("event_type", "") 
            
            # Track latest timestamp for any event type for a given conv_id
            if conv_id:
                if conv_id not in all_conversations or timestamp > all_conversations[conv_id]["timestamp"]:
                    all_conversations[conv_id] = {
                        "timestamp": timestamp,
                        # "event_type": event_type # Store type if needed later for title generation
                    }
        
        # Format the response
        conversations_data = [
            {
                "id": conv_id, 
                "title": f"Conversation {conv_id[:8]}...", # Simple title
                "timestamp": data["timestamp"]
            } 
            for conv_id, data in all_conversations.items()
        ]
        
        # Sort by timestamp (most recent first)
        conversations_data.sort(key=lambda x: x["timestamp"], reverse=True)
                
        return {"conversations": conversations_data}
        
    except Exception as e:
        print(f"Error fetching conversation list from Weaviate (Python processing): {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history list.")

# Model to represent the structure returned by the history endpoint
class HistoryEventResponse(BaseModel):
    conversation_id: str
    timestamp: str
    type: str # e.g., user_message, agent_yield
    sender: Optional[str] = None
    name: Optional[str] = None # e.g., user_input, plan, step_result
    tags: Optional[List[str]] = None
    data_json: str # The JSON string data payload

@app.get("/api/history/{conversation_id}", response_model=List[HistoryEventResponse])
async def get_conversation_history(conversation_id: str):
    """
    Fetches the user messages and agent yields for a specific conversation ID,
    ordered by timestamp, to reconstruct the conversation history as seen by the user.
    """
    client = get_weaviate_client()
    if not client:
        raise HTTPException(status_code=503, detail="History persistence is unavailable.")

    history_collection = client.collections.get(settings.WEAVIATE_HISTORY_CLASS)

    try:
        # Fetch only user_message and agent_yield types using event_type field
        response = history_collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("conversation_id").equal(conversation_id)
            & (
                # Use event_type for filtering
                wvc.query.Filter.by_property("event_type").equal("user_message") |
                wvc.query.Filter.by_property("event_type").equal("agent_yield")
            ),
            sort=wvc.query.Sort.by_property("timestamp"),
            limit=2000 # Increased limit for longer conversations
        )
        
        # Reconstruct events from Weaviate objects into the response model
        events = []
        for obj in response.objects:
            props = obj.properties
            print(f"Props: {props}")
            # Convert timestamp to string to match Pydantic model
            timestamp_str = props.get("timestamp").isoformat() if props.get("timestamp") else ""
            events.append(HistoryEventResponse(
                conversation_id=props.get("conversation_id"),
                timestamp=timestamp_str,  # Ensure it's a string
                type=props.get("event_type"),
                sender=props.get("sender"),
                name=props.get("name"),
                tags=props.get("tags", []),  # Ensure tags is a list
                data_json=props.get("data_json")
            ))
            
        return events
    except Exception as e:
        print(f"Error fetching history for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history for conversation {conversation_id}.")

# --- Run with Uvicorn (for local development) --- #
# You would typically run this using: uvicorn main:app --reload --port 8000

# Need asyncio for await sleep
import asyncio 