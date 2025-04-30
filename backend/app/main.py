from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
import os
from dotenv import load_dotenv
import asyncio
import json

# Import the agent stream generator function
from .agent import stream_agent_research

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

# --- Optional: Add logic for status updates later --- #

# --- Run with Uvicorn (for local development) --- #
# You would typically run this using: uvicorn main:app --reload --port 8000

# Need asyncio for await sleep
import asyncio 