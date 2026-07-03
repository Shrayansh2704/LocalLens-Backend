
"""
from fastapi import APIRouter
from schemas import ChatRequest, ChatResponse
from services.rag_service import ask

router = APIRouter(prefix="/chat", tags=["RAG Chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = ask(query=req.query, active_city=req.city)
    return ChatResponse(**result)
"""

from fastapi import APIRouter
from schemas import ChatRequest, ChatResponse
from services.rag_service import ask

router = APIRouter(prefix="/chat", tags=["RAG Chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    # debugging the input 
    print("QUERY:", req.query)
    print("CITY:", req.city)

    result = ask(query=req.query, active_city=req.city)

    # debugging the output
    print("RAG RESULT:", result)

    return ChatResponse(**result)
