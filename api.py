# api.py - CLEAN CHAT-LIKE RESPONSES (like ChatGPT)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List, Dict, Any
import logging
import json

from core import load_rag_system, load_examples
from utils import get_final_prompt
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Islamic RAG API")

# Load RAG system and few-shot examples on startup
retriever, llm = load_rag_system()
prompt_template = get_final_prompt()
few_shot_examples = load_examples("few_shot_examples.json")

logger.info(f"Loaded {len(few_shot_examples)} few-shot examples")

class Query(BaseModel):
    question: str

def load_examples(path="few_shot_examples.json"):
    """Load few-shot examples for the model"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)
            logger.info(f"Successfully loaded {len(examples)} examples from {path}")
            return examples
    except Exception as e:
        logger.warning(f"Could not load few-shot examples from {path}: {e}")
        return []

def format_few_shot_examples(examples: List[Dict]) -> str:
    """Format few-shot examples for the prompt"""
    if not examples:
        return ""
    
    formatted = "\n\nHere are some example Q&A pairs to guide your responses:\n\n"
    for i, example in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"Question: {example.get('question', '')}\n"
        formatted += f"Answer: {example.get('answer', '')}\n\n"
    
    return formatted

def extract_text_from_response(response) -> str:
    """
    Extract clean text from various response formats.
    This is the KEY function that makes the output clean!
    UPDATED: Now properly handles Gemini's response format
    """
    # If it's a list of dicts with 'type' and 'text' (Gemini format)
    if isinstance(response, list):
        if len(response) == 0:
            return ""
        
        # Handle Gemini's format: [{'type': 'text', 'text': '...'}]
        if isinstance(response[0], dict) and 'text' in response[0]:
            # Concatenate all text parts
            return "".join([item.get('text', '') for item in response if item.get('type') == 'text'])
        
        # Otherwise get first item and continue processing
        response = response[0]
    
    # If it's a dict, extract the text field
    if isinstance(response, dict):
        # Extract text from the dict
        text = (response.get('text') or 
                response.get('content') or 
                response.get('message') or 
                response.get('output'))
        return str(text) if text else str(response)
    
    # If it has a content attribute (AIMessage)
    if hasattr(response, 'content'):
        content = response.content
        # Handle if content is also a list (nested structure)
        if isinstance(content, list):
            if content and isinstance(content[0], dict) and 'text' in content[0]:
                return "".join([item.get('text', '') for item in content if item.get('type') == 'text'])
        return str(content)
    
    # If it has a text attribute
    if hasattr(response, 'text'):
        return str(response.text)
    
    # Default: convert to string
    return str(response)

def call_llm(llm_obj, prompt_text: str) -> str:
    """
    Call LLM and return ONLY the text - nothing else!
    """
    if llm_obj is None:
        return "LLM not configured."
    
    try:
        # Call the LLM
        response = llm_obj.invoke(prompt_text)
        
        # Extract clean text
        clean_text = extract_text_from_response(response)
        
        # Remove any extra whitespace
        clean_text = clean_text.strip()
        
        logger.info(f"Generated answer: {clean_text[:100]}...")
        return clean_text
        
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@app.post("/ask")
async def ask_question(q: Query):
    """
    Ask a question and get a CLEAN chat-like response (like ChatGPT)
    """
    def run_chain(question: str) -> str:
        try:
            logger.info(f"Question: {question}")
            
            # 1. Get relevant documents
            docs = retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # 2. Build context
            context = "\n\n".join([getattr(d, "page_content", "") for d in docs])
            
            # 3. Add few-shot examples
            few_shot_text = format_few_shot_examples(few_shot_examples)
            
            # 4. Format prompt
            prompt_text = prompt_template.format(context=context, question=question)
            
            if few_shot_text:
                prompt_text = prompt_text.replace(
                    question,
                    few_shot_text + "\nNow answer this question:\n" + question
                )
            
            # 5. Get answer
            answer = call_llm(llm, prompt_text)
            
            # 6. Extract sources for metadata
            sources = []
            for d in docs:
                meta = getattr(d, "metadata", {}) or {}
                source_info = {
                    "id": meta.get("id") or meta.get("parent_id"),
                    "source": meta.get("source") or meta.get("book_id")
                }
                if source_info["id"] or source_info["source"]:
                    sources.append(source_info)
            
            # Return clean response
            return {
                "answer": answer,  # Just the text!
                "sources": sources if sources else []
            }
            
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": []
            }

    try:
        result = await run_in_threadpool(run_chain, q.question)
        return result
        
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "retriever_loaded": retriever is not None,
        "llm_loaded": llm is not None,
        "few_shot_examples": len(few_shot_examples)
    }

@app.get("/examples")
async def get_examples():
    """View examples"""
    return {
        "count": len(few_shot_examples),
        "examples": few_shot_examples
    }

@app.post("/reload-examples")
async def reload_examples():
    """Reload examples"""
    global few_shot_examples
    few_shot_examples = load_examples("few_shot_examples.json")
    return {
        "status": "success",
        "count": len(few_shot_examples)
    }

@app.get("/")
async def root():
    """API info"""
    return {
        "message": "Islamic RAG API - Clean Chat Responses",
        "endpoints": {
            "/ask": "POST - Ask a question (returns clean text)",
            "/health": "GET - Health check",
            "/examples": "GET - View examples",
            "/reload-examples": "POST - Reload examples"
        }
    }