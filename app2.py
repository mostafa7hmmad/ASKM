import streamlit as st
import time
import logging
import json
from typing import List, Dict, Any

# Import your existing modules
# Make sure core.py, utils.py, and few_shot_examples.json are in the same folder!
from core import load_rag_system, load_examples
from utils import get_final_prompt

# --- Configuration & Logging ---
st.set_page_config(page_title="Islamic RAG", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Load Resources (Cached for Performance) ---
# We use @st.cache_resource so these heavy models load only once, not on every click.
@st.cache_resource
def get_rag_engine():
    """Load the RAG system and prompt templates once."""
    logger.info("Loading RAG system...")
    retriever, llm = load_rag_system()
    prompt_template = get_final_prompt()
    examples = load_examples("few_shot_examples.json")
    return retriever, llm, prompt_template, examples

# Initialize resources
try:
    retriever, llm, prompt_template, few_shot_examples = get_rag_engine()
    st.success("✅ System Loaded Successfully")
except Exception as e:
    st.error(f"Failed to load RAG system: {e}")
    st.stop()

# --- 2. Helper Functions (From your api.py) ---
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
    """Clean text extraction logic from your API"""
    if isinstance(response, list):
        if len(response) == 0: return ""
        if isinstance(response[0], dict) and 'text' in response[0]:
            return "".join([item.get('text', '') for item in response if item.get('type') == 'text'])
        response = response[0]
    
    if isinstance(response, dict):
        text = (response.get('text') or response.get('content') or 
                response.get('message') or response.get('output'))
        return str(text) if text else str(response)
    
    if hasattr(response, 'content'):
        return str(response.content)
    if hasattr(response, 'text'):
        return str(response.text)
    
    return str(response)

def call_llm(prompt_text: str) -> str:
    """Invoke the LLM and clean the response"""
    try:
        response = llm.invoke(prompt_text)
        clean_text = extract_text_from_response(response)
        return clean_text.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Error: {str(e)}"

# --- 3. Main Logic Function ---
def get_rag_answer(question: str):
    """The main RAG logic (formerly in your API endpoint)"""
    # 1. Retrieve Documents
    docs = retriever.get_relevant_documents(question)
    
    # 2. Build Context
    context = "\n\n".join([getattr(d, "page_content", "") for d in docs])
    
    # 3. Format Prompt with Examples
    few_shot_text = format_few_shot_examples(few_shot_examples)
    prompt_text = prompt_template.format(context=context, question=question)
    
    if few_shot_text:
        prompt_text = prompt_text.replace(
            question,
            few_shot_text + "\nNow answer this question:\n" + question
        )
    
    # 4. Get Answer
    answer = call_llm(prompt_text)
    
    # 5. Extract Sources
    sources = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        sources.append({
            "id": meta.get("id") or meta.get("parent_id"),
            "source": meta.get("source") or meta.get("book_id")
        })
    
    return answer, sources

# --- 4. Streamlit UI ---
st.title("📚 Islamic RAG Assistant")

query = st.text_input("Ask your question:")

if st.button("Ask") and query:
    placeholder = st.empty()
    sources_placeholder = st.empty()
    
    with st.spinner("Searching documents and generating answer..."):
        try:
            # DIRECT CALL - No requests.post anymore!
            answer, sources = get_rag_answer(query)

            # Streaming Simulation
            chunks = answer.split("\n") 
            displayed_text = ""
            for chunk in chunks:
                if chunk.strip():
                    displayed_text += chunk + "\n\n"
                    placeholder.markdown(displayed_text)
                    time.sleep(0.1) # Faster simulation

            # Show Sources
            with sources_placeholder.expander("📌 Sources"):
                if sources:
                    for src in sources:
                        st.write(f"- **{src['source']}** (ID: {src['id']})")
                else:
                    st.write("No specific sources found.")

        except Exception as e:
            st.error(f"An error occurred: {e}")