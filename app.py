from dotenv import load_dotenv
import os
import json
import numpy as np
import gradio as gr
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.staticfiles import StaticFiles
import re
import tempfile
from pathlib import Path
import sys

# Ensure `src/` is on the import path so `bgbpythonbot` package is importable at runtime
sys.path.insert(0, str(Path(__file__).resolve().parent.joinpath("src")))

from bgbpythonbot.document_parser import parse_document
from bgbpythonbot.document_store import DocumentStore

# === CONFIG ===
EMBEDDINGS_FILE = "bgb_embeddings_new_data.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 3
STATIC_URL_BASE = "http://localhost:8000/static/bgb_new.html"
PROMPT_PREAMBLE = (
   "You are a friendly and helpful legal assistant specialized in the German Civil Code (BGB). "
    "Please base your answers strictly on the provided BGB content, referencing the clickable section and paragraph links given below, including inline subsection links such as [327c](...). "
    "Avoid speculation or inventing information beyond the provided content. "
    "If the exact term or section requested is not available, provide the most relevant information based on related sections or legal principles, and explain clearly with background and context. "
    "You are assisting an audience of expats in Germany who may not be familiar with local legal terms, so use clear language and include important German legal terms with their English translations when appropriate. "
    "You must ALWAYS include the clickable links provided to support your claims.\n\n"
    "---\n"
)


# === Load environment variable ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=api_key)

# === Cosine similarity ===
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# === Initialize document store ===
document_store = DocumentStore(cosine_similarity_fn=cosine_similarity)

# === Load chunks from file ===
all_chunks = []
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_chunks.append(json.loads(line))

# Precompute section-number -> anchor id for faster subsection link replacement
SECTION_ANCHOR_MAP = {
    chunk.get("SectionNumber", "").lower(): chunk.get("SectionId", chunk.get("Id"))
    for chunk in all_chunks
    if chunk.get("SectionNumber")
}

# === Embedding ===
def get_query_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def rank_bgb_chunks(query_embedding, top_k=None):
    ranked = sorted(
        [
            {"chunk": chunk, "score": cosine_similarity(query_embedding, chunk["Embedding"])}
            for chunk in all_chunks
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked if top_k is None else ranked[:top_k]

# === Main RAG logic ===
def build_prompt_from_chunks(chunks, user_input):
    prompt = PROMPT_PREAMBLE
    
    for chunk in chunks:
        source = chunk.get("source", "bgb")
        
        if source == "uploaded":
            # Format citation for uploaded documents with specific text
            filename = chunk.get("filename", "Unknown")
            text = chunk.get("text", "")
            
            # Extract first 100 chars as snippet for citation
            snippet = text[:100].replace("\n", " ").strip()
            if len(text) > 100:
                snippet += "..."
            
            citation = f"[Uploaded: {filename} - \"{snippet}\"]"
            prompt += f"{citation}:\n{text}\n\n"
        
        else:
            # Format citation for BGB sections (existing logic)
            sec_num = chunk.get("SectionNumber", "N/A")
            sec_title = chunk.get("SectionTitle", "")
            sec_id = chunk.get("SectionId", chunk.get("Id"))
            chunk_id = chunk.get("Id")
            section_text = chunk.get("Text", "")
            
            # Replace subsection references in the chunk text with clickable links
            section_text_with_links = replace_subsection_links(section_text, SECTION_ANCHOR_MAP, STATIC_URL_BASE)
            
            # Build section and paragraph links
            section_link = f"[Section {sec_num} — {sec_title}]({STATIC_URL_BASE}#{sec_id})"
            paragraph_link = f"[paragraph]({STATIC_URL_BASE}#{chunk_id})"
            
            prompt += (
                f"{section_link}:\n"
                f"{section_text_with_links}\n"
                f"{paragraph_link}\n\n"
            )
    
    prompt += f"---\nQuestion: {user_input}"
    return prompt

def replace_subsection_links(text, section_anchor_map, base_url):
    import re

    # Regex to find subsection references (e.g. 327c, 327r (3))
    pattern = r'\b\d{1,4}[a-z]?(\s*\(\d+\))?\b'

    def replacer(match):
        ref = match.group(0).replace(" ", "")
        anchor = section_anchor_map.get(ref.lower())
        if anchor:
            link = f"[{ref}]({base_url}#{anchor})"
            return link
        return ref  # no match, return original

    return re.sub(pattern, replacer, text)


def answer_question(user_input, history):
    user_input_lc = user_input.lower()
    is_bgb_query = "section" in user_input_lc or "bgb" in user_input_lc
    has_uploaded_doc = document_store.has_documents()

    if has_uploaded_doc:
        print(f"DEBUG: Searching {document_store.get_chunk_count()} uploaded chunks")
    else:
        print("DEBUG: No uploaded document found")

    # If just asking a general question with no keywords, use GPT directly
    # UNLESS there's an uploaded document, then we should search it
    if not is_bgb_query and not has_uploaded_doc:
        chat_resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": user_input}]
        )
        return chat_resp.choices[0].message.content

    # If looking for a specific section
    match = re.search(r'section\s+(\d+[a-z]?)', user_input_lc)
    if match:
        requested_section = match.group(1)
        matched_chunks = [chunk for chunk in all_chunks if chunk.get("SectionNumber") == requested_section]

        if not matched_chunks:
            return f"Sorry, section {requested_section} does not exist in the current version of the BGB."

        prompt = build_prompt_from_chunks(matched_chunks, user_input)

    else:
        # Search both BGB and uploaded documents
        query_embedding = get_query_embedding(user_input)
        
        # Search uploaded document chunks FIRST (prioritize them)
        ranked_uploaded = document_store.search(query_embedding, top_k=TOP_K)
        
        # If we have uploaded results, use those + 1-2 BGB results as context
        if ranked_uploaded:
            # Get some BGB results for additional context
            ranked_bgb = rank_bgb_chunks(query_embedding, top_k=5)  # Get top 5 BGB for context
            
            # Prioritize uploaded: take all uploaded results, then fill with 1-2 BGB if needed
            chunks_list = [item["chunk"] for item in ranked_uploaded[:2]]
            if len(chunks_list) < TOP_K:
                chunks_list.extend([item["chunk"] for item in ranked_bgb[:1]])
            chunks = chunks_list
            print(f"DEBUG: Using {len(ranked_uploaded)} uploaded + BGB context")
        else:
            # No uploaded docs, search BGB normally
            ranked_bgb = rank_bgb_chunks(query_embedding)
            
            chunks = [item["chunk"] for item in ranked_bgb[:TOP_K]]
            print(f"DEBUG: Using {len(chunks)} BGB results (no uploaded)")
        
        prompt = build_prompt_from_chunks(chunks, user_input)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content



# === Gradio UI ===
# Multimodal chat interface with file upload support
def handle_message_with_upload(message, history):
    """Handle user messages in multimodal chat interface.
    
    Args:
        message: Dict with 'text' and optional 'files' from Gradio multimodal input
        history: Chat history
    
    Returns:
        Response text
    """
    # Extract text from message
    user_text = message.get("text", "")
    
    if not user_text.strip():
        return "Please enter a question."
    
    # Check if file was uploaded in this message
    files = message.get("files", [])
    upload_msg = ""
    
    if files and len(files) > 0:
        # Get first file (can be string path or file object)
        file_input = files[0]
        
        # Debug: figure out what type we got
        file_path = None
        
        print(f"DEBUG: File input type: {type(file_input)}, value: {file_input}")
        
        # If it's a string path
        if isinstance(file_input, str):
            file_path = file_input
        # If it's a file object with 'name' attribute
        elif hasattr(file_input, 'name'):
            file_path = file_input.name
        # If it's a dict with 'name' key (Gradio file object)
        elif isinstance(file_input, dict) and 'name' in file_input:
            file_path = file_input['name']
        else:
            file_path = str(file_input)
        
        filename = Path(file_path).name
        print(f"DEBUG: Processing file: {filename}, path: {file_path}")
        
        try:
            # Parse document directly
            chunks = parse_document(file_path, filename)
            print(f"DEBUG: Created {len(chunks)} chunks from {filename}")
            
            # Generate embeddings and store
            document_store.add_chunks(chunks, get_query_embedding)
            
            chunk_count = document_store.get_chunk_count()
            upload_msg = f"✓ Uploaded: {filename} ({chunk_count} chunks)\n\n"
            print(f"DEBUG: Document stored with {chunk_count} chunks")
        except Exception as e:
            print(f"DEBUG: Error parsing document: {e}")
            import traceback
            traceback.print_exc()
            upload_msg = f"⚠️ Upload error: {str(e)}\n\n"
    
    # Process question (with or without uploaded file)
    response_text = answer_question(user_text, history)
    
    return upload_msg + response_text


def create_app_ui():
    """Create Gradio multimodal chat interface with document upload."""
    demo = gr.ChatInterface(
        fn=handle_message_with_upload,
        title="BGB Legal Chatbot",
        theme="ocean",
        description="Ask me about German civil law. Upload documents (PDF/DOCX) for context-aware analysis. Type 'Section' or 'BGB' to trigger legal lookup.",
        examples=[
            {"text": "My landlord just increased my rent, what can i do according to the BGB?"},
            {"text": "Can you explain the exclusions for certain trips from the package travel contract rules in Section 651a?"},
            {"text": "I need to return an item I purchased - what does the BGB say about this?"},
            {"text": "What are my consumer rights according to the BGB?"}
        ],
        type="messages",
        multimodal=True,  # Enable multimodal input (text + files)
    )
    
    return demo

demo = create_app_ui()

# demo.launch() -- needed to host on server and comment out below code

# === Mount Gradio and static HTML ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app = gr.mount_gradio_app(app, demo, path="/")

# === FastAPI upload endpoint ===
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and parse a document (PDF or DOCX).
    
    Returns JSON with status, chunk count, or error message.
    """
    try:
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Parse document
            chunks = parse_document(tmp_path, file.filename)
            
            # Generate embeddings and store
            document_store.add_chunks(chunks, get_query_embedding)
            
            chunk_count = document_store.get_chunk_count()
            return {
                "status": "success",
                "message": f"Uploaded {file.filename} ({chunk_count} chunks)",
                "filename": file.filename,
                "chunk_count": chunk_count
            }
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# === To run ===
# uvicorn app:app --reload
