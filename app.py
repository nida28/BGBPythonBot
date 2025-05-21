from dotenv import load_dotenv
import os
import json
import numpy as np
import gradio as gr
from openai import OpenAI
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
import re

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

# === Load chunks from file ===
all_chunks = []
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_chunks.append(json.loads(line))

# === Cosine similarity ===
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# === Embedding ===
def get_query_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def find_section_chunks(section_number, all_chunks):
    # Return list of chunks matching SectionNumber exactly
    return [chunk for chunk in all_chunks if chunk.get("SectionNumber") == section_number]

# === Main RAG logic ===
def build_prompt_from_chunks(chunks, user_input):
    prompt = PROMPT_PREAMBLE
    for chunk in chunks:
        sec_num = chunk.get("SectionNumber", "N/A")
        sec_title = chunk.get("SectionTitle", "")
        sec_id = chunk.get("SectionId", chunk.get("Id"))
        chunk_id = chunk.get("Id")
        section_text = chunk.get("Text", "")

    # Replace subsection references in the chunk text with clickable links
    section_text_with_links = replace_subsection_links(section_text, all_chunks, STATIC_URL_BASE)

    # Build section and paragraph links as before
    section_link = f"[Section {sec_num} — {sec_title}]({STATIC_URL_BASE}#{sec_id})"
    paragraph_link = f"[paragraph]({STATIC_URL_BASE}#{chunk_id})"

    # Append prompt with links inline in text, and main section/paragraph links if you want
    prompt += (
        f"{section_link}:\n"
        f"{section_text_with_links}\n"
        f"{paragraph_link}\n\n"
    )

    prompt += f"---\nQuestion: {user_input}"
    return prompt

def replace_subsection_links(text, all_chunks, base_url):
    import re

    # Regex to find subsection references (e.g. 327c, 327r (3))
    pattern = r'\b\d{1,4}[a-z]?(\s*\(\d+\))?\b'

    def replacer(match):
        ref = match.group(0).replace(" ", "")
        # Find matching chunk by SectionNumber ignoring case
        for chunk in all_chunks:
            if chunk.get("SectionNumber", "").lower() == ref.lower():
                link = f"[{ref}]({base_url}#{chunk.get('SectionId', chunk.get('Id'))})"
                return link
        return ref  # no match, return original

    return re.sub(pattern, replacer, text)


def answer_question(user_input, history):
    user_input_lc = user_input.lower()
    is_bgb_query = "section" in user_input_lc or "bgb" in user_input_lc

    if not is_bgb_query:
        chat_resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": user_input}]
        )
        return chat_resp.choices[0].message.content

    match = re.search(r'section\s+(\d+[a-z]?)', user_input_lc)
    if match:
        requested_section = match.group(1)
        matched_chunks = [chunk for chunk in all_chunks if chunk.get("SectionNumber") == requested_section]

        if not matched_chunks:
            return f"Sorry, section {requested_section} does not exist in the current version of the BGB."

        prompt = build_prompt_from_chunks(matched_chunks, user_input)

    else:
        query_embedding = get_query_embedding(user_input)
        ranked = sorted(
            [
                {"chunk": chunk, "score": cosine_similarity(query_embedding, chunk["Embedding"])}
                for chunk in all_chunks
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:TOP_K]

        chunks = [item["chunk"] for item in ranked]
        prompt = build_prompt_from_chunks(chunks, user_input)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content



# === Gradio UI ===
demo = gr.ChatInterface(
    fn=answer_question,
    title="BGB Legal Chatbot",
    theme="ocean",
    description="Ask me about German civil law. Type 'Section' or 'BGB' to trigger legal lookup.",
    examples=[
        "My landlord just increased my rent, what can i do according to the BGB?",
        "Can you explain the exclusions for certain trips from the package travel contract rules in Section 651a?",
        "I need to return an item I purchased - what does the BGB say about this?",
        "What are my consumer rights according to the BGB?"
    ],
    type="messages"
)

# demo.launch() -- needed to host on server and comment out below code

# === Mount Gradio and static HTML ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app = gr.mount_gradio_app(app, demo, path="/")   

# === To run ===
# uvicorn app:app --reload
