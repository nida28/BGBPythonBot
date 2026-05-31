# BGB Bot
BGB Legal ChatBot is a friendly legal assistant focused on the German Civil Code (BGB). Using Retrieval-Augmented Generation (RAG), it provides clear, practical answers to questions about tenant rights, contracts, consumer protections, and more—tailored especially for expats in Germany. The bot references official BGB sections with clickable links for easy access to the original legal texts. Hosted at https://huggingface.co/spaces/nfm1708/BGBChatBot.

## Setup Instructions

This repository **does not include** the `.env` file or the embeddings JSONL file to protect sensitive API keys and proprietary data.

### What you need to add manually:

1. **Create a `.env` file** in the project root with your OpenAI API key:

    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    ```

2. **Add the embeddings file** to the root folder as:

    ```bash
    bgb_embeddings_new_data.jsonl
    ```

This file contains the precomputed embeddings used by the application and is required for the chatbot to function.

---

## Repository layout (reorganized)

Top-level structure now organizes code, docs, and data for clarity:

- `app.py` — FastAPI + Gradio application entrypoint (root)
- `src/bgbpythonbot/` — Python package containing library modules (`document_parser.py`, `document_store.py`)
- `tests/` — Test suite (`test_document_handling.py`)
- `scripts/` — Helper scripts (`build_validate.py`, etc.)
- `docs/` — Markdown documentation and reports (moved from root)
- `data/` — Large or binary data files (embeddings, samples)

This layout keeps executable entrypoints at the project root while packaging reusable modules under `src/`.


### Notes:

- Make sure your `.env` file is included in `.gitignore` to prevent accidental commits.
- The embeddings file is large and proprietary, so please request access separately or generate your own using the embedding script [**here**](https://github.com/nida28/BGBRagBot/blob/development/RAGBaseApp/RAGBaseApp/Program.cs).
- Without these files, the chatbot will not start or will fail to respond correctly.

---

## Document Upload Feature

The chatbot now supports uploading personal documents (PDFs or DOCX files) for context-aware legal analysis. When you upload a document, the bot can reference it alongside BGB sections when answering your questions.

### Supported File Types
- **PDF** (.pdf) — Extracts text from all pages
- **DOCX** (.docx) — Extracts text from all paragraphs

### How It Works

1. **Upload** — Attach a PDF or DOCX file in the chat message input (max 10MB)
2. **Parse** — The system extracts text and splits it into searchable chunks (512 tokens with 50-token overlap)
3. **Embed** — Each chunk is converted to an embedding for semantic search
4. **Search** — When you ask a question, the bot searches **both** your uploaded document and the BGB database
5. **Answer** — The bot cites specific text from your document along with relevant BGB sections

### Example Workflow

**Scenario:** You received a letter from your landlord about rent increases and want to know your rights.

1. Upload the landlord's letter as a PDF
2. Ask: "Check this letter from my landlord. Per my rights, can he do this?"
3. The bot will:
   - Extract key phrases from the letter
   - Search your uploaded document for relevant context
   - Search the BGB for applicable tenant rights
   - Provide an answer citing both sources

**Output Example:**
> According to the letter you provided: *"Your rent will increase by €100 starting next month"* — the BGB allows this under certain conditions in **[Section 558](...)** which states...

### Limitations

- **One file at a time:** Uploading a new document replaces the previous one
- **In-memory storage:** Uploaded documents are stored in the app's memory and lost when the app restarts
- **File size limit:** Maximum 10MB per document
- **Text extraction:** Complex layouts (multi-column PDFs, tables) may lose formatting
- **Language:** Best performance with German or English text

### Installation

No additional setup needed beyond the main installation. The required libraries are:
- `PyPDF2` — for PDF text extraction
- `python-docx` — for DOCX text extraction
- `python-multipart` — for file upload handling

---

## Running the Application

### Installation
```bash
pip install -r requirements.txt
```

### Start the server
```bash
uvicorn app:app --reload
```

The app will be available at `http://localhost:8000`

Note: `app.py` automatically adds `src/` to `sys.path` so the `bgbpythonbot` package is importable at runtime. You can also run with `PYTHONPATH=src` if you prefer.

### Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Gradio Multimodal Chat Interface                                │
│ User: [Attach file] + "Check this letter from my landlord..."  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │ handle_message_with_upload()    │
        │ Detects file in message         │
        └────────────────┬────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │ Document Parser Module                              │
    │ • Extract text from file (PDF/DOCX)                 │
    │ • Normalize formatting                              │
    │ • Split into 512-token chunks with 50-token overlap │
    └────────────────┬────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────┐
    │ Embedding Generation (OpenAI)                       │
    │ • Generate vector embedding for each chunk          │
    │ • Store in-memory (DocumentStore)                   │
    └────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌─────────────┐          ┌──────────────────┐
    │  Uploaded   │          │  BGB Database    │
    │  Document   │          │  (30K+ sections) │
    │  Chunks     │          │                  │
    └─────────────┘          └──────────────────┘
         │                          │
         └───────────┬──────────────┘
                     │
        ┌────────────┴────────────┐
        │ Embed user query        │
        │ (same query text)       │
        └────────────┬────────────┘
                     │
        │ Search both sources using       │
        │ cosine similarity               │
        └────────────┬────────────────────┘
                     │
         ┌───────────┴──────────────┐
         │                          │
         ▼                          ▼
    ┌─────────────┐          ┌──────────────────┐
    │ Top matches │          │ Top matches      │
    │ from upload │          │ from BGB         │
    └─────────────┘          └──────────────────┘
         │                          │
         └───────────┬──────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │ Merge results, select top 3     │
        │ total (best scoring across both)│
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │ Build prompt with citations:    │
        │ • [Uploaded: filename - text]   │
        │ • [Section §XXX - BGB link]     │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │ Send to GPT-4o for final answer │
        └────────────┬────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────┐
        │ Return answer with citations    │
        │ linking to both sources         │
        └─────────────────────────────────┘
```
