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

### Notes:

- Make sure your `.env` file is included in `.gitignore` to prevent accidental commits.
- The embeddings file is large and proprietary, so please request access separately or generate your own using the embedding script here: https://github.com/nida28/BGBRagBot/blob/development/RAGBaseApp/RAGBaseApp/Program.cs 
- Without these files, the chatbot will not start or will fail to respond correctly.
