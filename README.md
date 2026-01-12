# README

This application is a sandbox for me to explore LLM tools and frameworks and apply them to a working application. Here I build an AI chatbot and specialise it for different tasks using RAG.

# Chemistry Chat
Chemistry Chat is a Retrieval-Augmented Generation (RAG) AI chatbot built to answer high-school-level chemistry questions with higher factual reliability than a standard LLM chatbot.

The retriever model retrieves relevant content from a curated chemistry knowledge base (past year papers in pdf format) and injects it into the model’s prompt at inference time. This reduces hallucinations and keeps responses aligned with syllabus-level material.

1. Frontend (Rails, Claude-style UI)
   - User submits text or voice input
   - Messages are sent via AJAX (/chemistry_chat/message)
2. Rails Backend
   - Forwards user queries to the Python RAG service which returns structured JSON (answer, sources)
   - No server-side conversation memory required
3. Python RAG Service
   - Built with FastAPI
   - Uses SentenceTransformers for embeddings
   - Stores vectors in ChromaDB (persistent)
   - Retrieves top-k relevant chunks per query
   - Constructs a constrained prompt using retrieved context
4. LLM Inference
   - Uses meta-llama/Llama-3.1-8B-Instruct via Hugging Face API
   - Context-first prompting with strict role instructions
   - Stateless inference for scalability
5. Speech Input
   - Whisper-based transcription endpoint
