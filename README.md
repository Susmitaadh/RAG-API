# RAG – Document-Based AI Assistant

A Retrieval-Augmented Generation (RAG) application built with **FastAPI** that allows users to upload documents, search their content, and interact with an AI assistant using document context and conversation history.

## Features

- **Document Upload:** Supports PDF and TXT files.
- **Text Extraction:** Extracts text from uploaded documents using PyPDF.
- **Text Chunking:** Supports fixed-size and sentence-based chunking strategies.
- **Embeddings:** Generates vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- **Vector Storage:** Stores document embeddings in Qdrant.
- **Document Metadata:** Stores document information in SQLite.
- **Semantic Search:** Retrieves relevant document chunks based on user queries.
- **Conversational RAG:** Uses Gemini to generate context-aware responses.
- **Chat Memory:** Uses Redis to maintain conversation history across multiple turns.
- **Interview Booking:** Extracts interview details from conversations and stores confirmed bookings in SQLite.

## Tech Stack

- Python
- FastAPI
- Sentence Transformers
- Qdrant
- SQLite
- SQLAlchemy
- Redis
- Google Gemini API
- PyPDF
- Pydantic

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/Susmitaadh/palm-mind-rag.git
cd palm-mind-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your own API credentials:

```env
GEMINI_API_KEY=your_gemini_api_key
REDIS_URL=your_redis_url
```

Do not commit your `.env` file or share your API credentials.

### 5. Run the application

```bash
uvicorn main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

### 6. Explore API documentation

Open the interactive FastAPI documentation:

`http://127.0.0.1:8000/docs`

## Author

**Susmita Adhikari**

GitHub: https://github.com/Susmitaadh
