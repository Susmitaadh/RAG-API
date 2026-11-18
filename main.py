import os
import io
import json
import certifi
from pathlib import Path
from uuid import uuid4

import redis
from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

from database import engine, SessionLocal, Base
from document_model import Document, InterviewBooking


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not REDIS_URL or not REDIS_URL.startswith(("redis://", "rediss://")):
    raise ValueError(
        "Invalid REDIS_URL. Use your Upstash Redis connection URL."
    )

app = FastAPI(title="Palm Mind RAG API")

Base.metadata.create_all(bind=engine)

model = SentenceTransformer("all-MiniLM-L6-v2")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.5-flash-lite"

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    ssl_ca_certs=certifi.where(),
    socket_connect_timeout=15,
    socket_timeout=15
)

CHAT_MEMORY_TTL = 86400  

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

client = QdrantClient(path="./qdrant_storage")

COLLECTION_NAME = "documents"

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

class User(BaseModel):
    name: str
    email: str

class Product(BaseModel):
    name: str
    price: float

class SearchRequest(BaseModel):
    question: str
    limit: int = Field(default=5, ge=1, le=20)

class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    limit: int = Field(default=5, ge=1, le=20)

class BookingRequest(BaseModel):
    message: str
    conversation_id: str | None = None

@app.get("/")
def home():
    return {
        "message": "Welcome to Palm Mind RAG API"
    }

@app.get("/user/")
def get_users():
    return {"message": "List of users"}

@app.get("/user/{user_id}")
def get_user(user_id: int):
    return {
        "message": "User details",
        "user_id": user_id
    }

@app.post("/users")
def create_user(user: User):
    return {
        "message": "User created",
        "user": user
    }

@app.get("/documents")
def get_documents(page: int = 1, limit: int = 10):
    return {
        "page": page,
        "limit": limit
    }

@app.get("/product/{product_id}")
def get_product(product_id: int):
    return {"product_id": product_id}

@app.post("/product")
def create_product(product: Product):
    return {
        "message": "Product created",
        "product": product
    }

def fixed_size_chunking(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def sentence_based_chunking(text):
    sentences = text.split(".")
    chunks = []

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence:
            chunks.append(sentence)

    return chunks

def generate_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings.tolist()

def store_embeddings(chunks, embeddings, filename):
    points = []

    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": filename
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(points)

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("fixed")
):
    filename = file.filename

    if not filename or not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are allowed"
        )

    content = await file.read()

    
    try:
        if filename.lower().endswith(".pdf"):
            pdf_reader = PdfReader(io.BytesIO(content))
            text = ""

            for page in pdf_reader.pages:
                text += (page.extract_text() or "") + "\n"

        else:
            text = content.decode("utf-8")

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read file: {str(error)}"
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in document"
        )

    if strategy == "fixed":
        chunks = fixed_size_chunking(text, 500)

    elif strategy == "sentence":
        chunks = sentence_based_chunking(text)

    else:
        raise HTTPException(
            status_code=400,
            detail="Use fixed or sentence as chunking strategy"
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text chunks generated"
        )

    extension = Path(filename).suffix.lower()
    unique_filename = f"{uuid4()}{extension}"

    file_path = UPLOAD_DIR / unique_filename
    file_path.write_bytes(content)

    embeddings = generate_embeddings(chunks)

    stored_count = store_embeddings(
        chunks,
        embeddings,
        filename
    )

    db = SessionLocal()

    try:
        new_document = Document(
            filename=filename,
            saved_filename=unique_filename,
            file_type=extension.replace(".", "").upper(),
            strategy=strategy,
            total_chunks=len(chunks)
        )

        db.add(new_document)
        db.commit()
        db.refresh(new_document)

        document_id = new_document.id

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Could not save document metadata: {str(error)}"
        )

    finally:
        db.close()

    return {
        "message": "Document uploaded and processed successfully",
        "document_id": document_id,
        "filename": filename,
        "saved_filename": unique_filename,
        "strategy": strategy,
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]),
        "stored_vectors": stored_count
    }

@app.post("/documents/search")
def search_documents(request: SearchRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    query_embedding = model.encode(
        request.question
    ).tolist()

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=request.limit,
        with_payload=True
    )

    results = []

    for point in search_result.points:
        payload = point.payload or {}

        results.append({
            "text": payload.get("text", ""),
            "filename": payload.get("filename", ""),
            "similarity_score": point.score
        })

    return {
        "question": request.question,
        "total_results": len(results),
        "results": results
    }

@app.post("/chat")
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    conversation_id = request.conversation_id or str(uuid4())

    memory_key = f"chat:{conversation_id}"

    try:
        stored_history = redis_client.get(memory_key)

        if stored_history:
            chat_history = json.loads(stored_history)
        else:
            chat_history = []

    except redis.RedisError as error:
        print("REDIS READ ERROR:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Redis connection failed. Check your Redis URL."
        )

    query_embedding = model.encode(
        request.question
    ).tolist()

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=request.limit,
        with_payload=True
    )

    context = ""
    sources = []

    for point in search_result.points:
        payload = point.payload or {}

        context += payload.get("text", "") + "\n\n"

        sources.append({
            "filename": payload.get("filename", ""),
            "similarity_score": point.score
        })

    if not context.strip():

        answer = (
            "I couldn't find relevant information "
            "in the uploaded documents."
        )

    else:

        history_text = ""

        for message in chat_history:
            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        prompt =  f"""
        You are a helpful conversational RAG assistant.

        You have two sources of information:

        1. Conversation History:
        Use this to understand previous messages, references,
        and follow-up questions.

        2. Document Context:
        Use this to answer questions about uploaded documents.

        IMPORTANT RULES:
        - Always consider both conversation history and document context.
        - If the user provides information in the conversation,
        you may use it to answer follow-up questions.
        - Do not claim that user-provided information came from
        the uploaded document.
        - If information is missing from both sources, say so.
        - Do not invent facts.

        Previous Conversation:
        {history_text}

        Document Context:
        {context}

        Current Question:
        {request.question}

        Answer:
        """

        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            answer = response.text or "I could not generate an answer."

        except Exception as error:
            print("GEMINI ERROR:", repr(error))
            raise HTTPException(
                status_code=503,
                detail="Gemini API request failed. Please try again."
            )

    chat_history.append({
        "role": "user",
        "content": request.question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    chat_history = chat_history[-10:]

    try:
        redis_client.set(
            memory_key,
            json.dumps(chat_history),
            ex=CHAT_MEMORY_TTL
        )

    except redis.RedisError as error:
        print("REDIS SAVE ERROR:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Could not save conversation memory."
        )

    return {
        "conversation_id": conversation_id,
        "question": request.question,
        "answer": answer,
        "sources": sources
    }

@app.get("/redis-health")
def redis_health():

    try:
        redis_client.ping()

        return {
            "status": "connected",
            "message": "Redis connection successful"
        }

    except redis.RedisError as error:
        print("REDIS HEALTH ERROR:", repr(error))

        raise HTTPException(
            status_code=503,
            detail="Redis connection failed"
        )

@app.post("/book-interview")
def book_interview(request: BookingRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    conversation_id = request.conversation_id or str(uuid4())

    booking_key = f"booking:{conversation_id}"

    try:
        stored_booking = redis_client.get(booking_key)

        if stored_booking:
            booking_data = json.loads(stored_booking)
        else:
            booking_data = {
                "name": None,
                "email": None,
                "interview_date": None,
                "interview_time": None
            }

    except redis.RedisError as error:
        print("BOOKING REDIS ERROR:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve booking information"
        )

    prompt = f"""
    You are an interview booking assistant.

    Extract the following information from the user's message:
    - name
    - email
    - interview_date
    - interview_time

    Existing booking information:
    {json.dumps(booking_data)}

    User message:
    {request.message}

    Instructions:
    1. Keep previously provided information if not updated.
    2. Extract any new information from the user's message.
    3. Do not invent missing information.
    4. Return ONLY a valid JSON object.
    5. Use null for missing values.

    Required JSON format:
    {{
        "name": null,
        "email": null,
        "interview_date": null,
        "interview_time": null
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        extracted_data = json.loads(
            response.text.strip().replace("```json", "").replace("```", "")
        )

    except Exception as error:
        print("BOOKING GEMINI ERROR:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Could not process booking information"
        )

    for field in booking_data:
        value = extracted_data.get(field)

        if value:
            booking_data[field] = value

    missing_fields = [
        field for field, value in booking_data.items()
        if not value
    ]

    try:
        redis_client.set(
            booking_key,
            json.dumps(booking_data),
            ex=CHAT_MEMORY_TTL
        )

    except redis.RedisError as error:
        print("BOOKING REDIS SAVE ERROR:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Could not save booking information"
        )

    if missing_fields:
        return {
            "conversation_id": conversation_id,
            "status": "incomplete",
            "message": "Please provide the following information: "
                       + ", ".join(missing_fields),
            "booking_details": booking_data
        }

    if "@" not in booking_data["email"]:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid email address"
        )

    db = SessionLocal()

    try:
        new_booking = InterviewBooking(
            conversation_id=conversation_id,
            name=booking_data["name"],
            email=booking_data["email"],
            interview_date=booking_data["interview_date"],
            interview_time=booking_data["interview_time"],
            status="confirmed"
        )

        db.add(new_booking)
        db.commit()
        db.refresh(new_booking)

        booking_id = new_booking.id

        redis_client.delete(booking_key)

        return {
            "conversation_id": conversation_id,
            "booking_id": booking_id,
            "status": "confirmed",
            "message": "Interview booked successfully!",
            "booking_details": booking_data
        }

    except Exception as error:
        db.rollback()
        print("BOOKING DATABASE ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Could not save interview booking"
        )

    finally:
        db.close()

@app.get("/interview-bookings")
def get_interview_bookings():

    db = SessionLocal()

    try:
        bookings = db.query(InterviewBooking).all()

        return {
            "total_bookings": len(bookings),
            "bookings": [
                {
                    "id": booking.id,
                    "name": booking.name,
                    "email": booking.email,
                    "interview_date": booking.interview_date,
                    "interview_time": booking.interview_time,
                    "status": booking.status
                }
                for booking in bookings
            ]
        }

    finally:
        db.close()
