# backend.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pandas as pd
from transformers import pipeline

app = FastAPI()

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            content = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
        content = df.to_string()
    else:
        return {"error": "Unsupported file format"}

    return {"text": content}


@app.post("/ask/")
async def ask_question(question: str = Form(...), context: str = Form(...), user_type: str = Form(...)):
    # Adjust tone (optional)
    if user_type == "student":
        question = f"Explain simply for a student: {question}"
    elif user_type == "professional":
        question = f"Answer professionally: {question}"

    result = qa_pipeline(question=question, context=context)
    return {"answer": result["answer"]}
