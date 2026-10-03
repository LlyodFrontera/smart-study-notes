import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Study Notes Generator")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

class SummarizeRequest(BaseModel):
    text: str

def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

def extract_key_points(text: str, max_points: int = 4) -> list:
    """Extract key takeaways from sentences in the summary."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    return sentences[:max_points]

@app.post("/api/summarize")
def summarize(payload: SummarizeRequest):
    input_text = payload.text.strip()
    orig_count = count_words(input_text)

    if orig_count < 25:
        raise HTTPException(status_code=400, detail="Input text must be at least 25 words long.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    # Payload for Hugging Face BART model
    hf_payload = {
        "inputs": input_text,
        "parameters": {
            "max_length": max(40, int(orig_count * 0.6)),
            "min_length": 20,
            "do_sample": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=hf_payload, timeout=30)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Model Inference API Error: {response.text}"
        )

    result = response.json()
    if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
        summary_text = result[0]["summary_text"]
    else:
        raise HTTPException(status_code=500, detail="Unexpected response format from model.")

    summary_count = count_words(summary_text)
    
    # Calculate percentage reduction
    reduction_percentage = round(((orig_count - summary_count) / orig_count) * 100, 2)
    if reduction_percentage < 0:
        reduction_percentage = 0.0

    key_points = extract_key_points(summary_text)

    return {
        "original_word_count": orig_count,
        "summary_word_count": summary_count,
        "reduction_percentage": reduction_percentage,
        "summary": summary_text,
        "key_points": key_points
    }