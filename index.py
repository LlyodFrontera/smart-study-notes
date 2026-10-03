import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Study Notes Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Smart Study Notes Generator</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800 p-6 flex flex-col items-center">
  <main class="w-full max-w-3xl space-y-6">
    <header class="text-center">
      <h1 class="text-3xl font-extrabold tracking-tight text-slate-900">Smart Study Notes Generator</h1>
      <p class="text-slate-500 mt-1">Transform dense paragraphs into concise summaries and key takeaways.</p>
    </header>

    <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
      <label class="block font-semibold mb-2 text-slate-700">Source Paragraph</label>
      <textarea id="inputText" rows="6" placeholder="Paste your study material or paragraph here (min 25 words)..." 
        class="w-full p-3 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"></textarea>
      
      <div class="mt-4 flex items-center justify-between">
        <span id="charCount" class="text-xs text-slate-400">0 characters</span>
        <button id="submitBtn" onclick="generateNotes()" 
          class="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-lg transition duration-150">
          Generate Notes
        </button>
      </div>
    </div>

    <div id="statusBox" class="hidden p-4 rounded-lg text-sm"></div>

    <div id="results" class="hidden space-y-6">
      <div class="grid grid-cols-3 gap-4">
        <div class="bg-white p-4 rounded-xl border border-slate-200 text-center">
          <p class="text-xs text-slate-500 uppercase tracking-wider">Original Words</p>
          <p id="origCount" class="text-2xl font-bold text-slate-800 mt-1">0</p>
        </div>
        <div class="bg-white p-4 rounded-xl border border-slate-200 text-center">
          <p class="text-xs text-slate-500 uppercase tracking-wider">Summary Words</p>
          <p id="summaryCount" class="text-2xl font-bold text-slate-800 mt-1">0</p>
        </div>
        <div class="bg-white p-4 rounded-xl border border-emerald-200 bg-emerald-50/50 text-center">
          <p class="text-xs text-emerald-700 uppercase tracking-wider">Reduced By</p>
          <p id="reductionPct" class="text-2xl font-bold text-emerald-600 mt-1">0%</p>
        </div>
      </div>

      <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <h2 class="font-bold text-slate-800 mb-2">Summary</h2>
        <p id="summaryText" class="text-slate-600 leading-relaxed"></p>
      </div>

      <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <h2 class="font-bold text-slate-800 mb-2">Key Takeaways</h2>
        <ul id="keyPointsList" class="list-disc list-inside space-y-2 text-slate-600"></ul>
      </div>
    </div>
  </main>

  <script>
    const input = document.getElementById("inputText");
    input.addEventListener("input", () => {
      document.getElementById("charCount").innerText = `${input.value.length} characters`;
    });

    async function generateNotes() {
      const text = input.value.trim();
      const statusBox = document.getElementById("statusBox");
      const results = document.getElementById("results");
      const btn = document.getElementById("submitBtn");

      if (!text) {
        showStatus("Please paste some text first.", true);
        return;
      }

      btn.disabled = true;
      btn.innerText = "Analyzing...";
      showStatus("Generating summary with AI...", false);
      results.classList.add("hidden");

      try {
        const res = await fetch("/api/summarize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Request failed");

        statusBox.classList.add("hidden");
        document.getElementById("origCount").innerText = data.original_word_count;
        document.getElementById("summaryCount").innerText = data.summary_word_count;
        document.getElementById("reductionPct").innerText = `${data.reduction_percentage}%`;
        document.getElementById("summaryText").innerText = data.summary;
        
        const list = document.getElementById("keyPointsList");
        list.innerHTML = "";
        data.key_points.forEach(pt => {
          const li = document.createElement("li");
          li.innerText = pt;
          list.appendChild(li);
        });

        results.classList.remove("hidden");
      } catch (err) {
        showStatus(err.message, true);
      } finally {
        btn.disabled = false;
        btn.innerText = "Generate Notes";
      }
    }

    function showStatus(msg, isError) {
      const statusBox = document.getElementById("statusBox");
      statusBox.innerText = msg;
      statusBox.className = `p-4 rounded-lg text-sm block ${
        isError ? "bg-red-50 text-red-700 border border-red-200" : "bg-blue-50 text-blue-700 border border-blue-200"
      }`;
    }
  </script>
</body>
</html>
"""

class SummarizeRequest(BaseModel):
    text: str

def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

def extract_key_points(text: str, max_points: int = 4) -> list:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    return sentences[:max_points]

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return HTML_CONTENT

@app.post("/api/summarize")
def summarize(payload: SummarizeRequest):
    input_text = payload.text.strip()
    orig_count = count_words(input_text)

    if orig_count < 20:
        raise HTTPException(status_code=400, detail="Input text must be at least 20 words long.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    hf_payload = {
        "inputs": input_text,
        "parameters": {
            "max_length": max(40, int(orig_count * 0.6)),
            "min_length": 15,
            "do_sample": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=hf_payload, timeout=30)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Model Error: {response.text}")

    result = response.json()
    if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
        summary_text = result[0]["summary_text"]
    else:
        raise HTTPException(status_code=500, detail="Unexpected response format from model.")

    summary_count = count_words(summary_text)
    reduction_percentage = round(((orig_count - summary_count) / orig_count) * 100, 2)
    if reduction_percentage < 0:
        reduction_percentage = 0.0

    return {
        "original_word_count": orig_count,
        "summary_word_count": summary_count,
        "reduction_percentage": reduction_percentage,
        "summary": summary_text,
        "key_points": extract_key_points(summary_text)
    }
