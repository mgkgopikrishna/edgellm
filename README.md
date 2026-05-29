# 🔒 EdgeLLM — Privacy-First On-Device AI

> **An open source platform where your text NEVER leaves your device — AI that respects your privacy**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white)
![Status](https://img.shields.io/badge/Status-Building-orange?style=flat-square)

---

## 🧒 What Is This? (Explain Like I Am 10 Years Old)

Imagine you have a diary with all your secrets. You want a robot to help you search through your diary and answer questions. But you do NOT want to show the diary to the robot creator!

Normal AI like ChatGPT works like this:
1. You type your text
2. Your text travels to a company server far away
3. The AI reads your text on THEIR computer
4. The answer comes back to you

The problem? Your private information left your device!

**EdgeLLM works differently:**
1. You type your text
2. Your text stays on YOUR computer — it NEVER goes anywhere
3. A special code called a vector is created from your text
4. Only that tiny code travels to the server
5. The AI answers using only that code

Think of it like this: instead of sending a photo of your face to someone, you send only a fingerprint. They can recognize you from the fingerprint without ever seeing your face!

---

## 🧠 The Core Concept — What Are Vectors?

When you write "The dog ran fast" — that sentence has a meaning. A vector captures that meaning as a list of numbers.

```
"The dog ran fast"          → [0.23, -0.87, 0.45, 0.12, -0.34, ...]
"The puppy sprinted quickly" → [0.21, -0.85, 0.44, 0.11, -0.33, ...]
"I love pizza"               → [-0.67, 0.34, -0.12, 0.89, 0.23, ...]
```

Same meaning = similar numbers. Different meaning = very different numbers.
The server only sees numbers — NEVER your actual words!

---

## 🗺️ Complete Architecture

```
YOUR DEVICE (Nothing private leaves here!)
┌─────────────────────────────────────────┐
│  Your Text: "What is my salary?"        │
│          ↓                              │
│  Local Embedding Model                  │
│  (converts text to numbers locally)     │
│          ↓                              │
│  Vector: [0.23, -0.45, 0.67, ...]      │
└─────────────────────────────────────────┘
                    ↓
           Only the VECTOR travels
                    ↓
EDGELLM SERVER (Never sees your actual words!)
┌─────────────────────────────────────────┐
│  Receives Vector                        │
│          ↓                              │
│  Vector Database Search                 │
│          ↓                              │
│  LLM generates answer                   │
│          ↓                              │
│  Sends answer back to you              │
└─────────────────────────────────────────┘
```

---

## 🗺️ Step-By-Step Build Guide

### Step 1 — Understanding the Privacy Problem

When you use ChatGPT or any cloud AI:
- Your medical questions go to OpenAI servers
- Your business documents go to their servers
- Your personal conversations go to their servers

For hospitals, law firms, and banks — this is a huge problem.

**EdgeLLM solution:** Run the understanding part of AI locally. Only send mathematical summaries (vectors) to the server.

---

### Step 2 — Setting Up the Local Embedding Model

**What is an embedding model?**
A special AI that converts text into vectors. We use sentence-transformers/all-MiniLM-L6-v2:
- Very small (only 22MB)
- Runs on CPU — no GPU needed
- Understands meaning very well

```python
# embed_local.py — This runs ON YOUR DEVICE
from sentence_transformers import SentenceTransformer

# Load the model locally — runs offline after first download
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str) -> list:
    """Convert text to vector — NEVER sends text anywhere!"""
    vector = model.encode(text)
    return vector.tolist()

# Example
my_private_text = "My salary is high"
vector = embed_text(my_private_text)
print(f"Vector length: {len(vector)} numbers")
# Text never left your device!
```

---

### Step 3 — Building the FastAPI Server

**What is FastAPI?**
FastAPI is a framework for building web servers in Python. Think of it like a restaurant waiter — it takes requests, processes them, and returns responses.

```python
# server/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import chromadb

app = FastAPI(title="EdgeLLM Server")
client = chromadb.Client()
collection = client.create_collection("documents")

class VectorQuery(BaseModel):
    vector: List[float]   # The vector — NOT the raw text!
    top_k: int = 3

@app.post("/query")
async def query(request: VectorQuery):
    """Answer a question — we only receive a VECTOR, never raw text!"""
    results = collection.query(
        query_embeddings=[request.vector],
        n_results=request.top_k
    )
    return {
        "results": results,
        "privacy": "Your original text was never sent to this server"
    }

@app.get("/health")
async def health():
    return {"status": "running", "privacy": "guaranteed"}
```

---

### Step 4 — The Client Library

```python
# client/edgellm_client.py
from sentence_transformers import SentenceTransformer
import requests

class EdgeLLMClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        # Load embedding model LOCALLY
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def ask(self, question: str) -> str:
        """Ask a question — question text never leaves this device!"""
        # Convert to vector LOCALLY
        question_vector = self.embedder.encode(question).tolist()
        
        # Only the VECTOR is sent
        response = requests.post(
            f"{self.server_url}/query",
            json={"vector": question_vector}
        )
        return response.json()

# Usage
client = EdgeLLMClient("http://localhost:8000")
answer = client.ask("What are my private documents about?")
# Your question text never left your machine!
```

---

### Step 5 — Docker Container

**What is Docker?**
Docker is like a shipping container for software. It packages your app with everything it needs so it runs the same way on any computer.

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t edgellm-server .
docker run -p 8000:8000 edgellm-server
```

---

### Step 6 — Kubernetes Deployment

**What is Kubernetes?**
Kubernetes is like a manager for many Docker containers. It keeps them running, replaces crashed ones, and adds more when busy.

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edgellm-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: edgellm
  template:
    metadata:
      labels:
        app: edgellm
    spec:
      containers:
      - name: edgellm-server
        image: mgkgopikrishna/edgellm-server:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl get pods
```

---

## 📊 Privacy Comparison

| Feature | Normal Cloud AI | EdgeLLM |
|---------|----------------|--------|
| Raw text sent to server | YES — Privacy risk! | NEVER |
| Server sees your questions | YES | NO |
| Works offline | NO | YES |
| Open source | Sometimes | Always |
| GDPR/HIPAA compliant | Depends | By design |

---

## 🛠️ Tech Stack

| Tool | What It Does |
|------|--------------|
| **Python** | Main programming language |
| **FastAPI** | Web server framework |
| **Sentence Transformers** | Local embedding model |
| **ChromaDB** | Vector database |
| **Docker** | Containerization |
| **Kubernetes** | Production orchestration |

---

## 🔧 How To Run This

```bash
git clone https://github.com/mgkgopikrishna/edgellm
cd edgellm
docker-compose up -d
pip install -r client/requirements.txt
python client/example.py
```

---

## 💡 Key Lessons

1. **Privacy by design** — build privacy in from the start
2. **Vectors are powerful** — compare meaning without seeing actual words
3. **Docker makes deployment easy** — package once, run anywhere
4. **Kubernetes handles scale** — from 1 user to 1 million, same code
5. **Open source builds trust** — anyone can verify the privacy claims

---

## 👨‍💻 Built By

**Gopi Krishna Marka** — MLOps Engineer | Cloud Engineer | DevOps Engineer

*Building AI systems that respect human privacy*
