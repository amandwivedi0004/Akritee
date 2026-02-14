from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from fastapi.responses import FileResponse
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat")
def chat(msg: Message):

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": msg.message}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    print(res.text)

    reply = res.json()["choices"][0]["message"]["content"]
    return {"reply": reply}



# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

# Conversation memory
conversation_history = []


def build_history():
    lines=[]
    for m in conversation_history[-6:]:
        if m["role"]=="user":
            lines.append(f"User: {m['text']}")
        else:
            lines.append(f"Akritee: {m['text']}")
    return "\n".join(lines)


@app.post("/chat")
def chat(msg: Message):

    global conversation_history

    conversation_history.append({"role":"user","text":msg.message})

    history_text = build_history()

    # ⭐ CLEAN PROMPT (HINGLISH FIXED)
    prompt = f"""
You are Akritee — a friendly girl chatting casually.

RULES:
- Reply in same language as user (Hindi / Hinglish / English).
- Use only one language per reply, no mixing.
- Short natural replies.
- No assistant tone.
- Avoid repeating greetings.
- Write clean Hinglish with correct spelling.
- Use simple clear sentences.

Previous chat:
{history_text}

User: {msg.message}
Akritee:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.65,
                "top_p": 0.9,
                "repeat_penalty": 1.3,
                "presence_penalty": 1.0,
                "num_predict": 55,
                "num_ctx": 2048,
                "seed": 42
            }
        }
    )

    data = response.json()
    reply = data.get("response","")

    # ⭐ UTF-8 FIX (emoji + broken characters)
    reply = reply.encode("utf-8","ignore").decode("utf-8")

    conversation_history.append({"role":"girl","text":reply})

    return {"reply": reply}
