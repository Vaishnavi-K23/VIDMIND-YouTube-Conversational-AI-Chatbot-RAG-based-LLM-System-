# rag_core.py

import os
import re
import json
from urllib.parse import urlparse, parse_qs
from difflib import SequenceMatcher
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv() 
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ======================
# Config
# ======================
EMBED_MODEL = "intfloat/multilingual-e5-base"
INDEX_DIR = "indexes"
CHUNKS_DIR = "chunks_cache"
QA_MODEL = "llama-3.1-8b-instant"  # Groq-hosted — fast + current (replaces decommissioned llama3-8b-8192)

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ======================
# Embeddings — cached, loaded once per process
# ======================
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )

# ======================
# LLM — Groq API (replaces ChatOllama)
# Reads GROQ_API_KEY from environment variable
# ======================
@lru_cache(maxsize=1)
def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. "
            "Get a free key at https://console.groq.com and set it with:\n"
            "  export GROQ_API_KEY=your_key_here   (Linux/Mac)\n"
            "  set GROQ_API_KEY=your_key_here       (Windows CMD)\n"
            "  $env:GROQ_API_KEY='your_key_here'   (PowerShell)"
        )
    return ChatGroq(
        model=QA_MODEL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=512,
        max_retries=2,
    )

# ======================
# Parsing
# ======================
def parse_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    if "youtu.be" in s:
        path = urlparse(s).path.strip("/")
        return path
    q = parse_qs(urlparse(s).query)
    return q.get("v", [None])[0]

# ======================
# Transcript
# ======================
def get_transcript(video_id: str):
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)

    for lang in ["en", "en-US", "en-GB"]:
        try:
            t = transcript_list.find_transcript([lang])
            return t.fetch(), t.language_code
        except:
            pass

    for t in transcript_list:
        return t.fetch(), t.language_code

# ======================
# Cleaning & Chunking
# ======================
def clean_text(text):
    t = re.sub(r"\[(music|applause|laughter)\]", "", text, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

def build_docs(items):
    docs = []
    last = None
    for item in items:
        txt = clean_text(item.text)
        if not txt or len(txt.split()) < 3:
            continue
        if last and SequenceMatcher(None, txt, last).ratio() > 0.88:
            continue
        docs.append({
            "page_content": txt,
            "metadata": {
                "start": float(item.start),
                "duration": float(item.duration),
            }
        })
        last = txt
    return docs

def window_chunks(docs, window_s=60):
    merged = []
    buf = []
    start_time = docs[0]["metadata"]["start"]
    end_time = start_time

    for d in docs:
        t0 = d["metadata"]["start"]
        t1 = t0 + d["metadata"]["duration"]

        if (t1 - start_time) > window_s:
            merged.append({
                "page_content": " ".join(buf),
                "metadata": {"start": start_time, "end": end_time}
            })
            buf = [d["page_content"]]
            start_time = t0
        else:
            buf.append(d["page_content"])
            end_time = t1

    if buf:
        merged.append({
            "page_content": " ".join(buf),
            "metadata": {"start": start_time, "end": end_time}
        })

    return merged

# ======================
# Index
# ======================
def save_chunks(video_id, chunks):
    with open(f"{CHUNKS_DIR}/{video_id}.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

def load_chunks(video_id):
    path = f"{CHUNKS_DIR}/{video_id}.json"
    if not os.path.exists(path):
        return None
    return json.load(open(path, "r", encoding="utf-8"))

def load_or_create_index(video_id, chunks):
    embeddings = get_embeddings()
    index_path = f"{INDEX_DIR}/{video_id}"

    if os.path.exists(index_path):
        vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        cached = load_chunks(video_id)
        docs = [Document(**c) for c in cached]
        return vs, docs

    save_chunks(video_id, chunks)
    docs = [Document(**c) for c in chunks]
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_path)
    return vs, docs

# ======================
# QA
# ======================
def answer_question(video_id, question, retriever, history, lc_docs=None):
    llm = get_llm()

    # 1) Summary path
    if is_summary_question(question):
        if lc_docs is None:
            return "I don't know based on the transcript.", []
        return summarize_video_fast(lc_docs, llm), []

    # 2) Follow-up rewrite
    rewritten = question
    if history and is_followup_question(question):
        rewritten = condense_question(question, history, llm)

    # 3) Retrieve + answer
    docs = retriever.invoke(rewritten)

    context = "\n\n".join(
        f"[{int(d.metadata.get('start', 0))}s]\n{(d.page_content or '')[:450]}"
        for d in docs
    )

    prompt = f"""You are a helpful assistant. Answer in English using ONLY the transcript context below.
If the answer is not in the context, say: "I don't know based on the transcript."

Transcript context:
{context}

User question: {question}

Answer in 2-5 sentences:"""

    resp = llm.invoke(prompt)
    out = resp.content if hasattr(resp, "content") else str(resp)
    return out.strip(), docs


def is_summary_question(q: str) -> bool:
    ql = (q or "").lower()
    triggers = [
        "summarize", "summary", "what is this video about", "overview",
        "main points", "key takeaways", "tl;dr", "tldr", "recap"
    ]
    return any(t in ql for t in triggers)


def format_history(history, max_turns: int = 6) -> str:
    if not history:
        return ""
    tail = history[-max_turns:]
    return "\n".join([f"User: {q}\nAssistant: {a}" for q, a in tail])


def chunk_list(items, size: int):
    for i in range(0, len(items), size):
        yield items[i:i+size]


def summarize_video_fast(lc_docs, llm, max_sections: int = 10):
    if not lc_docs:
        return "I don't know based on the transcript."

    step = max(1, len(lc_docs) // max_sections)
    sampled = lc_docs[::step][:max_sections]

    context = "\n\n".join(
        f"[{int(d.metadata.get('start', 0))}s]\n{(d.page_content or '')[:500]}"
        for d in sampled
    )

    prompt = f"""Summarize this YouTube video in English using ONLY the transcript excerpts.
Return:
- 1-2 sentence overview
- 6 key takeaways (bullets)

Transcript excerpts:
{context}

Answer:"""
    resp = llm.invoke(prompt)
    out = resp.content if hasattr(resp, "content") else str(resp)
    return out.strip()


def is_followup_question(q: str) -> bool:
    ql = (q or "").lower().strip()
    return (
        len(ql.split()) <= 10 and
        any(x in ql for x in ["that", "this", "he", "she", "they", "it", "those", "there", "him", "her"])
    )


def condense_question(question: str, history, llm) -> str:
    hist = format_history(history, max_turns=4)

    prompt = f"""Rewrite the user's question as a standalone search query (one sentence, English).

Conversation:
{hist}

Question: {question}
Standalone query:"""
    resp = llm.invoke(prompt)
    out = resp.content if hasattr(resp, "content") else str(resp)
    return out.strip()