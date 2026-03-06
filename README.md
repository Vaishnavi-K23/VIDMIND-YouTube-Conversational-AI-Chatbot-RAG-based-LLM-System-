# VIDMIND — Chat With Any YouTube Video

VIDMIND is an AI-powered system that lets you chat with any YouTube video using its transcript. Instead of watching an entire video, ask questions and get answers grounded in the transcript with timestamped sources.

---

## Demo

1. Paste a YouTube video URL
2. The transcript is extracted and indexed automatically
3. Ask questions about the video
4. Receive answers with timestamped source links

**Example query:**
> *"What does the speaker say about AGI risk?"*

The system retrieves the most relevant transcript segments and generates a grounded answer with links to the exact moments in the video.

---

## Features

- 💬 Chat with any YouTube video
- 📄 Automatic transcript extraction and cleaning
- 🔍 Hybrid retrieval (semantic + keyword search)
- 🕐 Timestamped source citations with clickable links
- 🔄 Follow-up question support
- 🌍 Multilingual transcript support
- ⚡ Fast inference via Groq LLM API
- 🖥️ Interactive chat UI built with Streamlit

---

## System Architecture

VIDMIND uses a Retrieval-Augmented Generation (RAG) pipeline:

```
User
 ↓
Streamlit UI
 ↓
YouTube URL Input
 ↓
Transcript Extraction
 ↓
Transcript Cleaning
 ↓
Window-Based Chunking
 ↓
Embedding Generation (multilingual-e5-base)
 ↓
Vector Storage (FAISS)
 ↓
Hybrid Retrieval
   ├── Semantic Search (FAISS)
   └── Keyword Search (BM25)
 ↓
Reciprocal Rank Fusion
 ↓
Context Selection
 ↓
Groq LLaMA Model
 ↓
Answer + Timestamped Sources
```

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Transcript Extraction | youtube-transcript-api |
| Embeddings | intfloat/multilingual-e5-base |
| Vector Database | FAISS |
| Keyword Search | BM25 |
| LLM | Groq (Llama-3.1-8B-Instant) |
| Framework | LangChain |

---

## How It Works

### 1. Transcript Extraction
Subtitles are retrieved using `youtube_transcript_api`, providing caption text and timestamps for each segment.

### 2. Transcript Cleaning
Noise is removed from the transcript (e.g. `[music]`, `[applause]`), and duplicate captions are filtered using text similarity checks.

### 3. Window-Based Chunking
Captions are merged into ~60-second time-based chunks:

```json
{
  "page_content": "...transcript text...",
  "metadata": {
    "start": 120,
    "end": 180
  }
}
```

This provides better semantic context, fewer embeddings, and improved retrieval quality.

### 4. Embedding Generation
Transcript chunks are converted to vector embeddings using `intfloat/multilingual-e5-base`, enabling multilingual retrieval.

### 5. Vector Indexing
Embeddings are stored in a persistent FAISS index:

```
indexes/
└── {video_id}/
    └── index.faiss
```

Previously processed videos reload instantly from the cached index.

### 6. Hybrid Retrieval
Two retrieval methods are combined for better accuracy:

- **Semantic Search (FAISS)** — Best for paraphrased questions and conceptual meaning
- **Keyword Search (BM25)** — Best for names, numbers, and exact phrases (e.g. *"Sam Altman"*, *"300 million workers"*)

### 7. Reciprocal Rank Fusion
Results from both methods are merged using Reciprocal Rank Fusion (RRF) for improved ranking stability.

### 8. Answer Generation
Relevant chunks are passed to the LLM along with the user query. The model generates an answer based solely on the retrieved transcript context.

**Model:** `Groq / llama-3.1-8b-instant`

### 9. Source Attribution
Every answer includes clickable timestamped sources, e.g.:

> `[120s–180s]` → jumps directly to that moment in the video

---

## Getting Started

### Prerequisites
- Python 3.8+
- A [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vidmind.git
cd vidmind

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
vidmind/
├── app.py            # Streamlit UI
├── RAG_core.py       # RAG pipeline logic
├── requirements.txt
├── README.md
├── indexes/          # Persisted FAISS indexes
└── chunks_cache/     # Cached transcript chunks
```

---

## Performance

Early versions had response times of 3–4 minutes. Several optimizations reduced this to **~1–3 seconds**:

- Switched from local LLM to Groq GPU inference
- Cached embedding model and transcript chunks
- Persisted FAISS indexes for previously seen videos
- Reduced retrieval size (`k = 6 → k = 3`)
- Removed duplicate retrieval calls
- Optimized prompt structure

---

## Limitations

Some videos may not have transcripts available due to:
- Captions being disabled by the uploader
- Private or unlisted videos
- YouTube blocking requests from cloud/server IPs

---

## Roadmap

- [ ] Streaming responses
- [ ] Multi-video knowledge base
- [ ] Timeline topic extraction
- [ ] Automatic video summaries
- [ ] Improved UI interactions

---

## Lessons Learned

- Retrieval quality can matter as much as the LLM itself
- Hybrid retrieval meaningfully improves search robustness
- Context size significantly impacts latency
- Caching is essential for production-grade systems
- UX improvements strongly affect perceived performance

---

## License

MIT
