VIDMIND — Chat with Any YouTube Video

VIDMIND is an AI system that allows users to chat with any YouTube video by querying its transcript.

Instead of watching an entire video, users can ask questions and receive answers grounded in the transcript with timestamped sources.

The system uses a Retrieval-Augmented Generation (RAG) pipeline that combines transcript processing, hybrid retrieval, and large language models to generate accurate responses.

Demo

Example workflow:

Paste a YouTube video URL

Transcript is extracted automatically

Transcript is processed and indexed

Ask questions about the video

Receive answers with timestamped sources

Example query:

What does the speaker say about AGI risk?

The system retrieves the relevant transcript segments and generates a grounded answer.

Features

• Chat with any YouTube video
• Transcript extraction and processing
• Hybrid retrieval (semantic + keyword search)
• Timestamped source citations
• Follow-up question support
• Multilingual transcript support
• Fast inference using Groq LLM API
• Interactive chat UI built with Streamlit

System Architecture

VIDMIND uses a Retrieval-Augmented Generation (RAG) architecture.

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
Embedding Generation
(multilingual-e5-base)
 ↓
Vector Storage (FAISS)
 ↓
Hybrid Retrieval
   ├ Semantic Search (FAISS)
   └ Keyword Search (BM25)
 ↓
Reciprocal Rank Fusion
 ↓
Context Selection
 ↓
Groq LLaMA Model
 ↓
Answer Generation
 ↓
Answer + Timestamp Sources

How It Works
1. Transcript Extraction

The system retrieves subtitles using:

youtube_transcript_api

This provides the caption text and timestamps for each segment.

2. Transcript Cleaning

Transcript data is cleaned to remove noise such as:

[music]
[applause]
[laughter]

Duplicate captions are filtered using text similarity checks.

3. Window-Based Chunking

Captions are merged into time-based chunks (~60 seconds).

Example chunk:

{
 page_content: "...transcript text...",
 metadata:
   start: 120
   end: 180
}

Benefits:

• Better semantic context
• Fewer embeddings
• Improved retrieval quality

4. Embedding Generation

Transcript chunks are converted into vector embeddings using:

intfloat/multilingual-e5-base

This model supports multilingual retrieval, allowing users to query videos in different languages.

5. Vector Indexing

Embeddings are stored in a FAISS vector index.

Directory structure:

indexes/
   video_id/
      index.faiss

Benefits:

• Fast similarity search
• Persistent storage
• Quick reload for previously processed videos

6. Hybrid Retrieval

VIDMIND uses hybrid retrieval to improve search accuracy.

Semantic Search

Uses FAISS vector similarity.

Best for:

• paraphrased questions
• semantic meaning

Keyword Search

Uses BM25 keyword matching.

Best for:

• names
• numbers
• exact phrases

Example:

"Sam Altman"
"300 million workers"
7. Reciprocal Rank Fusion

Results from both retrieval methods are combined using:

Reciprocal Rank Fusion (RRF)

This improves ranking stability and retrieval accuracy.

8. Answer Generation

Relevant transcript chunks are sent to the LLM with the user query.

Model used:

Groq
llama-3.1-8b-instant

The LLM generates an answer based only on retrieved transcript context.

9. Source Attribution

Each answer includes timestamped sources.

Example:

[120s–180s] → link to video timestamp

Clicking the source jumps directly to the relevant part of the video.

Performance Optimizations

Early versions of the system had very slow responses (~3–4 minutes).

Several improvements significantly reduced latency:

• Switched from local LLM inference to Groq GPU inference
• Cached embedding model
• Persisted FAISS indexes
• Cached transcript chunks
• Reduced retrieval size (k = 6 → k = 3)
• Removed duplicate retrieval calls
• Optimized prompt structure

Final performance:

Response time: ~1–3 seconds
Running Locally
1. Clone the repository
git clone https://github.com/yourusername/vidmind.git
cd vidmind
2. Install dependencies
pip install -r requirements.txt
3. Add environment variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
4. Run the app
streamlit run app.py

Open:

http://localhost:8501
Project Structure
vidmind/
│
├── app.py
├── RAG_core.py
├── requirements.txt
├── README.md
├── indexes/
├── chunks_cache/
Limitations

Some videos may not have transcripts available.

Possible reasons:

• captions disabled
• private videos
• YouTube blocking requests from cloud IPs

Future Improvements

Possible extensions:

• streaming responses
• multi-video knowledge base
• timeline topic extraction
• automatic video summaries
• improved UI interaction

Lessons Learned

Building this system highlighted several important insights:

• Retrieval quality can be as important as the LLM itself
• Hybrid retrieval improves search robustness
• Context size significantly impacts latency
• Caching is essential for production systems
• UX improvements strongly affect perceived performance
