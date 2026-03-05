import streamlit as st
from RAG_core import *

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="VIDMIND",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force sidebar always visible — prevents it being hidden by dark CSS
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    min-width: 280px !important;
    transform: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
section[data-testid="stSidebarContent"] {
    display: block !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Custom CSS (dark + modern)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

:root{
  --bg:#07070a;
  --surface:#0f0f14;
  --surface2:#14141c;
  --border:#232332;
  --text:#f3f3f7;
  --muted:#9b9bb2;
  --accent:#ff2a45;
  --accent2:#ff5c6f;
}

html, body, [data-testid="stApp"]{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

#MainMenu, footer, header {visibility:hidden;}

[data-testid="stSidebar"]{
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

h1, h2, h3 {font-family: Syne, sans-serif !important;}

.hero{
  padding: 1.4rem 1.6rem;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, #0d0d14 60%, #140008 100%);
  border-radius: 14px;
  margin-bottom: 1rem;
}
.hero .tag{
  font-family: "IBM Plex Mono", monospace;
  color: var(--accent);
  letter-spacing: .22em;
  text-transform: uppercase;
  font-size: .7rem;
}
.hero .title{
  font-family: Syne, sans-serif;
  font-weight: 800;
  font-size: 2.6rem;
  margin-top: .25rem;
  line-height: 1;
}
.hero .title span{
  color: var(--accent);
  text-shadow: 0 0 28px rgba(255,42,69,.20);
}
.hero .sub{
  margin-top: .45rem;
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: .78rem;
  letter-spacing: .10em;
  text-transform: uppercase;
}

.stTextInput input{
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}

.stButton > button{
  background: var(--accent) !important;
  border: 1px solid transparent !important;
  color: white !important;
  border-radius: 10px !important;
  padding: .55rem 1.05rem !important;
  font-weight: 700 !important;
  letter-spacing: .06em !important;
}
.stButton > button:hover{
  background: var(--accent2) !important;
}

.chip{
  display:inline-block;
  padding:.25rem .55rem;
  border: 1px solid var(--border);
  background: rgba(255,255,255,.03);
  border-radius: 999px;
  font-family:"IBM Plex Mono", monospace;
  font-size:.72rem;
  color: var(--muted);
  margin-right: .35rem;
}
.chip b{color: var(--text); font-weight: 600;}
.small-note{
  color: var(--muted);
  font-size: .85rem;
}
hr{border-color: var(--border) !important;}

[data-testid="stChatMessage"] {
    color: #ffffff !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] span {
    color: #f5f5f5 !important;
}
[data-testid="stChatMessage"] strong {
    color: #ffffff !important;
}
[data-testid="stChatMessage"] li {
    color: #eaeaf5 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "history" not in st.session_state:
    st.session_state.history = []
if "lc_docs" not in st.session_state:
    st.session_state.lc_docs = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "lang" not in st.session_state:
    st.session_state.lang = None
if "chunks_n" not in st.session_state:
    st.session_state.chunks_n = 0

# -----------------------------
# Sidebar: Load video
# -----------------------------
with st.sidebar:
    st.markdown("### ⬡ VIDMIND")
    st.markdown('<div class="small-note">Paste a YouTube URL/ID, build the index, then chat.</div>', unsafe_allow_html=True)
    youtube_url = st.text_input("YouTube URL or Video ID", placeholder="https://youtube.com/watch?v=...")

    colA, colB = st.columns(2)
    with colA:
        load = st.button("Load Video", use_container_width=True)
    with colB:
        clear = st.button("Clear Chat", use_container_width=True)

    if clear:
        st.session_state.history = []
        st.rerun()

    if load and youtube_url.strip():
        video_id = parse_video_id(youtube_url)

        if video_id != st.session_state.video_id:
            st.session_state.history = []

        st.session_state.video_id = video_id

        with st.spinner("Processing video (transcript → chunks → index)..."):
            items, lang = get_transcript(video_id)
            docs = build_docs(items)
            chunks = window_chunks(docs)
            vs, lc_docs = load_or_create_index(video_id, chunks)

            st.session_state.lc_docs = lc_docs
            st.session_state.retriever = vs.as_retriever(search_kwargs={"k": 3})
            st.session_state.lang = lang
            st.session_state.chunks_n = len(chunks)

        st.success("Ready!")

    st.markdown("---")
    if st.session_state.video_id:
        st.markdown(f"**Active Video**: `{st.session_state.video_id}`")
        st.markdown(
            f'<span class="chip"><b>lang</b> {str(st.session_state.lang).upper()}</span>'
            f'<span class="chip"><b>chunks</b> {st.session_state.chunks_n}</span>',
            unsafe_allow_html=True
        )

# -----------------------------
# Main: Hero + video + chat
# -----------------------------
st.markdown("""
<div class="hero">
  <div class="tag">AI-powered chat</div>
  <div class="title">VID<span>MIND</span></div>
  <div class="sub">// paste link → ask anything → get grounded answers</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.video_id:
    vid = st.session_state.video_id
    st.markdown("#### Video")
    st.video(f"https://www.youtube.com/watch?v={vid}")
    st.markdown("---")
else:
    st.info("Paste a YouTube link in the sidebar and click **Load Video** to start.")
    st.stop()

# Render chat history
for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

# Chat input
question = st.chat_input("Ask a question about the video...")

if question:
    st.chat_message("user").write(question)

    with st.spinner("Thinking..."):
        answer, docs_used = answer_question(
            st.session_state.video_id,
            question,
            st.session_state.retriever,
            st.session_state.history,
            st.session_state.lc_docs
        )

    st.chat_message("assistant").write(answer)

    # FIX: reuse docs_used returned by answer_question — no second retriever.invoke() call
    if docs_used:
        with st.expander("Sources"):
            for d in docs_used[:4]:
                t = int(d.metadata.get("start", 0))
                st.markdown(f"- [{t}s](https://www.youtube.com/watch?v={vid}&t={t}s) — {(d.page_content or '')[:140]}…")

    st.session_state.history.append((question, answer))