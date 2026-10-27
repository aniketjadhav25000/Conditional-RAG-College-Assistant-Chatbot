import streamlit as st
import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] { display: none; }

/* ── Top header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 28px 0 20px;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 8px;
}
.app-header .icon {
    font-size: 2.2rem;
    line-height: 1;
}
.app-header h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.app-header p {
    margin: 2px 0 0;
    font-size: 0.82rem;
    color: #5c6380;
}

/* ── Programme badge ── */
.prog-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a1f2e;
    border: 1px solid #2a3050;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    color: #7c8db5;
    margin-bottom: 20px;
}
.prog-badge strong { color: #a5b4fc; }

/* ── Chat messages ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 16px 0;
}

.msg-row {
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar.ai   { background: #1e2d4a; }
.avatar.user { background: #1f1b3a; }

.bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 0.88rem;
    line-height: 1.6;
}
.bubble.ai {
    background: #161b27;
    border: 1px solid #1e2540;
    color: #d0d5e8;
    border-top-left-radius: 4px;
}
.bubble.user {
    background: #1a1042;
    border: 1px solid #2d1f6e;
    color: #cdc5f5;
    border-top-right-radius: 4px;
}

/* ── Tag chips ── */
.tag {
    display: inline-block;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 10px;
    margin-bottom: 5px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.tag-academic { background: #0d2b1f; color: #4ade80; border: 1px solid #1a4a30; }
.tag-fee      { background: #2b1a08; color: #fb923c; border: 1px solid #4a300d; }
.tag-general  { background: #0d1a2b; color: #60a5fa; border: 1px solid #1a2f4a; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1a1f2e; margin: 10px 0; }

/* ── Input box override ── */
div[data-testid="stChatInput"] > div {
    background: #161b27 !important;
    border: 1px solid #2a3050 !important;
    border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea {
    color: #e8eaf0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: #312e81;
    color: #c7d2fe;
    border: 1px solid #4338ca;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 8px 18px;
    transition: background 0.18s;
}
div.stButton > button:hover {
    background: #3730a3;
    color: #e0e7ff;
}

/* ── Programme selector ── */
.programme-card {
    background: #161b27;
    border: 1px solid #1e2540;
    border-radius: 14px;
    padding: 28px 24px 32px;
    margin-top: 12px;
}
.programme-card h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 6px;
}
.programme-card p {
    font-size: 0.82rem;
    color: #5c6380;
    margin-bottom: 20px;
}

div[data-testid="stRadio"] label {
    color: #9aa3c2 !important;
    font-size: 0.88rem !important;
}
div[data-testid="stRadio"] div[role="radio"][aria-checked="true"] + label {
    color: #a5b4fc !important;
    font-weight: 600 !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px 0;
    color: #3a4060;
    font-size: 0.88rem;
}
.empty-state .big { font-size: 2.5rem; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKEND  (unchanged from original)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_graph():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_retriever(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(document)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)

    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        qurey_type: str
        retrieved_context: str

    def classifier_node(state: State) -> dict:
        last_message = state["messages"][-1].content
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )
        response = llm.invoke(prompt)
        category = response.content.strip().lower()
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"
        return {"qurey_type": category}

    def academic_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]
        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer."
            )
        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    def route_query(state: State):
        if state["qurey_type"] == "academic":
            return "academic_rag"
        elif state["qurey_type"] == "fee":
            return "fee_rag"
        else:
            return "general"

    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)
    return graph.compile()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "programme" not in st.session_state:
    st.session_state.programme = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role", "content", "tag"}


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="icon">🎓</div>
    <div>
        <h1>College Assistant</h1>
        <p>Ask about academics, fees, or anything campus-related</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PROGRAMME SELECTION SCREEN
# ─────────────────────────────────────────────
if st.session_state.programme is None:
    st.markdown('<div class="programme-card">', unsafe_allow_html=True)
    st.markdown("**Which programme are you enrolled in?**")
    st.markdown('<p>This helps personalise answers from college documents.</p>', unsafe_allow_html=True)

    choice = st.radio(
        label="Programme",
        options=["BCA", "BBA", "B.Com (H)"],
        label_visibility="collapsed",
    )

    if st.button("Start chatting →"):
        st.session_state.programme = choice
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# CHAT SCREEN
# ─────────────────────────────────────────────
programme = st.session_state.programme

# Programme badge + reset
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown(
        f'<div class="prog-badge">Logged in as <strong>{programme}</strong> student</div>',
        unsafe_allow_html=True,
    )
with col2:
    if st.button("Switch", help="Change your programme"):
        st.session_state.programme = None
        st.session_state.chat_history = []
        st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Tag colour helper ──
TAG_CLASSES = {"academic": "tag-academic", "fee": "tag-fee", "general": "tag-general"}
TAG_LABELS  = {"academic": "📚 Academic", "fee": "💰 Fee", "general": "💬 General"}

# ── Render history ──
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <div class="big">💬</div>
        Ask about attendance rules, fee payments, exam schedules, or anything else.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar user">🧑</div>
                <div class="bubble user">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            tag_html = ""
            if msg.get("tag"):
                cls = TAG_CLASSES.get(msg["tag"], "tag-general")
                lbl = TAG_LABELS.get(msg["tag"], msg["tag"].title())
                tag_html = f'<div><span class="tag {cls}">{lbl}</span></div>'
            st.markdown(f"""
            <div class="msg-row ai">
                <div class="avatar ai">🤖</div>
                <div class="bubble ai">{tag_html}{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input("Type your question…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    app = load_graph()

    with st.spinner("Thinking…"):
        result = app.invoke({
            "programme": programme,
            "messages": [("human", user_input)],
        })

    ai_reply = result["messages"][-1].content
    query_tag = result.get("qurey_type", "general")

    st.session_state.chat_history.append({
        "role": "ai",
        "content": ai_reply,
        "tag": query_tag,
    })

    st.rerun()