import os

# Keep CPU/thread usage predictable on small Render instances.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st
import time
import uuid
import json
import logging
from datetime import datetime
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conditional_rag")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Conditional RAG Campus Bot",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME + STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #fbfaf7;
    --bg-elev: #ffffff;
    --bg-sunken: #f4f2ec;
    --ink: #16161a;
    --ink-2: #3c3c44;
    --ink-3: #6f6f7c;
    --ink-4: #a6a6b2;
    --line: #e6e3da;
    --line-strong: #d5d1c4;
    --accent: #0f4c3a;
    --accent-soft: #e8f0ec;
    --warn: #9a5b00;
    --warn-soft: #fbf1e0;
    --info: #1f3f7a;
    --info-soft: #eaf0fb;
    --radius: 10px;
    --radius-sm: 6px;
}

/* ── Global Layout ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: var(--bg);
}

header[data-testid="stHeader"] { display: none; }
footer { display: none; }
#MainMenu { display: none; }
div[data-testid="stToolbar"] { display: none; }

.block-container {
    padding-top: 0.45rem !important;
    padding-bottom: 0.35rem !important;
    max-width: 1040px;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em !important;
    color: var(--ink) !important;
}

code, pre, .mono {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-elev);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] > div {
    padding: 20px 20px 40px;
}

.sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0 18px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 20px;
}

.sb-brand .mark {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: var(--ink);
    color: var(--bg) !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Instrument Serif', serif !important;
    font-size: 1.15rem;
    font-weight: 500;
}

.sb-brand .name {
    font-family: 'Instrument Serif', serif !important;
    font-size: 1.15rem;
    color: var(--ink) !important;
    line-height: 1.1;
}

.sb-brand .sub {
    font-size: 0.68rem;
    color: var(--ink-3) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

.sb-section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-4) !important;
    font-weight: 600;
    margin: 22px 0 10px;
}

.sb-kv {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    font-size: 0.82rem;
    border-bottom: 1px solid var(--line);
}

.sb-kv:last-child {
    border-bottom: none;
}

.sb-kv .k {
    color: var(--ink-3) !important;
}

.sb-kv .v {
    color: var(--ink) !important;
    font-weight: 500;
}

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0 16px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 0;
}

.top-bar .left {
    display: flex;
    align-items: baseline;
    gap: 14px;
}

.brand-home {
    color: inherit !important;
    text-decoration: none !important;
    cursor: pointer;
}
.brand-home:hover { color: var(--accent) !important; }

.top-bar h1 {
    font-size: 1.75rem;
    margin: 0;
}

.top-bar .crumb {
    font-size: 0.74rem;
    color: var(--ink-4) !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 600;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 11px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent) !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.status-pill .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2.4s ease-in-out infinite;
}

@keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* ── Onboarding ── */
.onboard-aside {
    background: linear-gradient(145deg, #f1f0e9 0%, #e8f0ec 100%);
    padding: 38px 32px;
    border: 1px solid var(--line);
    border-radius: 10px 0 0 10px;
    border-right: none;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.onboard-aside .eyebrow {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ink-4) !important;
    font-weight: 600;
    margin-bottom: 14px;
}

.onboard-aside h2 {
    font-size: 1.85rem;
    margin: 0 0 14px;
    line-height: 1.15;
    color: var(--ink) !important;
}

.onboard-aside p {
    font-size: 0.86rem;
    color: var(--ink-2) !important;
    line-height: 1.6;
    margin: 0;
}

.onboard-aside .meta {
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid var(--line-strong);
    font-size: 0.75rem;
    color: var(--ink-3) !important;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.7;
}

.onboard-main {
    padding: 40px 36px;
    border: 1px solid var(--line);
    border-radius: 0 10px 10px 0;
    border-left: none;
    background: var(--bg-elev);
    height: 100%;
}

.onboard-main h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--ink-4) !important;
    font-weight: 600;
    margin: 0 0 16px;
}

/* ── Landing page content ── */
.landing-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--accent) !important;
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(15,76,58,.14);
    border-radius: 999px;
    padding: 6px 11px;
    font-size: .66rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.landing-kicker .kicker-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
}
.onboard-aside h2 { font-size: 2.15rem; line-height: 1.12; margin-bottom: 15px; }
.onboard-aside .lead { font-size: .9rem; line-height: 1.7; max-width: 390px; }
.feature-list { display: grid; gap: 12px; margin-top: 28px; }
.feature-item { display: flex; gap: 12px; align-items: flex-start; }
.feature-icon {
    flex: 0 0 30px; height: 30px; display: grid; place-items: center;
    border: 1px solid rgba(15,76,58,.16); border-radius: 8px;
    background: rgba(255,255,255,.7); color: var(--accent);
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem; font-weight: 600;
}
.feature-copy strong { display: block; color: var(--ink) !important; font-size: .82rem; margin: 1px 0 3px; }
.feature-copy span { display: block; color: var(--ink-3) !important; font-size: .74rem; line-height: 1.5; }
.onboard-main { padding: 34px 30px; }
.st-key-onboard_main .panel-intro { color: var(--ink-3) !important; font-size: .8rem; line-height: 1.55; margin: -7px 0 20px; }
.programme-note {
    margin-top: 20px; padding: 14px 15px; border-radius: 7px;
    background: var(--bg-sunken); border: 1px solid var(--line);
}
.programme-note .note-title { color: var(--ink) !important; font-size: .75rem; font-weight: 600; margin-bottom: 5px; }
.programme-note p { color: var(--ink-3) !important; font-size: .72rem; line-height: 1.5; margin: 0; }
.top-bar h1 { font-size: 1.55rem; }
@media (max-width: 640px) {
    .onboard-aside, .onboard-main { padding: 28px 22px; }
    .onboard-aside h2 { font-size: 1.8rem; }
    .top-bar h1 { font-size: 1.25rem; }
}

/* ── Radio ── */
div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: var(--bg-elev);
    border: 1px solid var(--line-strong);
    border-radius: 6px;
    padding: 14px 16px !important;
    margin: 0 !important;
    transition: all 0.14s ease;
    cursor: pointer;
}

div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: var(--ink-3);
    background: var(--bg-sunken);
}

div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {
    border-color: var(--accent);
    background: var(--accent-soft);
    box-shadow: inset 3px 0 0 var(--accent);
}

div[data-testid="stRadio"] label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] label[data-baseweb="radio"] p {
    color: var(--ink) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] div[data-testid="stMarkdownContainer"] p {
    color: var(--accent) !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: var(--ink) !important;
    color: #ffffff !important;
    border: 1px solid var(--ink) !important;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.15s ease;
}

div.stButton > button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}

div.stButton > button p {
    color: #ffffff !important;
}

.btn-primary div.stButton > button {
    width: 100%;
    padding: 13px 20px !important;
}

/* ══════════════════════════════════════════
   CHAT PAGE LAYOUT
   ══════════════════════════════════════════ */

/* Back button beside the connection status (chat page only) */
.st-key-back_to_programme { margin: 0 !important; padding: 0 !important; }
.st-key-back_to_programme div.stButton > button {
    width: auto !important;
    min-height: 28px;
    padding: 3px 9px !important;
    border: 1px solid #16161a !important;
    border-radius: 7px !important;
    background: #16161a !important;
    transform: translateY(8px);
    color: #ffffff !important;
    font-size: .76rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    white-space: nowrap !important;
}
.st-key-back_to_programme div.stButton > button p,
.st-key-back_to_programme div.stButton > button span {
    color: #ffffff !important;
}
.st-key-back_to_programme div.stButton > button:hover {
    background: #303036 !important;
    border-color: #303036 !important;
    color: #ffffff !important;
}

/* Chat meta bar */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 20px;
    background: var(--bg-sunken);
    border: 1px solid var(--line);
    border-radius: 10px 10px 0 0;
    font-size: 0.72rem;
    color: var(--ink-3) !important;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 16px;
    margin-bottom: 0;
}

.chat-header .session-id {
    color: var(--ink-4) !important;
}

.chat-header .prog-tag {
    padding: 4px 10px;
    background: var(--bg-elev);
    border: 1px solid var(--line-strong);
    border-radius: 5px;
    color: var(--ink) !important;
    font-weight: 600;
}

/* Scrollable chat container */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 0 0 10px 10px !important;
    border-color: var(--line) !important;
    background: var(--bg-elev) !important;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    border-bottom: 1px solid var(--line);
    padding: 16px 22px !important;
    background: transparent;
    gap: 12px !important;
}

div[data-testid="stChatMessage"]:last-child {
    border-bottom: none;
}

div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(90deg, var(--bg-sunken) 0%, var(--bg-elev) 60%);
}

div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(270deg, var(--accent-soft) 0%, var(--bg-elev) 50%);
}

div[data-testid="chatAvatarIcon-user"] {
    background-color: var(--ink) !important;
    color: var(--bg) !important;
    border-radius: 8px !important;
}

div[data-testid="chatAvatarIcon-assistant"] {
    background-color: var(--accent) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Tables inside AI responses */
div[data-testid="stChatMessage"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 14px 0 !important;
    font-size: 0.85rem !important;
    background: var(--bg-elev) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

div[data-testid="stChatMessage"] th {
    background: var(--bg-sunken) !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--line-strong) !important;
}

div[data-testid="stChatMessage"] td {
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--line) !important;
    color: var(--ink-2) !important;
    vertical-align: top !important;
}

div[data-testid="stChatMessage"] tr:last-child td {
    border-bottom: none !important;
}

/* Chat input */
div[data-testid="stChatInput"] {
    background: transparent !important;
    padding: 0 !important;
    margin-top: 12px !important;
}

div[data-testid="stChatInput"] > div {
    background: var(--bg-elev) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}

div[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--ink) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

div[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: var(--accent) !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--ink-4) !important;
}

div[data-testid="stChatInput"] button {
    background: var(--ink) !important;
    border-radius: 6px !important;
    border: none !important;
}

div[data-testid="stChatInput"] button:hover {
    background: var(--accent) !important;
}

div[data-testid="stChatInput"] button svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* Keep Streamlit's fixed chat-input area consistent with the page */
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
div[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
    box-shadow: none !important;
}
div[data-testid="stBottom"]::before {
    background: var(--bg) !important;
}
div[data-testid="stChatInput"] {
    max-width: 860px;
    margin: 0 auto !important;
    padding: 10px 0 14px !important;
}

/* Empty state */
.empty-wrap {
    padding: 30px 28px;
    width: 100%;
    max-width: 680px;
    margin: 0 auto;
    box-sizing: border-box;
}

.empty-wrap .eyebrow {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--ink-4) !important;
    font-weight: 600;
    margin-bottom: 12px;
}

.empty-wrap h2 {
    font-size: 1.9rem;
    margin: 0 0 12px;
    line-height: 1.2;
    color: var(--ink) !important;
    font-family: 'Instrument Serif', serif !important;
    font-weight: 400 !important;
}

.empty-wrap p {
    font-size: 0.87rem;
    color: var(--ink-2) !important;
    line-height: 1.65;
    margin: 0 0 20px;
    max-width: 600px;
}

.suggestion-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}

.suggestion {
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 14px 16px;
    background: var(--bg-elev);
    font-size: 0.82rem;
    color: var(--ink-2) !important;
    line-height: 1.5;
    transition: all 0.14s ease;
    cursor: default;
    min-width: 0;
}

.suggestion:hover {
    border-color: var(--ink-3);
    background: var(--bg-sunken);
}

.suggestion .cat {
    display: block;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--ink-4) !important;
    font-weight: 600;
    margin-bottom: 5px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Assistant tag label */
.msg-tag {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--accent) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    display: block;
}

/* Spinner */
div[data-testid="stSpinner"] > div {
    border-color: var(--line-strong) !important;
    border-top-color: var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--line-strong);
    border-radius: 10px;
    border: 2px solid var(--bg);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--ink-4);
}

/* ── Text colors ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] ul {
    color: var(--ink-2) !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span {
    color: var(--ink-2) !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
}

strong, b, [data-testid="stMarkdownContainer"] strong {
    color: var(--ink) !important;
    font-weight: 600 !important;
}

div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4 {
    color: var(--ink) !important;
    margin-top: 16px !important;
    margin-bottom: 8px !important;
}

a, [data-testid="stMarkdownContainer"] a {
    color: var(--accent) !important;
    text-decoration: underline;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: var(--ink-2) !important;
}

/* ── Programme selection panel ── */
.st-key-onboard_main {
    background: #ffffff; border: 1px solid var(--line);
    border-radius: 14px; padding: 30px !important;
    min-height: 100%; box-sizing: border-box;
}
.selection-kicker {
    display: inline-block; color: var(--accent) !important;
    background: var(--accent-soft); border-radius: 999px;
    padding: 6px 11px; font-size: .65rem; font-weight: 700;
    letter-spacing: .11em; margin-bottom: 15px;
}
.st-key-onboard_main .selection-title {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: 2rem !important; line-height: 1.15;
    color: var(--ink) !important; margin: 0 0 10px;
}
.st-key-onboard_main .panel-intro {
    color: var(--ink-3) !important; font-size: .84rem;
    line-height: 1.65; margin: 0 0 22px;
}
.choose-label { color: var(--ink) !important; font-size: .75rem; font-weight: 700; margin: 0 0 10px; }
.st-key-programme_selector { display: none; }
.st-key-programme_selector div.stButton > button {
    width: 100% !important; min-height: 50px !important;
    padding: 12px 16px !important; border-radius: 9px !important;
    text-align: left !important; white-space: normal !important;
    font-size: .84rem !important; box-shadow: none !important;
}
.st-key-programme_selector div.stButton > button[kind="secondary"] {
    background: #fff !important; color: var(--ink) !important;
    border: 1px solid var(--line-strong) !important;
}
.st-key-programme_selector div.stButton > button[kind="secondary"]:hover {
    background: var(--bg-sunken) !important; border-color: var(--accent) !important; color: var(--accent) !important;
}
.st-key-programme_selector div.stButton > button[kind="primary"] {
    background: var(--accent) !important; color: #fff !important; border: 1px solid var(--accent) !important;
}
.st-key-programme_selector div.stButton > button p {
    color: inherit !important; font-weight: 600 !important; text-align: left !important; white-space: normal !important;
}
.workflow-panel {
    margin-top: 8px; padding: 17px 16px; border: 1px solid var(--line);
    border-radius: 10px; background: linear-gradient(135deg, #f7f6f1 0%, #edf3ef 100%);
}
.workflow-heading { color: var(--ink) !important; font-size: .78rem; font-weight: 700; margin-bottom: 13px; }
.workflow-steps { display: grid; gap: 11px; }
.workflow-step { display: flex; align-items: flex-start; gap: 10px; }
.workflow-step > span {
    flex: 0 0 23px; width: 23px; height: 23px; display: grid; place-items: center;
    border-radius: 50%; background: var(--accent); color: white !important; font-size: .68rem; font-weight: 700;
}
.workflow-step strong { display: block; color: var(--ink) !important; font-size: .73rem; margin: 0 0 3px; }
.workflow-step small { display: block; color: var(--ink-3) !important; font-size: .68rem; line-height: 1.45; }
.landing-footnote { margin-top: 28px; padding-top: 15px; border-top: 1px solid rgba(15,76,58,.15); color: var(--ink-3) !important; font-size: .72rem; line-height: 1.5; }
.onboard-aside { border-radius: 14px !important; border: 1px solid var(--line) !important; padding: 34px 32px; height: 620px !important; min-height: 620px !important; box-sizing: border-box; }
.st-key-onboard_main { height: 620px !important; min-height: 620px !important; box-sizing: border-box; display: flex; flex-direction: column; }
.st-key-onboard_main [data-testid="stSelectbox"] { margin: 4px 0 8px; }
.st-key-onboard_main [data-testid="stSelectbox"] > div > div { min-height: 46px; border-radius: 9px; border-color: var(--line-strong); background: #fff; }
.st-key-onboard_main [data-testid="stSelectbox"] * { color: var(--ink) !important; }
.st-key-onboard_main div.stButton > button { min-height: 48px; border-radius: 9px !important; margin-top: 10px !important; font-size: .86rem; }
@media (max-width: 760px) {
    .st-key-onboard_main { padding: 24px 20px !important; height: auto !important; min-height: 0 !important; }
    .st-key-onboard_main .selection-title { font-size: 1.8rem !important; }
}
@media (max-width: 420px) {
    .st-key-back_to_programme div.stButton > button { min-height: 30px; padding: 4px 8px !important; font-size: .68rem !important; }
 .st-key-onboard_main { padding: 20px 16px !important; } }

/* ── Responsive layout: tablet and mobile ── */
@media (max-width: 900px) {
    .block-container {
        max-width: 100% !important;
        padding-left: 1.35rem !important;
        padding-right: 1.35rem !important;
    }
}

@media (max-width: 760px) {
    .block-container {
        padding-top: .45rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 5rem !important;
    }

    /* Stack the two landing-page panels in a single column */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 14px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
    }

    .top-bar {
        align-items: flex-start;
        gap: 10px;
        padding: 8px 0 13px;
    }
    .top-bar .left {
        gap: 3px;
        flex-direction: column;
        align-items: flex-start;
        min-width: 0;
    }
    .top-bar h1 { font-size: 1.35rem !important; line-height: 1.15; }
    .top-bar .crumb { font-size: .59rem; letter-spacing: .09em; }
    .status-pill { font-size: .62rem; padding: 5px 8px; white-space: nowrap; }

    .onboard-aside, .onboard-main, .st-key-onboard_main {
        height: auto !important;
        min-height: 0 !important;
        min-height: 0 !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        padding: 25px 21px !important;
    }
    .onboard-aside h2 { font-size: 1.85rem !important; }
    .onboard-aside .lead { font-size: .86rem; }
    .feature-list { margin-top: 22px; gap: 13px; }
    .onboard-main { padding: 23px 19px !important; }

    div[data-testid="stRadio"] div[role="radiogroup"] { gap: 8px; }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        padding: 12px 13px !important;
        min-height: 46px;
    }

    .chat-header { margin-top: 12px; padding: 10px 12px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: min(58vh, 430px) !important;
        min-height: 270px !important;
    }
    .empty-wrap { padding: 24px 16px; }
    .empty-wrap h2 { font-size: 1.65rem; }
    .suggestion-grid { grid-template-columns: 1fr; gap: 9px; }
    .suggestion { padding: 12px 13px; }
    div[data-testid="stChatMessage"] { padding: 13px 11px !important; }

    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
    div[data-testid="stChatInput"] { width: 100% !important; }
    div[data-testid="stChatInput"] textarea { font-size: 16px !important; }
}

@media (max-width: 420px) {
    .block-container { padding-left: .75rem !important; padding-right: .75rem !important; }
    .top-bar h1 { font-size: 1.2rem !important; }
    .status-pill { font-size: .56rem; }
    .landing-kicker { font-size: .58rem; padding: 5px 9px; }
    .onboard-aside, .onboard-main, .st-key-onboard_main { padding: 21px 16px !important; height: auto !important; min-height: 0 !important; }
    .onboard-aside h2 { font-size: 1.65rem !important; }
    .feature-copy strong { font-size: .78rem; }
    .feature-copy span { font-size: .7rem; }
    .programme-card-grid { grid-template-columns: 1fr !important; }
    .chat-header { font-size: .62rem; }
}


/* Mobile-only refinements: desktop rules remain untouched */
@media (max-width: 760px) {
    html, body, .stApp { overflow-x: hidden !important; }
    .block-container {
        width: 100% !important;
        max-width: 100% !important;
        padding: 12px 14px 88px !important;
    }

    .top-bar {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 10px !important;
        padding: 8px 0 14px !important;
    }
    .top-bar .left { flex: 1 1 auto; min-width: 0; gap: 3px !important; }
    .top-bar h1 { font-size: clamp(1.12rem, 5vw, 1.38rem) !important; line-height: 1.12 !important; }
    .top-bar .crumb { font-size: .56rem !important; letter-spacing: .08em !important; }
    .status-pill { flex: 0 0 auto; font-size: .58rem !important; padding: 5px 8px !important; }

    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 12px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .onboard-aside, .onboard-main, .st-key-onboard_main {
        width: 100% !important;
        height: auto !important;
        min-height: 0 !important;
        padding: 22px 18px !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        box-sizing: border-box !important;
    }
    .onboard-aside h2 { font-size: 1.75rem !important; line-height: 1.12 !important; }
    .onboard-aside .lead { font-size: .84rem !important; line-height: 1.6 !important; }
    .landing-kicker { margin-bottom: 14px !important; font-size: .59rem !important; }
    .feature-list { margin-top: 19px !important; gap: 12px !important; }
    .feature-item { gap: 10px !important; }
    .feature-icon { flex-basis: 28px !important; height: 28px !important; }
    .feature-copy strong { font-size: .78rem !important; }
    .feature-copy span { font-size: .70rem !important; line-height: 1.45 !important; }
    .landing-footnote { margin-top: 20px !important; padding-top: 12px !important; }

    .st-key-onboard_main .selection-title { font-size: 1.75rem !important; }
    .st-key-onboard_main .panel-intro { font-size: .78rem !important; line-height: 1.55 !important; margin-bottom: 16px !important; }
    .st-key-onboard_main [data-testid="stSelectbox"] { margin: 2px 0 8px !important; }
    .st-key-onboard_main [data-testid="stSelectbox"] > div > div { min-height: 46px !important; }
    .workflow-panel {
    margin-top: 10px !important;
    padding: 14px 13px !important;
    }
    .workflow-steps { gap: 10px !important; }

    .chat-header {
        margin-top: 10px !important;
        padding: 10px 12px !important;
        min-height: 44px;
        gap: 8px !important;
        border-radius: 9px !important;
        font-size: .62rem !important;
    }
    .chat-header .session-id { min-width: 0; overflow-wrap: anywhere; }
    .chat-header .prog-tag { flex: 0 0 auto; padding: 4px 8px !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: min(59svh, 470px) !important;
        min-height: 270px !important;
        border-radius: 9px !important;
    }
    .empty-wrap { padding: 19px 14px !important; }
    .empty-wrap .eyebrow { margin-bottom: 9px !important; font-size: .62rem !important; }
    .empty-wrap h2 { font-size: 1.55rem !important; line-height: 1.16 !important; margin-bottom: 9px !important; }
    .empty-wrap p { font-size: .79rem !important; line-height: 1.55 !important; margin-bottom: 14px !important; }
    .suggestion-grid { grid-template-columns: 1fr !important; gap: 8px !important; }
    .suggestion { padding: 10px 12px !important; font-size: .76rem !important; line-height: 1.45 !important; }
    .suggestion .cat { font-size: .59rem !important; margin-bottom: 3px !important; }
    div[data-testid="stChatMessage"] { padding: 12px 10px !important; gap: 9px !important; }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li { font-size: .84rem !important; line-height: 1.55 !important; }
    div[data-testid="stChatMessage"] table { display: block; overflow-x: auto !important; white-space: nowrap; }

    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }
    div[data-testid="stChatInput"] { width: 100% !important; max-width: 100% !important; padding: 7px 0 10px !important; }
    div[data-testid="stChatInput"] > div { border-radius: 10px !important; }
    div[data-testid="stChatInput"] textarea { font-size: 16px !important; }
}

@media (max-width: 420px) {
    .block-container { padding-left: 11px !important; padding-right: 11px !important; }
    .top-bar h1 { font-size: 1.08rem !important; }
    .top-bar .crumb { font-size: .52rem !important; }
    .status-pill { font-size: .54rem !important; padding: 4px 7px !important; }
    .onboard-aside, .onboard-main, .st-key-onboard_main { padding: 19px 15px !important; }
    .chat-header { padding: 9px 10px !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { height: min(56svh, 420px) !important; min-height: 250px !important; }
    .empty-wrap { padding: 16px 11px !important; }
    .empty-wrap h2 { font-size: 1.42rem !important; }
}

/* Ensure the full welcome suggestions fit on small mobile screens */
@media (max-width: 420px) {
    /* Give the welcome panel enough room for all four suggestions */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: min(72svh, 540px) !important;
        min-height: 390px !important;
    }
    .empty-wrap { padding: 12px 10px !important; }
    .empty-wrap .eyebrow { margin-bottom: 7px !important; }
    .empty-wrap h2 { margin-bottom: 8px !important; }
    .empty-wrap p { margin-bottom: 9px !important; line-height: 1.45 !important; }
    .suggestion-grid { gap: 6px !important; }
    .suggestion { padding: 8px 10px !important; line-height: 1.35 !important; }
    .suggestion .cat { margin-bottom: 2px !important; }
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKEND (Unchanged)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_graph():
    """Build and cache the embedding model, PDF retrievers, LLM and LangGraph."""
    logger.info("Starting Conditional RAG knowledge-base initialization")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it under Render > Environment."
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    academic_pdf = os.path.join(base_dir, "academics_handbook.pdf")
    fee_pdf = os.path.join(base_dir, "fee_structure.pdf")

    for pdf_path in (academic_pdf, fee_pdf):
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(
                f"Required knowledge-base PDF was not found: {pdf_path}. "
                "Make sure the PDF is committed to the repository."
            )

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it under Render > Environment."
        )

    logger.info("Initializing Gemini embedding API")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key,
    )

    # Build each FAISS retriever only when its category is first requested.
    retrievers = {}

    def build_retriever(pdf_path: str, label: str):
        logger.info("Loading %s PDF: %s", label, pdf_path)
        documents = PyPDFLoader(pdf_path).load()
        if not documents:
            raise RuntimeError(f"The {label} PDF contains no readable pages.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise RuntimeError(f"No text could be extracted from the {label} PDF.")

        logger.info("Creating %s FAISS index from %s chunks", label, len(chunks))
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(key: str):
        if key not in retrievers:
            if key == "academic":
                retrievers[key] = build_retriever(academic_pdf, "academic handbook")
            elif key == "fee":
                retrievers[key] = build_retriever(fee_pdf, "fee structure")
            else:
                raise ValueError(f"Unknown retriever category: {key}")
        return retrievers[key]

    logger.info("Initializing Groq chat model")
    llm = ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0.4,
        timeout=60,
        max_retries=2,
    )

    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        query_type: str
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
        logger.info("Classifying user query")
        response = llm.invoke(prompt)
        category_text = str(response.content).strip().lower()

        if "academic" in category_text:
            category = "academic"
        elif "fee" in category_text:
            category = "fee"
        else:
            category = "general"

        logger.info("Query classified as %s", category)
        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        logger.info("Retrieving academic context")
        docs = get_retriever("academic").invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        if not context.strip():
            logger.warning("Academic retriever returned no context")
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        logger.info("Retrieving fee context")
        docs = get_retriever("fee").invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        if not context.strip():
            logger.warning("Fee retriever returned no context")
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state.get("retrieved_context", "")

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                "Use the following context from the official college documents to answer "
                "the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context or 'No relevant text was retrieved from the college documents.'}\n\n"
                f"Question: {query}\n\n"
                "If the provided context does not contain the answer, say so clearly "
                "instead of inventing college-specific rules or figures. "
                "Give a clear, friendly, and precise answer."
            )

        logger.info("Generating final answer with Groq")
        response = llm.invoke(prompt)
        answer = str(response.content).strip()
        if not answer:
            raise RuntimeError("The language model returned an empty response.")
        return {"messages": [("ai", answer)]}

    def route_query(state: State):
        category = state.get("query_type", "general")
        if category == "academic":
            return "academic_rag"
        if category == "fee":
            return "fee_rag"
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

    logger.info("Conditional RAG knowledge base initialized successfully")
    return graph.compile()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
DEFAULTS = {
    "programme": None,
    "selected_programme": "BCA",
    "chat_history": [],
    "session_id": uuid.uuid4().hex[:8].upper(),
    "session_started": datetime.now().strftime("%d %b %Y, %H:%M"),
    "message_count": 0,
    "last_latency_ms": None,
    "last_error": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Clicking the app title returns to programme selection and starts a fresh session.
if st.query_params.get("home") == "1":
    st.session_state.programme = None
    st.session_state.chat_history = []
    st.session_state.message_count = 0
    st.session_state.last_latency_ms = None
    st.session_state.last_error = None
    st.session_state.session_id = uuid.uuid4().hex[:8].upper()
    st.session_state.session_started = datetime.now().strftime("%d %b %Y, %H:%M")
    st.query_params.clear()
    st.rerun()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_transcript() -> str:
    lines = [
        "CONDITIONAL RAG CAMPUS BOT — SESSION TRANSCRIPT",
        f"Session ID   : {st.session_state.session_id}",
        f"Programme    : {st.session_state.programme or '—'}",
        f"Started      : {st.session_state.session_started}",
        f"Exported     : {datetime.now().strftime('%d %b %Y, %H:%M')}",
        "=" * 52,
        "",
    ]

    for m in st.session_state.chat_history:
        who = (
            "STUDENT"
            if m["role"] == "user"
            else f"ASSISTANT [{(m.get('tag') or 'general').upper()}]"
        )
        lines.append(who)
        lines.append(m["content"])
        lines.append("")

    return "\n".join(lines)


def export_json() -> str:
    payload = {
        "session_id": st.session_state.session_id,
        "programme": st.session_state.programme,
        "started_at": st.session_state.session_started,
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.chat_history,
    }

    return json.dumps(payload, indent=2)


def clear_conversation():
    st.session_state.chat_history = []
    st.session_state.message_count = 0
    st.session_state.last_latency_ms = None
    st.session_state.last_error = None
    st.session_state.session_id = uuid.uuid4().hex[:8].upper()
    st.session_state.session_started = datetime.now().strftime("%d %b %Y, %H:%M")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="mark">CR</div>
        <div>
            <div class="name">Conditional RAG Campus Bot</div>
            <div class="sub">Knowledge Retrieval</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-section-label">Session</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="sb-kv"><span class="k">Session ID</span><span class="v mono">{st.session_state.session_id}</span></div>
    <div class="sb-kv"><span class="k">Programme</span><span class="v">{st.session_state.programme or '—'}</span></div>
    <div class="sb-kv"><span class="k">Started</span><span class="v">{st.session_state.session_started}</span></div>
    <div class="sb-kv"><span class="k">Messages</span><span class="v">{st.session_state.message_count}</span></div>
    <div class="sb-kv"><span class="k">Last latency</span><span class="v mono">{f"{st.session_state.last_latency_ms} ms" if st.session_state.last_latency_ms else "—"}</span></div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-section-label">Knowledge Sources</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="sb-kv"><span class="k">Academics handbook</span><span class="v">PDF</span></div>
    <div class="sb-kv"><span class="k">Fee structure</span><span class="v">PDF</span></div>
    <div class="sb-kv"><span class="k">Classifier</span><span class="v">gpt-oss-20b</span></div>
    <div class="sb-kv"><span class="k">Embeddings</span><span class="v">Gemini</span></div>
    """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown(
            '<div class="sb-section-label">Export</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            "Download transcript (.txt)",
            data=build_transcript(),
            file_name=f"transcript_{st.session_state.session_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.download_button(
            "Download session (.json)",
            data=export_json(),
            file_name=f"session_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown(
        '<div class="sb-section-label">Actions</div>',
        unsafe_allow_html=True
    )

    if st.button("Clear conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    if st.session_state.programme and st.button(
        "Change programme",
        use_container_width=True
    ):
        st.session_state.programme = None
        clear_conversation()
        st.rerun()

    st.markdown(
        '<div style="font-size:0.7rem;color:var(--ink-4);margin-top:28px;'
        'line-height:1.6;font-family:IBM Plex Mono,monospace;">'
        'v1.0 &middot; production build<br>'
        'Answers derived from institutional documents.'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────
top_left = """
<div class="left">
    <h1><a class="brand-home" href="?home=1" title="Return to programme selection">Conditional RAG Campus Bot</a></h1>
    <span class="crumb">Campus Knowledge Assistant</span>
</div>
"""

top_right = f"""
<div class="status-pill">
    <span class="dot"></span>
    {"Connected &middot; " + st.session_state.programme if st.session_state.programme else "Awaiting programme"}
</div>
"""

if st.session_state.programme:
    title_col, back_col, status_col = st.columns([5.8, 0.6, 1.8], vertical_alignment="center")
    with title_col:
        st.markdown(
        '<div class="top-bar"><div class="left"><h1><a class="brand-home" '
        'href="?home=1" title="Return to programme selection" '
        'onclick="window.location.href=this.href; return false;">'
        'Conditional RAG Campus Bot</a></h1>'
        '<span class="crumb">Campus Knowledge Assistant</span></div></div>',
        unsafe_allow_html=True,
        )
    with back_col:
        if st.button("← Back", key="back_to_programme", use_container_width=True):
            st.session_state.programme = None
            st.session_state.selected_programme = "BCA"
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.session_state.last_latency_ms = None
            st.session_state.last_error = None
            st.session_state.session_id = uuid.uuid4().hex[:8].upper()
            st.session_state.session_started = datetime.now().strftime("%d %b %Y, %H:%M")
            st.query_params.clear()
            st.rerun()
    with status_col:
        st.markdown(top_right, unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="top-bar">{top_left}{top_right}</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# PROGRAMME SELECTION (PAGE 1)
# ─────────────────────────────────────────────
if st.session_state.programme is None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            """<div class="onboard-aside">
<div>
<div class="landing-kicker"><span class="kicker-dot"></span> Your campus information hub</div>
<h2>Campus answers, made simpler.</h2>
<p class="lead">Get clear answers to common academic and fee-related questions. Ask naturally — the assistant identifies the topic and finds relevant information from the available college documents.</p>
<div class="feature-list">
<div class="feature-item"><div class="feature-icon">01</div><div class="feature-copy"><strong>Academic information</strong><span>Attendance rules, examinations, credits, course structure and academic requirements.</span></div></div>
<div class="feature-item"><div class="feature-icon">02</div><div class="feature-copy"><strong>Fees and payments</strong><span>Tuition details, payment rules, late fees, refunds and scholarship information.</span></div></div>
<div class="feature-item"><div class="feature-icon">03</div><div class="feature-copy"><strong>Programme-aware answers</strong><span>Responses are tailored to the programme you select before starting.</span></div></div>
</div>
</div>
<div class="landing-footnote">One place for academic guidance and campus-related questions.</div>
</div>""",
            unsafe_allow_html=True,
        )

    with col2:
        with st.container(key="onboard_main"):
            st.markdown("""
            <div class="selection-kicker">GET STARTED</div>
            <h3 class="selection-title">Choose your programme</h3>
            <p class="panel-intro">Select the course you are currently enrolled in. Your selection helps the assistant provide answers relevant to your programme.</p>
            <div class="choose-label">YOUR PROGRAMME</div>
            """, unsafe_allow_html=True)

            programme_options = ["BCA", "BBA", "B.Com (H)"]
            st.session_state.selected_programme = st.selectbox(
                "Select your course",
                options=programme_options,
                index=programme_options.index(st.session_state.selected_programme)
                if st.session_state.selected_programme in programme_options else 0,
                label_visibility="collapsed",
                key="programme_dropdown",
            )

            st.markdown("""
            <div class="workflow-panel">
                <div class="workflow-heading">How your question is answered</div>
                <div class="workflow-steps">
                    <div class="workflow-step"><span>1</span><div><strong>Understand</strong><small>Your question is classified as academic, fee-related, or general.</small></div></div>
                    <div class="workflow-step"><span>2</span><div><strong>Retrieve</strong><small>Relevant college information is retrieved when the question needs it.</small></div></div>
                    <div class="workflow-step"><span>3</span><div><strong>Respond</strong><small>The assistant generates an answer tailored to your selected programme.</small></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Continue to assistant", use_container_width=True, type="primary"):
                st.session_state.programme = st.session_state.selected_programme
                st.rerun()

    st.stop()


# ─────────────────────────────────────────────
# CHAT SCREEN (PAGE 2)
# ─────────────────────────────────────────────
programme = st.session_state.programme

TAG_LABELS = {
    "academic": "Academic",
    "fee": "Fee",
    "general": "General",
}

# ── 1. Chat header ────────────────────────────
st.markdown(f"""
<div class="chat-header">
    <span class="session-id">SESSION {st.session_state.session_id}</span>
    <span class="prog-tag">{programme}</span>
</div>
""", unsafe_allow_html=True)

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    st.session_state.last_error = None


# ── 2. Chat history / welcome state ──────────
CHAT_HEIGHT = 430

if not st.session_state.chat_history:
    # Keep the welcome suggestions in normal page flow.
    # A fixed-height Streamlit container clips the last card on short mobile screens.
    st.markdown(
        """<div class="empty-wrap"><div class="eyebrow">Ready</div><h2>Ask anything about your programme.</h2><p>The assistant routes each question to the correct source — academic regulations, fee schedules, or general guidance — and retrieves relevant information when needed.</p><div class="suggestion-grid"><div class="suggestion"><span class="cat">Academic</span>What is the minimum attendance requirement for exams?</div><div class="suggestion"><span class="cat">Fee</span>What is the late fee for semester tuition?</div><div class="suggestion"><span class="cat">Academic</span>How many credits are required to advance to the next year?</div><div class="suggestion"><span class="cat">Fee</span>Am I eligible for a refund if I withdraw mid-semester?</div></div></div>""",
        unsafe_allow_html=True,
    )
else:
    # Only the actual conversation uses a fixed-height scrollable region.
    with st.container(height=CHAT_HEIGHT, border=True):
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"

            with st.chat_message(role):
                if role == "assistant":
                    tag = msg.get("tag", "general")
                    lbl = TAG_LABELS.get(tag, tag.title())

                    st.markdown(
                        f'<span class="msg-tag">Assistant &middot; {lbl}</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown(msg["content"])


# ── 3. Chat input ─────────────────────────────
user_input = st.chat_input(
    "Ask a question about academics, fees, or campus policy…",
    key="main_chat_input",
)


# ── 4. Process input ──────────────────────────
if user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
    })

    t_start = time.perf_counter()

    try:
        with st.spinner("Loading knowledge base and generating response…"):
            app = load_graph()
            result = app.invoke({
                "programme": programme,
                "messages": [("human", user_input)],
            })

        ai_reply = result["messages"][-1].content
        query_tag = result.get("query_type", "general")
        if not ai_reply or not str(ai_reply).strip():
            raise RuntimeError("The assistant returned an empty response.")

        st.session_state.chat_history.append({
            "role": "ai",
            "content": str(ai_reply).strip(),
            "tag": query_tag,
        })
        st.session_state.message_count += 1
        st.session_state.last_error = None

    except Exception as exc:
        logger.exception("Failed to process user query")
        st.session_state.last_error = (
            f"Unable to generate a response: {type(exc).__name__}: {exc}"
        )

    finally:
        st.session_state.last_latency_ms = int(
            (time.perf_counter() - t_start) * 1000
        )

    st.rerun()
