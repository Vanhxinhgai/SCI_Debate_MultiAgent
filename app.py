"""Scientific Debate Simulator — Streamlit UI.

Run:
    streamlit run app.py
"""
import os
import re
import time
import streamlit as st
import plotly.graph_objects as go
from scidebate import Debate, ConsensusMetrics, load_config
from scidebate.llms import OllamaLLM, OpenRouterLLM, GroqLLM


# ── CITATION HIGHLIGHTING ─────────────────────────────────────

def highlight_citations(text: str, pro_papers: list[dict], con_papers: list[dict]) -> str:
    if not pro_papers and not con_papers:
        return text

    paragraphs = text.split("\n")
    highlighted_paragraphs = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            highlighted_paragraphs.append("")
            continue

        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        highlighted_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                highlighted_sentences.append(sentence)
                continue

            sentence_lower = sentence.lower()
            matched_pro = []
            matched_con = []

            def is_paper_matched(idx, paper):
                pattern_doc = f"[{idx}]"
                pattern_doc_alt = f"document [{idx}]"
                author_lastnames = []
                authors = paper.get("authors", "")
                if authors and authors not in ("Unknown Authors", "SciFact corpus"):
                    for author in authors.split(","):
                        parts = author.strip().split()
                        if parts and len(parts[-1]) > 3:
                            author_lastnames.append(parts[-1])
                title_words = [w for w in paper.get("title", "").lower().split() if len(w) > 4]
                title_snippet = " ".join(title_words[:2]) if len(title_words) >= 2 else ""
                if pattern_doc in sentence:
                    return True
                if pattern_doc_alt in sentence_lower:
                    return True
                if any(ln.lower() in sentence_lower for ln in author_lastnames if len(ln) > 2):
                    return True
                if title_snippet and len(title_snippet) > 5 and title_snippet in sentence_lower:
                    return True
                return False

            for idx, paper in enumerate(pro_papers, 1):
                if is_paper_matched(idx, paper):
                    matched_pro.append((idx, paper))
            for idx, paper in enumerate(con_papers, 1):
                if is_paper_matched(idx, paper):
                    matched_con.append((idx, paper))

            if matched_pro or matched_con:
                badges = []
                for doc_idx, doc_paper in matched_pro:
                    pdf_url = doc_paper.get("pdf_link", "#")
                    paper_title = doc_paper.get("title", "").replace('"', '&quot;')
                    badges.append(
                        f'<a href="{pdf_url}" target="_blank" style="text-decoration:none">'
                        f'<span class="cite-badge cite-pro" title="{paper_title}">Doc {doc_idx} &middot; PRO</span>'
                        f'</a>'
                    )
                for doc_idx, doc_paper in matched_con:
                    pdf_url = doc_paper.get("pdf_link", "#")
                    paper_title = doc_paper.get("title", "").replace('"', '&quot;')
                    badges.append(
                        f'<a href="{pdf_url}" target="_blank" style="text-decoration:none">'
                        f'<span class="cite-badge cite-con" title="{paper_title}">Doc {doc_idx} &middot; CON</span>'
                        f'</a>'
                    )
                badge_str = "".join(badges)
                if matched_pro:
                    hl = 'background:rgba(6,95,70,0.07);border-left:3px solid #065F46;padding:1px 6px;border-radius:2px;'
                else:
                    hl = 'background:rgba(153,27,27,0.07);border-left:3px solid #991B1B;padding:1px 6px;border-radius:2px;'
                highlighted_sentences.append(f'<mark style="{hl}">{sentence}</mark>{badge_str}')
            else:
                highlighted_sentences.append(sentence)

        highlighted_paragraphs.append(" ".join(highlighted_sentences))

    return "\n".join(highlighted_paragraphs)


# ── PAPER COLUMN RENDERER ─────────────────────────────────────

def render_papers_column(
    papers: list[dict],
    column_title: str,
    is_pro: bool,
    result_transcript=None,
    verdict_justification=None,
):
    accent = "#065F46" if is_pro else "#991B1B"
    stance_str = "PRO" if is_pro else "CON"

    st.markdown(
        f'<div class="papers-col-title" style="color:{accent}">{column_title}</div>',
        unsafe_allow_html=True,
    )

    if not papers:
        st.markdown(
            '<p class="papers-empty">No evidence retrieved for this stance.</p>',
            unsafe_allow_html=True,
        )
        return

    for i, paper in enumerate(papers, 1):
        with st.expander(f"[{i}] {paper['title']}", expanded=False):
            if paper.get("source"):
                st.markdown(f"**Source:** `{paper['source']}`")
            if paper.get("doc_id"):
                st.markdown(f"**Document ID:** `{paper['doc_id']}`")
            if paper.get("label"):
                label_note = "gold evidence" if paper.get("is_gold_evidence") else "retrieved"
                st.markdown(f"**Label:** {paper['label']} ({label_note})")
            if paper.get("evidence_sentences"):
                st.markdown(f"**Evidence sentences:** {paper['evidence_sentences']}")
            st.markdown(f"**Authors:** {paper['authors']}")
            st.markdown(f"**Published:** {paper['published']}")
            if paper.get("pdf_link"):
                st.markdown(f"**PDF:** [{paper['pdf_link']}]({paper['pdf_link']})")
            st.markdown(f"**Abstract:** {paper['summary']}")

            if result_transcript is not None:
                citations = []
                first_author_lastname = ""
                authors = paper.get("authors", "")
                if authors and authors != "Unknown Authors":
                    first_author = authors.split(",")[0].strip()
                    name_parts = first_author.split()
                    if name_parts:
                        first_author_lastname = name_parts[-1]
                title_words = [w for w in paper.get("title", "").lower().split() if len(w) > 4]
                title_snippet = " ".join(title_words[:2]) if len(title_words) >= 2 else ""

                for turn in result_transcript:
                    if turn.filtered_out:
                        continue
                    content_lower = turn.content.lower()
                    cited = False
                    if f"[{i}]" in turn.content:
                        cited = True
                    elif f"document [{i}]" in content_lower:
                        cited = True
                    elif first_author_lastname and len(first_author_lastname) > 2 and first_author_lastname.lower() in content_lower:
                        cited = True
                    elif title_snippet and len(title_snippet) > 5 and title_snippet in content_lower:
                        cited = True
                    if cited:
                        citations.append(f"{turn.speaker} (Round {turn.round_num})")

                if verdict_justification:
                    verdict_lower = verdict_justification.lower()
                    cited_by_judge = False
                    if f"[{i}]" in verdict_justification:
                        cited_by_judge = True
                    elif f"document [{i}]" in verdict_lower:
                        cited_by_judge = True
                    elif first_author_lastname and len(first_author_lastname) > 2 and first_author_lastname.lower() in verdict_lower:
                        cited_by_judge = True
                    elif title_snippet and len(title_snippet) > 5 and title_snippet in verdict_lower:
                        cited_by_judge = True
                    if cited_by_judge:
                        citations.append("JUDGE (Final Verdict)")

                citations = list(dict.fromkeys(citations))
                st.divider()
                if citations:
                    st.markdown("**Cited by:**")
                    for cit in citations:
                        st.markdown(f"- {cit}")
                else:
                    st.markdown("*Not explicitly cited in the debate.*")


# ── LLM FACTORY ───────────────────────────────────────────────

def _create_llm(cfg: dict, temperature: float, max_tokens: int):
    if "Ollama" in cfg["backend"]:
        return OllamaLLM(model=cfg["model"], temperature=temperature, max_tokens=max_tokens)
    elif "Groq" in cfg["backend"]:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY is not set. Add it to your .env file or run: export GROQ_API_KEY='gsk_...'")
            st.stop()
        return GroqLLM(model=cfg["model"], temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("OPENROUTER_API_KEY is not set. Add it to your .env file or run: export OPENROUTER_API_KEY='sk-or-v1-...'")
            st.stop()
        return OpenRouterLLM(model=cfg["model"], temperature=temperature, max_tokens=max_tokens, api_key=api_key)


# ── PAGE CONFIG ───────────────────────────────────────────────

st.set_page_config(
    page_title="SciDebate — Claim Verifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN SYSTEM & CSS ───────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

/* --- DESIGN TOKENS --- */
:root {
    --bg:            #F4F3EE;
    --bg-sidebar:    #2B44EF;
    --bg-card:       #FFFFFF;
    --border-hard:   #0F0F1A;
    --border-soft:   #D0CCC4;
    --text:          #0F0F1A;
    --text-muted:    #52505A;
    --text-sidebar:  #E8E6F0;
    --text-sidebar-muted: #A09CB0;
    --accent:        #2B44EF;
    --accent-light:  #EEF1FF;
    --pro:           #065F46;
    --pro-bg:        rgba(6, 95, 70, 0.08);
    --pro-border:    rgba(6, 95, 70, 0.30);
    --con:           #9B1C1C;
    --con-bg:        rgba(155, 28, 28, 0.08);
    --con-border:    rgba(155, 28, 28, 0.30);
    --judge:         #78350F;
    --judge-bg:      rgba(120, 53, 15, 0.08);
    --judge-border:  rgba(120, 53, 15, 0.30);
    --filtered-bg:   rgba(80, 78, 90, 0.08);
    --font-head:     'Space Grotesk', system-ui, sans-serif;
    --font-mono:     'JetBrains Mono', 'Courier New', monospace;
    --font-body:     'DM Sans', system-ui, sans-serif;
    --transition:    cubic-bezier(0.4, 0, 0.2, 1);
}

/* --- BASE --- */
html, body {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

.stApp {
    background-color: var(--bg) !important;
}

/* Force Streamlit main content text to be readable */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: var(--text) !important;
}

/* Caption text in main area */
[data-testid="stCaptionContainer"] p,
.stCaption p {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}

/* Expander header */
[data-testid="stExpander"] summary span {
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* Metric label / value */
[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] div {
    color: var(--text) !important;
    font-family: var(--font-head) !important;
}

/* Info / success / warning / error alerts */
[data-testid="stAlert"] p,
[data-testid="stAlert"] strong {
    color: inherit !important;
}

/* --- SIDEBAR --- */
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 2px solid #000000 !important;
    font-size: 0.75rem !important;
}

/* All text in sidebar defaults to light */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] strong {
    color: var(--text-sidebar) !important;
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
}

/* ── Sidebar: override ALL Streamlit native accent/primary to blue ── */
section[data-testid="stSidebar"] input[type="radio"],
section[data-testid="stSidebar"] input[type="checkbox"],
section[data-testid="stSidebar"] input[type="range"] {
    accent-color: #2B44EF !important;
}

/* ── Force blue on radio selected indicator & slider fill (override Streamlit red) ── */
section[data-testid="stSidebar"] [data-testid="stRadio"] svg circle {
    fill: #2B44EF !important;
    stroke: #2B44EF !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"][aria-checked="true"] > div {
    background-color: #2B44EF !important;
    border-color: #2B44EF !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background-color: #2B44EF !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: #2B44EF !important;
    border-color: #2B44EF !important;
}

/* ── Sidebar caption / help text ── */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] .stCaption p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p small {
    color: var(--text-sidebar-muted) !important;
    font-size: 0.76rem !important;
}

/* ── Slider: fix ALL text inside slider widget ── */
section[data-testid="stSidebar"] [data-testid="stSlider"] p,
section[data-testid="stSidebar"] [data-testid="stSlider"] span,
section[data-testid="stSidebar"] [data-testid="stSlider"] div,
section[data-testid="stSidebar"] [data-testid="stSlider"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: var(--text-sidebar) !important;
}

/* ── Checkbox: card style on dark bg ── */
section[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    background: #252534;
    border: 1px solid #353548;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 4px 0;
    transition: border-color 0.18s ease, background 0.18s ease;
}
section[data-testid="stSidebar"] [data-testid="stCheckbox"]:hover {
    border-color: #2B44EF !important;
    background: #2C2C40 !important;
}
section[data-testid="stSidebar"] [data-testid="stCheckbox"] p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] span,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: var(--text-sidebar) !important;
}

/* ── Radio: card-style group ── */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div > div {
    background: #252534;
    border: 1px solid #353548;
    border-radius: 4px;
    padding: 6px 10px;
    margin: 2px 0;
    transition: border-color 0.18s ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div > div:hover {
    border-color: #2B44EF !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] p,
section[data-testid="stSidebar"] [data-testid="stRadio"] span,
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: var(--text-sidebar) !important;
}

/* ── Selectbox: dark bg ── */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #252534 !important;
    border: 1px solid #353548 !important;
    border-radius: 4px !important;
    color: var(--text-sidebar) !important;
}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] p {
    color: var(--text-sidebar) !important;
}

/* ── Text input: dark bg ── */
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: #252534 !important;
    border: 1px solid #353548 !important;
    border-radius: 4px !important;
    color: var(--text-sidebar) !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #2B44EF !important;
    box-shadow: 0 0 0 2px rgba(43,68,239,0.25) !important;
}

/* ── Expander in sidebar ── */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #252534 !important;
    border: 1px solid #353548 !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: var(--text-sidebar) !important;
}

/* Sidebar section headers */
.sidebar-section {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-sidebar-muted) !important;
    padding: 20px 0 8px 0;
    border-top: 1px solid #2E2E40;
    margin-top: 6px;
}

.sidebar-section:first-child {
    border-top: none;
    padding-top: 4px;
}

/* Sidebar module descriptions */
.module-desc {
    font-size: 0.72rem;
    color: var(--text-sidebar-muted) !important;
    margin: 0;
    line-height: 1.4;
    padding-left: 26px;
}

/* --- PAGE HEADER --- */
@keyframes title-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.page-header {
    padding: 36px 0 28px 0;
    border-bottom: 2px solid var(--border-hard);
    margin-bottom: 32px;
}

.page-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
    display: block;
}

.page-title {
    font-family: var(--font-head) !important;
    font-weight: 700;
    font-size: clamp(2rem, 3.8vw, 3.2rem);
    line-height: 1.0;
    letter-spacing: -0.03em;
    white-space: nowrap;
    background: linear-gradient(135deg, #0F0F1A 0%, #2B44EF 50%, #0F0F1A 100%);
    background-size: 250% 250%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: title-shift 7s ease infinite;
    margin: 0;
}

/* --- CLAIM INPUT SECTION --- */
.input-section {
    margin-bottom: 28px;
}

.input-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text);
    display: block;
    margin-bottom: 10px;
}

/* Text area override */
[data-testid="stTextArea"] textarea {
    font-family: var(--font-body) !important;
    font-size: 1rem !important;
    border: 2px solid var(--border-hard) !important;
    border-radius: 0 !important;
    background: var(--bg-card) !important;
    color: var(--text) !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s var(--transition) !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 3px 3px 0 var(--accent) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border: 2px solid var(--border-hard) !important;
    border-radius: 0 !important;
    background: var(--bg-card) !important;
    font-family: var(--font-body) !important;
    transition: border-color 0.2s var(--transition) !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    border: 2px solid var(--border-hard) !important;
    border-radius: 0 !important;
    background: var(--bg-card) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s var(--transition) !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 3px 3px 0 var(--accent) !important;
}

/* --- BUTTONS --- */
.stButton > button {
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: var(--border-hard) !important;
    color: var(--bg) !important;
    border: 2px solid var(--border-hard) !important;
    border-radius: 0 !important;
    padding: 14px 28px !important;
    transition:
        background 0.22s var(--transition),
        color 0.22s var(--transition),
        transform 0.18s var(--transition),
        box-shadow 0.18s var(--transition) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--accent);
    transform: translateX(-102%);
    transition: transform 0.28s var(--transition);
    z-index: 0;
}

.stButton > button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #FFFFFF !important;
    transform: translateY(-3px) !important;
    box-shadow: 4px 4px 0 var(--border-hard) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
    box-shadow: 2px 2px 0 var(--border-hard) !important;
}

/* Prevent global text-color rule from bleeding into button children */
.stButton > button p,
.stButton > button span,
.stButton > button div,
.stButton > button small {
    color: inherit !important;
    font-size: inherit !important;
    font-family: inherit !important;
    letter-spacing: inherit !important;
    text-transform: inherit !important;
}

/* --- SLIDERS --- */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--border-hard) !important;
}

[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background-color: var(--accent) !important;
}

/* --- RADIO BUTTONS --- */
[data-testid="stRadio"] label {
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
}

/* --- SECTION HEADINGS --- */
.section-heading {
    font-family: var(--font-head);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border-hard);
    margin: 36px 0 20px 0;
}

.section-heading-mono {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
    display: block;
}

/* --- EXAMPLES BLOCK --- */
.examples-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.examples-list li {
    font-size: 0.9rem;
    color: var(--text-muted);
    padding: 8px 0;
    border-bottom: 1px solid var(--border-soft);
    line-height: 1.5;
    font-style: italic;
}

.examples-list li:last-child {
    border-bottom: none;
}

/* --- TRANSCRIPT CARDS --- */
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}

.turn-card {
    border-left: 4px solid var(--border-soft);
    background: var(--bg-card);
    padding: 14px 18px 10px 18px;
    margin-bottom: 4px;
    animation: slideInLeft 0.28s var(--transition) forwards;
}

.turn-card.role-pro {
    border-left-color: var(--pro);
    background: var(--pro-bg);
    margin-right: 14%;
}

.turn-card.role-con {
    border-left: none;
    border-right: 4px solid var(--con);
    background: var(--con-bg);
    margin-left: 14%;
}

.turn-card.role-con .turn-header-row {
    flex-direction: row-reverse;
}

.turn-card.role-judge {
    border-left-color: var(--judge);
    background: var(--judge-bg);
}

.turn-card.role-filtered {
    border-left-color: var(--text-muted);
    background: var(--filtered-bg);
    opacity: 0.6;
}

.turn-header-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}

.turn-role {
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    padding: 2px 8px;
    border-radius: 2px;
    line-height: 1.8;
}

.turn-role.pro  { color: var(--pro);   background: rgba(6,95,70,0.12);   }
.turn-role.con  { color: var(--con);   background: rgba(153,27,27,0.12); }
.turn-role.judge{ color: var(--judge); background: rgba(120,53,15,0.12); }
.turn-role.filtered { color: var(--text-muted); background: rgba(107,102,112,0.12); }

.turn-meta {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}

.tag-filtered {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #FFFFFF;
    background: var(--text-muted);
    padding: 2px 7px;
    border-radius: 2px;
    margin-left: auto;
}

.tag-retained {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #FFFFFF;
    background: var(--accent);
    padding: 2px 7px;
    border-radius: 2px;
    margin-left: auto;
}

.turn-body {
    font-family: var(--font-body);
    font-size: 0.95rem;
    line-height: 1.72;
    color: var(--text);
    margin: 0;
}

.turn-body-filtered {
    opacity: 0.55;
    font-style: italic;
}

.turn-filter-reason {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed var(--border-soft);
}

.turn-spacer {
    height: 12px;
}

/* --- CITATION BADGES --- */
.cite-badge {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 2px 7px;
    border-radius: 2px;
    margin-left: 6px;
    vertical-align: middle;
    display: inline-block;
    transition: opacity 0.15s ease;
}

.cite-badge:hover { opacity: 0.75; }

.cite-pro {
    background: var(--pro);
    color: #FFFFFF;
}

.cite-con {
    background: var(--con);
    color: #FFFFFF;
}

/* --- VERDICT CARD --- */
@keyframes verdict-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes border-pulse {
    0%, 100% { border-left-width: 5px; }
    50%       { border-left-width: 8px; }
}

.verdict-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 0;
    padding: 28px 32px;
    margin: 20px 0;
    animation: verdict-in 0.4s var(--transition) forwards, border-pulse 2s ease 0.4s 2;
}

.verdict-card.supported { border-left: 5px solid var(--pro); }
.verdict-card.refuted   { border-left: 5px solid var(--con); }
.verdict-card.inconclusive { border-left: 5px solid var(--judge); }

.verdict-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
    display: block;
}

.verdict-result {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 20px;
}

.verdict-result.supported   { color: var(--pro); }
.verdict-result.refuted     { color: var(--con); }
.verdict-result.inconclusive{ color: var(--judge); }

.verdict-stats {
    display: flex;
    gap: 40px;
    margin-bottom: 22px;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--border-soft);
}

.verdict-stat-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    display: block;
    margin-bottom: 4px;
}

.verdict-stat-value {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
}

.verdict-justification-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
    display: block;
}

.verdict-justification-body {
    font-family: var(--font-body);
    font-size: 1rem;
    line-height: 1.72;
    color: var(--text);
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* --- CONSENSUS METRICS --- */
.quadrant-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-left: 4px solid var(--accent);
    padding: 18px 22px;
    margin-bottom: 20px;
}

.quadrant-name {
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    margin-bottom: 6px;
    display: block;
}

.quadrant-explanation {
    font-size: 0.92rem;
    line-height: 1.6;
    color: var(--text);
}

/* Metric values */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    padding: 14px 16px;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* --- PAPERS COLUMNS --- */
.papers-col-title {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding-bottom: 10px;
    border-bottom: 2px solid currentColor;
    margin-bottom: 14px;
}

.papers-empty {
    font-size: 0.88rem;
    color: var(--text-muted);
    font-style: italic;
}

/* --- EXPANDER STYLING --- */
[data-testid="stExpander"] {
    border: 1px solid var(--border-soft) !important;
    border-radius: 0 !important;
    background: var(--bg-card) !important;
}

[data-testid="stExpander"] summary {
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* --- ALERTS & MESSAGES --- */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    border-left-width: 4px !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
}

/* --- SPINNER TEXT --- */
[data-testid="stSpinner"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
}

/* --- DIVIDER --- */
hr {
    border-color: var(--border-soft) !important;
    margin: 28px 0 !important;
}

/* --- PLOTLY CHARTS --- */
.js-plotly-plot {
    border: 1px solid var(--border-soft);
}

/* --- CODE BLOCKS (raw judge output) --- */
[data-testid="stCode"] pre,
[data-testid="stCode"] code,
.stCodeBlock pre,
.stCodeBlock code {
    background: #EEEAE0 !important;
    color: var(--text) !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 0 !important;
}

/* Primary action button → accent blue */
[data-testid="baseButton-primary"],
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #FFFFFF !important;
}
[data-testid="baseButton-primary"] p,
[data-testid="baseButton-primary"] span,
[data-testid="baseButton-primary"] div {
    color: #FFFFFF !important;
}
[data-testid="baseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: #1B30CC !important;
    border-color: #1B30CC !important;
    transform: translateY(-3px) !important;
    box-shadow: 4px 4px 0 var(--border-hard) !important;
}

/* --- TOP HEADER BAR (Deploy / menu) --- */
[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid var(--border-soft) !important;
}
[data-testid="stHeader"] button,
[data-testid="stHeader"] a,
[data-testid="stHeader"] svg,
[data-testid="stHeader"] svg path {
    color: var(--text) !important;
    fill: var(--text) !important;
    stroke: var(--text) !important;
}

/* --- COLLAPSED SIDEBAR TOGGLE --- */
[data-testid="collapsedControl"] {
    background: var(--accent) !important;
    border-radius: 4px !important;
}
[data-testid="collapsedControl"]:hover {
    background: #1B30CC !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg path {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
}
[data-testid="collapsedControl"] button {
    color: #FFFFFF !important;
}

/* --- HIDE NUMBER INPUT KEYBOARD HINT --- */
[data-testid="InputInstructions"],
.stNumberInput [data-testid="InputInstructions"],
section[data-testid="stSidebar"] [data-testid="InputInstructions"] {
    display: none !important;
    visibility: hidden !important;
}

/* ═══ SIDEBAR: BLUE BACKGROUND THEME (all overrides cascade last) ═══ */

/* Subtle dark border at the right edge */
section[data-testid="stSidebar"] {
    border-right-color: rgba(0, 0, 0, 0.35) !important;
}

/* Native form element accent → white on blue bg */
section[data-testid="stSidebar"] input[type="radio"],
section[data-testid="stSidebar"] input[type="checkbox"],
section[data-testid="stSidebar"] input[type="range"] {
    accent-color: #FFFFFF !important;
}

/* Custom radio selected circle → white */
section[data-testid="stSidebar"] [data-testid="stRadio"] svg circle {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] > div:first-child,
section[data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] > span:first-child {
    background-color: #FFFFFF !important;
    border-color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] > div > div {
    background-color: #FFFFFF !important;
}

/* Checkbox + radio option cards */
section[data-testid="stSidebar"] [data-testid="stCheckbox"],
section[data-testid="stSidebar"] [data-testid="stRadio"] > div > div {
    background: rgba(0, 0, 0, 0.18) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
section[data-testid="stSidebar"] [data-testid="stCheckbox"]:hover,
section[data-testid="stSidebar"] [data-testid="stRadio"] > div > div:hover {
    background: rgba(0, 0, 0, 0.32) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
}

/* Selectbox */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(0, 0, 0, 0.22) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Text input */
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(0, 0, 0, 0.22) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: rgba(255, 255, 255, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.15) !important;
}

/* Expander */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(0, 0, 0, 0.18) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Number inputs */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: rgba(0, 0, 0, 0.22) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #FFFFFF !important;
    font-size: 0.78rem !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
}

/* Section label + muted text */
.sidebar-section {
    border-top-color: rgba(255, 255, 255, 0.18) !important;
    color: rgba(255, 255, 255, 0.65) !important;
}
.module-desc {
    color: rgba(255, 255, 255, 0.65) !important;
}

</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ── LOAD CONFIG ───────────────────────────────────────────────

config = load_config()

# ── SIDEBAR ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-section" style="border-top:none;padding-top:0">LLM Engine</div>', unsafe_allow_html=True)

    llm_backends_config = config.get("llm_backends", {})
    mode_val = llm_backends_config.get("mode", "Heter-MAD (different models)")
    mode_index = 1 if mode_val == "Heter-MAD (different models)" else 0

    backend_mode = st.radio(
        "Agent Mode",
        ["Same model (all agents)", "Heter-MAD (different models)"],
        index=mode_index,
    )

    backend_options = ["Ollama (local)", "Groq (cloud)", "OpenRouter (cloud)"]

    if backend_mode == "Same model (all agents)":
        same_model_cfg = llm_backends_config.get("same_model", {})
        default_backend = same_model_cfg.get("backend", "Ollama (local)")
        backend_idx = backend_options.index(default_backend) if default_backend in backend_options else 0
        backend_type = st.selectbox("Backend", backend_options, index=backend_idx)
        model_name = st.text_input("Model name", value=same_model_cfg.get("model", "qwen2.5:3b"))
        pro_config = con_config = judge_config = {"backend": backend_type, "model": model_name}
    else:
        st.caption("Set a different LLM for each agent role.")
        heter_mad_cfg = llm_backends_config.get("heter_mad", {})

        col_pro, col_con = st.columns(2)
        with col_pro:
            st.markdown("**PRO**")
            pro_cfg = heter_mad_cfg.get("pro", {})
            pro_b = pro_cfg.get("backend", "OpenRouter (cloud)")
            pro_b_idx = backend_options.index(pro_b) if pro_b in backend_options else 2
            pro_backend = st.selectbox("Backend", backend_options, index=pro_b_idx, key="pro_backend")
            pro_model = st.text_input("Model", value=pro_cfg.get("model", "openai/gpt-oss-120b:free"), key="pro_model")
            pro_config = {"backend": pro_backend, "model": pro_model}

        with col_con:
            st.markdown("**CON**")
            con_cfg = heter_mad_cfg.get("con", {})
            con_b = con_cfg.get("backend", "Groq (cloud)")
            con_b_idx = backend_options.index(con_b) if con_b in backend_options else 1
            con_backend = st.selectbox("Backend", backend_options, index=con_b_idx, key="con_backend")
            con_model = st.text_input("Model", value=con_cfg.get("model", "llama-3.3-70b-versatile"), key="con_model")
            con_config = {"backend": con_backend, "model": con_model}

        st.markdown("**JUDGE**")
        judge_cfg = heter_mad_cfg.get("judge", {})
        judge_b = judge_cfg.get("backend", "Groq (cloud)")
        judge_b_idx = backend_options.index(judge_b) if judge_b in backend_options else 1
        judge_backend = st.selectbox("Backend", backend_options, index=judge_b_idx, key="judge_backend")
        judge_model = st.text_input("Model", value=judge_cfg.get("model", "meta-llama/llama-4-scout-17b-16e-instruct"), key="judge_model")
        judge_config = {"backend": judge_backend, "model": judge_model}

    # ── DEBATE PARAMETERS ─────────────────────────────────────
    st.markdown('<div class="sidebar-section">Debate Parameters</div>', unsafe_allow_html=True)

    debate_settings = config.get("debate_settings", {})
    max_rounds = st.number_input("Rounds", min_value=1, max_value=4, value=int(debate_settings.get("max_rounds", 3)), step=1)

    if backend_mode == "Same model (all agents)":
        same_model_cfg = llm_backends_config.get("same_model", {})
        temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=float(same_model_cfg.get("temperature", 0.5)), step=0.1, format="%.1f")
        pro_temp = con_temp = judge_temp = temperature
    else:
        heter_mad_cfg = llm_backends_config.get("heter_mad", {})
        with st.expander("Per-Agent Temperatures", expanded=True):
            pro_temp   = st.number_input("PRO",   min_value=0.0, max_value=1.0, value=float(heter_mad_cfg.get("pro",   {}).get("temperature", 0.5)), step=0.1, format="%.1f", key="pro_temp")
            con_temp   = st.number_input("CON",   min_value=0.0, max_value=1.0, value=float(heter_mad_cfg.get("con",   {}).get("temperature", 0.7)), step=0.1, format="%.1f", key="con_temp")
            judge_temp = st.number_input("JUDGE", min_value=0.0, max_value=1.0, value=float(heter_mad_cfg.get("judge", {}).get("temperature", 0.1)), step=0.1, format="%.1f", key="judge_temp")

    max_tokens = st.number_input("Max tokens/turn", min_value=100, max_value=800, value=int(debate_settings.get("max_tokens", 300)), step=50)

    # ── MODULES ───────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Modules</div>', unsafe_allow_html=True)

    parallel_opening = st.checkbox(
        "Parallel Start",
        value=debate_settings.get("parallel_opening", False),
    )
    st.markdown('<p class="module-desc">PRO & CON open at the same time</p>', unsafe_allow_html=True)

    dar_settings = config.get("dar_settings", {})
    use_dar = st.checkbox(
        "Argument Filter",
        value=dar_settings.get("enable_dar", False),
    )
    st.markdown('<p class="module-desc">Filter redundant arguments each round</p>', unsafe_allow_html=True)

    rag_settings = config.get("rag_settings", {})
    use_rag = st.checkbox(
        "Scientific Evidence",
        value=rag_settings.get("enable_rag", True),
    )
    st.markdown('<p class="module-desc">Ground agents in scientific literature</p>', unsafe_allow_html=True)

    if use_rag:
        source_options = ["hybrid", "scifact", "arxiv"]
        source_default = rag_settings.get("source", "hybrid")
        source_idx = source_options.index(source_default) if source_default in source_options else 0
        rag_source = st.selectbox(
            "Evidence source",
            source_options,
            index=source_idx,
            format_func=lambda x: {
                "hybrid":   "Hybrid — SciFact + arXiv",
                "scifact":  "SciFact local dataset",
                "arxiv":    "arXiv realtime API",
            }.get(x, x),
        )
        rag_max_results = st.number_input(
            "Max docs", min_value=1, max_value=5,
            value=int(rag_settings.get("max_results", 5)), step=1,
        )
    else:
        rag_source = rag_settings.get("source", "hybrid")
        rag_max_results = rag_settings.get("max_results", 5)

    uncertainty_settings = config.get("uncertainty_settings", {})
    compute_uncertainty = st.checkbox(
        "Consensus Metrics",
        value=uncertainty_settings.get("compute_consensus", True),
    )
    if compute_uncertainty:
        n_samples = st.number_input(
            "Analysis samples", min_value=3, max_value=10,
            value=int(uncertainty_settings.get("n_samples", 4)), step=1,
        )
        enable_early_stopping = st.checkbox(
            "Early Stopping",
            value=uncertainty_settings.get("enable_early_stopping", True),
        )
        use_logprobs = st.checkbox(
            "Fast Estimation",
            value=uncertainty_settings.get("use_logprobs", True),
        )
        u_modes = [
            "Default",
            "Use local Ollama (free, slower)",
            "Use same models as debate",
        ]
        u_mode_val = uncertainty_settings.get("uncertainty_backend_mode", "Use same models as debate")
        u_mode_idx = u_modes.index(u_mode_val) if u_mode_val in u_modes else 2
        uncertainty_mode = st.radio("Analysis backend", u_modes, index=u_mode_idx)
        u_backend = uncertainty_settings.get("uncertainty_backend", "Groq (cloud)")
        u_model = uncertainty_settings.get("uncertainty_model", "llama-3.3-70b-versatile")
        uncertainty_model_local = uncertainty_settings.get("uncertainty_model_local", "qwen2.5:3b")
        if uncertainty_mode == "Default":
            backend_options = ["Ollama (local)", "Groq (cloud)", "OpenRouter (cloud)"]
            u_backend_idx = backend_options.index(u_backend) if u_backend in backend_options else 1
            
            # Chọn backend tương ứng trong cấu hình default
            u_backend = st.selectbox("Backend for default uncertainty", backend_options, index=u_backend_idx)
            # Nhập model tương ứng
            u_model = st.text_input("Model for default uncertainty", value=u_model)
            
        elif "Ollama" in uncertainty_mode:
            # Option 2: Sử dụng uncertainty_model_local
            uncertainty_model_local = st.text_input("Ollama model for uncertainty", value=uncertainty_model_local)
    else:
        n_samples = uncertainty_settings.get("n_samples", 5)
        enable_early_stopping = uncertainty_settings.get("enable_early_stopping", False)
        use_logprobs = uncertainty_settings.get("use_logprobs", True)
        uncertainty_mode = "Use same models as debate"
        uncertainty_model = uncertainty_settings.get("uncertainty_model", "qwen2.5:3b")



# ── MAIN CONTENT ──────────────────────────────────────────────

st.markdown("""
<div class="page-header">
    <span class="page-eyebrow">Multi-Agent Debate System</span>
    <h1 class="page-title">Scientific Claim Verifier</h1>
</div>
""", unsafe_allow_html=True)

# ── CLAIM INPUT ───────────────────────────────────────────────

st.markdown('<span class="input-label">Scientific claim to verify</span>', unsafe_allow_html=True)

claim = st.text_area(
    label="claim_input",
    label_visibility="collapsed",
    value="",
    height=90,
    max_chars=500,
    placeholder="Enter a scientific claim to verify (max 500 characters) — e.g., 'Vitamin C prevents the common cold.'",
)

start = st.button("Initiate Debate", type="primary", use_container_width=True)


# ── DEBATE EXECUTION ──────────────────────────────────────────

if start:
    if not claim.strip():
        st.error("Please enter a scientific claim before initiating the debate.")
        st.stop()

    pre_retrieved_pro = None
    pre_retrieved_con = None

    try:
        pro_llm   = _create_llm(pro_config,   pro_temp,   max_tokens)
        con_llm   = _create_llm(con_config,   con_temp,   max_tokens)
        judge_llm = _create_llm(judge_config, judge_temp, max_tokens)

        if compute_uncertainty:
            if uncertainty_mode == "Default":
                # Option 1: Tạo LLM động dựa trên backend và model đã cấu hình
                u_pro_llm = _create_llm({"backend": u_backend, "model": u_model}, 0.8, 100)
                u_con_llm = _create_llm({"backend": u_backend, "model": u_model}, 0.8, 100)
                u_judge_llm = _create_llm({"backend": u_backend, "model": u_model}, 0.9, 100)
            elif "Ollama" in uncertainty_mode:
                # Option 2: Sử dụng Ollama cục bộ với model local
                u_pro_llm = OllamaLLM(model=uncertainty_model_local, temperature=0.8, max_tokens=100)
                u_con_llm = OllamaLLM(model=uncertainty_model_local, temperature=0.8, max_tokens=100)
                u_judge_llm = OllamaLLM(model=uncertainty_model_local, temperature=0.9, max_tokens=100)
            else:
                # Option 3: Giữ nguyên logic (dùng chính model của debate)
                u_pro_llm = None
                u_con_llm = None
                u_judge_llm = None
        
        else:
            u_pro_llm = None, 
            u_con_llm = None,
            u_judge_llm = None

        if use_dar:
            dar_cfg  = config.get("dar_settings", {})
            f_backend = dar_cfg.get("filter_backend", "Groq (cloud)")
            f_model   = dar_cfg.get("filter_model", "llama-3.1-8b-instant")
            f_temp    = dar_cfg.get("filter_temperature", 0.0)
            if "Groq" in f_backend:
                groq_key = os.environ.get("GROQ_API_KEY")
                if not groq_key:
                    st.error("GROQ_API_KEY is not set. Required for the DAR filter agent when using Groq.")
                    st.stop()
                filter_llm = GroqLLM(model=f_model, temperature=f_temp, max_tokens=300, api_key=groq_key)
            elif "Ollama" in f_backend:
                filter_llm = OllamaLLM(model=f_model, temperature=f_temp, max_tokens=300)
            else:
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    st.error("OPENROUTER_API_KEY is not set. Required for the DAR filter agent when using OpenRouter.")
                    st.stop()
                filter_llm = OpenRouterLLM(model=f_model, temperature=f_temp, max_tokens=300, api_key=api_key)
        else:
            filter_llm = None

    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()

    debate = Debate(
        pro_llm=pro_llm,
        con_llm=con_llm,
        judge_llm=judge_llm,
        max_rounds=max_rounds,
        parallel_opening=parallel_opening,
        compute_uncertainty=compute_uncertainty,
        n_uncertainty_samples=n_samples,
        verbose=False,
        uncertainty_pro_llm=u_pro_llm,
        uncertainty_con_llm=u_con_llm,
        uncertainty_judge_llm=u_judge_llm,
        use_dar=use_dar,
        filter_llm=filter_llm,
        enable_early_stopping=enable_early_stopping,
        use_logprobs=use_logprobs,
        enable_rag=use_rag,
        rag_max_results=rag_max_results,
        rag_source=rag_source,
    )

    st.divider()

    # ── TRANSCRIPT ────────────────────────────────────────────
    st.markdown('<div class="section-heading">Debate Transcript</div>', unsafe_allow_html=True)
    transcript_container = st.container()

    def on_turn(turn):
        with transcript_container:
            pro_papers = pre_retrieved_pro.get("papers", []) if (use_rag and pre_retrieved_pro) else []
            con_papers = pre_retrieved_con.get("papers", []) if (use_rag and pre_retrieved_con) else []
            highlighted_content = highlight_citations(turn.content, pro_papers, con_papers)

            if turn.filtered_out:
                role_key   = "filtered"
                role_label = "FILTERED"
            elif turn.speaker == "PRO":
                role_key   = "pro"
                role_label = "PRO"
            elif turn.speaker == "CON":
                role_key   = "con"
                role_label = "CON"
            else:
                role_key   = "judge"
                role_label = "JUDGE"

            filter_tag   = '<span class="tag-filtered">DAR FILTERED</span>' if turn.filtered_out else ""
            retained_tag = (
                '<span class="tag-retained">DAR RETAINED</span>'
                if (use_dar and turn.filter_reason and not turn.filtered_out) else ""
            )

            card_placeholder = st.empty()

            def stream_typewriter(placeholder, raw, full_card_html):
                def gen():
                    for w in raw.split(" "):
                        yield w + " "
                        time.sleep(0.006)
                placeholder.write_stream(gen())
                placeholder.markdown(full_card_html, unsafe_allow_html=True)

            meta_text = (
                f"Round {turn.round_num}"
                if role_key in ("pro", "con")
                else f"Round {turn.round_num}&nbsp;&nbsp;&middot;&nbsp;&nbsp;{turn.model}"
            )
            header_html = (
                f'<div class="turn-header-row">'
                f'<span class="turn-role {role_key}">{role_label}</span>'
                f'<span class="turn-meta">{meta_text}</span>'
                f'{filter_tag}{retained_tag}'
                f'</div>'
            )

            if turn.filtered_out:
                reason_html = (
                    f'<div class="turn-filter-reason">Filtered: {turn.filter_reason}</div>'
                    if turn.filter_reason else ""
                )
                body_html = f'<div class="turn-body turn-body-filtered">{highlighted_content}</div>{reason_html}'
            else:
                body_html = f'<div class="turn-body">{highlighted_content}</div>'

            full_card_html = (
                f'<div class="turn-card role-{role_key}">'
                f'{header_html}'
                f'{body_html}'
                f'</div>'
            )

            stream_typewriter(card_placeholder, turn.content, full_card_html)
            st.markdown('<div class="turn-spacer"></div>', unsafe_allow_html=True)

    def on_consensus(round_num, consensus):
        with transcript_container:
            if isinstance(consensus, dict):
                # 🟢 INTERMEDIATE: Round 2-n-1 (JSD only)
                jsd = consensus.get("jsd", 0.0)
                pro_dist = consensus.get("pro_dist", {})
                con_dist = consensus.get("con_dist", {})
                
                # Format distributions
                pro_str = ", ".join(f"{k}={v:.0%}" for k, v in pro_dist.items() if v > 0)
                con_str = ", ".join(f"{k}={v:.0%}" for k, v in con_dist.items() if v > 0)
                
                # Color coding for JSD
                if jsd < 0.3:
                    jsd_color = "🟢"  # Green: strong consensus
                    jsd_desc = "Strong Agreement"
                elif jsd < 0.5:
                    jsd_color = "🟡"  # Yellow: moderate
                    jsd_desc = "Moderate Disagreement"
                else:
                    jsd_color = "🔴"  # Red: high disagreement
                    jsd_desc = "High Disagreement"
                
                st.info(
                    f"{jsd_color} **Round {round_num} — Pro/Con Disagreement (JSD)**\n\n"
                    f"- **JSD Score**: {jsd:.3f} ({jsd_desc})\n"
                    f"- **Pro Distribution**: [{pro_str}]\n"
                    f"- **Con Distribution**: [{con_str}]"
                )
            
            else:
                # 🔵 FINAL: Full ConsensusMetrics object (final round or early stop)
                st.success(
                    f"✅ **Round {round_num} — Final Consensus Metrics**\n\n"
                    f"- **Consensus Level**: {consensus.consensus_level}\n"
                    f"- **Quadrant**: {consensus.consensus_quadrant}\n"
                    f"- **JSD (Pro/Con Disagreement)**: {consensus.jsd:.3f}\n"
                    f"- **Entropy (Judge Stability)**: {consensus.normalized_entropy:.2f}\n"
                    f"- **Calibrated Confidence**: {consensus.calibrated_confidence:.2f}"
                )
                
                if enable_early_stopping and consensus.consensus_quadrant == "Strong Consensus" and round_num < max_rounds:
                    st.warning(f"🛑 **[EARLY STOPPING]** Strong Consensus reached at Round {round_num}. Ending debate early!")
            # if isinstance(consensus, dict):
            #     jsd      = consensus.get("jsd", 0.0)
            #     pro_dist = consensus.get("pro_dist", {})
            #     con_dist = consensus.get("con_dist", {})
            #     pro_str  = ", ".join(f"{k}={v:.0%}" for k, v in pro_dist.items() if v > 0)
            #     con_str  = ", ".join(f"{k}={v:.0%}" for k, v in con_dist.items() if v > 0)

            #     # Compute a semantically accurate label based on actual distributions,
            #     # not just raw JSD value (JSD=0.2 with a 50/50 PRO split is NOT
            #     # "Strong Agreement" even though the number looks small).
            #     def _jsd_label(jsd_val, pd, cd):
            #         if not pd or not cd:
            #             return f"JSD {jsd_val:.3f}"
            #         pt = max(pd, key=pd.get)
            #         ct = max(cd, key=cd.get)
            #         pc = pd.get(pt, 0)
            #         cc = cd.get(ct, 0)
            #         if pt == ct and pc >= 0.70 and cc >= 0.70 and jsd_val < 0.15:
            #             return f"Strong Consensus — both lean {pt}"
            #         if pt == ct and jsd_val < 0.35:
            #             return f"Weak Alignment — both lean {pt}"
            #         if jsd_val < 0.40:
            #             return f"Low Divergence — PRO→{pt}, CON→{ct}"
            #         if jsd_val < 0.60:
            #             return "Moderate Disagreement"
            #         return "High Disagreement"
            #     jsd_desc = _jsd_label(jsd, pro_dist, con_dist)

            #     st.info(
            #         f"**Round {round_num} — Pro/Con Disagreement (JSD)**\n\n"
            #         f"- **JSD Score:** {jsd:.3f} — {jsd_desc}\n"
            #         f"- **PRO Distribution:** [{pro_str}]\n"
            #         f"- **CON Distribution:** [{con_str}]"
            #     )
            # else:
            #     st.success(
            #         f"**Round {round_num} — Final Consensus Metrics**\n\n"
            #         f"- **Consensus Level:** {consensus.consensus_level}\n"
            #         f"- **Quadrant:** {consensus.consensus_quadrant}\n"
            #         f"- **JSD (Pro/Con Disagreement):** {consensus.jsd:.3f}\n"
            #         f"- **Entropy (Judge Stability):** {consensus.normalized_entropy:.2f}\n"
            #         f"- **Calibrated Confidence:** {consensus.calibrated_confidence:.2f}"
            #     )
            #     if enable_early_stopping and consensus.consensus_quadrant == "Strong Consensus":
            #         st.warning(
            #             f"**Early Stopping triggered at Round {round_num}.** "
            #             "Strong consensus detected — debate concluded early."
            #         )


    debate.on_turn_complete    = on_turn
    debate.on_consensus_complete = on_consensus

    # ── RAG PRE-FETCH ─────────────────────────────────────────
    rag_placeholder = st.empty()
    if use_rag:
        source_label = {
            "hybrid":  "SciFact + arXiv",
            "scifact": "SciFact dataset",
            "arxiv":   "arXiv API",
        }.get(rag_source, rag_source)

        with st.spinner(f"Retrieving evidence from {source_label}..."):
            try:
                from scidebate.tools import retrieve_evidence
                pre_retrieved_pro = retrieve_evidence(
                    claim, max_results=rag_max_results,
                    generator_llm=pro_llm, stance="PRO", source=rag_source,
                )
                pre_retrieved_con = retrieve_evidence(
                    claim, max_results=rag_max_results,
                    generator_llm=con_llm, stance="CON", source=rag_source,
                )
            except Exception as e:
                st.error(f"Evidence retrieval failed ({source_label}): {e}")

        pro_papers = pre_retrieved_pro.get("papers", []) if pre_retrieved_pro else []
        con_papers = pre_retrieved_con.get("papers", []) if pre_retrieved_con else []

        with rag_placeholder.container():
            st.markdown(
                f'<span class="section-heading-mono">Evidence</span>'
                f'<div class="section-heading">{source_label} — Retrieved Literature</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                render_papers_column(pro_papers, "PRO — Supporting Evidence", is_pro=True)
            with col2:
                render_papers_column(con_papers, "CON — Opposing Evidence", is_pro=False)
            st.divider()

    # ── RUN DEBATE ────────────────────────────────────────────
    spinner_msg = (
        f"Running debate — {max_rounds} round(s)"
        + (" — parallel opening" if parallel_opening else "")
        + (" — computing consensus" if compute_uncertainty else "")
    )
    with st.spinner(spinner_msg):
        try:
            result = debate.run(
                claim,
                pre_retrieved_pro=pre_retrieved_pro,
                pre_retrieved_con=pre_retrieved_con,
            )
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower() or "too many requests" in err_str.lower():
                st.error(
                    "**Rate Limit Exceeded (HTTP 429)**\n\n"
                    "The API provider has temporarily throttled requests. To resolve this:\n"
                    "- Wait 30–60 seconds and retry\n"
                    "- Reduce **Rounds** or **Max tokens / turn** in the sidebar\n"
                    "- Disable **Parallel Opening** to reduce concurrent API calls\n"
                    "- Upgrade to a paid plan for higher rate limits\n"
                    "- Switch to a different backend (e.g., Ollama for local inference)"
                )
            else:
                st.error(f"Debate failed: {e}")
            st.stop()

    # ── RE-RENDER RAG WITH CITATIONS ──────────────────────────
    if use_rag:
        pro_papers = pre_retrieved_pro.get("papers", []) if pre_retrieved_pro else []
        con_papers = pre_retrieved_con.get("papers", []) if pre_retrieved_con else []
        source_label = {"hybrid": "SciFact + arXiv", "scifact": "SciFact", "arxiv": "arXiv"}.get(rag_source, rag_source)

        with rag_placeholder.container():
            st.markdown(
                f'<span class="section-heading-mono">Evidence</span>'
                f'<div class="section-heading">{source_label} — Literature with Citations</div>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                render_papers_column(
                    pro_papers, "PRO — Supporting Evidence", is_pro=True,
                    result_transcript=result.transcript,
                    verdict_justification=result.verdict.justification if result.verdict else None,
                )
            with col2:
                render_papers_column(
                    con_papers, "CON — Opposing Evidence", is_pro=False,
                    result_transcript=result.transcript,
                    verdict_justification=result.verdict.justification if result.verdict else None,
                )
            st.divider()

    # ── VERDICT ───────────────────────────────────────────────
    st.markdown(
        '<span class="section-heading-mono">Judgment</span>'
        '<div class="section-heading">Final Verdict</div>',
        unsafe_allow_html=True,
    )

    if result.verdict:
        v = result.verdict
        v_class = (
            "supported"    if v.verdict == "SUPPORTED"    else
            "refuted"      if v.verdict == "REFUTED"      else
            "inconclusive"
        )
        v_color = (
            "var(--pro)"   if v.verdict == "SUPPORTED"    else
            "var(--con)"   if v.verdict == "REFUTED"      else
            "var(--judge)"
        )

        pro_papers = pre_retrieved_pro.get("papers", []) if (use_rag and pre_retrieved_pro) else []
        con_papers = pre_retrieved_con.get("papers", []) if (use_rag and pre_retrieved_con) else []
        highlighted_justification = highlight_citations(v.justification, pro_papers, con_papers)

        st.markdown(f"""
<div class="verdict-card {v_class}">
    <span class="verdict-label">Verdict</span>
    <div class="verdict-result {v_class}">{v.verdict}</div>
    <div class="verdict-stats">
        <div>
            <span class="verdict-stat-label">Confidence</span>
            <span class="verdict-stat-value" style="color:{v_color}">{v.confidence:.0%}</span>
        </div>
        <div>
            <span class="verdict-stat-label">Elapsed</span>
            <span class="verdict-stat-value">{result.elapsed_seconds:.1f}s</span>
        </div>
        <div>
            <span class="verdict-stat-label">Rounds</span>
            <span class="verdict-stat-value">{result.num_rounds}</span>
        </div>
    </div>
    <span class="verdict-justification-label">Judge Justification</span>
    <div class="verdict-justification-body">{highlighted_justification}</div>
</div>
""", unsafe_allow_html=True)

        if result.parallel_opening_used:
            st.caption("Parallel opening was used for this debate.")

        with st.expander("Raw judge output"):
            st.code(v.raw_output)

    # ── CONSENSUS METRICS ─────────────────────────────────────
    consensus_error = getattr(result, '_consensus_error', None)
    if consensus_error:
        err_lower = consensus_error.lower()
        if "429" in consensus_error or "rate" in err_lower or "throttl" in err_lower:
            st.warning(
                "**Consensus sampling skipped — API rate limit reached.**  \n"
                "The final verdict above is based on the deterministic Judge output and is valid. "
                "Re-run with **Fast Estimation** enabled or wait 30 seconds to also compute "
                "entropy/JSD metrics."
            )
        else:
            st.warning(f"Consensus sampling skipped: {consensus_error[:120]}")

    if result.consensus:
        st.markdown(
            '<span class="section-heading-mono">Analysis</span>'
            '<div class="section-heading">Consensus Metrics</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Entropy measures verdict stability across repeated judge samples. "
            "JSD measures divergence between PRO and CON verdict distributions. "
            "Calibrated confidence combines both."
        )

        c = result.consensus

#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Consensus Level",       c.consensus_level,             help="Overall scientific consensus strength")
#         with col2:
#             st.metric("Entropy (Judge)",       f"{c.normalized_entropy:.2f}", help="Verdict stability — 0 = stable, 1 = uncertain")
#         with col3:
#             st.metric("JSD (Pro vs Con)",      f"{c.jsd:.3f}",                help="Agent divergence — 0 = aligned, 1 = opposite")
#         with col4:
#             st.metric("Calibrated Confidence", f"{c.calibrated_confidence:.2f}", help="Final confidence adjusted by entropy and JSD")

#         st.markdown(f"""
# <div class="quadrant-card">
#     <span class="quadrant-name">{c.consensus_quadrant}</span>
#     <div class="quadrant-explanation">{c.explanation}</div>
# </div>
# """, unsafe_allow_html=True)

        # ── Consensus Quadrant Map ─────────────────────────────
        st.markdown(
            '<div class="section-heading" style="font-size:1.1rem;margin-top:8px">Consensus Quadrant Map</div>',
            unsafe_allow_html=True,
        )

        fig_q = go.Figure()

        quadrant_labels = [
            (0.25, 0.75, "Genuine<br>Controversy"),
            (0.75, 0.75, "Confused /<br>Insufficient"),
            (0.25, 0.25, "Strong<br>Consensus"),
            (0.75, 0.25, "Aligned<br>Uncertainty"),
        ]
        for qx, qy, qt in quadrant_labels:
            fig_q.add_annotation(
                x=qx, y=qy, text=qt,
                font=dict(size=13, color="rgba(15,15,26,0.40)", family="JetBrains Mono, monospace"),
                showarrow=False,
            )

        fig_q.add_hline(y=0.4, line_dash="dot", line_color="rgba(15,15,26,0.45)", line_width=2)
        fig_q.add_vline(x=0.5, line_dash="dot", line_color="rgba(15,15,26,0.45)", line_width=2)

        level_color = {"HIGH": "#065F46", "MEDIUM": "#78350F", "LOW": "#991B1B"}
        fig_q.add_trace(go.Scatter(
            x=[c.normalized_entropy],
            y=[c.jsd],
            mode="markers+text",
            marker=dict(
                size=24,
                color=level_color.get(c.consensus_level, "#6B6670"),
                line=dict(width=3, color="#FFFFFF"),
            ),
            text=[c.consensus_level],
            textposition="top center",
            textfont=dict(
                size=13,
                family="JetBrains Mono, monospace",
                color=level_color.get(c.consensus_level, "#6B6670"),
            ),
            name="Current debate",
        ))

        fig_q.update_layout(
            xaxis=dict(
                title=dict(text="Entropy (normalized)  →", font=dict(size=13, color="#0F0F1A")),
                range=[0, 1],
                autorange=False,
                tickfont=dict(size=12, color="#0F0F1A"),
                gridcolor="rgba(15,15,26,0.12)",
                zerolinecolor="rgba(15,15,26,0.25)",
                showgrid=True,
            ),
            yaxis=dict(
                title=dict(text="JSD (disagreement)  →", font=dict(size=13, color="#0F0F1A")),
                range=[0, 1],
                autorange=False,
                tickfont=dict(size=12, color="#0F0F1A"),
                gridcolor="rgba(15,15,26,0.12)",
                zerolinecolor="rgba(15,15,26,0.25)",
                showgrid=True,
            ),
            height=420,
            margin=dict(t=20, b=50, l=10, r=10),
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#F8F7F2",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#0F0F1A"),
        )
        st.plotly_chart(fig_q, use_container_width=True)

    if not compute_uncertainty and not result.consensus:
        st.info(
            "Consensus metrics are disabled. To compute entropy, JSD, and calibrated confidence, "
            "enable **Consensus Metrics** in the sidebar and re-run the debate."
        )
