"""app.py
Persona-Adaptive Customer Support Agent NovaSaas
streamlit web UI with full RAG, persona detection, and escalation support."""


import os
import sys
import json 
import time 
from pathlib import Path


import streamlit as st
from dotenv import load_dotenv
from google import genai

sys.path.insert(0, str(Path(__file__).parent))



from src.persona_detector import detect_persona
from src.rag_pipeline import get_or_create_collection, retrieve
from src.response_generator import generate_response, generate_no_context_response
from src.escalation import check_escalation, generate_handoff_summary




load_dotenv()
DATA_DIR = "./data"
PERSONA_ICONS = {
    "Technical Expert": "🔧",
    "Frustrated User": "😤",
    "Business Executive": "💼",
    "Unknown": ""
}
PERSONA_COLORS = {
    "Technical Expert": "#1e88e5",
    "Frustrated User": "#e53935",
    "Business Executive": "#43a047"
}


st.set_page_config(
    page_title="NovaSaaS Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main container */
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Persona badge */
    .persona-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        color: white;
        margin-bottom: 8px;
    }

    /* Source pill */
    .source-pill {
        display: inline-block;
        background: #f0f4ff;
        color: #3b5bdb;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
        border: 1px solid #c5d0fa;
    }

    /* Escalation box */
    .escalation-box {
        background: #fff3e0;
        border-left: 4px solid #ff6d00;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
    }

    /* Handoff summary box */
    .handoff-box {
        background: #f3e5f5;
        border: 1px solid #ce93d8;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Chat message styling */
    .user-msg {
        background: #e3f2fd;
        padding: 10px 14px;
        border-radius: 12px 12px 2px 12px;
        margin: 4px 0;
    }
    .agent-msg {
        background: #f9f9f9;
        padding: 10px 14px;
        border-radius: 12px 12px 12px 2px;
        border: 1px solid #e8e8e8;
        margin: 4px 0;
    }

    /* Confidence indicator */
    .conf-high { color: #2e7d32; font-weight: 600; }
    .conf-medium { color: #e65100; font-weight: 600; }
    .conf-low { color: #c62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)



def init_session():
    defaults = {
        "messages": [],               # {role, content, persona, sources, confidence}
        "conversation_history": [],   # {role, content} for LLM context
        "all_sources_used": [],       # track all sources across session
        "turn_count": 0,
        "dissatisfied_turns": 0,
        "escalated": False,
        "handoff_summary": None,
        "gemini_client": None,
        "chroma_collection": None,
        "kb_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()



@st.cache_resource(show_spinner=False)
def initialize_backend(api_key: str):
    """Initialize Gemini client and ChromaDB collection (cached across reruns)."""
    client = genai.Client(api_key=api_key)
    collection = get_or_create_collection(client, DATA_DIR)
    return client, collection



with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chatbot.png", width=60)
    st.title("NovaSaaS\nSupport Agent")
    st.caption("Persona-Adaptive AI Support")
    st.divider()

    # API Key input
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="Paste your API key here")

    if api_key and not st.session_state.kb_ready:
        with st.spinner("Loading knowledge base (first run takes ~60s)..."):
            try:
                client, collection = initialize_backend(api_key)
                st.session_state.gemini_client = client
                st.session_state.chroma_collection = collection
                st.session_state.kb_ready = True
                st.success(f" Knowledge base ready ({collection.count()} chunks)")
            except Exception as e:
                st.error(f"Initialization failed: {e}")

    if st.session_state.kb_ready:
        col_count = st.session_state.chroma_collection.count()
        st.metric(" KB Chunks", col_count)
        st.metric(" Turns", st.session_state.turn_count)

    st.divider()

    # Escalation config
    st.subheader(" Escalation Settings")
    max_turns = st.slider("Max turns before escalation", 2, 10, 4)
    auto_billing = st.checkbox("Auto-escalate billing/legal", value=True)

    st.divider()

    # Reset button
    if st.button(" Clear Conversation", use_container_width=True):
        for key in ["messages", "conversation_history", "all_sources_used",
                    "turn_count", "dissatisfied_turns", "escalated", "handoff_summary"]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else (
                False if isinstance(st.session_state[key], bool) else
                0 if isinstance(st.session_state[key], int) else None
            )
        st.rerun()

    st.divider()
    st.caption(" **Try these example queries:**")
    examples = [
        "My API key returns 401 even though it's correct",
        "I've been trying to login for hours and nothing works!!",
        "What's the business impact of the current outage?",
        "How do I set up Jira integration?",
        "I want a refund for last month's invoice",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["prefill"] = ex
            st.rerun()



st.title(" NovaSaaS Customer Support")
st.caption("Powered by Gemini · RAG · Persona Detection · Human Escalation")

if not st.session_state.kb_ready:
    st.info(" Enter your Gemini API key in the sidebar to start the support agent.")
    st.stop()

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

    else:  # assistant
        with st.chat_message("assistant", avatar=""):
            # Persona badge
            persona = msg.get("persona", "Unknown")
            color = PERSONA_COLORS.get(persona, "#888")
            icon = PERSONA_ICONS.get(persona, )
            st.markdown(
                f'<span class="persona-badge" style="background:{color}">'
                f'{icon} {persona}</span>',
                unsafe_allow_html=True
            )

            # Response
            st.markdown(msg["content"])

            # Sources
            sources = msg.get("sources", [])
            conf = msg.get("confidence", "")
            if sources:
                conf_class = f"conf-{conf}" if conf else "conf-medium"
                st.markdown(
                    f'<small>📎 Sources: '
                    + "".join([f'<span class="source-pill">{s}</span>' for s in sources])
                    + f' &nbsp;|&nbsp; Retrieval: <span class="{conf_class}">{conf.upper()}</span></small>',
                    unsafe_allow_html=True
                )

            # Escalation notice
            if msg.get("escalated"):
                st.markdown(
                    f'<div class="escalation-box"> <strong>Escalated to Human Agent</strong> — '
                    f'{msg.get("escalation_reason", "")}</div>',
                    unsafe_allow_html=True
                )

            # Handoff summary
            if msg.get("handoff_summary"):
                with st.expander(" Human Handoff Summary (for support agent)"):
                    st.json(msg["handoff_summary"])

# Escalation banner
if st.session_state.escalated:
    st.warning(" This conversation has been escalated to a human support agent. A specialist will reach out shortly.")


prefill = st.session_state.pop("prefill", "") if "prefill" in st.session_state else ""
user_input = st.chat_input(
    "Type your support question here...",
    disabled=st.session_state.escalated
)

# Use prefill if available
if prefill and not user_input:
    user_input = prefill

if user_input:
    client = st.session_state.gemini_client
    collection = st.session_state.chroma_collection

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    st.session_state.turn_count += 1

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing your message..."):
            
            persona_result = detect_persona(client, user_input)
            # app.py line ~297
            from google.genai.errors import ServerError

            try:
                persona_result = detect_persona(client, user_input)
            except ServerError as e:
                st.error("⚠️ AI model is temporarily busy. Please try again in a moment.")
            st.stop()
            persona = persona_result.get("persona", "Frustrated User")
            persona_reasoning = persona_result.get("reasoning", "")

            color = PERSONA_COLORS.get(persona, "#888")
            icon = PERSONA_ICONS.get(persona, "❓")
            st.markdown(
                f'<span class="persona-badge" style="background:{color}">'
                f'{icon} {persona}</span>',
                unsafe_allow_html=True
            )

        with st.spinner("Searching knowledge base..."):
            retrieval = retrieve(client, collection, user_input)
            sources = retrieval["sources"]
            confidence = retrieval["confidence"]
            context_text = retrieval["context_text"]

            # Track all sources used in this session
            for s in sources:
                if s not in st.session_state.all_sources_used:
                    st.session_state.all_sources_used.append(s)

        
        from src.escalation import ESCALATION_CONFIG
        ESCALATION_CONFIG["max_turns_before_escalation"] = max_turns
        ESCALATION_CONFIG["billing_legal_auto_escalate"] = auto_billing

        escalation = check_escalation(
            user_message=user_input,
            retrieval_confidence=confidence,
            conversation_turns=st.session_state.turn_count,
            dissatisfied_turns=st.session_state.dissatisfied_turns,
            persona=persona
        )

        # Detect dissatisfaction in message for counter
        dissatisfaction_words = ["doesn't work", "not working", "broken", "useless",
                                  "frustrated", "angry", "ridiculous", "terrible"]
        if any(w in user_input.lower() for w in dissatisfaction_words):
            st.session_state.dissatisfied_turns += 1


        with st.spinner("Generating response..."):
            if confidence == "low" and not context_text.strip():
                response_text = generate_no_context_response(client, persona, user_input)
            else:
                response_text = generate_response(
                    client=client,
                    persona=persona,
                    user_message=user_input,
                    context_text=context_text,
                    conversation_history=st.session_state.conversation_history[:-1]
                )

        st.markdown(response_text)

        # Show sources
        if sources:
            conf_class = f"conf-{confidence}"
            st.markdown(
                f'<small>📎 Sources: '
                + "".join([f'<span class="source-pill">{s}</span>' for s in sources])
                + f' &nbsp;|&nbsp; Retrieval: <span class="{conf_class}">{confidence.upper()}</span></small>',
                unsafe_allow_html=True
            )

        
        handoff_summary = None
        if escalation["should_escalate"] and not st.session_state.escalated:
            st.session_state.escalated = True
            with st.spinner("Generating handoff summary..."):
                handoff_summary = generate_handoff_summary(
                    client=client,
                    persona=persona,
                    conversation_history=st.session_state.conversation_history,
                    sources_used=st.session_state.all_sources_used,
                    escalation_reason=escalation["reason"]
                )

            st.markdown(
                f'<div class="escalation-box">🚨 <strong>Escalating to Human Agent</strong><br>'
                f'{escalation["reason"]}</div>',
                unsafe_allow_html=True
            )

            with st.expander("📋 Human Handoff Summary (for support agent)", expanded=True):
                st.json(handoff_summary)

    # Update conversation history with assistant response
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": response_text
    })

    # Store message in display history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "persona": persona,
        "sources": sources,
        "confidence": confidence,
        "escalated": escalation["should_escalate"],
        "escalation_reason": escalation.get("reason"),
        "handoff_summary": handoff_summary
    })

    st.rerun()
