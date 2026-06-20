# Support-Agent
# 🤖 Persona-Adaptive Customer Support Agent

An intelligent AI-powered customer support agent that detects the user's communication persona, retrieves relevant information from a knowledge base using RAG, generates persona-adapted responses, and escalates to a human agent when necessary.

**Built for: Adsparkx AI Engineering Intern Assignment**

---

## 📸 Demo

> _See `demo.mp4` for a full walkthrough of all features._

---

## 🧠 Project Overview

This system acts as a first-line support agent for **NovaSaaS** (a fictional SaaS project management platform). It:

1. **Detects** whether the user is a _Technical Expert_, _Frustrated User_, or _Business Executive_
2. **Retrieves** the most relevant support documentation from a 13-document knowledge base
3. **Generates** a response in the exact tone and style appropriate for that persona
4. **Escalates** the conversation to a human agent when it detects billing issues, low retrieval confidence, repeated dissatisfaction, or account-sensitive queries
5. **Produces** a structured JSON handoff summary for the human agent

---

## 🏗️ Architecture Diagram

```
User Query
    │
    ▼
┌─────────────────────┐
│   Persona Detection  │  ← Gemini 1.5 Flash (zero-shot JSON classification)
│  (gemini-1.5-flash) │
└─────────┬───────────┘
          │  persona label
          ▼
┌─────────────────────┐
│   RAG Pipeline      │
│  1. Embed query     │  ← Gemini text-embedding-004
│  2. Query ChromaDB  │  ← cosine similarity search
│  3. Retrieve top-4  │  ← chunks with source + page metadata
└─────────┬───────────┘
          │  context chunks + confidence score
          ▼
┌─────────────────────┐
│  Escalation Check   │  ← rule-based: billing keywords, low confidence,
│                     │    dissatisfied_turns, max conversation turns
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
[No Escalation] [Escalate]
    │               │
    ▼               ▼
┌──────────┐   ┌───────────────────┐
│ Response │   │ Handoff Summary   │  ← Gemini generates structured JSON
│Generator │   │ (JSON for agent)  │
└──────────┘   └───────────────────┘
    │
    ▼
Persona-Adaptive Response (Technical / Empathetic / Executive tone)
    │
    ▼
Streamlit Chat UI (displays persona badge, sources, confidence, escalation status)
```

---

## 🔧 Tech Stack

| Layer | Tool | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.11+ | Core runtime |
| LLM | Google Gemini 1.5 Flash | via `google-genai` | Persona detection, response generation, handoff summary |
| Embeddings | Gemini text-embedding-004 | via `google-genai` | Semantic vector representation of text |
| RAG Framework | LangChain | ≥0.1.0 | `RecursiveCharacterTextSplitter` for document chunking |
| Vector DB | ChromaDB | ≥0.4.0 | Local persistent vector store for embeddings |
| PDF Parsing | pypdf | ≥3.0.0 | Extract text from PDF knowledge base documents |
| UI | Streamlit | ≥1.30.0 | Interactive chat web interface |
| Env Management | python-dotenv | ≥1.0.0 | Secure API key handling |

---

## 📂 Project Structure

```
support-agent/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Excludes .env, chroma_db/, __pycache__/
├── README.md                       # This file
│
├── src/                            # Core modules
│   ├── __init__.py
│   ├── persona_detector.py         # Gemini-based persona classification
│   ├── rag_pipeline.py             # Document loading, chunking, embedding, retrieval
│   ├── response_generator.py       # Persona-adaptive response generation
│   └── escalation.py              # Escalation logic + handoff summary generation
│
└── data/                           # Knowledge base documents
    ├── 01_getting_started.md
    ├── 02_password_reset.md
    ├── 03_api_authentication.md
    ├── 04_billing_plans.md
    ├── 05_integrations.md
    ├── 06_performance_uptime.md
    ├── 07_security_privacy.md
    ├── 08_troubleshooting.md
    ├── 09_enterprise_features.md
    ├── 10_data_migration.md
    ├── 11_refund_policy.md
    ├── 12_notifications_settings.md
    └── 13_service_level_agreement.pdf   ← Required PDF document
```

---

## 🧑‍🔬 Persona Detection Strategy

### Classification Method
Zero-shot classification using **Gemini 1.5 Flash** with a structured JSON output prompt.

### How It Works
The user's message is sent to Gemini with a carefully designed prompt that:
1. Defines each persona with clear characteristics and example phrases
2. Instructs the model to output a strict JSON object with `persona`, `confidence`, and `reasoning` fields
3. Handles JSON parsing failures gracefully with a regex fallback

### Persona Definitions

| Persona | Signal Words & Patterns | Response Style |
|---|---|---|
| **Technical Expert** | API, error code, 401, OAuth, config, logs, stack trace, endpoint | Detailed, technical, root cause analysis, step-by-step commands |
| **Frustrated User** | "doesn't work", "I've tried everything", "terrible", urgent/emotional tone | Empathetic, simple language, reassuring, action-oriented |
| **Business Executive** | impact, operations, resolution, timeline, cost, business continuity | Concise, outcome-focused, professional, no jargon |

### Prompt Design
```python
"Classify the following customer message into exactly one of:
1. Technical Expert — uses technical terminology, asks about APIs/logs/configs
2. Frustrated User — emotional language, repeated failed attempts, urgent
3. Business Executive — outcome-focused, asks about impact and timeline

Respond ONLY as JSON: {persona, confidence, reasoning}"
```

---

## 🔍 RAG Pipeline Design

### Document Ingestion
- `.md` and `.txt` files: read with Python's built-in `open()` / `Path.read_text()`
- `.pdf` files: parsed page-by-page with **pypdf**, each page stored as a separate document section
- All document sections tagged with `source` (filename) and `page` metadata

### Chunking Strategy
- **Algorithm**: `RecursiveCharacterTextSplitter` from LangChain
- **Chunk Size**: 500 characters — small enough for focused retrieval, large enough for complete thoughts
- **Chunk Overlap**: 50 characters — prevents important context (e.g., an API endpoint or reset step) from being cut across chunk boundaries
- **Separators**: `["\n\n", "\n", ".", " ", ""]` — splits at paragraph breaks first, then sentences, then words

### Embedding Model
- **Model**: `text-embedding-004` (Google Gemini)
- **Dimensions**: 768 floating-point values per chunk
- **Semantic matching**: Two semantically similar sentences (e.g., "reset password" vs "forgot credentials") produce vectors close in cosine distance, enabling accurate retrieval even when exact keywords don't match

### Vector Database
- **ChromaDB** with cosine similarity space (`hnsw:space: cosine`)
- **Persistent storage**: `./chroma_db/` — embeddings are computed once and reused across sessions
- **Collection**: `novasaas_kb` — single collection for all 13 documents

### Retrieval Strategy
- Query is embedded using the same `text-embedding-004` model
- Top-4 (`TOP_K=4`) most similar chunks retrieved by cosine distance
- **Confidence scoring** based on best chunk's cosine distance:
  - `< 0.4` → **HIGH** confidence
  - `0.4 – 0.7` → **MEDIUM** confidence
  - `> 0.7` → **LOW** confidence (triggers escalation)
- Retrieved chunks injected into LLM prompt as `[KNOWLEDGE BASE CONTEXT]` with source attribution

---

## 🚨 Escalation Logic

### Escalation Triggers (in priority order)

1. **Billing/Legal/Account Keywords** — auto-escalate if message contains: `refund`, `chargeback`, `legal`, `fraud`, `data breach`, `account compromised`, `gdpr`, etc.
2. **Low Retrieval Confidence** — no sufficiently relevant KB documents found (cosine distance > 0.7)
3. **User Dissatisfaction Keywords** — message contains: `speak to a manager`, `useless`, `terrible service`, `escalate`, `cancel my account`, etc.
4. **Repeated Dissatisfaction** — dissatisfied language detected in 2+ consecutive turns
5. **Prolonged Unresolved Conversation** — conversation exceeds configurable `max_turns` with unresolved issue

### Configuration
Escalation thresholds are configurable from the Streamlit sidebar at runtime:
- `Max turns before escalation`: 2–10 (default: 4)
- `Auto-escalate billing/legal`: toggle on/off

### Handoff Summary
When escalation triggers, Gemini generates a structured JSON summary:
```json
{
  "persona": "Frustrated User",
  "issue_summary": "Customer unable to login after password reset. Has tried browser cache clearing and backup codes.",
  "escalation_reason": "User expressed strong dissatisfaction across multiple turns.",
  "documents_used": ["02_password_reset.md", "08_troubleshooting.md"],
  "attempted_steps": ["Password reset via email", "Backup code login", "Browser cache clear"],
  "customer_sentiment": "Frustrated",
  "recommended_next_steps": ["Check account lock status", "Verify 2FA configuration", "Contact customer directly"],
  "priority": "High"
}
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11 or higher
- A Google Gemini API key ([get one free here](https://aistudio.google.com/app/apikey))

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/persona-support-agent.git
cd persona-support-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
# Open .env and paste your Gemini API key:
# GEMINI_API_KEY=your_actual_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first launch, the knowledge base will be ingested and indexed (~60 seconds). Subsequent launches load from the persisted ChromaDB store instantly.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Your Google Gemini API key from Google AI Studio |

---

## 💬 Example Queries (5 Demonstrations)

### 1. Technical Expert — API Authentication
> _"My API call returns 401 even though I just generated a new key. The prefix is nsk_live. What could cause this?"_

**Expected behavior**: Detected as Technical Expert → Retrieves `03_api_authentication.md` → Provides detailed response about scope mismatches, key validation, and rate limits.

### 2. Frustrated User — Password Reset
> _"I've been trying to login for 3 hours!!! Nothing works, I reset my password twice and it STILL says invalid credentials. This is ridiculous."_

**Expected behavior**: Detected as Frustrated User → Retrieves `02_password_reset.md` + `08_troubleshooting.md` → Empathetic response with simple steps → May trigger escalation on follow-up.

### 3. Business Executive — Billing Impact
> _"How does the current service disruption impact our enterprise contract and what are the SLA credits we're entitled to?"_

**Expected behavior**: Detected as Business Executive → Retrieves `13_service_level_agreement.pdf` + `04_billing_plans.md` → Concise response about SLA credit schedule and claim process.

### 4. Escalation — Refund Request
> _"I need a full refund for the past 3 months. My team never got value from this product."_

**Expected behavior**: Billing keyword (`refund`) detected → Immediate escalation → Handoff summary generated with conversation history.

### 5. Technical Expert — Jira Integration
> _"The Jira sync is one-directional. Is bidirectional sync available and what API scopes does the NovaSaaS Jira app require?"_

**Expected behavior**: Technical Expert detected → Retrieves `05_integrations.md` → Detailed response about sync behavior, plan requirements, and permissions.

---

## ⚠️ Known Limitations and Future Improvements

### Current Limitations
- **No offline mode**: Requires an active internet connection for Gemini API calls
- **English only**: Persona detection and responses optimized for English messages
- **Single-workspace knowledge base**: KB is generic SaaS; would need customization for specific products
- **No persistent memory across sessions**: Conversation history resets when the browser tab closes
- **Gemini rate limits**: Heavy usage may hit API rate limits on the free tier

### Future Improvements
- [ ] Add **LangGraph workflow** for multi-agent orchestration (research agent, response agent, quality check agent)
- [ ] Implement **sentiment analysis** with scores to track user mood continuously
- [ ] Add **confidence scoring** displayed as a visual gauge in the UI
- [ ] **Multi-language support** using Gemini's multilingual capabilities
- [ ] **Human approval workflow** — human agent can reply directly through the UI
- [ ] **Analytics dashboard** — track escalation rates, common issues, persona distribution
- [ ] **Feedback collection** — thumbs up/down on each response to improve retrieval
- [ ] **PostgreSQL backend** for persistent conversation history across sessions

---

## 👤 Author

**Balakrishna Velamala**  
B.Tech Mechanical Engineering | NovaSaaS Support Agent Assignment  
GitHub: [@balakrishnavelamala339](https://github.com/balakrishnavelamala339)
