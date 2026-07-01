---
title: ThyraX CDSS
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# ThyraX CDSS 🩺

**Unified Clinical Decision Support System API for Thyroid Cancer Diagnosis**

ThyraX CDSS (v4.0.0 - Continuous Context Orchestrator) is a comprehensive, AI-driven Clinical Decision Support System designed to assist medical professionals in the diagnosis, assessment, and synthesis of thyroid cancer cases.

## 🚀 Key Features & AI Nodes

ThyraX integrates multiple AI models and workflows into a unified, agentic pipeline:

- **Node 1 & 2: Clinical Assessment & Agentic Routing**
  Uses XGBoost for initial clinical data predictions and intelligently routes the patient's diagnostic state.
- **Node 3: Ultrasound Gatekeeper**
  Employs ONNX MobileNetV2 to validate and preprocess ultrasound images before further analysis.
- **Node 4: ONNX Segmentation & Classification**
  Provides precise segmentation and ACR TI-RADS classification for ultrasound imaging.
- **Node 5: Medical AI Assistant Chat**
  A robust agentic chatbot powered by Groq (Llama-3), RAG (ChromaDB + Sentence Transformers), and web search fallback to assist doctors with clinical queries.
- **Synthesis Node:**
  Synthesizes clinical data, radiomic features (shape, margin, echogenicity), and ultrasound predictions into a comprehensive final diagnostic report with an Image Compositor.
- **FNAC Cytopathology Node:**
  Analyzes Fine Needle Aspiration Cytology (FNAC) data based on the Bethesda System for Reporting Thyroid Cytopathology (Categories I–VI).

## 🧠 Continuous Context Orchestration

- **Dynamic Patient State Manager:** Seamlessly tracks and orchestrates patient diagnostic context across all AI nodes (`/state/{session_id}`).
- **Memory Management:** In-memory TTL caching and global Redis LLM caching for blazing-fast context retrieval.
- **Clinical Traceability:** JSONL audit logging provides a complete history of diagnostic decisions and system actions.

## ⚙️ Enterprise-Grade Architecture

- **Circuit Breaker Pattern:** Ensures resilience across all LLM-dependent and external services to prevent cascading failures.
- **Asynchronous & Scalable:** Built on FastAPI, SQLAlchemy (asyncpg), and Gunicorn for high-throughput API serving.
- **Background Task Processing:** Utilizes Celery, Redis, and RabbitMQ (Pika) for heavy processing tasks like image compositing and background report generation.
- **Database Storage:** Robust integration with PostgreSQL and Supabase for persistent, secure medical data storage.
- **Model Context Protocol (MCP):** Implements MCP client servers for scalable agentic tool integration.

## 🛠 Technology Stack

- **Backend:** FastAPI, Python 3.12+, Uvicorn, Gunicorn
- **AI / ML:** XGBoost, ONNX Runtime, scikit-learn, OpenCV, Pillow
- **LLM & Agents:** LangChain, LangChain-Groq, MCP (Model Context Protocol), ddgs (Web Search)
- **RAG System:** ChromaDB, Sentence-Transformers
- **Database:** PostgreSQL, Supabase, SQLAlchemy 2.0, asyncpg, Alembic
- **Caching & Tasks:** Redis, Celery, RabbitMQ (Pika), Cachetools

## 📖 API Endpoints Overview

- `POST /clinical/assess` — XGBoost prediction & routing
- `POST /image/validate` — Ultrasound gatekeeper (ONNX)
- `POST /image/predict` — Segmentation + ACR TI-RADS
- `POST /fnac/predict` — FNAC cytopathology analysis
- `POST /agent/chat` — Agentic medical assistant
- `POST /synthesis/review` — Final report synthesis & image composition
- `GET /state/{session_id}` — Retrieve patient diagnostic context
- `DELETE /state/{session_id}` — Clear patient session
- `GET /health` — Check system and circuit breaker health
- `GET /audit/logs` — Retrieve clinical audit logs

For full API documentation, please refer to the OpenAPI (`/docs`) endpoint when running the server, or view the `API_DOCUMENTATION.md` file.
