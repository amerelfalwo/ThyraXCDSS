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
- **Node 4: ONNX Segmentation, Classification & Radiomics**
  Provides precise segmentation, extracts radiomic features (Shape, Margin, Echogenicity), and applies ATA (American Thyroid Association) guidelines for risk stratification.
- **Node 5: Patient-Specific Clinical Chatbot**
  A fast LLM assistant powered by AsyncGroq (2-3s latency) with strict anti-hallucination guardrails. Dedicated exclusively to querying and analyzing the patient's diagnostic context.
- **Node 6: Diagnostic Synthesis & Compositing**
  Synthesizes clinical data, radiomic features, and ultrasound predictions into a comprehensive final diagnostic report with an Image Compositor.
- **Node 7: FNAC Cytopathology Node**
  Analyzes Fine Needle Aspiration Cytology (FNAC) data based on the Bethesda System for Reporting Thyroid Cytopathology (Categories I–VI).
- **Node 8: General Knowledge, RAG & Web Search**
  An AI assistant for general medical queries, retrieval-augmented generation (RAG), and web search, isolated from patient-specific data to prevent context mixing.

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
- **LLM & Inference:** Native AsyncGroq, Direct LLM Invocation, Model Context Protocol (MCP)
- **Database:** PostgreSQL, Supabase, SQLAlchemy 2.0, asyncpg, Alembic
- **Caching & Tasks:** Redis, Celery, RabbitMQ (Pika), Cachetools

## 📖 API Endpoints Overview

- `POST /clinical/assess` — XGBoost prediction & routing
- `POST /image/validate` — Ultrasound gatekeeper (ONNX)
- `POST /image/predict` — Segmentation + ATA Risk Stratification
- `POST /fnac/predict` — FNAC cytopathology analysis
- `POST /agent/chat` — Fast Dual-Mode Medical Assistant (Direct LLM)
- `POST /synthesis/review` — Final report synthesis & image composition
- `GET /state/{session_id}` — Retrieve patient diagnostic context
- `DELETE /state/{session_id}` — Clear patient session
- `GET /health` — Check system and circuit breaker health
- `GET /audit/logs` — Retrieve clinical audit logs

For full API documentation, please refer to the OpenAPI (`/docs`) endpoint when running the server, or view the `API_DOCUMENTATION.md` file.
