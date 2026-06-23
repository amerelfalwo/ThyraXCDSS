# ThyraX CDSS — API Documentation

> **Version:** 4.0.0 — Continuous Context Orchestrator
>
> **Base URL:** `https://your-domain.com` or `http://localhost:8000`

---

## 🔐 Authentication

All endpoints (except `/health`) require an internal API key passed via header:

```
X-AI-Service-Key: your-secret-key
```

| Header             | Type   | Required | Description                        |
| :----------------- | :----- | :------- | :--------------------------------- |
| `X-AI-Service-Key` | string | ✅       | Internal service authentication key |

> [!CAUTION]
> Requests without a valid `X-AI-Service-Key` will receive `403 Forbidden`.

---

## 📑 Table of Contents

| #   | Endpoint                  | Method | Description                          |
| :-- | :------------------------ | :----- | :----------------------------------- |
| 1   | `/health`                 | GET    | System health & circuit breaker status |
| 2   | `/audit/logs`             | GET    | Retrieve audit trail logs            |
| 3   | `/clinical/assess`        | POST   | Clinical assessment (XGBoost + Agentic Routing) |
| 4   | `/image/validate`         | POST   | Ultrasound image gatekeeper          |
| 5   | `/image/predict`          | POST   | Ultrasound segmentation + classification |
| 6   | `/fnac/predict`           | POST   | FNAC cytopathology (Bethesda System) |
| 7   | `/agent/chat/stream`      | POST   | Dual-mode streaming chat (multipart) |
| 8   | `/agent/chat`             | POST   | Dual-mode streaming chat (JSON)      |

---

## 1. `GET /health`

Health check endpoint. **No authentication required.**

### Response `200 OK`

```json
{
  "status": "healthy",
  "service": "ThyraX AI Engine",
  "version": "4.0.0",
  "llm_backend": "Groq (Llama-3)",
  "nodes": [
    "clinical_assessment",
    "agentic_routing",
    "ultrasound_gatekeeper",
    "onnx_segmentation",
    "fnac_cytopathology",
    "medical_agent_chat"
  ],
  "circuit_breakers": {
    "agent_chat": { "state": "closed", "failures": 0 }
  }
}
```

### Usage

```bash
curl https://your-domain.com/health
```

---

## 2. `GET /audit/logs`

Retrieve recent audit log entries for clinical traceability.

### Query Parameters

| Parameter | Type | Default | Description                     |
| :-------- | :--- | :------ | :------------------------------ |
| `limit`   | int  | 50      | Max entries to return (cap: 200) |

### Response `200 OK`

```json
{
  "entries": [
    {
      "timestamp": "2026-06-23T10:30:00Z",
      "node": "fnac_predict",
      "action": "bethesda_classification",
      "result": "Bethesda IV",
      "confidence": 0.87,
      "metadata": { "session_id": "sess-abc", "filename": "slide1.png" }
    }
  ],
  "total": 1
}
```

### Usage

```bash
curl -H "X-AI-Service-Key: YOUR_KEY" \
     "https://your-domain.com/audit/logs?limit=20"
```

---

## 3. `POST /clinical/assess`

Run the full CDSS clinical workflow: **XGBoost disease model (Node 1)** + **Agentic routing (Node 2)**.

### Request Body (`application/json`)

```json
{
  "patient_id": "P001",
  "session_id": "sess-abc-123",
  "doctor_id": "dr-ahmed",
  "age": 45,
  "on_thyroxine": 0,
  "thyroid_surgery": 0,
  "query_hyperthyroid": 1,
  "TSH": 0.3,
  "T3": 3.5,
  "TT4": 15.0,
  "FTI": 160,
  "T4U": 1.1,
  "nodule_present": true
}
```

| Field               | Type    | Required | Description                                     |
| :------------------ | :------ | :------- | :---------------------------------------------- |
| `patient_id`        | string  | ✅       | Unique patient identifier                        |
| `session_id`        | string  | ❌       | Links results to the patient's diagnostic journey |
| `doctor_id`         | string  | ❌       | Doctor ID for data isolation check               |
| `age`               | int     | ✅       | Patient age (0–120)                              |
| `on_thyroxine`      | int     | ✅       | On thyroxine? (0 or 1)                           |
| `thyroid_surgery`   | int     | ✅       | Thyroid surgery history? (0 or 1)                |
| `query_hyperthyroid`| int     | ✅       | Suspected hyperthyroidism? (0 or 1)              |
| `TSH`               | float   | ✅       | Thyroid Stimulating Hormone (µIU/mL)             |
| `T3`                | float   | ✅       | Triiodothyronine (ng/mL)                         |
| `TT4`               | float   | ✅       | Total T4 (µg/dL)                                 |
| `FTI`               | float   | ✅       | Free Thyroxine Index                             |
| `T4U`               | float   | ✅       | T4 Uptake                                        |
| `nodule_present`    | boolean | ✅       | Palpable nodule detected?                        |

### Response `200 OK`

```json
{
  "status": "success",
  "patient_id": "P001",
  "functional_status": "hyperthyroid",
  "probabilities": {
    "hypothyroid": 0.05,
    "euthyroid": 0.10,
    "hyperthyroid": 0.85
  },
  "model_confidence": 0.85,
  "needs_manual_review": false,
  "risk_level": "High",
  "clinical_recommendation": "Refer to endocrinologist for further evaluation...",
  "ai_recommendation": "Based on the clinical data...",
  "next_step": "ultrasound",
  "next_step_details": {
    "reason": "Nodule detected with hyperthyroid status",
    "urgency": "routine"
  }
}
```

### Error Responses

| Status | Condition                           |
| :----- | :---------------------------------- |
| `403`  | Doctor does not own session/patient |
| `422`  | Validation error (missing fields)  |
| `500`  | Disease model inference failure    |
| `503`  | LLM temporarily overloaded         |

### Usage

```bash
curl -X POST https://your-domain.com/clinical/assess \
  -H "Content-Type: application/json" \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -d '{
    "patient_id": "P001",
    "session_id": "sess-abc-123",
    "doctor_id": "dr-ahmed",
    "age": 45,
    "on_thyroxine": 0,
    "thyroid_surgery": 0,
    "query_hyperthyroid": 1,
    "TSH": 0.3,
    "T3": 3.5,
    "TT4": 15.0,
    "FTI": 160,
    "T4U": 1.1,
    "nodule_present": true
  }'
```

---

## 4. `POST /image/validate`

**Gatekeeper (Node 3)** — Verify that uploaded images are valid medical ultrasound images using MobileNetV2 ONNX model.

### Request (`multipart/form-data`)

| Field   | Type     | Required | Description                                  |
| :------ | :------- | :------- | :------------------------------------------- |
| `files` | File[]   | ✅       | One or more image files                       |
| `force` | boolean  | ❌       | `true` to bypass validation (human-in-the-loop) |

### Response `200 OK` — `List[ImageValidationResponse]`

```json
[
  {
    "filename": "thyroid_scan.png",
    "is_ultrasound": true,
    "confidence": 0.9832,
    "reason": "Classified as ultrasound with high confidence.",
    "status": "success"
  }
]
```

| Field           | Type    | Description                                    |
| :-------------- | :------ | :--------------------------------------------- |
| `filename`      | string  | Original filename                              |
| `is_ultrasound` | boolean | `true` if image passes ultrasound verification |
| `confidence`    | float   | Model confidence (0.0–1.0)                     |
| `reason`        | string  | Human-readable explanation                     |
| `status`        | string  | `"success"` or `"error"`                       |

### Usage

```bash
curl -X POST https://your-domain.com/image/validate \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -F "files=@thyroid_scan.png"
```

---

## 5. `POST /image/predict`

**Full ONNX Pipeline (Node 4)** — U-Net segmentation → ROI extraction → Classification with ACR TI-RADS risk assessment.

### Request (`multipart/form-data`)

| Field        | Type    | Required | Description                                    |
| :----------- | :------ | :------- | :--------------------------------------------- |
| `files`      | File[]  | ✅       | One or more ultrasound image files              |
| `force`      | boolean | ❌       | `true` to bypass gatekeeper (adds warning)      |
| `session_id` | string  | ❌       | Links result to the patient's session           |
| `doctor_id`  | string  | ❌       | Doctor ID for ownership verification            |

### Response `200 OK` — `List[ImagePredictionResponse]`

```json
[
  {
    "filename": "thyroid_scan.png",
    "status": "success",
    "ai_recommendation": "The ultrasound analysis shows...",
    "bbox": [120, 80, 340, 290],
    "classification": {
      "prediction": 1,
      "label": "suspicious",
      "confidence_pct": 87.34,
      "risk_level": "Intermediate Suspicion",
      "acr_tirads_level": "TR4",
      "clinical_recommendation": "FNA biopsy recommended for nodules ≥ 1.5 cm.",
      "needs_manual_review": false
    },
    "segmentation": {
      "method": "U-Net ONNX",
      "roi_extraction": "bounding_box_crop"
    },
    "images": {
      "mask_url": "https://storage.example.com/mask.png",
      "overlay_url": "https://storage.example.com/overlay.png",
      "roi_url": "https://storage.example.com/roi.png"
    },
    "validation_bypassed": false,
    "warning": null,
    "medical_disclaimer": "⚕️ DISCLAIMER: This is an AI-assisted risk assessment..."
  }
]
```

### Key Response Fields

| Field                    | Type   | Description                                            |
| :----------------------- | :----- | :----------------------------------------------------- |
| `classification.label`   | string | `"benign"` or `"suspicious"` (NOT "malignant")        |
| `classification.acr_tirads_level` | string | AI-estimated TI-RADS level (TR2–TR5)         |
| `classification.risk_level` | string | Very Low → Very High Suspicion                      |
| `images.mask_url`        | string | Binary segmentation mask image URL                     |
| `images.overlay_url`     | string | Mask overlaid on original ultrasound URL               |
| `images.roi_url`         | string | Cropped Region of Interest (nodule) URL                |

### Usage

```bash
curl -X POST https://your-domain.com/image/predict \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -F "files=@thyroid_scan.png" \
  -F "session_id=sess-abc-123" \
  -F "doctor_id=dr-ahmed"
```

---

## 6. `POST /fnac/predict`

**FNAC Cytopathology (Bethesda System)** — Classify FNAC slides into Bethesda Categories I–VI using EfficientNet-B4 ONNX model.

### Request (`multipart/form-data`)

| Field        | Type    | Required | Description                               |
| :----------- | :------ | :------- | :---------------------------------------- |
| `files`      | File[]  | ✅       | One or more FNAC cytopathology image files |
| `session_id` | string  | ❌       | Links result to the patient's session      |
| `doctor_id`  | string  | ❌       | Doctor ID for data isolation               |
| `patient_id` | string  | ❌       | Patient ID for data isolation              |

### Response `200 OK` — `List[FnacPredictionResponse]`

```json
[
  {
    "filename": "fnac_slide_01.png",
    "status": "success",
    "ai_recommendation": "The cytopathological analysis reveals...",
    "classification": {
      "prediction": 3,
      "bethesda_category": "IV",
      "bethesda_label": "Bethesda IV — Follicular Neoplasm / Suspicious for FN",
      "confidence_pct": 82.45,
      "malignancy_risk": "15–30%",
      "recommendation": "Diagnostic lobectomy recommended.",
      "needs_manual_review": false
    },
    "session_id": "sess-abc-123",
    "medical_disclaimer": "⚕️ DISCLAIMER: This is an AI-assisted cytopathological risk assessment..."
  }
]
```

### Bethesda Categories Reference

| Category | Label                                              | Malignancy Risk |
| :------- | :------------------------------------------------- | :-------------- |
| I        | Non-diagnostic / Unsatisfactory                    | 1–4%            |
| II       | Benign                                             | 0–3%            |
| III      | AUS / FLUS                                         | 6–18%           |
| IV       | Follicular Neoplasm / Suspicious for FN            | 15–30%          |
| V        | Suspicious for Malignancy                          | 60–75%          |
| VI       | Malignant                                          | 97–99%          |

### Usage

```bash
curl -X POST https://your-domain.com/fnac/predict \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -F "files=@fnac_slide.png" \
  -F "session_id=sess-abc-123" \
  -F "doctor_id=dr-ahmed" \
  -F "patient_id=P001"
```

---

## 7. `POST /agent/chat/stream`

**Dual-Mode Streaming Chat (Multipart)** — Accepts text, optional images, and returns SSE (Server-Sent Events).

> [!IMPORTANT]
> This endpoint supports direct file uploads via `multipart/form-data` and automatically routes images through the internal CV pipeline before sending to the LLM.

### Request (`multipart/form-data`)

| Field          | Type   | Required | Description                                     |
| :------------- | :----- | :------- | :---------------------------------------------- |
| `query`        | string | ❌       | The user's medical question                      |
| `session_id`   | string | ❌       | `null` → Mode 1 (General), provided → Mode 2    |
| `patient_id`   | string | ❌       | Required for Mode 2 (Contextual)                 |
| `doctor_id`    | string | ❌       | Required for Mode 2 (Contextual)                 |
| `chat_history` | string | ❌       | JSON string of previous messages (default `"[]"`) |
| `image`        | File   | ❌       | Optional image file (ultrasound, lab report)     |

### Modes

#### Mode 1 — General Medical Chat (`session_id = null`)
- No database validation
- Uses generic medical-assistant persona
- **Does NOT persist** conversation
- Works as a standalone medical Q&A

#### Mode 2 — Contextual Patient Chat (`session_id` provided)
- Validates doctor → session → patient ownership
- Injects full patient context (long-term + short-term memory + diagnostics)
- **Persists** conversation to `sessions` table
- Auto-injects diagnostic context (Ultrasound + FNAC results) into query

### Response (SSE Stream — `text/event-stream`)

Events are streamed line-by-line as `data: {...}\n\n`:

```
data: {"status": "thinking", "message": "Processing your query..."}

data: {"status": "streaming", "chunk": "Based on the "}

data: {"status": "streaming", "chunk": "clinical findings..."}

data: {"status": "success", "query": "What does TSH 0.3 mean?", "response": "Based on the clinical findings...", "tools_used": ["thyroid_search"]}
```

### SSE Event Status Types

| Status         | Description                                        |
| :------------- | :------------------------------------------------- |
| `thinking`     | Agent is processing the query                      |
| `streaming`    | Real-time token chunks                             |
| `success`      | Final complete response                            |
| `rejected`     | Non-medical query blocked by guardrail             |
| `retrying`     | Transient LLM error, auto-retrying                 |
| `error`        | Permanent failure                                  |
| `circuit_open` | Circuit breaker open, service temporarily disabled |

### Error Responses

| Status | Condition                                       |
| :----- | :---------------------------------------------- |
| `403`  | Doctor does not own session/patient (Mode 2)    |
| `422`  | `doctor_id` missing when `session_id` provided  |

### Usage — Mode 1 (General Chat)

```bash
curl -X POST https://your-domain.com/agent/chat/stream \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -F "query=What are the symptoms of Hashimoto's thyroiditis?"
```

### Usage — Mode 2 (Contextual Chat with Image)

```bash
curl -X POST https://your-domain.com/agent/chat/stream \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -F "query=Interpret this thyroid ultrasound" \
  -F "session_id=sess-abc-123" \
  -F "patient_id=P001" \
  -F "doctor_id=dr-ahmed" \
  -F "image=@thyroid_scan.png"
```

### Frontend Integration (JavaScript SSE)

```javascript
const formData = new FormData();
formData.append("query", userMessage);
formData.append("session_id", sessionId);     // null for Mode 1
formData.append("patient_id", patientId);
formData.append("doctor_id", doctorId);
formData.append("chat_history", JSON.stringify(chatHistory));
// formData.append("image", fileInput.files[0]);  // optional

const response = await fetch(`${BASE_URL}/agent/chat/stream`, {
  method: "POST",
  headers: { "X-AI-Service-Key": API_KEY },
  body: formData,
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  
  const text = decoder.decode(value);
  const lines = text.split("\n").filter(l => l.startsWith("data: "));
  
  for (const line of lines) {
    const payload = JSON.parse(line.replace("data: ", ""));
    
    switch (payload.status) {
      case "thinking":
        showLoader(payload.message);
        break;
      case "streaming":
        appendToChat(payload.chunk);
        break;
      case "success":
        finalizeChat(payload.response, payload.tools_used);
        break;
      case "rejected":
        showRejection(payload.response);
        break;
      case "error":
        showError(payload.response);
        break;
    }
  }
}
```

---

## 8. `POST /agent/chat`

**Dual-Mode Streaming Chat (JSON body)** — Same logic as `/agent/chat/stream` but accepts a JSON payload instead of multipart.

### Request Body (`application/json`)

```json
{
  "user_message": "What are the treatment options for papillary thyroid cancer?",
  "session_id": null,
  "patient_id": null,
  "doctor_id": null
}
```

| Field          | Type   | Required | Description                                  |
| :------------- | :----- | :------- | :------------------------------------------- |
| `user_message` | string | ✅       | The doctor's medical question                 |
| `session_id`   | string | ❌       | `null` → Mode 1, provided → Mode 2           |
| `patient_id`   | string | ❌       | Required for Mode 2                           |
| `doctor_id`    | string | ❌       | Required for Mode 2                           |

### Response (SSE Stream — `text/event-stream`)

Same SSE format as `/agent/chat/stream` (see above).

### Usage — Mode 1

```bash
curl -X POST https://your-domain.com/agent/chat \
  -H "Content-Type: application/json" \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -d '{"user_message": "What is TSH?"}'
```

### Usage — Mode 2

```bash
curl -X POST https://your-domain.com/agent/chat \
  -H "Content-Type: application/json" \
  -H "X-AI-Service-Key: YOUR_KEY" \
  -d '{
    "user_message": "Summarize this patient diagnostic journey",
    "session_id": "sess-abc-123",
    "patient_id": "P001",
    "doctor_id": "dr-ahmed"
  }'
```

---

## 🔒 Data Isolation Model

> [!WARNING]
> All Mode 2 (Contextual) endpoints enforce strict data isolation. Violations return `403 Forbidden`.

```
Doctor dr-ahmed
  └── Session sess-abc-123   ← owned by dr-ahmed
        └── Patient P001     ← owned by dr-ahmed
```

**Rules:**
1. A doctor can only access sessions they own.
2. A doctor can only access patients they own.
3. `doctor_id` is **mandatory** when `session_id` is provided.
4. Cross-doctor access attempts are logged and blocked.

---

## 🧠 Diagnostic Context Flow

The system accumulates diagnostic data across all nodes into a single session:

```
POST /clinical/assess  →  saves "clinical" key
POST /image/predict    →  saves "ultrasound" key
POST /fnac/predict     →  saves "fnac" key
POST /agent/chat       →  reads ALL keys (injected into system prompt)
```

The `diagnostic_context` JSONB in the sessions table merges (not overwrites):

```json
{
  "clinical": {
    "functional_status": "hyperthyroid",
    "risk_level": "High",
    "timestamp": "2026-06-23T10:30:00Z"
  },
  "ultrasound": {
    "label": "suspicious",
    "acr_tirads_level": "TR4",
    "confidence_pct": 87.34,
    "timestamp": "2026-06-23T10:32:00Z"
  },
  "fnac": {
    "bethesda_category": "IV",
    "bethesda_label": "Bethesda IV — Follicular Neoplasm",
    "confidence_pct": 82.45,
    "timestamp": "2026-06-23T10:35:00Z"
  }
}
```

---

## 🛡️ Medical Guardrails

The chat endpoints include a **pre-LLM guardrail** that rejects non-medical queries before incurring API costs:

**Blocked topics:** code generation, recipes, weather, stocks, jokes, poems, sports, movies, travel, math, translation.

**Response when blocked:**

```json
{
  "status": "rejected",
  "response": "I appreciate your question, but I'm ThyraX — a medical AI assistant designed to help with clinical and healthcare questions..."
}
```

> [!NOTE]
> The guardrail supports bilingual rejection messages (English/Arabic) based on the detected input language.

---

## 📊 Static Files

| Path      | Description                           |
| :-------- | :------------------------------------ |
| `/media/` | Locally saved segmentation result images (fallback when Supabase is unavailable) |

---

## ⚡ Rate Limiting & Resilience

| Feature             | Description                                                    |
| :------------------ | :------------------------------------------------------------- |
| **Circuit Breaker** | Auto-opens after repeated LLM failures; auto-recovers in ~2 min |
| **Auto-Retry**      | Transient errors (429, 503) retry 3× with delays [5s, 15s, 30s] |
| **API Key Rotation**| Cycles through multiple Groq keys on quota exhaustion          |
| **Confidence Guard**| Results with confidence < 65% are flagged `needs_manual_review: true` |
