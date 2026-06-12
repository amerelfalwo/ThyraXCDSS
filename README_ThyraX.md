# ThyraX: AI Architecture and Technical Implementation

## 1. Executive Summary
ThyraX is an advanced Clinical Decision Support System (CDSS) specifically architected for the diagnostic lifecycle of Thyroid pathology. The system bridges the gap between fragmented clinical data and the rigorous requirements of international clinical guidelines (e.g., ATA, TI-RADS). By integrating a specialized Multi-Agent AI system with advanced Deep Learning architectures, ThyraX provides clinicians with a unified platform for risk assessment, automated medical image interpretation, and evidence-based conversational assistance.

## 2. Data Acquisition and Preprocessing
The reliability and accuracy of ThyraX's multimodal intelligence are grounded in high-fidelity medical datasets and rigorous preprocessing protocols.

### 2.1 Data Sources and Repositories
The AI models were trained, validated, and tested using the following primary repositories:
* **Ultrasound Datasets:** **TN5000** (containing over 5,000 annotated thyroid images) and **ThyroidXL** (curated by hunglc007) were utilized for nodule detection, precise segmentation, and classification.
* **Cytopathology Data:** A specialized **FNAC** (Fine-Needle Aspiration Cytology) dataset was incorporated to train the system on cellular features for Bethesda category classification.
* **Clinical Analysis:** Structured laboratory results and patient demographic data were sourced from the **Kaggle Thyroid Disease dataset**.

### 2.2 Preprocessing Pipeline
A comprehensive preprocessing stage was implemented across all data modalities:
* **Clinical Tabular Data:** Included statistical normalization and handling of missing values.
* **Radiological & Cytological Imaging:** Techniques such as artifact removal, contrast enhancement, and data augmentation were applied to ensure model robustness across varying ultrasound machine qualities and diverse clinical environments.

## 3. Advanced AI Diagnostic Pipeline
ThyraX employs a modular, sequential AI pipeline designed to mimic and augment the clinical diagnostic reasoning process.

### 3.1 Nodule Segmentation (U-Net + Attention)
To achieve precise anatomical localization, the system utilizes a **U-Net architecture enhanced with Attention Gates**. This model intelligently focuses on relevant features to identify tumor boundaries within the ultrasound scan. Once segmented, the system automatically crops the **Region of Interest (ROI)** and generates a **Bounding Box** to isolate the nodule, filtering out irrelevant background noise before classification.

### 3.2 Diagnostic Classification (EfficientNet-B4 + Attention)
The isolated ROI bounding box is then passed to a high-performance classification engine based on **EfficientNet-B4 with an integrated Attention mechanism**. This model performs a dual-role analysis:
* **Malignancy Risk Assessment:** Determining the probability of the nodule being benign or malignant based on visual textures and margins.
* **Tumor Staging:** Identifying the visual stage and morphological characteristics of the tumor to assist the surgeon in operative planning.

### 3.3 Specialized Diagnostic Nodes
Beyond ultrasound, the pipeline incorporates specialized nodes for other diagnostic modalities:
* **FNAC Classification (Custom CNN):** A proprietary Custom Convolutional Neural Network (CNN) architecture was developed to classify microscopic cytological features into the standard Bethesda I-VI categories.
* **Laboratory Analysis (XGBoost):** Structured clinical markers (e.g., TSH, T3, T4 levels) are analyzed using an **XGBoost** ensemble model to generate a biochemical risk profile.
* **Medical OCR Node:** A dedicated computer vision node utilizes localized OCR to extract structured textual data from scanned lab reports, prescriptions, and historical medical documents, seamlessly digitizing physical records into the patient's dynamic state.

## 4. Clinical Orchestration, LLM, and Dynamic RAG
The core innovation of ThyraX lies in the seamless orchestration between specialized predictive models, a highly constrained Large Language Model (LLM), and a continuously evolving medical knowledge base.

### 4.1 The Large Language Model (LLM) & Guardrails
The conversational reasoning core is powered by **Llama-3.1-8B**, deployed via the **Groq** inference engine to support ultra-low latency streaming (SSE). To ensure absolute clinical safety and eliminate hallucinations, the LLM is governed by rigorous Prompt Engineering and Code-Level Guardrails:
* **Strict Routing Protocol:** The LLM follows a deterministic multi-path routing logic. It must use the `search_medical_guidelines` tool for clinical queries, or immediately fall back to its internal clinical weights if external guidelines yield no results.
* **Non-Medical Rejection Engine:** Python-level regex patterns acting as a perimeter defense intercept and reject out-of-scope queries (e.g., coding, recipes, sports) before they reach the LLM, enforcing a strictly clinical environment.
* **Language Mirroring:** The agent dynamically adapts to the user's language, providing responses in English or localized Professional Egyptian Medical Arabic.

### 4.2 Model-Specific RAG & Clinical Advice
Each diagnostic node (XGBoost, ONNX, Custom CNN) is uniquely connected to the **Retrieval-Augmented Generation (RAG)** engine. Instead of merely outputting raw numerical probabilities, the models trigger the RAG system to interpret their specific outputs against international guidelines (such as ATA and ACR TI-RADS) to generate context-aware, evidence-based recommendations for the clinician's next diagnostic step.

### 4.3 Continuous Learning via Self-Updating RAG
Unlike traditional CDSS platforms that rely on static databases, ThyraX features a Dynamic, Self-Updating RAG mechanism (Semantic Caching) powered by Sentence-Transformers and ChromaDB:
* **Knowledge Extraction:** When a clinician asks a complex question that yields no direct results from the current ChromaDB guidelines, the LLM is forced to generate an evidence-based answer utilizing its internal clinical weights.
* **Automated Indexing:** The system automatically captures this generated clinical insight, processes it, and embeds it back into the local vector store.
* **Organic Evolution:** By externalizing the LLM's implicit knowledge into explicit, searchable vectors, the database continuously updates and enriches itself. This ensures that the system caches specialist knowledge and grows smarter with every clinical interaction.

## 5. Technical Stack Summary
* **Backend Infrastructure:** FastAPI (Uvicorn) with Multi-Agent Orchestration.
* **Computer Vision & ML Models:** U-Net+Attention, EfficientNet-B4+Attention, XGBoost, Custom CNN, Tesseract OCR.
* **LLM & Orchestration:** Groq API (Llama-3.1-8B), LangChain routing logic.
* **Vector Store & Embeddings:** ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`).
* **Deployment Strategy:** Dockerized containers hosted on Hugging Face Spaces / Render.
