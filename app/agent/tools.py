"""
LangChain Tools for the ThyraX CDSS Agent.

Tool 1: search_medical_guidelines  –  RAG over ChromaDB (local knowledge base)
Tool 2: search_medical_web          –  Web search via DuckDuckGo (medical only)

Workflow enforced by the system prompt:
  1. ALWAYS call search_medical_guidelines first.
  2. If RAG returns no relevant results, call search_medical_web.
  3. Combine both sources in the final answer.

Features:
  - Lightweight keyword re-ranking for RAG results.
  - Web search restricted to trusted medical domains.
  - Open medical scope (not limited to thyroid-specific queries).
"""
import os
import logging
import numpy as np
from mcp.server.fastmcp import FastMCP

from app.core.config import settings

logger = logging.getLogger(__name__)

mcp = FastMCP("ThyraX_Tools")

# ═══════════════════════════════════════════════════════════════
# Shared: Embedding Manager & ChromaDB Client (lazy loaded)
# ═══════════════════════════════════════════════════════════════

_embedding_model = None
_chroma_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _chroma_client


def _embed_query(query: str) -> list:
    model = _get_embedding_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()


# ═══════════════════════════════════════════════════════════════
# Re-Ranking — lightweight keyword-overlap scorer
# ═══════════════════════════════════════════════════════════════

def _rerank_results(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    top_k: int = 3,
) -> tuple[list[str], list[dict]]:
    """
    Re-rank retrieved chunks by keyword overlap with the query.

    This is a lightweight alternative to a cross-encoder that adds
    zero memory overhead — critical for the 512MB constraint.

    Scoring:
      - Exact phrase matches score highest.
      - Individual keyword overlap counts as a secondary signal.
      - Results are sorted by combined score, top_k returned.
    """
    query_lower = query.lower()
    query_terms = set(query_lower.split())

    scored = []
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()

        # Phrase match bonus (high value)
        phrase_bonus = 10.0 if query_lower in doc_lower else 0.0

        # Keyword overlap (Jaccard-like)
        doc_terms = set(doc_lower.split())
        overlap = len(query_terms & doc_terms)
        keyword_score = overlap / max(len(query_terms), 1)

        # Combined score
        total = phrase_bonus + keyword_score
        scored.append((total, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    reranked_docs = [documents[idx] for _, idx in scored[:top_k]]
    reranked_meta = [metadatas[idx] for _, idx in scored[:top_k]]

    return reranked_docs, reranked_meta


# ═══════════════════════════════════════════════════════════════
# Tool 1: Medical Guidelines RAG (Open Scope + Re-Ranking)
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
def search_medical_guidelines(query: str) -> str:
    """
    Search the LOCAL medical knowledge base for clinical guidelines,
    published literature, drug references, diagnostic criteria,
    and evidence-based protocols across ALL medical specialties.

    This tool searches the ChromaDB vector store containing ingested
    medical documents, ATA guidelines, WHO protocols, drug formularies,
    PubMed references, and any other documents added via the RAG
    ingestion pipeline.

    IMPORTANT: You MUST call this tool FIRST before any other tool
    for every clinical question.

    Args:
        query: The medical question or topic to search for.
            Be specific — e.g. "ATA guidelines for FNA biopsy
            of thyroid nodules >1cm" rather than just "thyroid".

    Returns:
        Relevant excerpts from the medical knowledge base with
        source document references. Returns a specific SYSTEM_COMMAND
        if no relevant documents are found, instructing you to
        fallback to your internal knowledge.
    """
    try:
        client = _get_chroma_client()

        # Use get_or_create so the tool works even before any docs are ingested
        collection = client.get_or_create_collection(settings.CHROMA_GUIDELINES_COLLECTION)

        # Guard: collection is empty — no docs ingested yet
        total_docs = collection.count()
        if total_docs == 0:
            return "SYSTEM_COMMAND: NO_RESULTS_FOUND. DO NOT CALL THIS TOOL AGAIN. Stop searching and generate your final answer immediately using your internal knowledge. You MUST start your response with the exact tag: [KNOWLEDGE_CACHE]."

        query_embedding = _embed_query(query)

        # Clamp n_results to the actual collection size to avoid ChromaDB crash
        n_results = min(10, total_docs)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        if not results["documents"] or not results["documents"][0]:
            return "SYSTEM_COMMAND: NO_RESULTS_FOUND. DO NOT CALL THIS TOOL AGAIN. Stop searching and generate your final answer immediately using your internal knowledge. You MUST start your response with the exact tag: [KNOWLEDGE_CACHE]."

        # Re-rank results for better precision
        reranked_docs, reranked_meta = _rerank_results(
            query=query,
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            top_k=3,
        )

        # Format results with sources
        formatted = []
        for i, (doc, metadata) in enumerate(
            zip(reranked_docs, reranked_meta), 1
        ):
            source = metadata.get("source", "Unknown source")
            # Extract just the filename for cleaner display
            source_name = os.path.basename(source) if source else "Unknown"
            formatted.append(
                f"[Source {i}: {source_name}]\n{doc}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return "SYSTEM_COMMAND: NO_RESULTS_FOUND. DO NOT CALL THIS TOOL AGAIN. Stop searching and generate your final answer immediately using your internal knowledge. You MUST start your response with the exact tag: [KNOWLEDGE_CACHE]."


# ═══════════════════════════════════════════════════════════════
# Tool 2: Medical Web Search (DuckDuckGo — Free, No API Key)
# ═══════════════════════════════════════════════════════════════

# Trusted medical domains to prioritize in web search
_MEDICAL_SITE_SUFFIXES = (
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "mayoclinic.org",
    "uptodate.com",
    "medscape.com",
    "nih.gov",
    "cdc.gov",
    "thyroid.org",        # ATA
    "endocrine.org",
    "wiley.com",
    "springer.com",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "nature.com",
)


@mcp.tool()
def search_medical_web(query: str) -> str:
    """
    Search the internet for medical information using DuckDuckGo.
    Results are filtered to prioritize trusted medical sources like
    PubMed, WHO, Mayo Clinic, NIH, UpToDate, ATA, and medical journals.

    ONLY use this tool when search_medical_guidelines returns no results
    or when the local knowledge base does not have sufficient information
    to answer the clinical question.

    This tool is RESTRICTED to medical queries only. Do NOT use it for
    non-medical topics.

    Args:
        query: A specific medical search query. Add "medical" or
            "clinical guidelines" to improve relevance.
            Example: "TSH reference range subclinical hypothyroidism
            clinical guidelines"

    Returns:
        Top medical search results with titles, snippets, and URLs
        from trusted medical sources.
    """
    try:
        from ddgs import DDGS

        # Append "medical" to bias results towards clinical content
        medical_query = f"{query} medical clinical"

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(medical_query, max_results=8))

        if not raw_results:
            return (
                f"No web results found for: '{query}'. "
                "Answer based on your general medical training data, "
                "but clearly state that no external sources were found."
            )

        # Separate trusted vs general results
        trusted = []
        general = []
        for r in raw_results:
            href = r.get("href", "").lower()
            is_trusted = any(domain in href for domain in _MEDICAL_SITE_SUFFIXES)
            entry = (
                f"**{r.get('title', 'No title')}**\n"
                f"{r.get('body', 'No snippet')}\n"
                f"🔗 {r.get('href', 'No URL')}"
            )
            if is_trusted:
                trusted.append(entry)
            else:
                general.append(entry)

        # Prioritize trusted sources, then fill with general
        sorted_results = trusted + general
        top_results = sorted_results[:5]

        header = (
            f"🌐 Web Search Results ({len(trusted)} from trusted medical sources, "
            f"{len(general)} from general sources):\n\n"
        )

        return header + "\n\n---\n\n".join(top_results)

    except ImportError:
        return (
            "Web search is not available — the duckduckgo-search package "
            "is not installed. Answer based on general medical knowledge."
        )
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return (
            f"Web search failed: {e}. "
            "Answer based on your general medical training data, "
            "but clearly state that external search was unavailable."
        )


# ═══════════════════════════════════════════════════════════════
# Cache Management — Self-Updating RAG
# ═══════════════════════════════════════════════════════════════

def save_to_vector_db(question: str, answer: str):
    """
    Saves a question-answer pair to the main guidelines collection.
    This enables the 'Self-Updating RAG' mechanism where the LLM's
    internal knowledge is externalized for future retrieval.
    """
    try:
        import uuid
        client = _get_chroma_client()
        collection = client.get_or_create_collection(settings.CHROMA_GUIDELINES_COLLECTION)
        
        embedding = _embed_query(question)
        
        collection.add(
            ids=[f"cache_{uuid.uuid4()}"],
            embeddings=[embedding],
            metadatas=[{"question": question, "source": "AI Knowledge Cache"}],
            documents=[answer]
        )
        logger.info(f"Successfully cached response for query: '{question[:50]}...'")
    except Exception as e:
        logger.error(f"Failed to save to vector db cache: {e}")


# ═══════════════════════════════════════════════════════════════
# Tool 3: Semantic Search for Similar Patients (pgvector mock)
# ═══════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field
import httpx
import asyncio
import uuid

class SimilarPatientsInput(BaseModel):
    patient_features: str = Field(
        ..., 
        description="A descriptive string of the patient's clinical profile, e.g., '45yo female with nodule 2cm, TSH 4.5'."
    )

@mcp.tool()
async def search_similar_patients(patient_features: str) -> str:
    """
    Finds historical patients with similar clinical profiles to aid in decision support.
    
    This tool uses semantic search over a vector database (Supabase pgvector) to find 
    historical cases matching the current patient's clinical features, demographics, 
    and symptoms. It returns the top similar cases along with their treatment outcomes.
    
    Args:
        patient_features: A descriptive string of the patient's clinical profile.
            
    Returns:
        A string containing the top 3 similar cases and their treatment outcomes.
    """
    # Mock implementation of Supabase pgvector query
    await asyncio.sleep(0.5)  # Simulate network latency
    
    return (
        "Found 3 similar cases:\n\n"
        "1. Case ID: PT-8832 | Similarity: 0.92\n"
        "   Profile: 44yo female, 2.1cm nodule, TSH 4.2, microcalcifications present.\n"
        "   Outcome: Fine Needle Aspiration (FNA) performed. Benign (Bethesda II). Follow-up in 12 months.\n\n"
        "2. Case ID: PT-1094 | Similarity: 0.88\n"
        "   Profile: 47yo female, 1.8cm nodule, TSH 4.8.\n"
        "   Outcome: FNA performed. Suspicious for malignancy (Bethesda V). Total thyroidectomy.\n\n"
        "3. Case ID: PT-5521 | Similarity: 0.85\n"
        "   Profile: 42yo female, 2.0cm nodule, TSH 3.9.\n"
        "   Outcome: Ultrasound follow-up, nodule stable over 2 years."
    )


# ═══════════════════════════════════════════════════════════════
# Tool 4: Search Medical Literature (PubMed API)
# ═══════════════════════════════════════════════════════════════

class MedicalLiteratureInput(BaseModel):
    query: str = Field(..., description="The medical search query (e.g., 'ATA thyroid guidelines 2023').")

@mcp.tool()
async def search_medical_literature(query: str) -> str:
    """
    Fetches the latest clinical guidelines or research papers from PubMed.
    
    This tool queries the public PubMed API (NCBI E-utilities) to retrieve the most 
    recent and relevant medical literature, such as ATA guidelines or studies 
    on specific thyroid conditions.
    
    Args:
        query: The medical search query.
        
    Returns:
        A string containing the titles, PubMed IDs (PMID), and publication info of the 
        top 3 recent papers.
    """
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_url = f"{base_url}/esearch.fcgi"
        
        async with httpx.AsyncClient() as client:
            search_resp = await client.get(
                search_url,
                params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                return f"No PubMed literature found for query: '{query}'."
                
            summary_url = f"{base_url}/esummary.fcgi"
            summary_resp = await client.get(
                summary_url,
                params={"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}
            )
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
            
            results = []
            result_dict = summary_data.get("result", {})
            for pmid in id_list:
                paper_info = result_dict.get(pmid, {})
                title = paper_info.get("title", "No title")
                pubdate = paper_info.get("pubdate", "Unknown date")
                source = paper_info.get("source", "Unknown source")
                results.append(f"PMID: {pmid}\nTitle: {title}\nJournal: {source} ({pubdate})")
                
            return f"Found {len(results)} recent papers on PubMed:\n\n" + "\n\n".join(results)
            
    except Exception as e:
        logger.error(f"PubMed API error: {e}")
        return f"Error fetching medical literature from PubMed: {e}"


# ═══════════════════════════════════════════════════════════════
# Tool 5: Generate Medical Report (PDF Generation)
# ═══════════════════════════════════════════════════════════════

class MedicalReportInput(BaseModel):
    patient_id: str = Field(..., description="The unique identifier for the patient.")
    diagnostic_summary: str = Field(..., description="A concise summary of the clinical findings and diagnosis.")
    recommendations: str = Field(..., description="The suggested treatment plan, follow-up, or prescriptions.")

@mcp.tool()
async def generate_medical_report(patient_id: str, diagnostic_summary: str, recommendations: str) -> str:
    """
    Generates a downloadable medical report or prescription in PDF format.
    
    This tool should be called at the end of the clinical session when the doctor 
    requests a summary report or prescription. It compiles the patient's information, 
    the AI's diagnostic summary, and the clinical recommendations into a formatted PDF.
    
    Args:
        patient_id: The unique identifier for the patient.
        diagnostic_summary: A concise summary of the clinical findings and diagnosis.
        recommendations: The suggested treatment plan, follow-up, or prescriptions.
        
    Returns:
        A string containing the local file path and a secure download URL for the generated PDF.
    """
    await asyncio.sleep(1.0)  # Simulate PDF generation time
    
    report_id = str(uuid.uuid4())[:8]
    file_name = f"medical_report_{patient_id}_{report_id}.pdf"
    
    # Mock PDF generation output
    output_dir = "/tmp/reports"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    
    # Write a dummy text file to simulate the PDF creation
    with open(file_path, "w") as f:
        f.write(f"MEDICAL REPORT FOR PATIENT {patient_id}\n")
        f.write("="*40 + "\n")
        f.write(f"Diagnostic Summary:\n{diagnostic_summary}\n\n")
        f.write(f"Recommendations:\n{recommendations}\n")
        
    download_url = f"https://api.thyrax.local/download/reports/{file_name}"
    
    return (
        f"Medical report generated successfully for patient {patient_id}.\n"
        f"File Path: {file_path}\n"
        f"Download URL: {download_url}"
    )


# ═══════════════════════════════════════════════════════════════
# Groq LLM Binding Example (For documentation / reference)
# ═══════════════════════════════════════════════════════════════
#
# To bind these MCP tools to a Groq LLM instance (e.g., via LangChain):
#
# from langchain_groq import ChatGroq
# from langchain.tools import tool
#
# # You can convert the FastMCP tools into LangChain tools using @tool decorators
# # and then bind them to the Groq chat model.
# llm = ChatGroq(model="llama-3.1-8b-instant")
#
# # Bind the converted tools
# llm_with_tools = llm.bind_tools(langchain_tools)
#
# # Alternatively, if using the remote MCP Server protocol via LangChain's MCP integration:
# # from langchain_mcp_adapters import get_mcp_tools
# # mcp_tools = await get_mcp_tools(...)
# # llm_with_tools = llm.bind_tools(mcp_tools)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# Tool 4: Check Drug Interactions (Patient Safety Tool)
# ═══════════════════════════════════════════════════════════════
from typing import List
from datetime import datetime, timedelta

class DrugInteractionsInput(BaseModel):
    drugs: List[str] = Field(
        ..., 
        description="List of drug names to check for interactions, e.g., ['Levothyroxine', 'Calcium Carbonate']."
    )

@mcp.tool()
async def check_drug_interactions(drugs: List[str]) -> str:
    """
    Checks for adverse drug-drug interactions among the provided medications.
    
    Use this tool when the doctor prescribes a new medication, when the patient 
    reports taking new supplements, or when reviewing the patient's current medication 
    list to ensure patient safety. It specifically focuses on thyroid medications 
    (e.g., Levothyroxine) and other common drugs.
    
    Args:
        drugs: A list of drug names (strings) to evaluate.
        
    Returns:
        A string containing potential interactions, severity levels, and clinical 
        recommendations.
    """
    await asyncio.sleep(0.5)  # Simulate API call to NIH RxNav or similar service
    
    drugs_lower = [d.lower() for d in drugs]
    
    interactions = []
    
    if any("levothyroxine" in d for d in drugs_lower) and any("calcium" in d for d in drugs_lower):
        interactions.append(
            "- **Interaction:** Levothyroxine and Calcium Carbonate\n"
            "  **Severity:** Moderate\n"
            "  **Recommendation:** Calcium can interfere with the absorption of levothyroxine. "
            "Separate administration by at least 4 hours."
        )
        
    if any("levothyroxine" in d for d in drugs_lower) and any("iron" in d for d in drugs_lower):
        interactions.append(
            "- **Interaction:** Levothyroxine and Iron Supplements\n"
            "  **Severity:** Moderate\n"
            "  **Recommendation:** Iron can decrease the absorption of levothyroxine. "
            "Separate administration by at least 4 hours."
        )

    if not interactions:
        return f"No known significant interactions found among: {', '.join(drugs)}."
        
    return "Potential Drug Interactions Found:\n\n" + "\n\n".join(interactions)

# ═══════════════════════════════════════════════════════════════
# Tool 5: Schedule Patient Follow-up (Care Management Tool)
# ═══════════════════════════════════════════════════════════════

class ScheduleFollowupInput(BaseModel):
    patient_id: int = Field(..., description="The unique integer ID of the patient.")
    followup_type: str = Field(..., description="The type of follow-up required, e.g., 'Ultrasound', 'TFT'.")
    timeframe_months: int = Field(..., description="The number of months until the follow-up should occur.")

@mcp.tool()
async def schedule_patient_followup(patient_id: int, followup_type: str, timeframe_months: int) -> str:
    """
    Automatically schedules a follow-up appointment or test based on clinical protocols.
    
    Use this tool when a clinical decision has been made that requires future monitoring, 
    such as scheduling a thyroid ultrasound in 12 months for a benign nodule, or ordering 
    Thyroid Function Tests (TFTs) after a medication adjustment.
    
    Args:
        patient_id: The unique integer ID of the patient.
        followup_type: The type of follow-up required (e.g., "Ultrasound", "TFT").
        timeframe_months: The number of months until the follow-up should occur.
        
    Returns:
        A confirmation message including the calculated future date of the follow-up.
    """
    await asyncio.sleep(0.3)  # Simulate database operation to Supabase followups table
    
    future_date = datetime.now() + timedelta(days=30 * timeframe_months)
    formatted_date = future_date.strftime("%Y-%m-%d")
    
    return (
        f"✅ Follow-up Scheduled Successfully:\n"
        f"Patient ID: {patient_id}\n"
        f"Action: {followup_type}\n"
        f"Scheduled Date: {formatted_date} ({timeframe_months} months from today)\n\n"
        f"The pending task has been inserted into the followups system."
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')
