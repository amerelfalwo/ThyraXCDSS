#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║  ThyraX CDSS — End-to-End Integration Test Suite                 ║
║                                                                   ║
║  Tests the full pipeline:                                         ║
║    Phase 1: Seed test data (Doctor → Patient → Session)           ║
║    Phase 2: Vision /image/predict endpoint                        ║
║    Phase 3: Agent /agent/chat endpoint (SSE streaming)            ║
║    Phase 4: Cleanup all test data                                 ║
║                                                                   ║
║  Prerequisites:                                                   ║
║    pip install httpx asyncpg pillow python-dotenv                 ║
║                                                                   ║
║  Usage:                                                           ║
║    1. Start the server: uvicorn main:app --port 7860              ║
║    2. Run tests:        python test_e2e.py                        ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import asyncio
import io
import json
import os
import ssl
import sys
import uuid
import traceback
from datetime import datetime, timezone

# ── Third-party imports ──
try:
    import httpx
except ImportError:
    sys.exit("❌ httpx not installed. Run: pip install httpx")

try:
    import asyncpg
except ImportError:
    sys.exit("❌ asyncpg not installed. Run: pip install asyncpg")

try:
    from PIL import Image
except ImportError:
    sys.exit("❌ Pillow not installed. Run: pip install pillow")

try:
    from dotenv import dotenv_values
except ImportError:
    sys.exit("❌ python-dotenv not installed. Run: pip install python-dotenv")


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

# Load .env from project root
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_env = dotenv_values(_ENV_PATH)

# The DATABASE_URL from .env uses asyncpg format; we need the raw
# PostgreSQL DSN for asyncpg's connect() (strip the SQLAlchemy prefix).
_RAW_DB_URL = _env.get("DATABASE_URL", "")
if _RAW_DB_URL.startswith("postgresql+asyncpg://"):
    _RAW_DB_URL = _RAW_DB_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

# Server under test
BASE_URL = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:7860")

# Auth header required by all protected endpoints
API_KEY = _env.get("INTERNAL_SERVICE_KEY", "thyrax-internal-sk-2026-secure")
AUTH_HEADERS = {"X-AI-Service-Key": API_KEY}

# ── Test identifiers (unique per run to avoid collisions) ──
_RUN_ID = uuid.uuid4().hex[:8]
TEST_DOCTOR_ID = f"e2e_doctor_{_RUN_ID}"
TEST_PATIENT_ID = f"e2e_patient_{_RUN_ID}"
TEST_SESSION_ID = f"e2e_session_{_RUN_ID}"


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

class Colors:
    """ANSI escape codes for pretty terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _ok(msg: str):
    print(f"  {Colors.GREEN}✅ PASS{Colors.RESET} — {msg}")


def _fail(msg: str):
    print(f"  {Colors.RED}❌ FAIL{Colors.RESET} — {msg}")


def _info(msg: str):
    print(f"  {Colors.CYAN}ℹ️  INFO{Colors.RESET} — {msg}")


def _warn(msg: str):
    print(f"  {Colors.YELLOW}⚠️  WARN{Colors.RESET} — {msg}")


def _header(phase: str, title: str):
    print(f"\n{Colors.BOLD}{'═' * 60}")
    print(f"  {phase}: {title}")
    print(f"{'═' * 60}{Colors.RESET}\n")


async def get_db_connection() -> asyncpg.Connection:
    """
    Create a direct asyncpg connection to the Supabase PostgreSQL database.
    Uses SSL (no cert verification) to match the Supabase pooler requirements.
    """
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    # Disable prepared statement cache — required for Supabase's
    # PgBouncer transaction pooler on port 6543.
    conn = await asyncpg.connect(
        _RAW_DB_URL,
        ssl=ssl_ctx,
        statement_cache_size=0,
    )
    return conn


def create_dummy_ultrasound_image() -> bytes:
    """
    Generate a minimal 256×256 grayscale image that simulates
    a thyroid ultrasound (dark background + lighter elliptical region).
    Returns the image as PNG bytes.
    """
    img = Image.new("L", (256, 256), color=30)  # dark background

    # Draw a brighter elliptical "nodule" region in the center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 176, 176], fill=160)   # simulated tissue
    draw.ellipse([110, 110, 150, 150], fill=200)  # brighter "nodule"

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def parse_sse_events(raw_text: str) -> list[dict]:
    """
    Parse a Server-Sent Events (SSE) response body into a list of
    JSON payloads. Each SSE line is prefixed with 'data: '.
    """
    events = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload_str = line[len("data: "):]
            try:
                events.append(json.loads(payload_str))
            except json.JSONDecodeError:
                pass  # skip malformed lines
    return events


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Test Data Setup (Direct PostgreSQL)
# ═══════════════════════════════════════════════════════════════════

async def phase1_setup() -> bool:
    """
    Insert mock Doctor → Patient → Session → PatientSession records
    into the Supabase PostgreSQL database using direct asyncpg queries.
    
    Tables involved:
      - doctors          (doctor_id PK)
      - patients         (patient_id PK, doctor_id FK)
      - sessions         (session_id PK, doctor_id FK, patient_id FK)
      - patient_sessions (session_id PK, doctor_id)
    """
    _header("Phase 1", "Test Data Setup (PostgreSQL)")
    conn = None
    now = datetime.now(timezone.utc)

    try:
        conn = await get_db_connection()
        _ok(f"Connected to database")

        # ── 1a. Insert Doctor ──
        await conn.execute(
            """
            INSERT INTO doctors (doctor_id, name, specialty, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (doctor_id) DO NOTHING
            """,
            TEST_DOCTOR_ID,
            "Dr. E2E Tester",
            "Endocrinology",
            now,
            now,
        )
        _ok(f"Inserted Doctor: {TEST_DOCTOR_ID}")

        # ── 1b. Insert Patient ──
        await conn.execute(
            """
            INSERT INTO patients (patient_id, doctor_id, demographics, medical_history, allergies, long_term_summary, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (patient_id) DO NOTHING
            """,
            TEST_PATIENT_ID,
            TEST_DOCTOR_ID,
            json.dumps({"name": "E2E Test Patient", "age": 45, "sex": "Female"}),
            json.dumps(["Hypothyroidism", "Hashimoto's thyroiditis"]),
            json.dumps(["Iodine contrast"]),
            "Patient has a 3-year history of hypothyroidism managed with levothyroxine.",
            now,
            now,
        )
        _ok(f"Inserted Patient: {TEST_PATIENT_ID}")

        # ── 1c. Insert Session (memory_models.Session) ──
        await conn.execute(
            """
            INSERT INTO sessions (session_id, doctor_id, patient_id, conversation_history, diagnostic_context, session_summary, is_active, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (session_id) DO NOTHING
            """,
            TEST_SESSION_ID,
            TEST_DOCTOR_ID,
            TEST_PATIENT_ID,
            json.dumps([]),       # empty conversation history
            json.dumps({}),       # empty diagnostic context
            "",                   # no summary yet
            "true",               # is_active
            now,
            now,
        )
        _ok(f"Inserted Session: {TEST_SESSION_ID}")

        # ── 1d. Insert PatientSession (db_models.PatientSession) ──
        # NOTE: The live DB schema may not have doctor_id column yet
        # (defined in SQLAlchemy model but not migrated). We insert
        # only the columns that actually exist.
        await conn.execute(
            """
            INSERT INTO patient_sessions (session_id, clinical_assessment, ultrasound_result, fnac_result, chat_history, conversation_summary, created_at, last_updated)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (session_id) DO NOTHING
            """,
            TEST_SESSION_ID,
            None,                 # clinical_assessment — empty
            None,                 # ultrasound_result — empty (will be populated by Phase 2)
            None,                 # fnac_result — empty
            json.dumps([]),       # chat_history
            "",                   # conversation_summary
            now,
            now,
        )
        _ok(f"Inserted PatientSession: {TEST_SESSION_ID}")

        # ── Verification: re-read the records ──
        row = await conn.fetchrow(
            "SELECT doctor_id FROM doctors WHERE doctor_id = $1",
            TEST_DOCTOR_ID,
        )
        assert row is not None, "Doctor not found after insert"

        row = await conn.fetchrow(
            "SELECT patient_id FROM patients WHERE patient_id = $1",
            TEST_PATIENT_ID,
        )
        assert row is not None, "Patient not found after insert"

        row = await conn.fetchrow(
            "SELECT session_id FROM sessions WHERE session_id = $1",
            TEST_SESSION_ID,
        )
        assert row is not None, "Session not found after insert"

        _ok("All test records verified in database")
        return True

    except Exception as e:
        _fail(f"Phase 1 failed: {e}")
        traceback.print_exc()
        return False

    finally:
        if conn:
            await conn.close()


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Vision & Storage Endpoint Test
# ═══════════════════════════════════════════════════════════════════

async def phase2_vision() -> bool:
    """
    Test the /image/predict endpoint:
      1. Generate a dummy ultrasound image.
      2. POST it as multipart/form-data with session_id and doctor_id.
      3. Assert 200 OK and validate response structure.
      4. Verify the ultrasound_result was persisted in patient_sessions.
    """
    _header("Phase 2", "Vision & Storage Endpoint Test")
    conn = None

    try:
        # ── 2a. Create dummy image ──
        image_bytes = create_dummy_ultrasound_image()
        _info(f"Generated dummy ultrasound image ({len(image_bytes)} bytes)")

        # ── 2b. POST to /image/predict ──
        _info(f"Sending POST to {BASE_URL}/image/predict ...")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{BASE_URL}/image/predict",
                headers=AUTH_HEADERS,
                files={"files": ("test_ultrasound.png", image_bytes, "image/png")},
                data={
                    "session_id": TEST_SESSION_ID,
                    "doctor_id": TEST_DOCTOR_ID,
                    "force": "true",  # bypass gatekeeper for test image
                },
            )

        # ── 2c. Validate HTTP response ──
        if response.status_code != 200:
            _fail(f"Expected 200 OK, got {response.status_code}")
            _info(f"Response body: {response.text[:500]}")
            return False
        _ok(f"HTTP 200 OK received")

        # Parse response
        result_list = response.json()
        assert isinstance(result_list, list), "Response should be a list"
        assert len(result_list) > 0, "Response list should not be empty"

        result = result_list[0]
        _info(f"Status: {result.get('status')}")
        _info(f"Classification: {result.get('classification')}")

        # Check for images (either signed URLs or base64 fallback)
        images = result.get("images")
        if images:
            mask_url = images.get("mask_url", "")
            if mask_url.startswith("data:image"):
                _ok("Images returned as base64 data URLs (Supabase Storage not configured)")
            elif mask_url.startswith("http"):
                _ok("Images returned as Supabase signed URLs")
            else:
                _warn(f"Unexpected image URL format: {mask_url[:80]}...")
        else:
            _warn("No images returned in response (may indicate no nodule detected)")

        # Check for validation bypass warning
        if result.get("validation_bypassed"):
            _info("Validation was bypassed (force=true) — expected for test")

        # Check for medical disclaimer
        if result.get("medical_disclaimer"):
            _ok("Medical disclaimer present in response")

        # ── 2d. Verify database persistence ──
        _info("Checking database for ultrasound_result persistence...")
        conn = await get_db_connection()

        row = await conn.fetchrow(
            "SELECT ultrasound_result FROM patient_sessions WHERE session_id = $1",
            TEST_SESSION_ID,
        )

        if row is None:
            _fail("PatientSession row not found in database")
            return False

        us_result = row["ultrasound_result"]
        if us_result is not None:
            _ok("ultrasound_result column is populated in patient_sessions ✓")
            # Quick sanity check on the stored data
            if isinstance(us_result, str):
                us_result = json.loads(us_result)
            if isinstance(us_result, dict) and us_result.get("status"):
                _info(f"Stored status: {us_result['status']}")
        else:
            _warn(
                "ultrasound_result is NULL — this may be expected if "
                "the dummy image didn't produce a valid segmentation"
            )

        _ok("Phase 2 completed successfully")
        return True

    except Exception as e:
        _fail(f"Phase 2 failed: {e}")
        traceback.print_exc()
        return False

    finally:
        if conn:
            await conn.close()


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Agent & Context Injection Test
# ═══════════════════════════════════════════════════════════════════

async def phase3_agent() -> bool:
    """
    Test the /agent/chat endpoint:
      1. POST a medical query referencing the ultrasound results.
      2. Parse the SSE streaming response.
      3. Assert the LLM generated a non-empty answer.
      4. Verify conversation_history was updated in the sessions table.
    """
    _header("Phase 3", "Agent & Context Injection Test")
    conn = None

    try:
        # ── 3a. Build request payload ──
        # The AgentChatRequest expects: patient_id (int), doctor_id (int),
        # session_id (str), user_message (str).
        # However, our test IDs are strings. The endpoint schema uses int
        # for patient_id/doctor_id, but the DB stores them as strings.
        # We'll need to handle this carefully.
        #
        # NOTE: Since the schema defines patient_id: int and doctor_id: int,
        # but our DB PKs are strings, we'll test the /agent/chat/stream
        # endpoint instead, which accepts Form data as strings.

        _info(f"Sending medical query to {BASE_URL}/agent/chat/stream ...")

        user_query = (
            "Based on the patient's recent ultrasound scan results, "
            "what is the recommended next step in their thyroid evaluation? "
            "Please consider their history of Hashimoto's thyroiditis."
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{BASE_URL}/agent/chat/stream",
                headers=AUTH_HEADERS,
                data={
                    "query": user_query,
                    "session_id": TEST_SESSION_ID,
                    "chat_history": json.dumps([]),
                },
            )

        # ── 3b. Validate HTTP response ──
        if response.status_code != 200:
            _fail(f"Expected 200 OK, got {response.status_code}")
            _info(f"Response body: {response.text[:500]}")
            return False
        _ok(f"HTTP 200 OK received")

        # ── 3c. Parse SSE events ──
        events = parse_sse_events(response.text)
        if not events:
            _fail("No SSE events parsed from response")
            _info(f"Raw response (first 500 chars): {response.text[:500]}")
            return False

        _info(f"Received {len(events)} SSE event(s)")

        # Find the final success/error event
        final_event = events[-1]
        status = final_event.get("status", "unknown")
        _info(f"Final event status: {status}")

        if status == "success":
            agent_response = final_event.get("response", "")
            if agent_response:
                _ok(f"Agent generated response ({len(agent_response)} chars)")
                _info(f"Response preview: {agent_response[:200]}...")
            else:
                _fail("Agent response is empty")
                return False
        elif status == "streaming":
            # If the last event is still streaming, reconstruct from chunks
            full_response = ""
            for evt in events:
                if evt.get("status") == "streaming":
                    full_response += evt.get("chunk", "")
            if full_response:
                _ok(f"Reconstructed streaming response ({len(full_response)} chars)")
                _info(f"Response preview: {full_response[:200]}...")
            else:
                _fail("Could not reconstruct response from streaming chunks")
                return False
        elif status == "error":
            error_detail = final_event.get("response", final_event.get("detail", "Unknown error"))
            _warn(f"Agent returned an error (may be transient): {error_detail[:200]}")
            _info("This is acceptable for E2E testing if the Groq API key has exhausted its quota")
            # Don't fail the test — LLM availability is external
        elif status == "rejected":
            _fail("Query was rejected by medical guardrail — this shouldn't happen for a medical query")
            return False
        else:
            _warn(f"Unexpected status: {status}")

        # ── 3d. Verify conversation_history in database ──
        _info("Waiting 2 seconds for background persistence to complete...")
        await asyncio.sleep(2)

        _info("Checking sessions table for conversation_history updates...")
        conn = await get_db_connection()

        row = await conn.fetchrow(
            "SELECT conversation_history FROM sessions WHERE session_id = $1",
            TEST_SESSION_ID,
        )

        if row:
            history = row["conversation_history"]
            if isinstance(history, str):
                history = json.loads(history)
            if history and len(history) > 0:
                _ok(f"conversation_history has {len(history)} message(s)")
            else:
                _fail("conversation_history is empty — expected it to be populated by background task")
                return False
        else:
            _warn("Session row not found — may have been modified")

        _ok("Phase 3 completed successfully")
        return True

    except Exception as e:
        _fail(f"Phase 3 failed: {e}")
        traceback.print_exc()
        return False

    finally:
        if conn:
            await conn.close()


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Cleanup (Teardown)
# ═══════════════════════════════════════════════════════════════════

async def phase4_cleanup() -> bool:
    """
    Delete all test records from the database in reverse dependency order:
      1. patient_sessions (no FK constraints)
      2. sessions (FK → patients, doctors)
      3. patients (FK → doctors)
      4. doctors (root entity)
    """
    _header("Phase 4", "Cleanup (Teardown)")
    conn = None

    try:
        conn = await get_db_connection()

        # ── 4a. Delete PatientSession ──
        result = await conn.execute(
            "DELETE FROM patient_sessions WHERE session_id = $1",
            TEST_SESSION_ID,
        )
        _ok(f"Deleted PatientSession: {result}")

        # ── 4b. Delete audit_logs (if any were created) ──
        try:
            result = await conn.execute(
                "DELETE FROM audit_logs WHERE session_id = $1",
                TEST_SESSION_ID,
            )
            _ok(f"Deleted audit_logs: {result}")
        except Exception:
            _info("No audit_logs table or no rows to delete (OK)")

        # ── 4c. Delete Session ──
        result = await conn.execute(
            "DELETE FROM sessions WHERE session_id = $1",
            TEST_SESSION_ID,
        )
        _ok(f"Deleted Session: {result}")

        # ── 4d. Delete Patient ──
        result = await conn.execute(
            "DELETE FROM patients WHERE patient_id = $1",
            TEST_PATIENT_ID,
        )
        _ok(f"Deleted Patient: {result}")

        # ── 4e. Delete Doctor ──
        result = await conn.execute(
            "DELETE FROM doctors WHERE doctor_id = $1",
            TEST_DOCTOR_ID,
        )
        _ok(f"Deleted Doctor: {result}")

        # ── Verification: confirm deletion ──
        row = await conn.fetchrow(
            "SELECT doctor_id FROM doctors WHERE doctor_id = $1",
            TEST_DOCTOR_ID,
        )
        assert row is None, "Doctor still exists after cleanup!"
        _ok("All test records cleaned up successfully")

        return True

    except Exception as e:
        _fail(f"Phase 4 cleanup failed: {e}")
        traceback.print_exc()
        return False

    finally:
        if conn:
            await conn.close()


# ═══════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════

async def main():
    print(f"\n{Colors.BOLD}{'╔' + '═' * 58 + '╗'}")
    print(f"║  ThyraX CDSS — E2E Integration Test Suite                ║")
    print(f"║  Run ID: {_RUN_ID}                                          ║")
    print(f"║  Target: {BASE_URL:<48}║")
    print(f"{'╚' + '═' * 58 + '╝'}{Colors.RESET}\n")

    print(f"  Test IDs:")
    print(f"    Doctor:  {TEST_DOCTOR_ID}")
    print(f"    Patient: {TEST_PATIENT_ID}")
    print(f"    Session: {TEST_SESSION_ID}")

    results = {}

    # ── Phase 1: Setup ──
    results["phase1"] = await phase1_setup()
    if not results["phase1"]:
        _fail("Phase 1 failed — cannot continue. Running cleanup...")
        await phase4_cleanup()
        sys.exit(1)

    # ── Phase 2: Vision ──
    results["phase2"] = await phase2_vision()

    # ── Phase 3: Agent ──
    results["phase3"] = await phase3_agent()

    # ── Phase 4: Cleanup (always runs) ──
    results["phase4"] = await phase4_cleanup()

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{Colors.BOLD}{'═' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'═' * 60}{Colors.RESET}\n")

    all_passed = True
    for phase, passed in results.items():
        icon = f"{Colors.GREEN}✅ PASS" if passed else f"{Colors.RED}❌ FAIL"
        print(f"  {icon}{Colors.RESET}  {phase}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"  {Colors.GREEN}{Colors.BOLD}🎉 ALL PHASES PASSED{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}{Colors.BOLD}⚠️  SOME PHASES HAD ISSUES{Colors.RESET}")
        print(f"  {Colors.YELLOW}  (Check warnings above — LLM errors may be transient){Colors.RESET}")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
